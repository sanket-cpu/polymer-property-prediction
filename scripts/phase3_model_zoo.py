"""
Phase 3 -- Model zoo (v4: + Gasteiger charges + target transform + selective Mol2Vec).

Mol2Vec update: a 100-dim skip-gram embedding trained fresh each run on a
150K PI1M SMILES sample (ported from phase2_pi1m.py's isolated ablation --
ei/eea/eps/nc/egb are the small targets where sample size limits everything
else, and PI1M lets an embedding learn substructure semantics from ~1M
unlabeled polymers instead of the ~250-row labeled slice). That ablation
showed a real but *target-dependent* effect (egb +0.020, ei +0.025, eps
-0.023, nc -0.007, rest flat), so it's wired in as opt-in per target
(MOL2VEC_TARGETS = {egb, ei}, gated via target_feature_cols()) rather than
folded into the global feature set the way MACCS was -- unlike MACCS, this
one actively hurt two targets, so making it universal would trade a real
gain on 2 targets for a real loss on 2 others.

Feature set (build_feature_matrix()): prajwal's original featurize()
output (RDKit descriptors + physics ratios + dimer-delta + 256-bit/
radius-2 Morgan) + 167-bit MACCS keys + Gasteiger partial-charge summary
stats, pruned with fit_feature_pruner. Revision note on MACCS: Phase 2's
original ablation rejected it because no single target's improvement
individually cleared the CV noise floor -- but it improved *all 7 of 7*
targets over the 256-bit-only baseline, the same "every target moves the
same direction, even if none individually clears its own noise band"
pattern that later made stacking's Phase 4 win convincing. In hindsight
the noise-floor-only bar was too strict for a change that's directionally
consistent across independent targets, so it's folded in as a real part
of the feature set. Gasteiger charges are new this round, targeting eps
(dielectric constant) and ei (ionization energy) specifically -- both are
polarity/charge-driven properties the existing mostly-topological
descriptor set doesn't directly capture.

Target transform: eps and ei are also the two most right-skewed targets
(Phase 0 skew 1.22 and 0.79) and, after the MACCS+wider-search round, the
two weakest CV scores -- neither has had a transform tried. Every model
factory in this file is now built via cv_harness.wrap_for_target(), which
applies a log/exp TransformedTargetRegressor wrapper for eps/ei and is a
no-op for the other 5 targets. This makes the transform genuinely load-
bearing for the tuning search too, not just the final scoring: Optuna
now searches hyperparameters *for the wrapped (transformed-space) model*
on eps/ei, since that's what actually gets used downstream -- searching
on the unwrapped model would tune the wrong objective.

build_feature_matrix() and the model-factory functions below are also
imported directly by final_pipeline.py, so the two scripts are
structurally unable to drift onto different feature sets or transform
behavior (unlike the earlier PHASE3_BEST-hardcoding disconnect, which had
to be fixed after the fact).

Zoo (8 models):
  Linear   : Ridge, ElasticNet -- each wrapped in a Pipeline with
             SelectKBest(f_regression, k=SELECT_K) ahead of the model, so
             feature selection is refit on the training fold only (no
             leakage). k=100 is a heuristic cap, not tuned -- chosen to
             sit well below the smallest small-target training fold
             (~176-184 rows), since the mostly-binary, correlated
             descriptor+fingerprint space here would otherwise let a
             linear model overfit on a small fold. Optuna tuning is
             scoped to the boosting models per the brief, not to k.
  Tree     : RandomForest, GradientBoosting, HistGradientBoosting --
             prajwal's original 3, unchanged. RF/GBM need imputation
             (no native NaN support); HGB doesn't.
  Boosting : XGBoost, CatBoost, LightGBM. All three handle NaN natively,
             no imputer. Each gets an Optuna search (N_OPTUNA_TRIALS
             trials, 3-fold GroupKFold via get_search_splits) per
             target_type; the tuned config only replaces the default if
             it beats the default's score on the *real* Phase 1 harness
             (get_harness_splits) by more than the noise floor (larger of
             the two configs' fold-to-fold std) -- same accept/reject
             rule used for the Phase 2 feature-variant decisions, so a
             config that just got lucky on the 3-fold search doesn't get
             promoted. Trial count raised from 15 to 50 and the search
             ranges widened (deeper trees, lower learning rates, stronger
             regularization all now reachable) versus the original run,
             since only 1 of 21 combos won with the narrower budget --
             either defaults were already good, or the search couldn't
             find better; a wider budget distinguishes the two.

Every model is scored via cv_harness.score_model, so every number here
is already leakage-safe (grouped on canonical SMILES) and, for the 5
small targets, averaged over 5-fold x3-seed repeated CV.
"""

import json
import os
import sys
import time
from pathlib import Path

# Reverted from an 8-core cap back to unrestricted (-1) -- the cap was
# meant to keep the laptop cool/battery-friendly during iteration, but
# CatBoost in particular (its split-search parallelizes well across cores)
# was taking a large hit from it once the search space got wider, so
# priority flipped to wall-clock speed for this run. -1 tells sklearn/
# XGBoost/LightGBM/CatBoost to use every core directly; unlike the model-
# level n_jobs/thread_count params, the OMP/BLAS env vars this used to set
# don't accept -1 (they want a literal thread count), so removing them
# entirely -- rather than setting them to -1 -- is what actually restores
# full-core behavior for HistGradientBoostingRegressor (which has no
# n_jobs param and only listens to those env vars).
N_JOBS = -1

import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdPartialCharges
from gensim.models import Word2Vec

RDLogger.DisableLog('rdApp.*')

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from prajwal_baseline import featurize, fit_feature_pruner, _parse_mol  # noqa: E402
from cv_harness import (  # noqa: E402
    load_train_with_groups, get_harness_splits, get_search_splits, score_model,
    wrap_for_target,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
SELECT_K = 100
N_OPTUNA_TRIALS = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PI1M_PATH = PROJECT_ROOT / "data" / "PI1M.csv"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "phase3_results.json"

# Mol2Vec-style embedding, ported from phase2_pi1m.py's isolated ablation
# (see that file's docstring for the full rules-compliance + method
# rationale: skip-gram trained fresh each run on PI1M's unlabeled SMILES,
# no property labels touched, nothing pretrained uploaded). That ablation
# found a target-dependent effect -- helps egb/ei, hurts eps/nc, wash on
# the rest -- so unlike MACCS/Gasteiger this is NOT folded into the global
# feature set; it's opt-in per target via MOL2VEC_TARGETS/target_feature_cols
# below, on the theory that 100 extra embedding dims are more likely to be
# noise a 220-row target overfits on than signal, for targets where the
# isolated test didn't show a benefit.
MOL2VEC_SAMPLE_N = 150_000
MOL2VEC_DIM = 100
MOL2VEC_RADIUS = 1
MOL2VEC_WINDOW = 10
MOL2VEC_MIN_COUNT = 3
MOL2VEC_EPOCHS = 10
MOL2VEC_TARGETS = {'egb', 'ei'}


def mol_to_sentence(mol, radius=MOL2VEC_RADIUS):
    """Mol2Vec 'alternating sentence': for each atom, walk radius 0..R and
    emit that atom's Morgan-environment identifier at each radius."""
    radii = list(range(radius + 1))
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    dict_atoms = {a.GetIdx(): {r: None for r in radii} for a in mol.GetAtoms()}
    for identifier, envs in info.items():
        for atom_idx, r in envs:
            if atom_idx in dict_atoms and r in dict_atoms[atom_idx]:
                dict_atoms[atom_idx][r] = identifier

    sentence = []
    for atom_idx in dict_atoms:
        for r in radii:
            ident = dict_atoms[atom_idx][r]
            if ident is not None:
                sentence.append(str(ident))
    return sentence


def load_pi1m_sample(n, seed):
    pi1m = pd.read_csv(PI1M_PATH)
    pi1m.columns = [c.strip() for c in pi1m.columns]
    smiles_col = 'SMILES' if 'SMILES' in pi1m.columns else pi1m.columns[0]
    sample = pi1m[smiles_col].dropna().sample(n=min(n, len(pi1m)), random_state=seed)
    return sample.reset_index(drop=True)


def train_mol2vec(smiles_sample):
    mols = smiles_sample.apply(_parse_mol).dropna()
    sentences = [mol_to_sentence(m) for m in mols]
    sentences = [s for s in sentences if len(s) > 0]
    model = Word2Vec(
        sentences, vector_size=MOL2VEC_DIM, window=MOL2VEC_WINDOW,
        min_count=MOL2VEC_MIN_COUNT, sg=1, workers=8, epochs=MOL2VEC_EPOCHS,
        seed=RANDOM_STATE,
    )
    return model


def mol2vec_embed(model, mol):
    tokens = mol_to_sentence(mol)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(MOL2VEC_DIM)
    return np.mean(vecs, axis=0)


def embed_dataframe(model, mols):
    embeds = np.stack([mol2vec_embed(model, m) for m in mols])
    return pd.DataFrame(embeds, columns=[f'mol2vec_{i}' for i in range(MOL2VEC_DIM)])


def target_feature_cols(feature_cols, target_type):
    """Column subset for one target: full set for egb/ei (Mol2Vec opted in),
    everything except the mol2vec_* block for the other 5. Both this file's
    main() and final_pipeline.py must call this at every per-target model
    fit/score site rather than using `feature_cols`/`X` directly -- passing
    the full mol2vec-inclusive matrix to e.g. eps or nc would reproduce the
    Phase 2 regression that motivated making this opt-in."""
    if target_type in MOL2VEC_TARGETS:
        return feature_cols
    return [c for c in feature_cols if not c.startswith('mol2vec_')]


def maccs_keys(mol):
    """167-bit MACCS keys for one molecule -- see phase2_fingerprints.py
    for the original isolated ablation this is folded in from."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.int8)
    for bit in fp.GetOnBits():
        if bit < 167:
            arr[bit] = 1
    return {f'maccs_{i}': int(arr[i]) for i in range(167)}


def gasteiger_features(mol):
    """
    Gasteiger partial charges, per atom, summarized (max/min/mean/sum/std).
    Targets eps (dielectric constant) and ei (ionization energy)
    specifically -- both are driven by molecular polarity/charge
    distribution, which the existing descriptor set (mostly topology- and
    counting-based) doesn't directly capture. RDKit-native, no QM engine,
    no new dependency -- cheap relative to a real electronic-structure
    calculation, at the cost of being a much rougher approximation.
    """
    mol = Chem.Mol(mol)  # ComputeGasteigerCharges mutates in place
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    charges = charges[np.isfinite(charges)]  # rare structures can produce nan/inf
    if len(charges) == 0:
        return {'gast_max': 0.0, 'gast_min': 0.0, 'gast_mean': 0.0,
                'gast_sum': 0.0, 'gast_std': 0.0}
    return {
        'gast_max': float(charges.max()), 'gast_min': float(charges.min()),
        'gast_mean': float(charges.mean()), 'gast_sum': float(charges.sum()),
        'gast_std': float(charges.std()),
    }


def build_feature_matrix(df, mol2vec_model):
    """
    Single source of truth for the feature set -- baseline (prajwal's
    featurize()) + MACCS + Gasteiger + Mol2Vec, pruned. Both this script and
    final_pipeline.py call this, so the two are structurally unable to
    drift onto different feature sets (which is exactly what happened
    with MACCS before build_feature_matrix existed: two independent
    hand-written copies of "concat these blocks, then prune").

    mol2vec_model is trained once by the caller (train_mol2vec) and passed
    in rather than trained inside here, since it's expensive-ish (~40s) and
    both callers only need one instance per run. The returned feature_cols
    includes the mol2vec_* columns unconditionally -- callers must filter
    per target via target_feature_cols() before fitting/scoring a model,
    since Mol2Vec only helps egb/ei (see MOL2VEC_TARGETS above).

    Takes a df with a 'smiles' column (any row count/order), handles the
    featurize()-can-return-None filtering itself, and returns
    (X, feature_cols, valid_df) -- valid_df is `df` reduced to the rows
    that featurized successfully, index reset to align with X.
    """
    raw_feats = df['smiles'].apply(featurize)
    valid_mask = raw_feats.notna()
    if (~valid_mask).sum():
        print(f"  dropping {(~valid_mask).sum()} rows that failed to featurize")
    valid_df = df[valid_mask].reset_index(drop=True)
    raw_baseline_df = pd.DataFrame(list(raw_feats[valid_mask])).reset_index(drop=True)

    mols = valid_df['smiles'].apply(_parse_mol)
    maccs_df = pd.DataFrame([maccs_keys(m) for m in mols])
    gast_df = pd.DataFrame([gasteiger_features(m) for m in mols])
    m2v_df = embed_dataframe(mol2vec_model, mols)

    raw_df = pd.concat([raw_baseline_df, maccs_df, gast_df, m2v_df], axis=1)
    feature_cols = fit_feature_pruner(raw_df)
    X = raw_df[feature_cols]
    return X, feature_cols, valid_df


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def linear_models():
    return {
        'Ridge': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('kbest', SelectKBest(f_regression, k=SELECT_K)),
            ('m', Ridge(alpha=5.0, random_state=RANDOM_STATE)),
        ]),
        'ElasticNet': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('kbest', SelectKBest(f_regression, k=SELECT_K)),
            ('m', ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=5000)),
        ]),
    }


def tree_models():
    return {
        'RF': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ]),
        'GBM': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=RANDOM_STATE)),
        ]),
        'HGB': HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.06,
            l2_regularization=0.1, random_state=RANDOM_STATE),
    }


def boosting_default_params():
    return {
        'XGB': dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0),
        'CatBoost': dict(iterations=300, depth=6, learning_rate=0.05,
                          l2_leaf_reg=3.0, random_state=RANDOM_STATE, verbose=False,
                          thread_count=N_JOBS),
        'LightGBM': dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.8,
                          random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1),
    }


BOOSTING_CTORS = {'XGB': XGBRegressor, 'CatBoost': CatBoostRegressor, 'LightGBM': LGBMRegressor}


def boosting_search_space(trial, name):
    if name == 'XGB':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.01, 20.0, log=True),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
        )
    if name == 'CatBoost':
        return dict(
            iterations=trial.suggest_int('iterations', 100, 800),
            depth=trial.suggest_int('depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 20.0, log=True),
            random_state=RANDOM_STATE, verbose=False, thread_count=N_JOBS,
        )
    if name == 'LightGBM':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            num_leaves=trial.suggest_int('num_leaves', 15, 255),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1,
        )
    raise ValueError(name)


def tune_boosting_model(name, target_type, X_df, y_series, search_sub_index, search_folds,
                         n_trials=N_OPTUNA_TRIALS):
    ctor = BOOSTING_CTORS[name]

    def objective(trial):
        params = boosting_search_space(trial, name)
        factory = wrap_for_target(lambda: ctor(**params), target_type)
        mean_r2, _ = score_model(factory, X_df, y_series,
                                  search_sub_index, [search_folds])
        return mean_r2

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    train = load_train_with_groups()
    print(f"Loaded train={train.shape}")

    print("Training Mol2Vec on a PI1M sample (egb/ei only, see MOL2VEC_TARGETS)...")
    m2v_model = train_mol2vec(load_pi1m_sample(MOL2VEC_SAMPLE_N, RANDOM_STATE))
    print(f"  vocab size {len(m2v_model.wv)} [{time.time()-t0:.0f}s]")

    print("Featurizing (baseline + MACCS + Gasteiger + Mol2Vec)...")
    X, feature_cols, train_valid = build_feature_matrix(train, m2v_model)
    y_all = train_valid['target']
    print(f"  {len(feature_cols)} features after pruning [{time.time()-t0:.0f}s]")

    target_types = sorted(train_valid['target_type'].unique())

    # ---- untuned zoo: linear + tree + boosting defaults ----
    print("\n" + "=" * 100)
    print("UNTUNED ZOO -- per-target CV R2")
    print("=" * 100)

    results = {tt: {} for tt in target_types}  # tt -> model_name -> (mean, std)

    static_zoo = {}
    static_zoo.update(linear_models())
    static_zoo.update(tree_models())

    for tt in target_types:
        sub_index, repeats = get_harness_splits(train_valid, tt)
        X_tt = X[target_feature_cols(feature_cols, tt)]
        row = f"{tt:6s}"
        for name, pipeline_template in static_zoo.items():
            factory = wrap_for_target(lambda p=pipeline_template: clone(p), tt)
            mean_r2, std_r2 = score_model(factory, X_tt, y_all, sub_index, repeats)
            results[tt][name] = (mean_r2, std_r2)
            row += f"  {name}={mean_r2:+.3f}"
        print(row + f"  [{time.time()-t0:.0f}s]")

    boosting_defaults = boosting_default_params()
    for name, params in boosting_defaults.items():
        ctor = BOOSTING_CTORS[name]
        for tt in target_types:
            sub_index, repeats = get_harness_splits(train_valid, tt)
            X_tt = X[target_feature_cols(feature_cols, tt)]
            factory = wrap_for_target(lambda p=params: ctor(**p), tt)
            mean_r2, std_r2 = score_model(factory, X_tt, y_all, sub_index, repeats)
            results[tt][f'{name}_default'] = (mean_r2, std_r2)
        print(f"{name} (default) done [{time.time()-t0:.0f}s]")

    print("\n--- Untuned zoo summary ---")
    all_model_names = list(static_zoo.keys()) + [f'{n}_default' for n in boosting_defaults]
    header = f"{'target':6s}" + "".join(f"{m:>14s}" for m in all_model_names)
    print(header)
    for tt in target_types:
        row = f"{tt:6s}"
        for name in all_model_names:
            mean_r2, std_r2 = results[tt][name]
            row += f"  {mean_r2:+.3f}({std_r2:.3f})".rjust(14)
        print(row)

    # ---- Optuna tuning for boosting models ----
    print("\n" + "=" * 100)
    print(f"OPTUNA TUNING -- {N_OPTUNA_TRIALS} trials, 3-fold search, per boosting model per target")
    print("=" * 100)

    tuned_choice = {}  # (name, tt) -> 'tuned' or 'default'
    tuned_params_store = {}  # (name, tt) -> full_params dict, regardless of accept/reject
    for name in BOOSTING_CTORS:
        ctor = BOOSTING_CTORS[name]
        for tt in target_types:
            X_tt = X[target_feature_cols(feature_cols, tt)]
            search_sub_index, search_folds = get_search_splits(train_valid, tt)
            best_params = tune_boosting_model(name, tt, X_tt, y_all, search_sub_index, search_folds)
            full_params = {**boosting_defaults[name], **best_params}
            tuned_params_store[(name, tt)] = full_params

            harness_sub_index, harness_repeats = get_harness_splits(train_valid, tt)
            factory = wrap_for_target(lambda p=full_params: ctor(**p), tt)
            tuned_mean, tuned_std = score_model(
                factory, X_tt, y_all, harness_sub_index, harness_repeats)
            default_mean, default_std = results[tt][f'{name}_default']

            noise_floor = max(tuned_std, default_std)
            if tuned_mean - default_mean > noise_floor:
                results[tt][f'{name}_tuned'] = (tuned_mean, tuned_std)
                tuned_choice[(name, tt)] = 'tuned'
                verdict = f"ACCEPT tuned ({default_mean:.4f} -> {tuned_mean:.4f})"
            else:
                results[tt][f'{name}_tuned'] = (default_mean, default_std)
                tuned_choice[(name, tt)] = 'default'
                verdict = f"keep default (tuned {tuned_mean:.4f} vs default {default_mean:.4f}, " \
                          f"didn't clear noise floor {noise_floor:.4f})"
            print(f"  {name:9s} {tt:5s}: {verdict}  [{time.time()-t0:.0f}s]")

    # ---- final zoo report: linear + tree + best-of(tuned/default) boosting ----
    print("\n" + "=" * 100)
    print("FINAL ZOO -- per-target CV R2 (boosting = tuned if accepted, else default)")
    print("=" * 100)
    final_model_names = list(static_zoo.keys()) + [f'{n}_tuned' for n in BOOSTING_CTORS]
    header = f"{'target':6s}" + "".join(f"{m:>16s}" for m in final_model_names)
    print(header)
    winners = {}
    for tt in target_types:
        row = f"{tt:6s}"
        best_name, best_mean = None, -np.inf
        for name in final_model_names:
            mean_r2, std_r2 = results[tt][name]
            row += f"  {mean_r2:+.4f}({std_r2:.3f})"
            if mean_r2 > best_mean:
                best_mean, best_name = mean_r2, name
        print(row)
        winners[tt] = (best_name, best_mean)

    print("\n--- Non-dominated models per target (within noise floor of the best) ---")
    for tt in target_types:
        best_name, best_mean = winners[tt]
        best_std = results[tt][best_name][1]
        earn_spot = []
        for name in final_model_names:
            mean_r2, std_r2 = results[tt][name]
            if best_mean - mean_r2 <= max(best_std, std_r2):
                earn_spot.append(name)
        print(f"  {tt:5s}: best={best_name} ({best_mean:.4f})  "
              f"non-dominated: {', '.join(earn_spot)}")

    # ---- persist results so final_pipeline.py can load them directly,
    # instead of someone hand-transcribing printed numbers into hardcoded
    # constants (which is exactly the disconnect that broke last time a
    # search config changed here) ----
    export = {
        'n_optuna_trials': N_OPTUNA_TRIALS,
        'feature_set': 'baseline+MACCS+Gasteiger+Mol2Vec(egb,ei only)',
        'target_types': target_types,
        # winners[tt][0] is a column name like 'CatBoost_tuned' (boosting
        # models always carry that suffix internally, tuned-or-not) --
        # strip it so this matches the plain model names (MODEL_NAMES in
        # final_pipeline.py) the consumer actually indexes by.
        'best_model_per_target': {
            tt: {
                'name': winners[tt][0][:-len('_tuned')] if winners[tt][0].endswith('_tuned') else winners[tt][0],
                'score': winners[tt][1],
            } for tt in target_types
        },
        'accepted_tuned_configs': {
            f'{name}__{tt}': tuned_params_store[(name, tt)]
            for (name, tt), choice in tuned_choice.items() if choice == 'tuned'
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\nSaved results -> {RESULTS_PATH}")

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
    print("Phase 3 complete -- stopping per workflow.")


if __name__ == "__main__":
    main()
