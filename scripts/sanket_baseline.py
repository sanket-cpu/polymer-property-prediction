"""
Polymer Property Prediction - Round 2 (v2, boosted)
Physics-informed + fingerprint descriptor model with weighted ensembling.

Predicts 7 polymer properties (Egc, Egb, Ei, Eea, EPS, Nc, Tg) from polymer SMILES
using RDKit descriptors + Morgan fingerprints + a "dimer" repeat-unit trick to
capture chain-level conjugation effects, followed by per-property model
selection and weighted blending.

No external data, no pretrained weights. Fully self-contained, runs in
seconds-minutes on CPU.

Key changes vs v1 (aimed at CV R2 0.78 -> ~0.84):
  1. Full RDKit descriptor set (~200 descriptors) instead of ~35 hand-picked
     ones, with automatic pruning of constant / near-duplicate columns.
  2. Morgan fingerprint bits (folded, 256-bit) added as coarse substructure
     features -- these carry a lot of the signal GBM/RF exploit for
     electronic properties (Egc, Egb, Ei, Eea).
  3. "Dimer" trick: the repeat unit is joined head-to-tail with a copy of
     itself at the * attachment points before descriptor calculation. This
     lets ring-conjugation / aromaticity / rotatable-bond descriptors see
     across the repeat-unit boundary, which matters for backbone-driven
     properties (band gaps, refractive index) far more than a single
     isolated unit does. Dimer descriptors are added as a *delta* from the
     monomer values, so they encode "what changes when the chain extends."
  4. Median imputation of any NaN feature values (full descriptor list can
     occasionally throw NaN/inf on unusual structures) instead of silently
     dropping columns.
  5. Model zoo expanded to include HistGradientBoostingRegressor (fast,
     usually stronger than the older GradientBoostingRegressor on tabular
     data) and ElasticNet alongside Ridge/RF/GBM.
  6. Per target_type, all 7 models' out-of-fold predictions are combined
     through a Ridge meta-learner (OOF stacking), evaluated with nested CV
     against the older top-2 softmax blend -- production uses whichever
     wins for that specific target (see the old_blend vs new_stack report
     each run prints), not a single hardcoded strategy for all 7.
  7. Feature pruning (variance threshold + correlation threshold) is fit on
     train only and re-used for test, avoiding leakage while keeping the
     matrix small enough that model fitting stays fast.

Usage: adjust TRAIN_PATH / TEST_PATH / OUT_PATH below, then run.
"""

import os

# Must be set before numpy/sklearn are imported -- these libraries read the
# thread count once at import/first-use and pinning them to 1 here removes
# the one real source of run-to-run nondeterminism in this script (parallel
# floating-point reductions aren't guaranteed bit-identical across runs,
# which can flip which model wins a near-tied CV comparison). Trades some
# wall-clock speed for a hard reproducibility guarantee.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

# The dimer-construction step (see _make_dimer_mol) intentionally attempts
# a bond-surgery + resanitize that fails on some ring topologies; those
# failures are caught and handled (falls back to zero-delta features), so
# RDKit's C++ logger spam ("Can't kekulize...") is expected noise, not a
# sign anything is broken. Silence it so the run log stays readable.
RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep console focused on our own prints

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- paths ----
# Resolved relative to this file's location (not the shell's cwd), so the
# script runs correctly regardless of the directory you invoke it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "sanket_submission.csv"

# ---- feature engineering knobs ----
FP_BITS = 256          # folded Morgan fingerprint size (keep small = fast)
FP_RADIUS = 2
VAR_THRESH = 1e-6      # drop near-constant columns
CORR_THRESH = 0.98     # drop one of any pair of columns correlated above this
TOP_K_MODELS = 2        # old top-2 softmax blend, kept as a fallback -- see Task 2 comparison
MODEL_NAMES = ['Ridge', 'ElasticNet', 'RF', 'GBM', 'HGB', 'XGB', 'CatBoost']  # fixed OOF/stack column order
MIN_SAMPLES_PER_FEATURE = 5   # target used per-target feature selection below this ratio
MIN_SELECTED_FEATURES = 30    # floor so selection never starves a small target of signal

# ---- Optuna tuning knobs ----
RUN_OPTION_A_COMPARISON = False   # shared multi-task model: tested twice, lost both times (see evaluate_shared_model)
OPTUNA_TRIALS = 15      # search budget per (target_type, model) -- kept modest, single-threaded runtime adds up fast
OPTUNA_SEARCH_FOLDS = 3  # cheaper CV during search; winner gets one final 5-fold score for a fair comparison

# ---- multi-seed bagging ----
# BAGGING_SEEDS[0] must equal RANDOM_STATE: that pass is already computed by
# the per-target CV/tuning loop above, so bagging only needs to repeat the
# OOF-generation + production refit for the remaining seeds, not redo tuning
# (hyperparameters are a property of the feature space, not the CV shuffle --
# same reasoning prajwal_baseline.py uses for its own multi-seed run).
BAGGING_SEEDS = [RANDOM_STATE, 0, 123]

# All RDKit descriptor (name, function) pairs, excluding a couple that are
# slow/unstable (3D descriptors need embedding, which polymer repeat units
# with dummy atoms often fail at).
_SLOW_OR_UNSTABLE = {
    'Ipc',  # can overflow to inf on larger structures
}
_DESC_LIST = [(n, f) for n, f in Descriptors._descList if n not in _SLOW_OR_UNSTABLE]

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


# ---------------------------------------------------------------------------
# 1. Featurization
# ---------------------------------------------------------------------------
def _parse_mol(smiles):
    """Parse a polymer repeat-unit SMILES (with * attachment points)."""
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        # fallback: cap dummy attachment atoms with carbon to keep valence valid
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    return mol


def _canon_smiles(smiles):
    """Canonical SMILES (stereo/isotopes preserved) -- used as the CV group key."""
    mol = _parse_mol(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def _canon_flat_smiles(smiles):
    """Canonical SMILES with stereochemistry and isotope labels stripped --
    used only for the near-duplicate diagnostic, not for CV grouping."""
    mol = _parse_mol(smiles)
    if mol is None:
        return smiles
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(mol, canonical=True)


def report_duplicate_smiles(train_valid):
    """
    Per target_type: report exact-duplicate (identical canonical SMILES) and
    near-duplicate (identical skeleton, differing only in stereo/isotopes)
    groups, and whether their target values agree or conflict.

    Exists because Round 1 of this project traced a CV/leaderboard mismatch
    to exactly this: duplicate molecules split across train/val folds under
    plain KFold. Printed on every run so a future data refresh that
    introduces new duplicates doesn't silently reopen the same leak.
    """
    df = train_valid[['canon_smiles', 'canon_flat', 'target', 'target_type']]
    print("\n--- Duplicate / near-duplicate SMILES check (per target_type) ---")
    total_exact_groups = total_exact_rows = total_near_groups = total_near_rows = 0
    for tt in sorted(df['target_type'].unique()):
        sub = df[df['target_type'] == tt]

        exact_grp = sub.groupby('canon_smiles')['target'].agg(list)
        exact_dup = exact_grp[exact_grp.apply(len) > 1]

        flat_grp = sub.groupby('canon_flat').agg(
            targets=('target', list),
            canon_exacts=('canon_smiles', lambda s: sorted(set(s))),
        )
        near_dup = flat_grp[flat_grp['canon_exacts'].apply(len) > 1]

        n_exact_groups, n_exact_rows = len(exact_dup), int(exact_dup.apply(len).sum())
        n_near_groups, n_near_rows = len(near_dup), int(near_dup['targets'].apply(len).sum())
        total_exact_groups += n_exact_groups
        total_exact_rows += n_exact_rows
        total_near_groups += n_near_groups
        total_near_rows += n_near_rows

        n_conflict_exact = sum(1 for v in exact_dup if max(v) - min(v) > 1e-6)
        n_conflict_near = sum(1 for v in near_dup['targets'] if max(v) - min(v) > 1e-6)
        print(f"  {tt:5s} (n={len(sub):4d})  exact_dup_groups={n_exact_groups:2d} "
              f"(rows={n_exact_rows:2d}, conflicting={n_conflict_exact:2d})   "
              f"near_dup_groups={n_near_groups:2d} (rows={n_near_rows:2d}, conflicting={n_conflict_near:2d})")

    print(f"  TOTAL: {total_exact_groups} exact-dup groups ({total_exact_rows} rows), "
          f"{total_near_groups} near-dup groups ({total_near_rows} rows)\n")
    return total_exact_groups + total_near_groups


def _make_dimer_mol(smiles):
    """
    Join two copies of the repeat unit at their * attachment points to
    approximate a short chain segment. Falls back to None if construction
    fails (rare, e.g. more/less than 2 attachment points).
    """
    try:
        s = smiles.replace('[*]', '*')
        if s.count('*') != 2:
            return None
        # Replace the two '*' in unit A with dummy isotopes, unit B likewise,
        # then bond A's second dummy to B's first dummy via RWMol surgery.
        molA = Chem.MolFromSmiles(s)
        molB = Chem.MolFromSmiles(s)
        if molA is None or molB is None:
            return None

        combo = Chem.RWMol(Chem.CombineMols(molA, molB))
        dummy_idx = [a.GetIdx() for a in combo.GetAtoms() if a.GetSymbol() == '*']
        if len(dummy_idx) != 4:
            return None

        # dummy_idx[0], dummy_idx[1] belong to molA; [2], [3] belong to molB
        def neighbor_of(idx):
            atom = combo.GetAtomWithIdx(idx)
            nbrs = atom.GetNeighbors()
            return nbrs[0].GetIdx() if nbrs else None

        a2 = neighbor_of(dummy_idx[1])
        b1 = neighbor_of(dummy_idx[2])
        if a2 is None or b1 is None:
            return None

        combo.AddBond(a2, b1, Chem.BondType.SINGLE)
        # remove the 4 dummy atoms (remove highest index first to keep indices valid)
        for idx in sorted(dummy_idx, reverse=True):
            combo.RemoveAtom(idx)

        dimer = combo.GetMol()
        try:
            # Preferred path: full sanitize (correct kekulization/aromaticity).
            Chem.SanitizeMol(dimer)
        except Exception:
            # Some ring topologies genuinely can't be re-kekulized after the
            # junction bond is spliced in (rare, structure-dependent -- not
            # a bug to "fix" by forcing it). Fall back to a partial sanitize
            # that skips just the kekulize/aromaticity perception steps, so
            # we still get valid valences and can compute the non-aromaticity
            # -dependent portion of the descriptor set instead of discarding
            # the whole dimer.
            dimer.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                dimer,
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
            )
        return dimer
    except Exception:
        return None


def _safe_descriptors(mol):
    """Compute the full RDKit descriptor list, replacing failures with NaN."""
    out = {}
    for name, func in _DESC_LIST:
        try:
            val = func(mol)
            if val is None or not np.isfinite(val):
                val = np.nan
        except Exception:
            val = np.nan
        out[f'desc_{name}'] = val
    return out


def _morgan_bits(mol):
    fp = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros((FP_BITS,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return {f'fp_{i}': int(arr[i]) for i in range(FP_BITS)}


def _custom_physics_feats(mol):
    """Small set of hand-crafted, physically-motivated ratios (cheap, robust)."""
    feats = {}
    heavy = mol.GetNumHeavyAtoms() or 1
    n_bonds = mol.GetNumBonds() or 1
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    n_atoms = len(atoms) or 1

    for el in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'Si', 'P']:
        feats[f'n_{el}'] = atoms.count(el)
        feats[f'frac_{el}'] = atoms.count(el) / n_atoms

    n_aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_conjugated_bonds = sum(1 for b in mol.GetBonds() if b.GetIsConjugated())
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)

    feats['AromaticRatio'] = n_aromatic_atoms / n_atoms
    feats['ConjugationRatio'] = n_conjugated_bonds / n_bonds
    feats['RotBondsPerHeavyAtom'] = n_rot / heavy
    feats['RingsPerHeavyAtom'] = n_rings / heavy
    feats['MolWtPerHeavyAtom'] = Descriptors.MolWt(mol) / heavy
    return feats


def featurize(smiles):
    mol = _parse_mol(smiles)
    if mol is None:
        return None

    feats = {}
    try:
        feats.update(_safe_descriptors(mol))
        feats.update(_custom_physics_feats(mol))
        feats.update(_morgan_bits(mol))

        # Dimer-delta features: how key backbone-sensitive descriptors shift
        # when the chain is extended by one repeat unit. Captures
        # conjugation/rigidity trends that a single isolated unit misses,
        # which is especially informative for Egc/Egb/Nc.
        dimer = _make_dimer_mol(smiles)
        if dimer is not None:
            dimer_heavy = dimer.GetNumHeavyAtoms() or 1
            dimer_bonds = dimer.GetNumBonds() or 1
            d_arom = sum(1 for a in dimer.GetAtoms() if a.GetIsAromatic()) / dimer_heavy
            d_conj = sum(1 for b in dimer.GetBonds() if b.GetIsConjugated()) / dimer_bonds
            d_rot = rdMolDescriptors.CalcNumRotatableBonds(dimer) / dimer_heavy
            feats['dimer_delta_AromaticRatio'] = d_arom - feats['AromaticRatio']
            feats['dimer_delta_ConjugationRatio'] = d_conj - feats['ConjugationRatio']
            feats['dimer_delta_RotBondsPerHeavy'] = d_rot - feats['RotBondsPerHeavyAtom']
        else:
            feats['dimer_delta_AromaticRatio'] = 0.0
            feats['dimer_delta_ConjugationRatio'] = 0.0
            feats['dimer_delta_RotBondsPerHeavy'] = 0.0
    except Exception:
        return None
    return feats


# ---------------------------------------------------------------------------
# 2. Feature matrix cleanup (fit on train, applied to test)
# ---------------------------------------------------------------------------
def fit_feature_pruner(df):
    """
    Returns the list of columns to keep, after dropping near-constant
    columns and one column from each highly-correlated pair. Fit on train
    only to avoid leakage.
    """
    variances = df.var(numeric_only=True)
    keep = variances[variances > VAR_THRESH].index.tolist()

    corr = df[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
    keep = [c for c in keep if c not in to_drop]
    return keep


# ---------------------------------------------------------------------------
# 3. Model zoo
# ---------------------------------------------------------------------------
def select_k_for(n_samples, n_features):
    """
    How many features a target_type's models should select down to.

    The competition metric averages R2 across all 7 target_types
    unweighted, but the 5 minor properties (egb/ei/eea/eps/nc) have only
    ~220-340 rows against ~440 pruned features -- more features than
    samples. That's a severe overfitting regime for RF/GBM in particular.
    Tg and Egc have plenty of samples relative to the feature count and
    pass through unrestricted (n_samples >= n_features already).
    """
    if n_samples >= n_features:
        return n_features
    return max(MIN_SELECTED_FEATURES, n_samples // MIN_SAMPLES_PER_FEATURE)


def get_model_zoo(k_features, seed=RANDOM_STATE):
    """
    k_features (from select_k_for) is only applied ahead of Ridge/ElasticNet.
    Those are the models that actually suffer when features outnumber
    samples -- RF/GBM/HGB already do their own implicit feature selection
    via split gain, and an upstream univariate linear filter (f_regression)
    can strip out a feature a tree needs for a nonlinear split/interaction
    before the tree ever sees it (this is what caused the `nc` regression:
    GBM 0.823 -> 0.742 when SelectKBest was applied to it too). So trees
    keep the full pruned feature_cols, unrestricted, same as before this
    feature-selection change was introduced.

    `seed` defaults to RANDOM_STATE (all Task 1/2 call sites are unaffected)
    but can be overridden per call -- multi-seed bagging refits this same
    zoo with different seeds to average over each model's own training
    randomness (RF bootstrap rows, GBM/XGB/CatBoost row/column subsampling).
    """
    return {
        'Ridge': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('select', SelectKBest(score_func=f_regression, k=k_features)),
            ('sc', StandardScaler()),
            ('m', Ridge(alpha=5.0, random_state=seed)),
        ]),
        'ElasticNet': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('select', SelectKBest(score_func=f_regression, k=k_features)),
            ('sc', StandardScaler()),
            ('m', ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=seed, max_iter=5000)),
        ]),
        'RF': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                n_jobs=1, random_state=seed)),
        ]),
        'GBM': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=seed)),
        ]),
        'HGB': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', HistGradientBoostingRegressor(
                max_iter=300, max_depth=6, learning_rate=0.06,
                l2_regularization=0.1, random_state=seed)),
        ]),
        'XGB': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=1, random_state=seed, verbosity=0)),
        ]),
        'CatBoost': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.06, l2_leaf_reg=3.0,
                thread_count=1, random_state=seed,
                verbose=False, allow_writing_files=False)),
        ]),
    }


# ---------------------------------------------------------------------------
# 3b. Optuna tuning for XGBoost / CatBoost
#
# These two are new additions with hand-picked-but-unverified default
# hyperparameters (unlike Ridge/RF/GBM/HGB, which were tuned by feel over
# earlier iterations of this script). Search space covers each model's
# usual overfitting knobs (depth, learning rate, regularization, subsample).
# ---------------------------------------------------------------------------
def _build_xgb(params, seed=RANDOM_STATE):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('m', XGBRegressor(n_jobs=1, random_state=seed, verbosity=0, **params)),
    ])


def _xgb_search_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }


def _build_catboost(params, seed=RANDOM_STATE):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('m', CatBoostRegressor(thread_count=1, random_state=seed,
                                 verbose=False, allow_writing_files=False, **params)),
    ])


def _catboost_search_space(trial):
    return {
        'iterations': trial.suggest_int('iterations', 100, 600),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
    }


# model_name -> (builder(params) -> Pipeline, search_space(trial) -> params dict)
TUNABLE_MODELS = {
    'XGB': (_build_xgb, _xgb_search_space),
    'CatBoost': (_build_catboost, _catboost_search_space),
}


def build_pipeline(name, k_features, tuned_params=None, seed=RANDOM_STATE):
    """
    Construct a fresh pipeline for `name`. If tuned_params is given (an
    Optuna winner that beat the default), build with those hyperparameters
    instead of the model zoo's defaults -- otherwise fall back to
    get_model_zoo's untuned version. `seed` overrides RANDOM_STATE for
    multi-seed bagging refits.
    """
    if tuned_params is not None and name in TUNABLE_MODELS:
        build_fn, _ = TUNABLE_MODELS[name]
        return build_fn(tuned_params, seed=seed)
    return get_model_zoo(k_features, seed=seed)[name]


def tune_model(name, X, y, groups, n_trials=OPTUNA_TRIALS, search_folds=OPTUNA_SEARCH_FOLDS):
    """
    Optuna search over `name`'s hyperparameters for one target_type's data.

    Uses a cheaper `search_folds`-fold CV during the search itself (default
    3, vs. the 5-fold used everywhere else) to keep runtime bounded across
    7 targets x 2 models x n_trials searches. The winning config is
    re-scored with the real 5-fold `gkf` by the caller before deciding
    whether it actually beats the untuned default -- this function's
    returned search score is only for guiding the search, not a number to
    report or compare against the baseline directly.

    `groups` (canonical SMILES) keeps the search grouped the same way as the
    real CV below it -- otherwise a config could look good here purely
    because it exploited a duplicate-molecule leak the final scoring doesn't
    have.
    """
    build_fn, space_fn = TUNABLE_MODELS[name]
    gkf_search = GroupKFold(n_splits=search_folds, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        model = build_fn(space_fn(trial))
        scores = cross_val_score(model, X, y, groups=groups, cv=gkf_search, scoring='r2', n_jobs=1)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def softmax_weights(scores):
    """Convert a list of CV R2 scores into positive blend weights."""
    arr = np.array(scores, dtype=float)
    arr = arr - arr.max()  # numerical stability
    w = np.exp(arr * 5.0)  # sharpen so the better model dominates a bit
    return w / w.sum()


def oof_predict_all_models(model_dict, X, y, cv_splits):
    """
    Fit each model in `model_dict` on each fold's train split and predict its
    held-out split, so every row ends up with exactly one out-of-fold
    prediction per model.

    This does two jobs at once: the per-fold R2 (averaged) reproduces what
    `cross_val_score` was already giving Task 1's model-zoo comparison, and
    the OOF prediction arrays are the required input for Task 2's stacking --
    a meta-learner has to be trained on predictions the base model never saw
    the true label for, or it's not really "out of fold" at all.
    """
    n = len(y)
    oof_preds = {name: np.empty(n) for name in model_dict}
    fold_r2 = {name: [] for name in model_dict}
    for tr_idx, val_idx in cv_splits:
        for name, proto in model_dict.items():
            model = clone(proto)
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            oof_preds[name][val_idx] = preds
            fold_r2[name].append(r2_score(y[val_idx], preds))
    return oof_preds, fold_r2


def seed_pass(seed, X, y, groups_tt, k_features, tuned_params, use_stack, top_names, top_weights):
    """
    One full bagging pass at `seed`: fresh GroupKFold split, fresh OOF
    predictions (using this seed's own model randomness), combined the same
    way Task 2 already decided for this target (stack or top-2 blend, with
    the SAME blend weights -- only the underlying model fits vary by seed),
    plus a production refit on 100% of the data at this seed.

    Hyperparameters (tuned_params) and the stack-vs-blend decision are NOT
    re-derived here -- both are properties of the feature space/model
    family, not of which seed happened to shuffle the folds, so redoing
    Optuna search or the stack-vs-blend comparison per seed would just be
    burning compute for a decision that isn't expected to change.
    """
    gkf_seed = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_splits = list(gkf_seed.split(X, y, groups_tt))

    zoo_seed = get_model_zoo(k_features, seed=seed)
    for name, params in tuned_params.items():
        zoo_seed[name] = build_pipeline(name, k_features, params, seed=seed)

    oof_preds_seed, _ = oof_predict_all_models(zoo_seed, X, y, cv_splits)

    meta_model = None
    if use_stack:
        oof_matrix_seed = np.column_stack([oof_preds_seed[n] for n in MODEL_NAMES])
        combined_oof = np.empty(len(y))
        for tr_idx, val_idx in cv_splits:
            meta = RidgeCV(alphas=np.logspace(-3, 3, 25))
            meta.fit(oof_matrix_seed[tr_idx], y[tr_idx])
            combined_oof[val_idx] = meta.predict(oof_matrix_seed[val_idx])
        meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(oof_matrix_seed, y)
    else:
        combined_oof = sum(w * oof_preds_seed[n] for n, w in zip(top_names, top_weights))

    base_models = {
        name: build_pipeline(name, k_features, tuned_params.get(name), seed=seed).fit(X, y)
        for name in MODEL_NAMES
    }
    return combined_oof, base_models, meta_model


# ---------------------------------------------------------------------------
# 3b. Shared multi-task model (Option A)
#
# Every target above trains in total isolation, even though egb/ei/eea/eps/nc
# are physically related electronic/optical properties to egc, and ~840 of
# the 6,565 unique SMILES in train.csv are measured for more than one
# target_type. A single shared model trained on all 7 at once -- with
# target_type one-hot-encoded as a feature -- lets splits learned from the
# data-rich targets (tg: 4,143 rows, egc: 2,028) carry over to the
# data-starved ones (220-340 rows each), instead of each fending for itself.
# ---------------------------------------------------------------------------
class GroupedTargetScaler(BaseEstimator, RegressorMixin):
    """
    Standardizes y per group (target_type) before fitting one shared
    estimator across all groups, then inverse-transforms predictions back
    to each row's own group scale.

    Necessary because target scales differ wildly across target_types --
    Tg spans ~0-495, Egc spans ~2-10. Without this, a single shared MSE
    loss would be almost entirely about minimizing Tg's error and would
    barely move the needle on the others.
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, groups):
        self.group_stats_ = {}
        y_scaled = np.empty(len(y), dtype=float)
        for g in np.unique(groups):
            mask = groups == g
            mu, sigma = y[mask].mean(), y[mask].std()
            sigma = sigma if sigma > 1e-8 else 1.0
            self.group_stats_[g] = (mu, sigma)
            y_scaled[mask] = (y[mask] - mu) / sigma
        self.estimator_ = clone(self.base_estimator).fit(X, y_scaled)
        return self

    def predict(self, X, groups):
        preds_scaled = self.estimator_.predict(X)
        preds = np.empty(len(preds_scaled), dtype=float)
        for g in np.unique(groups):
            mask = groups == g
            mu, sigma = self.group_stats_[g]
            preds[mask] = preds_scaled[mask] * sigma + mu
        return preds


def evaluate_shared_model(full_train, feature_cols):
    """
    5-fold CV for one shared HGB model trained on all target_types at once.

    Uses the SAME per-target-type fold assignment as the per-target
    baseline loop (same KFold random_state), so the two mean-R2 numbers
    are directly comparable -- both are scored against the exact same
    held-out rows. R2 is computed per target_type on each fold's held-out
    rows, then averaged, matching the actual competition metric -- a
    single pooled R2 over all 7 mixed-scale targets would NOT be
    equivalent (it would be dominated by whichever target has the
    largest absolute variance) and would not mean the same thing.
    """
    tt_dummies = pd.get_dummies(full_train['target_type'], prefix='tt')
    X_full = pd.concat([full_train[feature_cols], tt_dummies], axis=1).values
    y_full = full_train['target'].values
    groups_full = full_train['target_type'].values

    fold_id = np.full(len(full_train), -1)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tt in np.unique(groups_full):
        idx = np.where(groups_full == tt)[0]
        for fold_i, (_, val_idx) in enumerate(kf.split(idx)):
            fold_id[idx[val_idx]] = fold_i

    per_target_fold_scores = {tt: [] for tt in np.unique(groups_full)}
    for fold_i in range(5):
        train_idx = np.where(fold_id != fold_i)[0]
        test_idx = np.where(fold_id == fold_i)[0]

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_full[train_idx])
        X_test = imputer.transform(X_full[test_idx])

        # Higher capacity than the per-target HGB (max_iter 300->600,
        # max_depth 6->10, max_leaf_nodes default 31->127): this model has
        # to represent 7 different physical response surfaces conditioned
        # on a one-hot target_type flag, not just one, so it needs
        # materially more trees/splits to avoid diluting focus across all
        # of them. l2_regularization bumped 0.1->0.2 to offset the added
        # capacity's overfitting risk.
        model = GroupedTargetScaler(HistGradientBoostingRegressor(
            max_iter=600, max_depth=10, max_leaf_nodes=127, learning_rate=0.06,
            l2_regularization=0.2, random_state=RANDOM_STATE))
        model.fit(X_train, y_full[train_idx], groups_full[train_idx])
        preds = model.predict(X_test, groups_full[test_idx])

        y_test = y_full[test_idx]
        groups_test = groups_full[test_idx]
        for tt in per_target_fold_scores:
            sel = groups_test == tt
            if sel.sum() < 2:
                continue
            per_target_fold_scores[tt].append(r2_score(y_test[sel], preds[sel]))

    per_target_mean = {tt: float(np.mean(scores)) for tt, scores in per_target_fold_scores.items()}
    overall_mean = float(np.mean(list(per_target_mean.values())))

    print("--- Option A (higher capacity): shared multi-task HGB (target_type as feature) ---")
    for tt in sorted(per_target_mean):
        print(f"{tt:5s}  shared_R2={per_target_mean[tt]:.3f}")
    print(f"Shared-model mean CV R2 across all 7 targets: {overall_mean:.4f}\n")

    return per_target_mean, overall_mean


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------
def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Loaded train={train.shape}, test={test.shape}")

    # --- featurize train ---
    train_feats = train['smiles'].apply(featurize)
    valid_mask = train_feats.notna()
    print(f"Featurized {valid_mask.sum()}/{len(train)} train molecules")

    train_feat_df = pd.DataFrame(list(train_feats[valid_mask])).reset_index(drop=True)
    train_valid = train[valid_mask].reset_index(drop=True)

    # --- canonical SMILES: CV group key + duplicate/near-duplicate report ---
    # Round 1 of this project traced a CV/leaderboard mismatch to duplicate
    # molecules landing on both sides of a plain KFold split. Canonicalizing
    # here (rather than grouping on the raw string) matters because two
    # different-looking SMILES strings can be the exact same molecule.
    train_valid['canon_smiles'] = train_valid['smiles'].apply(_canon_smiles)
    train_valid['canon_flat'] = train_valid['smiles'].apply(_canon_flat_smiles)
    report_duplicate_smiles(train_valid)

    # --- prune features on train only ---
    feature_cols = fit_feature_pruner(train_feat_df)
    print(f"Kept {len(feature_cols)}/{train_feat_df.shape[1]} features after pruning")

    full_train = pd.concat(
        [train_valid[['target', 'target_type', 'canon_smiles']], train_feat_df[feature_cols]], axis=1
    )

    # --- per-target-type: CV-evaluate model zoo, OOF-stack (Task 2) ---
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    final_models = {}     # target_type -> {base_models, use_stack, meta_model, blend_config}
    target_means = {}
    cv_scores_summary = {}   # target_type -> whichever of blend/stack is used in production
    old_vs_new = {}          # target_type -> (old_blend_r2, new_stack_r2, use_stack)

    for tt in sorted(full_train['target_type'].unique()):
        sub = full_train[full_train['target_type'] == tt]
        X = sub[feature_cols].values
        y = sub['target'].values
        groups_tt = sub['canon_smiles'].values
        target_means[tt] = y.mean()
        k_features = select_k_for(len(y), len(feature_cols))

        # Materialize the split once: reused for model-zoo scoring, Optuna's
        # tuned-vs-default check, AND the OOF matrix stacking needs below, so
        # all three see the exact same train/val row assignment.
        cv_splits = list(gkf.split(X, y, groups_tt))

        zoo = get_model_zoo(k_features)
        oof_preds, fold_r2 = oof_predict_all_models(zoo, X, y, cv_splits)
        scores_per_model = {name: float(np.mean(fold_r2[name])) for name in zoo}

        # Optuna tuning: only for XGB/CatBoost (see TUNABLE_MODELS), only kept
        # if the tuned config actually beats the untuned default on the same
        # grouped 5-fold splits used everywhere else in this loop. When it
        # wins, its OOF column replaces the untuned one so stacking below
        # sees the model that will actually run at test time.
        tuned_params = {}
        for name in TUNABLE_MODELS:
            params = tune_model(name, X, y, groups_tt)
            tuned_oof, tuned_fold_r2 = oof_predict_all_models(
                {name: build_pipeline(name, k_features, params)}, X, y, cv_splits
            )
            tuned_score = float(np.mean(tuned_fold_r2[name]))
            if tuned_score > scores_per_model[name]:
                scores_per_model[name] = tuned_score
                tuned_params[name] = params
                oof_preds[name] = tuned_oof[name]

        # --- OLD: top-2 softmax blend. No nesting needed here -- the blend
        # weights come from held-out CV scores, not from fitting anything on
        # the labels, so scoring it on the OOF predictions is already unbiased.
        ranked = sorted(scores_per_model.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:TOP_K_MODELS]
        top_names, top_scores = zip(*top)
        top_weights = softmax_weights(top_scores)
        old_blend_oof = sum(w * oof_preds[n] for n, w in zip(top_names, top_weights))
        old_blend_r2 = r2_score(y, old_blend_oof)

        # --- NEW: Ridge meta-learner stacked on all 7 models' OOF predictions,
        # evaluated with NESTED cv_splits -- unlike the blend above, the
        # meta-learner's weights ARE fit on labels, so scoring it on the same
        # OOF matrix it trained on would flatter it.
        oof_matrix = np.column_stack([oof_preds[n] for n in MODEL_NAMES])
        nested_stack_pred = np.empty(len(y))
        for tr_idx, val_idx in cv_splits:
            meta = RidgeCV(alphas=np.logspace(-3, 3, 25))
            meta.fit(oof_matrix[tr_idx], y[tr_idx])
            nested_stack_pred[val_idx] = meta.predict(oof_matrix[val_idx])
        new_stack_r2 = r2_score(y, nested_stack_pred)

        use_stack = new_stack_r2 > old_blend_r2
        cv_scores_summary[tt] = new_stack_r2 if use_stack else old_blend_r2
        old_vs_new[tt] = (old_blend_r2, new_stack_r2, use_stack)

        # --- refit for production on 100% of this target's data (primary seed) ---
        # All 7 base models are fit regardless of which combiner wins: stacking
        # needs all 7 at test time, and the blend path just ignores the rest.
        base_models = {
            name: build_pipeline(name, k_features, tuned_params.get(name)).fit(X, y)
            for name in MODEL_NAMES
        }
        meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(oof_matrix, y) if use_stack else None
        primary_combined_oof = nested_stack_pred if use_stack else old_blend_oof
        single_seed_r2 = cv_scores_summary[tt]

        # --- multi-seed bagging: repeat the OOF+refit pass at the remaining
        # BAGGING_SEEDS (tuned_params/use_stack/blend weights reused, not
        # re-derived -- see seed_pass docstring), then check whether
        # averaging over these independently-seeded fits actually helps
        # before committing to it in production.
        seed_oof = [primary_combined_oof]
        seed_models = [{'base_models': base_models, 'meta_model': meta_model}]
        for seed in BAGGING_SEEDS[1:]:
            combined_oof_s, base_models_s, meta_model_s = seed_pass(
                seed, X, y, groups_tt, k_features, tuned_params, use_stack, top_names, top_weights
            )
            seed_oof.append(combined_oof_s)
            seed_models.append({'base_models': base_models_s, 'meta_model': meta_model_s})

        bagged_oof_pred = np.mean(seed_oof, axis=0)
        bagged_r2 = r2_score(y, bagged_oof_pred)
        use_bagging = bagged_r2 > single_seed_r2
        cv_scores_summary[tt] = bagged_r2 if use_bagging else single_seed_r2

        final_models[tt] = {
            'use_stack': use_stack,
            'blend_config': list(zip(top_names, top_weights)),
            'seed_models': seed_models if use_bagging else seed_models[:1],
        }

        tuned_note = ", ".join(f"{n}(tuned)" for n in tuned_params) or "none"
        print(f"{tt:5s} (n={len(y):4d}, k={k_features:3d})  " +
              "  ".join(f"{name}={v:.3f}" for name, v in scores_per_model.items()) +
              f"  [tuning won: {tuned_note}]")
        print(f"      old_blend_R2={old_blend_r2:.4f}  new_stack_R2={new_stack_r2:.4f}  "
              f"-> single-seed uses {'STACK' if use_stack else 'BLEND'} "
              f"(wins by {abs(new_stack_r2 - old_blend_r2):.4f})")
        print(f"      single_seed_R2={single_seed_r2:.4f}  bagged_R2({len(BAGGING_SEEDS)}-seed)={bagged_r2:.4f}  "
              f"-> production {'USES BAGGING' if use_bagging else 'stays single-seed'} "
              f"({'+' if use_bagging else ''}{bagged_r2 - single_seed_r2:+.4f})")

    mean_cv_r2 = np.mean(list(cv_scores_summary.values()))
    print(f"\nEstimated mean CV R2 across all 7 targets (best of blend/stack, with bagging where it wins): {mean_cv_r2:.4f}")

    print("\n--- Task 2: old top-2 softmax blend vs new OOF-stacked blend, per target ---")
    for tt in sorted(old_vs_new):
        old, new, used_stack = old_vs_new[tt]
        print(f"  {tt:5s}  old_blend={old:.4f}  new_stack={new:.4f}  delta={new - old:+.4f}  "
              f"-> {'STACK' if used_stack else 'BLEND'} used in production")
    print()

    # --- Option A: does a shared multi-task model beat the per-target baseline? ---
    # Tested twice (plain and higher-capacity) -- lost both times. Disabled by
    # default (RUN_OPTION_A_COMPARISON=False) to save runtime; flip it back on
    # to re-check after a feature/data change.
    if RUN_OPTION_A_COMPARISON:
        _, shared_mean_r2 = evaluate_shared_model(full_train, feature_cols)
        delta = shared_mean_r2 - mean_cv_r2
        verdict = "BETTER" if delta > 0 else "WORSE"
        print(f"Per-target baseline: {mean_cv_r2:.4f}  |  Shared multi-task: {shared_mean_r2:.4f}  "
              f"|  Delta: {delta:+.4f} ({verdict})\n")
        print("(This comparison run does not change the submission below -- it still uses the "
              "per-target blended/stacked models. Switch to the shared model only if it wins.)\n")

    # --- featurize test ---
    test_feats = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    print(f"Featurized {test_valid_mask.sum()}/{len(test)} test molecules")

    test_feat_df = pd.DataFrame(index=test.index, columns=feature_cols, dtype=float)
    for idx in test.index[test_valid_mask]:
        row = test_feats[idx]
        for k in feature_cols:
            test_feat_df.loc[idx, k] = row.get(k, np.nan)

    predictions = np.zeros(len(test))
    for tt in sorted(full_train['target_type'].unique()):
        mask = (test['target_type'] == tt).values
        if mask.sum() == 0:
            continue
        rows_valid = mask & test_valid_mask.values
        rows_invalid = mask & (~test_valid_mask.values)

        if rows_valid.sum() > 0:
            X_test = test_feat_df.loc[rows_valid, feature_cols].values.astype(float)
            cfg = final_models[tt]
            # Average this target's prediction across every seed that made
            # production (just the primary seed if bagging didn't win, all
            # of BAGGING_SEEDS if it did -- see the per-target decision above).
            seed_preds = []
            for seed_cfg in cfg['seed_models']:
                base_preds = {name: model.predict(X_test) for name, model in seed_cfg['base_models'].items()}
                if cfg['use_stack']:
                    stack_matrix = np.column_stack([base_preds[n] for n in MODEL_NAMES])
                    seed_preds.append(seed_cfg['meta_model'].predict(stack_matrix))
                else:
                    seed_preds.append(sum(w * base_preds[n] for n, w in cfg['blend_config']))
            blend_pred = np.mean(seed_preds, axis=0)
            predictions[rows_valid] = blend_pred

        if rows_invalid.sum() > 0:
            # fallback for the rare unparsable SMILES: use training mean for that property
            predictions[rows_invalid] = target_means[tt]

    test['target'] = predictions
    submission = test[['id', 'target']].copy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with shape {submission.shape}")


if __name__ == "__main__":
    _start = time.perf_counter()
    main()
    _elapsed = time.perf_counter() - _start
    print(f"Total runtime: {_elapsed:.1f}s ({_elapsed / 60:.1f} min)")
