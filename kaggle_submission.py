"""
ANRF AISEHack 2.0 -- Polymer Property Prediction. Single self-contained
script for a Kaggle Script/Notebook kernel: reproduces the pipeline that
scored 0.849 on the public leaderboard, in one file with no imports from
this repo's scripts/ directory (Kaggle's kernel environment doesn't have
those files -- everything importable only from this repo has been inlined
below).

What's IN, and why:
  - RDKit descriptors + physics ratios + dimer-delta + 256-bit Morgan
    fingerprints (prajwal's original featurize()), + 167-bit MACCS keys,
    + Gasteiger partial-charge summary stats. All computed fresh from
    train.csv/test.csv, no external data.
  - log/exp target transform for eps, ei (the two most right-skewed,
    weakest-CV targets -- see TARGET_TRANSFORMS below).
  - 8-model zoo (Ridge, ElasticNet, RF, GBM, HGB, XGBoost, CatBoost,
    LightGBM) per target_type, combined via out-of-fold stacking with a
    Ridge meta-learner.
  - Phase 5 output-safety clipping (train min/max + 10% margin, plus hard
    physical floors on band gap / refractive index / dielectric constant).

Rules compliance note on hyperparameter tuning: an earlier version of this
script hardcoded the 2 CatBoost configs (egc, tg) that a local Optuna search
found to beat their defaults. That's a violation of "all stages -- including
model definition/initialization and training -- must execute entirely
within the notebook during a single run, manual intervention at any stage
not permitted": those exact hyperparameter values are the *output* of a
data-dependent search run outside the graded execution, so injecting them
as constants is manual intervention at the training stage, even though the
models themselves still fit fresh on train.csv. Fixed by moving the Optuna
search itself into this script (see TUNE_BOOSTING below) -- the winning
configs are now discovered live, every run, inside the single execution.
Ordinary fixed hyperparameters elsewhere in this file (n_estimators=300,
SELECT_K=100, VAR_THRESH, the clipping margin, etc.) are NOT the same kind
of issue -- they're engineering defaults chosen by judgment, never fit or
searched against this dataset, so there's nothing external being replayed.

What's OUT versus the local dev pipeline, and why -- both are deliberate
simplifications, not omissions:
  1. No Mol2Vec / PI1M-derived features. A local ablation found Mol2Vec
     (skip-gram embedding trained on PI1M SMILES) helps egb/ei but hurts
     eps/nc, and that target-conditional version hasn't been validated on
     the real leaderboard yet as of this script. Leaving it out keeps this
     submission on the *confirmed* 0.849 pipeline rather than an untested
     one, and drops a dependency (gensim) + a data source (PI1M.csv) this
     script would otherwise need bundled as a Kaggle input.
  2. No Step 12 cross-target-feature ablation. That was diagnostic-only
     (never fed into predictions) and found "no clear effect" on all 4
     targets tested, so there's nothing to carry over.

Estimated runtime: several hours on CPU, dominated by the live Optuna
search (50 trials x 3 boosting models x 7 targets = 1050 model fits over a
wide search space, plus the outer accept/reject harness evaluation for
each). Well within a 10-hour notebook budget based on local timing, but
this is the stage to watch if you ever need to shorten it (N_OPTUNA_TRIALS
below is the one knob that trades search thoroughness for runtime).

Input path handling: looks for train.csv anywhere under /kaggle/input/ if
that directory exists (Kaggle mounts competition data there, under a
folder name that matches the competition slug, which this script doesn't
hardcode); otherwise falls back to ./data/ for local testing. Output is
written to ./submission.csv, which resolves to /kaggle/working/submission.csv
under Kaggle's default working directory -- exactly where a Code
Competition looks for it.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys, rdPartialCharges

# The dimer-construction step intentionally attempts a bond-surgery +
# resanitize that fails on some ring topologies; those failures are caught
# and handled (falls back to zero-delta features), so RDKit's C++ logger
# spam ("Can't kekulize...") is expected noise. Silenced so the run log
# stays readable.
RDLogger.DisableLog('rdApp.*')

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_JOBS = -1
N_OPTUNA_TRIALS = 50


# ---------------------------------------------------------------------------
# Paths -- Kaggle input dir auto-detected, local ./data/ as fallback
# ---------------------------------------------------------------------------
def _find_input_dir():
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        hits = list(kaggle_root.rglob("train.csv"))
        if hits:
            return hits[0].parent
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    return here / "data"


INPUT_DIR = _find_input_dir()
TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"
SAMPLE_SUB_PATH = INPUT_DIR / "sample_submission.csv"
OUT_PATH = Path("submission.csv")

# ---------------------------------------------------------------------------
# Featurization knobs
# ---------------------------------------------------------------------------
FP_BITS = 256
FP_RADIUS = 2
VAR_THRESH = 1e-6
CORR_THRESH = 0.98
SELECT_K = 100

_SLOW_OR_UNSTABLE = {'Ipc'}  # can overflow to inf on larger structures
_DESC_LIST = [(n, f) for n, f in Descriptors._descList if n not in _SLOW_OR_UNSTABLE]
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

SMALL_TARGETS = {'egb', 'ei', 'eea', 'eps', 'nc'}
LARGE_TARGETS = {'tg', 'egc'}
N_SPLITS = 5
N_REPEATS_SMALL = 3

# eps (dielectric constant) and ei (ionization energy) are the two most
# right-skewed targets and the two weakest CV scores; both are comfortably
# positive in train, so a plain log/exp is safe.
TARGET_TRANSFORMS = {
    'eps': (np.log, np.exp),
    'ei': (np.log, np.exp),
}

MARGIN_FRACTION = 0.10
PHYSICAL_FLOORS = {'egc': 0.0, 'egb': 0.0, 'nc': 1.0, 'eps': 1.0}
MODEL_NAMES = ['Ridge', 'ElasticNet', 'RF', 'GBM', 'HGB', 'XGB', 'CatBoost', 'LightGBM']
META_ALPHA = 1.0


# ---------------------------------------------------------------------------
# 1. Featurization (RDKit descriptors + physics ratios + dimer-delta +
#    Morgan fingerprints + MACCS keys + Gasteiger charges)
# ---------------------------------------------------------------------------
def _parse_mol(smiles):
    """Parse a polymer repeat-unit SMILES (with * attachment points)."""
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        # fallback: cap dummy attachment atoms with carbon to keep valence valid
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    return mol


def _make_dimer_mol(smiles):
    """Join two copies of the repeat unit at their * attachment points to
    approximate a short chain segment -- lets ring-conjugation/aromaticity/
    rotatable-bond descriptors see across the repeat-unit boundary, which
    matters for backbone-driven properties (band gaps, refractive index)
    more than a single isolated unit does. None if construction fails."""
    try:
        s = smiles.replace('[*]', '*')
        if s.count('*') != 2:
            return None
        molA = Chem.MolFromSmiles(s)
        molB = Chem.MolFromSmiles(s)
        if molA is None or molB is None:
            return None

        combo = Chem.RWMol(Chem.CombineMols(molA, molB))
        dummy_idx = [a.GetIdx() for a in combo.GetAtoms() if a.GetSymbol() == '*']
        if len(dummy_idx) != 4:
            return None

        def neighbor_of(idx):
            atom = combo.GetAtomWithIdx(idx)
            nbrs = atom.GetNeighbors()
            return nbrs[0].GetIdx() if nbrs else None

        a2 = neighbor_of(dummy_idx[1])
        b1 = neighbor_of(dummy_idx[2])
        if a2 is None or b1 is None:
            return None

        combo.AddBond(a2, b1, Chem.BondType.SINGLE)
        for idx in sorted(dummy_idx, reverse=True):
            combo.RemoveAtom(idx)

        dimer = combo.GetMol()
        try:
            Chem.SanitizeMol(dimer)
        except Exception:
            # Some ring topologies genuinely can't be re-kekulized after the
            # junction bond is spliced in -- fall back to a partial sanitize
            # that skips kekulize/aromaticity perception so we still get
            # valid valences instead of discarding the whole dimer.
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


def maccs_keys(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.int8)
    for bit in fp.GetOnBits():
        if bit < 167:
            arr[bit] = 1
    return {f'maccs_{i}': int(arr[i]) for i in range(167)}


def gasteiger_features(mol):
    """Gasteiger partial charges, per atom, summarized. Targets eps
    (dielectric constant) and ei (ionization energy) specifically -- both
    are driven by molecular polarity/charge distribution, which the mostly
    topology-/counting-based descriptor set above doesn't directly
    capture."""
    mol = Chem.Mol(mol)  # ComputeGasteigerCharges mutates in place
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    charges = charges[np.isfinite(charges)]
    if len(charges) == 0:
        return {'gast_max': 0.0, 'gast_min': 0.0, 'gast_mean': 0.0,
                'gast_sum': 0.0, 'gast_std': 0.0}
    return {
        'gast_max': float(charges.max()), 'gast_min': float(charges.min()),
        'gast_mean': float(charges.mean()), 'gast_sum': float(charges.sum()),
        'gast_std': float(charges.std()),
    }


def fit_feature_pruner(df):
    """Columns to keep after dropping near-constant columns and one column
    from each highly-correlated pair. Fit on train only to avoid leakage."""
    variances = df.var(numeric_only=True)
    keep = variances[variances > VAR_THRESH].index.tolist()

    corr = df[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
    keep = [c for c in keep if c not in to_drop]
    return keep


def build_feature_matrix(df):
    """Single source of truth for the feature set -- baseline + MACCS +
    Gasteiger, pruned. Takes a df with a 'smiles' column, handles the
    featurize()-can-return-None filtering itself, and returns
    (X, feature_cols, valid_df)."""
    raw_feats = df['smiles'].apply(featurize)
    valid_mask = raw_feats.notna()
    if (~valid_mask).sum():
        print(f"  dropping {(~valid_mask).sum()} rows that failed to featurize")
    valid_df = df[valid_mask].reset_index(drop=True)
    raw_baseline_df = pd.DataFrame(list(raw_feats[valid_mask])).reset_index(drop=True)

    mols = valid_df['smiles'].apply(_parse_mol)
    maccs_df = pd.DataFrame([maccs_keys(m) for m in mols])
    gast_df = pd.DataFrame([gasteiger_features(m) for m in mols])

    raw_df = pd.concat([raw_baseline_df, maccs_df, gast_df], axis=1)
    feature_cols = fit_feature_pruner(raw_df)
    X = raw_df[feature_cols]
    return X, feature_cols, valid_df


# ---------------------------------------------------------------------------
# 2. CV harness -- GroupKFold on canonical SMILES (duplicate molecules with
#    conflicting repeat measurements must never be split across train/val,
#    or the model gets leakage from having seen the ~same target during
#    training). Repeated (5-fold x3 seeds) for the 5 small targets
#    (221-337 rows), since which molecules land in the validation fold
#    materially swings R2 at that scale.
# ---------------------------------------------------------------------------
def canonical_smiles(smiles):
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_train_with_groups():
    train = pd.read_csv(TRAIN_PATH)
    train['canon'] = train['smiles'].apply(canonical_smiles)
    n_bad = train['canon'].isna().sum()
    if n_bad:
        print(f"  dropping {n_bad} unparsable train rows")
        train = train[train['canon'].notna()].reset_index(drop=True)
    return train


def n_repeats_for(target_type):
    return N_REPEATS_SMALL if target_type in SMALL_TARGETS else 1


def get_harness_splits(df, target_type, base_seed=RANDOM_STATE):
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values

    repeats = []
    for r in range(n_repeats_for(target_type)):
        gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=base_seed + r)
        repeats.append(list(gkf.split(np.zeros(len(sub_index)), groups=groups)))
    return sub_index, repeats


def get_search_splits(df, target_type, n_splits=3, base_seed=RANDOM_STATE):
    """Cheap 3-fold GroupKFold splitter for Optuna's *inner* search loop
    only -- deliberately smaller/cheaper than get_harness_splits so the
    search stays fast. Never used to decide whether a tuned config beats
    the default; that comparison always runs on get_harness_splits."""
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    return sub_index, list(gkf.split(np.zeros(len(sub_index)), groups=groups))


def drop_constant_columns(Xtr, Xva):
    """Some GBM implementations (HistGradientBoostingRegressor in
    particular) crash on a column that's constant within a training fold
    even if it varies globally -- common on the small-target slices here."""
    with np.errstate(invalid='ignore'):
        stds = np.nanstd(Xtr, axis=0)
    keep = (stds > 0) & ~np.isnan(stds)
    return Xtr[:, keep], Xva[:, keep]


def wrap_for_target(model_factory, target_type):
    """Wraps a model factory so the target gets log-transformed before fit
    and exp'd back after predict, for eps/ei -- no-op for the other 5."""
    if target_type not in TARGET_TRANSFORMS:
        return model_factory
    func, inverse_func = TARGET_TRANSFORMS[target_type]

    def wrapped():
        return TransformedTargetRegressor(
            regressor=model_factory(), func=func, inverse_func=inverse_func)
    return wrapped


# ---------------------------------------------------------------------------
# 3. Model zoo
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


def get_zoo_factories(tt, accepted_tuned_configs):
    """All 8 model factories for one target_type, boosting models using
    their accepted tuned config where one exists (accepted_tuned_configs,
    built live by tune_all_boosting_models() below), default
    hyperparameters otherwise -- every factory wrapped with the eps/ei
    log-transform where applicable."""
    factories = {}
    for name, pipeline_template in {**linear_models(), **tree_models()}.items():
        factories[name] = wrap_for_target(lambda p=pipeline_template: clone(p), tt)

    defaults = boosting_default_params()
    for name, ctor in BOOSTING_CTORS.items():
        params = accepted_tuned_configs.get((name, tt), defaults[name])
        factories[name] = wrap_for_target(lambda p=params, c=ctor: c(**p), tt)
    return factories


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
    """Optuna TPE search over boosting_search_space, scored on a cheap
    3-fold GroupKFold search split (get_search_splits) -- never the real
    accept/reject harness, or a config that just got lucky on the search
    split would look like a genuine improvement."""
    ctor = BOOSTING_CTORS[name]

    def objective(trial):
        params = boosting_search_space(trial, name)
        factory = wrap_for_target(lambda: ctor(**params), target_type)
        mean_r2, _ = score_model(factory, X_df, y_series, search_sub_index, [search_folds])
        return mean_r2

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_all_boosting_models(X, y_all, train_valid, target_types, t0):
    """Runs the live Optuna search for every (boosting model, target_type)
    combo -- this IS the model-selection/tuning stage, executing here
    inside the single run rather than being replayed from an offline
    result (see module docstring). A tuned config only replaces the
    default if it beats the default's score on the real harness
    (get_harness_splits) by more than the noise floor (larger of the two
    configs' fold-to-fold std), so a config that got lucky on the cheap
    3-fold search doesn't get promoted."""
    print("\n" + "=" * 100)
    print(f"Live Optuna tuning -- {N_OPTUNA_TRIALS} trials, 3-fold search, "
          f"per boosting model per target")
    print("=" * 100)

    defaults = boosting_default_params()
    accepted_tuned_configs = {}
    for name in BOOSTING_CTORS:
        ctor = BOOSTING_CTORS[name]
        for tt in target_types:
            search_sub_index, search_folds = get_search_splits(train_valid, tt)
            best_params = tune_boosting_model(name, tt, X, y_all, search_sub_index, search_folds)
            full_params = {**defaults[name], **best_params}

            harness_sub_index, harness_repeats = get_harness_splits(train_valid, tt)
            default_factory = wrap_for_target(lambda p=defaults[name]: ctor(**p), tt)
            default_mean, default_std = score_model(
                default_factory, X, y_all, harness_sub_index, harness_repeats)
            tuned_factory = wrap_for_target(lambda p=full_params: ctor(**p), tt)
            tuned_mean, tuned_std = score_model(
                tuned_factory, X, y_all, harness_sub_index, harness_repeats)

            noise_floor = max(tuned_std, default_std)
            if tuned_mean - default_mean > noise_floor:
                accepted_tuned_configs[(name, tt)] = full_params
                verdict = f"ACCEPT tuned ({default_mean:.4f} -> {tuned_mean:.4f})"
            else:
                verdict = f"keep default (tuned {tuned_mean:.4f} vs default {default_mean:.4f}, " \
                          f"didn't clear noise floor {noise_floor:.4f})"
            print(f"  {name:9s} {tt:5s}: {verdict}  [{time.time()-t0:.0f}s]")

    return accepted_tuned_configs


def score_model(model_factory, X_df, y_series, sub_index, repeats):
    scores = []
    for folds in repeats:
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            Xtr = X_df.loc[tr_idx].values.astype(float)
            Xva = X_df.loc[va_idx].values.astype(float)
            Xtr, Xva = drop_constant_columns(Xtr, Xva)
            ytr = y_series.loc[tr_idx].values
            yva = y_series.loc[va_idx].values
            model = model_factory()
            model.fit(Xtr, ytr)
            scores.append(r2_score(yva, model.predict(Xva)))
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


def generate_oof(model_factory, X_df, y_series, sub_index, repeats):
    """Per-row OOF prediction, averaged across repeats -- leak-free input
    to the stacking meta-learner."""
    n = len(sub_index)
    accum = np.zeros(n)
    for folds in repeats:
        fold_pred = np.full(n, np.nan)
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            Xtr = X_df.loc[tr_idx].values.astype(float)
            Xva = X_df.loc[va_idx].values.astype(float)
            Xtr, Xva = drop_constant_columns(Xtr, Xva)
            ytr = y_series.loc[tr_idx].values
            model = model_factory()
            model.fit(Xtr, ytr)
            fold_pred[va_pos] = model.predict(Xva)
        accum += fold_pred
    oof = accum / len(repeats)
    return pd.Series(oof, index=sub_index)


def fit_predict_full(model_factory, X_tr_df, y_tr_series, X_te_df):
    Xtr = X_tr_df.values.astype(float)
    Xte = X_te_df.values.astype(float)
    Xtr, Xte = drop_constant_columns(Xtr, Xte)
    model = model_factory()
    model.fit(Xtr, y_tr_series.values)
    return model.predict(Xte)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    print(f"Reading data from: {INPUT_DIR}")
    train = load_train_with_groups()
    print(f"Loaded train={train.shape}")

    print("Featurizing train (baseline + MACCS + Gasteiger)...")
    X, feature_cols, train_valid = build_feature_matrix(train)
    y_all = train_valid['target']
    print(f"  {len(feature_cols)} features after pruning [{time.time()-t0:.0f}s]")

    target_types = sorted(train_valid['target_type'].unique())

    # ---- live hyperparameter tuning (must run in this execution -- see
    # module docstring's rules-compliance note) ----
    accepted_tuned_configs = tune_all_boosting_models(X, y_all, train_valid, target_types, t0)
    print(f"\n  {len(accepted_tuned_configs)} accepted tuned config(s): "
          f"{list(accepted_tuned_configs.keys())}")

    # ---- OOF stacking ----
    print("\n" + "=" * 100)
    print("OOF stacking")
    print("=" * 100)

    oof_meta = {}
    harness_cache = {}
    cv_scores = {}  # tt -> (mean, std) of the stacked meta-learner, via its own CV
    for tt in target_types:
        sub_index, repeats = get_harness_splits(train_valid, tt)
        harness_cache[tt] = (sub_index, repeats)
        factories = get_zoo_factories(tt, accepted_tuned_configs)

        oof_cols = {name: generate_oof(factories[name], X, y_all, sub_index, repeats)
                    for name in MODEL_NAMES}
        oof_df = pd.DataFrame(oof_cols)
        print(f"  {tt:5s}: OOF generated for all {len(MODEL_NAMES)} models [{time.time()-t0:.0f}s]")

        # CV score of the meta-learner itself, evaluated directly on the OOF
        # matrix (own folds) -- this is the number that estimates what the
        # competition metric will actually show, not just a single model's.
        cv_scores[tt] = score_model(lambda: Ridge(alpha=META_ALPHA), oof_df, y_all, sub_index, repeats)

        meta_final = Ridge(alpha=META_ALPHA)
        meta_final.fit(oof_df.values, y_all.loc[sub_index].values)
        oof_meta[tt] = meta_final

    print(f"\n{'target':6s}{'stacked CV R2':>18s}")
    for tt in target_types:
        mean, std = cv_scores[tt]
        print(f"{tt:6s}{mean:14.4f} (+/-{std:.3f})")
    mean_r2 = float(np.mean([m for m, _ in cv_scores.values()]))
    print(f"\n>>> Mean CV R2 across all {len(target_types)} targets: {mean_r2:.4f} <<<")
    print("(this is the number that estimates the competition metric -- mean R2 across targets)")

    # ---- featurize test, full-refit base models on 100% train, predict ----
    print("\nFeaturizing test + full-refit base models, predicting...")
    test = pd.read_csv(TEST_PATH)
    test_feats = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    if (~test_valid_mask).sum():
        print(f"  {(~test_valid_mask).sum()} test SMILES failed to featurize -- "
              f"falling back to that target's train mean for those rows")
    test_mols = test.loc[test_valid_mask, 'smiles'].apply(_parse_mol)
    test_maccs = pd.Series([maccs_keys(m) for m in test_mols], index=test_mols.index)
    test_gast = pd.Series([gasteiger_features(m) for m in test_mols], index=test_mols.index)

    test_feat_df = pd.DataFrame(index=test.index, columns=feature_cols, dtype=float)
    for idx in test.index[test_valid_mask]:
        row = {**test_feats[idx], **test_maccs[idx], **test_gast[idx]}
        for k in feature_cols:
            test_feat_df.loc[idx, k] = row.get(k, np.nan)

    test_predictions = np.full(len(test), np.nan)
    for tt in target_types:
        sub_index, _ = harness_cache[tt]
        X_tr_tt = X.loc[sub_index]
        y_tr_tt = y_all.loc[sub_index]
        factories = get_zoo_factories(tt, accepted_tuned_configs)

        mask = (test['target_type'] == tt).values
        rows_valid = mask & test_valid_mask.values
        rows_invalid = mask & (~test_valid_mask.values)

        if rows_valid.sum() > 0:
            X_te_tt = test_feat_df.loc[rows_valid, feature_cols]
            base_test_preds = np.column_stack([
                fit_predict_full(factories[name], X_tr_tt, y_tr_tt, X_te_tt)
                for name in MODEL_NAMES
            ])
            test_predictions[rows_valid] = oof_meta[tt].predict(base_test_preds)

        if rows_invalid.sum() > 0:
            test_predictions[rows_invalid] = y_tr_tt.mean()
        print(f"  {tt:5s} done [{time.time()-t0:.0f}s]")

    # ---- output safety: clip to train range + margin, hard physical floors ----
    print("\n" + "=" * 100)
    print("Clipping + writing submission.csv")
    print("=" * 100)

    stacked_out = test[['id', 'target_type']].copy()
    stacked_out['target'] = test_predictions

    clipped_target = stacked_out['target'].copy()
    for tt in target_types:
        y = y_all.loc[train_valid['target_type'] == tt]
        lo, hi = y.min(), y.max()
        margin = MARGIN_FRACTION * (hi - lo)
        clip_lo, clip_hi = lo - margin, hi + margin
        if tt in PHYSICAL_FLOORS:
            clip_lo = max(clip_lo, PHYSICAL_FLOORS[tt])
        mask = stacked_out['target_type'] == tt
        vals = stacked_out.loc[mask, 'target']
        n_clipped = ((vals < clip_lo) | (vals > clip_hi)).sum()
        clipped_target.loc[mask] = vals.clip(lower=clip_lo, upper=clip_hi)
        print(f"  {tt:5s}: clip=[{clip_lo:.4g}, {clip_hi:.4g}]  "
              f"{n_clipped}/{mask.sum()} rows clipped")

    submission = stacked_out[['id']].copy()
    submission['target'] = clipped_target

    if SAMPLE_SUB_PATH.exists():
        # sample_submission.csv is a short format example (e.g. 10 rows), not
        # a full-length template -- only its columns are a real spec to check
        # against. Row count is checked against test.csv itself instead.
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        assert list(submission.columns) == list(sample_sub.columns), \
            f"column mismatch: {submission.columns.tolist()} vs {sample_sub.columns.tolist()}"
    assert len(submission) == len(test), \
        f"row count mismatch: {len(submission)} vs test.csv's {len(test)}"
    assert submission['target'].notna().all(), "unfilled predictions remain"

    submission.to_csv(OUT_PATH, index=False)
    print(f"\nFormat check OK -- saved {OUT_PATH} with shape {submission.shape}")
    print(f">>> Mean CV R2 across all {len(target_types)} targets: {mean_r2:.4f} <<<")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
