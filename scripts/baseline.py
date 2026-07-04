"""
Baseline model with proper validation harness.

Features  : RDKit descriptors (~210) + ECFP4 (2048 bits) + ECFP6 (2048 bits)
            + MACCS keys (167 bits) + topology features (2)
            + electronic features (5: aromatic rings, aromatic fraction,
              rotatable bonds, sp2 fraction, non-aromatic double bond count)
            + backbone_rotatable_bonds (monomer) + max_conjugation_path (monomer + chain)
Model     : LightGBM + XGBoost blend, hyperparameters tuned per target with Optuna
Validation: 5-fold CV grouped by *canonical* SMILES, stratified by target_type, scored mean R²
Prediction: multi-seed average of 5-fold LGB+XGB models across SEEDS
Pipeline  : features -> (per seed: prune -> tune LGB + XGB -> CV + fold ensemble) -> average

Run from project root:
    python3 scripts/baseline.py          (WSL)
    .venv/Scripts/python.exe scripts/baseline.py  (Windows)
Output: outputs/submission.csv
"""

import io
import os
import sys
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor

# Force UTF-8 on the terminal — reconfigure() is unreliable on Windows;
# wrapping the raw buffer directly is the guaranteed approach.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# Tee all stdout to a timestamped log file
os.makedirs("logs", exist_ok=True)
_log_path = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
_log_fh   = open(_log_path, "w", encoding="utf-8", buffering=1)

class _Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile  = logfile
    def write(self, data):
        self.terminal.write(data)
        self.logfile.write(data)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = _Tee(sys.stdout, _log_fh)  # type: ignore[assignment]
print(f"Logging to {_log_path}\n")

RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_PATH  = "data/train.csv"
TEST_PATH   = "data/test.csv"
OUTPUT_PATH = "outputs/submission.csv"
N_FOLDS     = 5
SEED        = 42
SEEDS       = [42, 0, 123]  # seeds for multi-seed prediction averaging; set to [42] for a single run

EARLY_STOPPING_ROUNDS = 50
N_TRIALS              = 30  # LGB Optuna trials per target; set to 2 for a quick smoke-test
N_XGB_TRIALS          = 10  # XGB Optuna trials per target
TUNE_N_ESTIMATORS     = 1000
FINAL_N_ESTIMATORS    = 3000
PROBE_N_ESTIMATORS    = 300
N_CHAIN_UNITS         = 3   # repeat units to stitch for chain-extension features

LGB_FIXED = {
    "n_estimators" : FINAL_N_ESTIMATORS,
    "random_state" : SEED,
    "n_jobs"       : -1,
    "verbose"      : -1,
}

# early_stopping_rounds must be in the constructor in XGBoost 3.x, not in .fit()
XGB_FIXED = {
    "n_estimators" : FINAL_N_ESTIMATORS,
    "random_state" : SEED,
    "n_jobs"       : -1,
    "tree_method"  : "hist",
    "device"       : "cpu",
}


# ── Feature computation ────────────────────────────────────────────────────────

def _desc_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {name: np.nan for name, _ in Descriptors._descList}
    return Descriptors.CalcMolDescriptors(mol)

def _ecfp4_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * 2048
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

def _ecfp6_one(smi):
    # radius=3 captures larger local chemical neighborhoods; more informative
    # for Egc where conjugation length beyond 2 bonds governs the band gap
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * 2048
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048))

def _maccs_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * 167
    return list(MACCSkeys.GenMACCSKeys(mol))

def _topo_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"star_distance": np.nan, "star_distance_frac": np.nan}
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(star_idx) != 2:
        return {"star_distance": np.nan, "star_distance_frac": np.nan}
    dmat = Chem.GetDistanceMatrix(mol)
    star_dist = dmat[star_idx[0], star_idx[1]]
    diameter  = dmat.max()
    return {
        "star_distance"     : star_dist,
        "star_distance_frac": star_dist / diameter if diameter > 0 else 0.0,
    }

def _electronic_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {
            "num_aromatic_rings"      : np.nan,
            "aromatic_atom_fraction"  : np.nan,
            "num_rotatable_bonds"     : np.nan,
            "sp2_atom_fraction"       : np.nan,
            "num_nonarom_double_bonds": np.nan,
        }
    n_atoms = mol.GetNumAtoms()
    n_arom  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    # sp2 hybridization = planar/conjugated atoms; fraction is a direct proxy
    # for conjugation extent which sets the polymer band gap (Egc)
    n_sp2   = sum(
        1 for a in mol.GetAtoms()
        if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2
    )
    # Non-aromatic C=C / C=N extend conjugation beyond ring systems
    n_dbl   = sum(
        1 for b in mol.GetBonds()
        if b.GetBondTypeAsDouble() == 2.0 and not b.GetIsAromatic()
    )
    return {
        "num_aromatic_rings"      : rdMolDescriptors.CalcNumAromaticRings(mol),
        "aromatic_atom_fraction"  : n_arom / n_atoms if n_atoms > 0 else 0.0,
        "num_rotatable_bonds"     : rdMolDescriptors.CalcNumRotatableBonds(mol),
        "sp2_atom_fraction"       : n_sp2 / n_atoms if n_atoms > 0 else 0.0,
        "num_nonarom_double_bonds": n_dbl,
    }

def _tg_specific_one(smi):
    """
    Backbone-path rotatable bond count: single non-ring bonds on the shortest
    path between the two * attachment points only. More precise Tg rigidity
    signal than whole-molecule rotatable bond count — side chains that don't
    affect backbone stiffness are excluded.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"backbone_rotatable_bonds": np.nan}
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(star_idx) != 2:
        return {"backbone_rotatable_bonds": np.nan}
    path = Chem.GetShortestPath(mol, star_idx[0], star_idx[1])
    rot = 0
    for i in range(len(path) - 1):
        bond = mol.GetBondBetweenAtoms(path[i], path[i + 1])
        if bond.GetBondTypeAsDouble() == 1.0 and not bond.IsInRing():
            rot += 1
    return {"backbone_rotatable_bonds": rot}


def _conjugation_one(smi):
    """
    Largest sp2-connected component size: number of sp2 atoms in the largest
    contiguous pi-conjugated subgraph. More faithful to Huckel-theory band gap
    physics than sp2 fraction — two molecules with the same sp2 fraction but
    different conjugation topology (isolated double bonds vs long conjugated
    backbone) have very different Egc values.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"max_conjugation_path": np.nan}
    sp2 = {
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2
    }
    if not sp2:
        return {"max_conjugation_path": 0}
    visited, max_comp = set(), 0
    for start in sp2:
        if start in visited:
            continue
        comp, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            for bond in mol.GetAtomWithIdx(node).GetBonds():
                nbr = bond.GetOtherAtomIdx(node)
                if nbr in sp2 and nbr not in comp:
                    stack.append(nbr)
        visited |= comp
        max_comp = max(max_comp, len(comp))
    return {"max_conjugation_path": max_comp}


def _build_chain(smi, n_units=N_CHAIN_UNITS):
    """
    Stitch n_units copies of a linear polymer repeat unit via * attachment points.
    Convention: sorted atom indices — lower-index * = left (head), higher = right (tail).
    Junction * atoms are removed and the neighbors bonded directly.
    Terminal * atoms are removed, leaving implicit H on the neighbor.
    Returns canonical SMILES, or None if the unit is invalid or stitching fails.
    """
    base = Chem.MolFromSmiles(smi)
    if base is None:
        return None
    base_stars = sorted(a.GetIdx() for a in base.GetAtoms() if a.GetAtomicNum() == 0)
    if len(base_stars) != 2:
        return None
    left_star_base, right_star_base = base_stars
    if (base.GetAtomWithIdx(left_star_base).GetDegree() != 1 or
            base.GetAtomWithIdx(right_star_base).GetDegree() != 1):
        return None

    chain = Chem.RWMol(base)
    chain_right_star = right_star_base

    for _ in range(n_units - 1):
        offset       = chain.GetNumAtoms()
        right_nbr    = chain.GetAtomWithIdx(chain_right_star).GetNeighbors()[0].GetIdx()
        bond_type    = chain.GetBondBetweenAtoms(chain_right_star, right_nbr).GetBondType()

        new_left_star  = left_star_base  + offset
        new_right_star = right_star_base + offset
        new_left_nbr   = base.GetAtomWithIdx(left_star_base).GetNeighbors()[0].GetIdx() + offset

        chain = Chem.RWMol(Chem.CombineMols(chain.GetMol(), base))
        chain.AddBond(right_nbr, new_left_nbr, bond_type)

        # Remove junction * atoms highest-index first to preserve lower indices
        for idx in sorted([chain_right_star, new_left_star], reverse=True):
            chain.RemoveAtom(idx)

        # Adjust new_right_star for the two removed atoms
        n_below = sum(1 for idx in [chain_right_star, new_left_star] if idx < new_right_star)
        chain_right_star = new_right_star - n_below

    # Remove remaining terminal * atoms
    for idx in sorted((a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
        chain.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(chain)
        return Chem.MolToSmiles(chain.GetMol())
    except Exception:
        return None


def compute_features(df):
    smiles = df["smiles"].tolist()

    # Build chain SMILES (trimer), fall back to monomer on failure
    chain_raw = [_build_chain(s) for s in smiles]
    n_fail    = sum(1 for c in chain_raw if c is None)
    chain_smi = [c if c is not None else s for c, s in zip(chain_raw, smiles)]
    if n_fail:
        print(f"    Chain build failures: {n_fail}/{len(smiles)} (fell back to monomer)")

    # Monomer features
    descs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_desc_one)(s)        for s in smiles)
    ecfp4  = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp4_one)(s)       for s in smiles)
    ecfp6  = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp6_one)(s)       for s in smiles)
    maccs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_maccs_one)(s)       for s in smiles)
    topo   = Parallel(n_jobs=-1, prefer="threads")(delayed(_topo_one)(s)        for s in smiles)
    elec   = Parallel(n_jobs=-1, prefer="threads")(delayed(_electronic_one)(s)  for s in smiles)
    tgfeat = Parallel(n_jobs=-1, prefer="threads")(delayed(_tg_specific_one)(s) for s in smiles)
    conj   = Parallel(n_jobs=-1, prefer="threads")(delayed(_conjugation_one)(s) for s in smiles)

    # Chain features (trimer) — no topology or tg_specific (needs * atoms)
    ch_descs = Parallel(n_jobs=-1, prefer="threads")(delayed(_desc_one)(s)        for s in chain_smi)
    ch_ecfp4 = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp4_one)(s)       for s in chain_smi)
    ch_ecfp6 = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp6_one)(s)       for s in chain_smi)
    ch_maccs = Parallel(n_jobs=-1, prefer="threads")(delayed(_maccs_one)(s)       for s in chain_smi)
    ch_elec  = Parallel(n_jobs=-1, prefer="threads")(delayed(_electronic_one)(s)  for s in chain_smi)
    ch_conj  = Parallel(n_jobs=-1, prefer="threads")(delayed(_conjugation_one)(s) for s in chain_smi)

    p = f"ch{N_CHAIN_UNITS}_"

    desc_df   = pd.DataFrame(descs,   index=df.index)
    ecfp4_df  = pd.DataFrame(ecfp4,   index=df.index, columns=[f"ecfp4_{i}" for i in range(2048)])
    ecfp6_df  = pd.DataFrame(ecfp6,   index=df.index, columns=[f"ecfp6_{i}" for i in range(2048)])
    maccs_df  = pd.DataFrame(maccs,   index=df.index, columns=[f"maccs_{i}" for i in range(167)])
    topo_df   = pd.DataFrame(topo,    index=df.index)
    elec_df   = pd.DataFrame(elec,    index=df.index)
    tgfeat_df = pd.DataFrame(tgfeat,  index=df.index)
    conj_df   = pd.DataFrame(conj,    index=df.index)

    ch_desc_df  = pd.DataFrame(ch_descs, index=df.index).add_prefix(p)
    ch_ecfp4_df = pd.DataFrame(ch_ecfp4, index=df.index, columns=[f"{p}ecfp4_{i}" for i in range(2048)])
    ch_ecfp6_df = pd.DataFrame(ch_ecfp6, index=df.index, columns=[f"{p}ecfp6_{i}" for i in range(2048)])
    ch_maccs_df = pd.DataFrame(ch_maccs, index=df.index, columns=[f"{p}maccs_{i}" for i in range(167)])
    ch_elec_df  = pd.DataFrame(ch_elec,  index=df.index).add_prefix(p)
    ch_conj_df  = pd.DataFrame(ch_conj,  index=df.index).add_prefix(p)

    combined = pd.concat([
        desc_df, ecfp4_df, ecfp6_df, maccs_df, topo_df, elec_df, tgfeat_df, conj_df,
        ch_desc_df, ch_ecfp4_df, ch_ecfp6_df, ch_maccs_df, ch_elec_df, ch_conj_df,
    ], axis=1)
    # Cast to float32 first: RDKit's Ipc descriptor can exceed float32 max (~3.4e38),
    # which overflows silently to inf when XGBoost casts internally, crashing QuantileDMatrix.
    # Doing it here surfaces the overflow as NaN before XGB ever sees the data.
    combined = combined.astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return combined


# ── Load ───────────────────────────────────────────────────────────────────────

start = datetime.now()
print(f"Started at {start.strftime('%H:%M:%S')}\n")

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  train: {len(train):,} rows  |  test: {len(test):,} rows")

print("\nValidating SMILES...")
train_bad = train["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is None).sum()
test_bad  = test["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is None).sum()
print(f"  train: {train_bad} unparseable  |  test: {test_bad} unparseable")

def _to_canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi

train["canon_smiles"] = train["smiles"].apply(_to_canon)
dup_groups = (
    train.groupby(["canon_smiles", "target_type"])["target"]
    .agg(list)
    .reset_index()
)
dup_groups = dup_groups[dup_groups["target"].apply(len) > 1]
print(f"\nDuplicate (canon SMILES, target_type) groups: {len(dup_groups)}")
if len(dup_groups) > 0:
    print("  [spread = max - min; large spread = labeling conflict that caps model R2]")
    for _, row in dup_groups.iterrows():
        vals   = row["target"]
        spread = max(vals) - min(vals)
        print(
            f"  {row['target_type'].upper()}  spread={spread:.2f}"
            f"  values={[round(v, 2) for v in vals]}"
            f"  smiles={row['canon_smiles'][:50]}"
        )

# Mean-merge duplicate (canon_smiles, target_type) rows before feature extraction
n_before = len(train)
train = (
    train.groupby(["canon_smiles", "target_type"], as_index=False)
    .agg(smiles=("smiles", "first"), target=("target", "mean"))
)
n_after = len(train)
print(f"\nDuplicate merge: {n_before:,} -> {n_after:,} rows  ({n_before - n_after} groups collapsed to mean)")


# ── Features ───────────────────────────────────────────────────────────────────

print("\nComputing features...")
X_train = compute_features(train)
X_test  = compute_features(test)
print(f"  Done.  {X_train.shape[1]} features per molecule")

y_train     = train["target"].values
strat_label = train["target_type"].values

# 70% of train SMILES are non-canonical — without canonicalizing, the same
# physical molecule can appear in both train and val folds under different strings.
groups = train["canon_smiles"].values

# Preserve full feature matrices; pruning is re-done per seed (fold 0 differs by seed)
X_train_full = X_train
X_test_full  = X_test

# Test subsets are the same across all seeds — define once
test_tg      = test[test["target_type"] == "tg"].copy()
test_egc     = test[test["target_type"] == "egc"].copy()
test_subsets = {"tg": test_tg, "egc": test_egc}

# Accumulators for multi-seed averaging
all_seed_test_tg  = []
all_seed_test_egc = []
seed_cv_scores    = []

print(f"\nMulti-seed run: SEEDS={SEEDS}  N_TRIALS={N_TRIALS}  N_XGB_TRIALS={N_XGB_TRIALS}")
# Calibrated from observed runtimes: features ~20 min (once), then per seed:
#   pruning + CV folds ~80 min (constant), LGB tuning ~5 min/trial, XGB ~5 min/trial
_per_seed_hr = (80 + N_TRIALS * 5 + N_XGB_TRIALS * 5) / 60
_est_hr      = 0.33 + len(SEEDS) * _per_seed_hr
print(f"Estimated runtime: ~{_est_hr:.1f} hours\n")


# ── Per-seed loop: prune -> tune -> CV -> collect test predictions ─────────────

for run_idx, run_seed in enumerate(SEEDS):
    print(f"\n{'#'*60}")
    print(f"  RUN {run_idx + 1}/{len(SEEDS)}  (seed={run_seed})")
    print(f"{'#'*60}")

    # Override random_state with this seed in both model configs
    _lgb_fixed = {**LGB_FIXED, "random_state": run_seed}
    _xgb_fixed = {**XGB_FIXED, "random_state": run_seed}

    sgkf      = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=run_seed)
    cv_splits = list(sgkf.split(X_train_full, strat_label, groups))

    # ── Feature pruning ────────────────────────────────────────────────────────
    # Re-done each seed: fold 0 (the probe fold) changes with the seed, so
    # keep_cols differs slightly across seeds — each seed prunes a different
    # random slice of the data, which is exactly what we want for variance reduction.

    print(f"\n{'='*60}")
    print(f"  FEATURE PRUNING  ({PROBE_N_ESTIMATORS}-tree probe, permutation importance)")
    print(f"{'='*60}\n")

    n_orig    = X_train_full.shape[1]
    keep_cols = set()
    probe_tr_idx, probe_val_idx = cv_splits[0]
    for ttype in ["tg", "egc"]:
        mask_tr  = strat_label[probe_tr_idx]  == ttype
        mask_val = strat_label[probe_val_idx] == ttype
        X_probe_tr  = X_train_full.iloc[probe_tr_idx][mask_tr]
        X_probe_val = X_train_full.iloc[probe_val_idx][mask_val]
        y_probe_tr  = y_train[probe_tr_idx][mask_tr]
        y_probe_val = y_train[probe_val_idx][mask_val]

        probe = LGBMRegressor(
            n_estimators=PROBE_N_ESTIMATORS, num_leaves=63,
            random_state=run_seed, n_jobs=-1, verbose=-1,
        )
        probe.fit(X_probe_tr, y_probe_tr)

        result = permutation_importance(
            probe, X_probe_val, y_probe_val,
            n_repeats=1, random_state=run_seed, n_jobs=-1,
        )
        imp      = pd.Series(result.importances_mean, index=X_train_full.columns)
        positive = imp[imp > 0].index
        keep_cols.update(positive)
        print(f"  {ttype.upper()}: {len(positive):,} / {n_orig:,} features with positive permutation importance")

    keep_cols = sorted(keep_cols)
    X_train   = X_train_full[keep_cols]
    X_test    = X_test_full[keep_cols]
    print(f"\n  Kept {len(keep_cols):,} features, dropped {n_orig - len(keep_cols):,} zero-importance")

    # ── LGB tuning ─────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  LGB TUNING  ({N_TRIALS} trials per target)")
    print(f"{'='*60}")

    # Default-argument capture avoids closure-in-loop bugs for the seed-local variables
    def make_lgb_objective(ttype, _cv=cv_splits, _X=X_train, _y=y_train,
                           _sl=strat_label, _fixed=_lgb_fixed):
        def objective(trial):
            params = {
                **_fixed,
                "n_estimators"     : TUNE_N_ESTIMATORS,
                "num_leaves"       : trial.suggest_int("num_leaves", 15, 255),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_split_gain"   : trial.suggest_float("min_split_gain", 0.0, 1.0),
                "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            scores = []
            for tr_idx, val_idx in _cv:
                mask_tr  = _sl[tr_idx]  == ttype
                mask_val = _sl[val_idx] == ttype
                X_tr  = _X.iloc[tr_idx][mask_tr]
                X_val = _X.iloc[val_idx][mask_val]
                y_tr  = _y[tr_idx][mask_tr]
                y_val = _y[val_idx][mask_val]
                model = LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
                )
                pred = model.predict(X_val)
                scores.append(r2_score(y_val, pred))
            return np.mean(scores)
        return objective

    best_lgb_params = {}
    for ttype in ["tg", "egc"]:
        print(f"\n  Tuning LGB {ttype.upper()}...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=run_seed),
        )
        study.optimize(make_lgb_objective(ttype), n_trials=N_TRIALS, show_progress_bar=False)
        best_lgb_params[ttype] = {**_lgb_fixed, **study.best_params}
        print(f"  Best R²({ttype.upper()}) = {study.best_value:+.4f}")
        print(f"  Best params: {study.best_params}")

    # ── XGB tuning ─────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  XGB TUNING  ({N_XGB_TRIALS} trials per target)")
    print(f"{'='*60}")

    def make_xgb_objective(ttype, _cv=cv_splits, _X=X_train, _y=y_train,
                           _sl=strat_label, _fixed=_xgb_fixed):
        def objective(trial):
            params = {
                **_fixed,
                "n_estimators"    : TUNE_N_ESTIMATORS,
                "max_depth"       : trial.suggest_int("max_depth", 3, 8),
                "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            }
            scores = []
            for tr_idx, val_idx in _cv:
                mask_tr  = _sl[tr_idx]  == ttype
                mask_val = _sl[val_idx] == ttype
                X_tr  = _X.iloc[tr_idx][mask_tr]
                X_val = _X.iloc[val_idx][mask_val]
                y_tr  = _y[tr_idx][mask_tr]
                y_val = _y[val_idx][mask_val]
                model = XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                pred = model.predict(X_val)
                scores.append(r2_score(y_val, pred))
            return np.mean(scores)
        return objective

    best_xgb_params = {}
    for ttype in ["tg", "egc"]:
        print(f"\n  Tuning XGB {ttype.upper()}...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=run_seed),
        )
        study.optimize(make_xgb_objective(ttype), n_trials=N_XGB_TRIALS, show_progress_bar=False)
        best_xgb_params[ttype] = {**_xgb_fixed, **study.best_params}
        print(f"  Best R²({ttype.upper()}) = {study.best_value:+.4f}")
        print(f"  Best params: {study.best_params}")

    # ── Final CV: score + accumulate fold-ensembled test predictions ───────────

    print(f"\n{'='*60}")
    print(f"  FINAL CV  (LGB+XGB blend)  +  FOLD-ENSEMBLED TEST PREDICTIONS")
    print(f"{'='*60}\n")

    test_pred_lgb  = {ttype: np.zeros(len(subset)) for ttype, subset in test_subsets.items()}
    test_pred_xgb  = {ttype: np.zeros(len(subset)) for ttype, subset in test_subsets.items()}
    fold_lgb_preds = {"tg": [], "egc": []}
    fold_xgb_preds = {"tg": [], "egc": []}
    fold_y_vals    = {"tg": [], "egc": []}

    for fold, (tr_idx, val_idx) in enumerate(cv_splits, 1):
        types_tr  = strat_label[tr_idx]
        types_val = strat_label[val_idx]

        scores = {}
        for ttype in ["tg", "egc"]:
            mask_tr  = types_tr  == ttype
            mask_val = types_val == ttype
            X_tr  = X_train.iloc[tr_idx][mask_tr]
            X_val = X_train.iloc[val_idx][mask_val]
            y_tr  = y_train[tr_idx][mask_tr]
            y_val = y_train[val_idx][mask_val]
            X_test_sub = X_test.loc[test_subsets[ttype].index]

            lgb_model = LGBMRegressor(**best_lgb_params[ttype])
            lgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
            )

            xgb_model = XGBRegressor(**best_xgb_params[ttype], early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            lgb_val  = lgb_model.predict(X_val)
            xgb_val  = xgb_model.predict(X_val)
            lgb_test = lgb_model.predict(X_test_sub)
            xgb_test = xgb_model.predict(X_test_sub)

            scores[ttype] = r2_score(y_val, (lgb_val + xgb_val) / 2)  # naive 50/50 for per-fold print
            fold_lgb_preds[ttype].append(lgb_val)
            fold_xgb_preds[ttype].append(xgb_val)
            fold_y_vals[ttype].append(y_val)
            test_pred_lgb[ttype] += lgb_test
            test_pred_xgb[ttype] += xgb_test

        mean = (scores["tg"] + scores["egc"]) / 2
        print(f"  Fold {fold}  R²(Tg)={scores['tg']:+.4f}  R²(Egc)={scores['egc']:+.4f}  mean={mean:+.4f}  (naive 50/50)")

    # ── Blend weight search ────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  BLEND WEIGHT SEARCH  (grid w=0.1...0.9, step 0.1, all OOF)")
    print(f"{'='*60}\n")

    blend_w = {}
    for ttype in ["tg", "egc"]:
        lgb_oof = np.concatenate(fold_lgb_preds[ttype])
        xgb_oof = np.concatenate(fold_xgb_preds[ttype])
        y_oof   = np.concatenate(fold_y_vals[ttype])

        naive_r2 = r2_score(y_oof, 0.5 * lgb_oof + 0.5 * xgb_oof)
        best_w, best_r2 = 0.5, naive_r2
        for w in np.round(np.arange(0.1, 1.0, 0.1), 1):
            r2 = r2_score(y_oof, w * lgb_oof + (1.0 - w) * xgb_oof)
            if r2 > best_r2:
                best_r2, best_w = r2, float(w)
        blend_w[ttype] = best_w
        print(
            f"  {ttype.upper()}: w(LGB)={best_w:.1f}  w(XGB)={(1.0 - best_w):.1f}"
            f"  OOF R²={best_r2:+.4f}  naive 50/50: {naive_r2:+.4f}  delta={best_r2 - naive_r2:+.5f}"
        )

    # Fold R² recomputed with optimal per-target blend weights
    fold_r2_opt = {"tg": [], "egc": []}
    for ttype in ["tg", "egc"]:
        for i in range(N_FOLDS):
            pred = blend_w[ttype] * fold_lgb_preds[ttype][i] + (1.0 - blend_w[ttype]) * fold_xgb_preds[ttype][i]
            fold_r2_opt[ttype].append(r2_score(fold_y_vals[ttype][i], pred))

    cv_tg  = np.mean(fold_r2_opt["tg"])
    cv_egc = np.mean(fold_r2_opt["egc"])
    cv_r2  = (cv_tg + cv_egc) / 2

    print(f"\n  Mean R²(Tg)  = {cv_tg:+.4f}  (std {np.std(fold_r2_opt['tg']):.4f})")
    print(f"  Mean R²(Egc) = {cv_egc:+.4f}  (std {np.std(fold_r2_opt['egc']):.4f})")
    print(f"\n  >>> CV score (seed={run_seed}) = {cv_r2:+.4f} <<<")

    # Collect blended, fold-averaged test predictions for this seed
    seed_tg_pred  = (blend_w["tg"]  * test_pred_lgb["tg"]  + (1 - blend_w["tg"])  * test_pred_xgb["tg"])  / N_FOLDS
    seed_egc_pred = (blend_w["egc"] * test_pred_lgb["egc"] + (1 - blend_w["egc"]) * test_pred_xgb["egc"]) / N_FOLDS
    all_seed_test_tg.append(seed_tg_pred)
    all_seed_test_egc.append(seed_egc_pred)
    seed_cv_scores.append(cv_r2)


# ── Multi-seed summary ─────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  MULTI-SEED SUMMARY")
print(f"{'='*60}")
for seed, score in zip(SEEDS, seed_cv_scores):
    print(f"  Seed {seed:3d}: CV = {score:+.4f}")
print(f"  Mean CV  : {np.mean(seed_cv_scores):+.4f}  (std {np.std(seed_cv_scores):.4f})")

# Average test predictions across seeds
test_tg["target"]  = np.mean(all_seed_test_tg,  axis=0)
test_egc["target"] = np.mean(all_seed_test_egc, axis=0)


# ── Submission ─────────────────────────────────────────────────────────────────

submission = (
    pd.concat([test_tg, test_egc])[["id", "target"]]
    .sort_values("id")
)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)

print(f"\nSubmission saved -> {OUTPUT_PATH}")
print(f"  {len(submission):,} rows  |  id range: {submission['id'].min()}-{submission['id'].max()}")
print(f"\n  First 5 rows:")
print(submission.head(5).to_string(index=False))

elapsed = datetime.now() - start
print(f"\nDone. Total time: {elapsed}")
