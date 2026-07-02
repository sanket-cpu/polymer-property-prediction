"""
Baseline model with proper validation harness.

Features  : RDKit descriptors (~210) + ECFP4 (2048 bits) + ECFP6 (2048 bits)
            + MACCS keys (167 bits) + topology features (2)
            + electronic features (5: aromatic rings, aromatic fraction,
              rotatable bonds, sp2 fraction, non-aromatic double bond count)
Model     : LightGBM + XGBoost blend, hyperparameters tuned per target with Optuna
Validation: 5-fold CV grouped by *canonical* SMILES, stratified by target_type, scored mean R²
Prediction: average of 5-fold LGB and XGB models (10 models total per target)
Pipeline  : features -> prune (300-tree probe) -> tune LGB + XGB -> CV + fold ensemble

Run from project root:
    python3 scripts/baseline.py          (WSL)
    .venv/Scripts/python.exe scripts/baseline.py  (Windows)
Output: outputs/submission.csv
"""

import sys
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor

sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_PATH  = "data/train.csv"
TEST_PATH   = "data/test.csv"
OUTPUT_PATH = "outputs/submission.csv"
N_FOLDS     = 5
SEED        = 42

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
    descs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_desc_one)(s)       for s in smiles)
    ecfp4  = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp4_one)(s)      for s in smiles)
    ecfp6  = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp6_one)(s)      for s in smiles)
    maccs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_maccs_one)(s)      for s in smiles)
    topo   = Parallel(n_jobs=-1, prefer="threads")(delayed(_topo_one)(s)       for s in smiles)
    elec   = Parallel(n_jobs=-1, prefer="threads")(delayed(_electronic_one)(s) for s in smiles)

    # Chain features (trimer) — no topology since chain has no * atoms
    ch_descs = Parallel(n_jobs=-1, prefer="threads")(delayed(_desc_one)(s)       for s in chain_smi)
    ch_ecfp4 = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp4_one)(s)      for s in chain_smi)
    ch_ecfp6 = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp6_one)(s)      for s in chain_smi)
    ch_maccs = Parallel(n_jobs=-1, prefer="threads")(delayed(_maccs_one)(s)      for s in chain_smi)
    ch_elec  = Parallel(n_jobs=-1, prefer="threads")(delayed(_electronic_one)(s) for s in chain_smi)

    p = f"ch{N_CHAIN_UNITS}_"

    desc_df  = pd.DataFrame(descs,  index=df.index)
    ecfp4_df = pd.DataFrame(ecfp4,  index=df.index, columns=[f"ecfp4_{i}" for i in range(2048)])
    ecfp6_df = pd.DataFrame(ecfp6,  index=df.index, columns=[f"ecfp6_{i}" for i in range(2048)])
    maccs_df = pd.DataFrame(maccs,  index=df.index, columns=[f"maccs_{i}" for i in range(167)])
    topo_df  = pd.DataFrame(topo,   index=df.index)
    elec_df  = pd.DataFrame(elec,   index=df.index)

    ch_desc_df = pd.DataFrame(ch_descs, index=df.index).add_prefix(p)
    ch_ecfp4_df = pd.DataFrame(ch_ecfp4, index=df.index, columns=[f"{p}ecfp4_{i}" for i in range(2048)])
    ch_ecfp6_df = pd.DataFrame(ch_ecfp6, index=df.index, columns=[f"{p}ecfp6_{i}" for i in range(2048)])
    ch_maccs_df = pd.DataFrame(ch_maccs, index=df.index, columns=[f"{p}maccs_{i}" for i in range(167)])
    ch_elec_df  = pd.DataFrame(ch_elec,  index=df.index).add_prefix(p)

    combined = pd.concat([
        desc_df, ecfp4_df, ecfp6_df, maccs_df, topo_df, elec_df,
        ch_desc_df, ch_ecfp4_df, ch_ecfp6_df, ch_maccs_df, ch_elec_df,
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

# Fix B: SMILES validation — catch failures before feature extraction
print("\nValidating SMILES...")
train_bad = train["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is None).sum()
test_bad  = test["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is None).sum()
print(f"  train: {train_bad} unparseable  |  test: {test_bad} unparseable")

# Fix C: canonical SMILES + duplicate label check
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
    print("  [spread = max − min; large spread = labeling conflict that caps model R²]")
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
print(f"\nDuplicate merge: {n_before:,} → {n_after:,} rows  ({n_before - n_after} groups collapsed to mean)")


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

sgkf      = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_splits = list(sgkf.split(X_train, strat_label, groups))


# ── Feature pruning (before tuning) ───────────────────────────────────────────
# Prune before tuning so Optuna finds hyperparameters valid for the actual
# feature set the final models will see. colsample_bytree on 4500 features
# is a different quantity than on 1500 — they are not interchangeable.

print(f"\n{'='*60}")
print(f"  FEATURE PRUNING  ({PROBE_N_ESTIMATORS}-tree probe, before tuning)")
print(f"{'='*60}\n")

n_orig    = X_train.shape[1]
keep_cols = set()
for ttype in ["tg", "egc"]:
    mask  = strat_label == ttype
    probe = LGBMRegressor(
        n_estimators=PROBE_N_ESTIMATORS, num_leaves=63,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    probe.fit(X_train[mask], y_train[mask])
    imp     = pd.Series(probe.feature_importances_, index=X_train.columns)
    nonzero = imp[imp > 0].index
    keep_cols.update(nonzero)
    print(f"  {ttype.upper()}: {len(nonzero):,} / {n_orig:,} features used")

keep_cols = sorted(keep_cols)
X_train   = X_train[keep_cols]
X_test    = X_test[keep_cols]
print(f"\n  Kept {len(keep_cols):,} features, dropped {n_orig - len(keep_cols):,} zero-importance")


# ── LGB hyperparameter tuning ─────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  LGB TUNING  ({N_TRIALS} trials per target)")
print(f"{'='*60}")

def make_lgb_objective(ttype):
    def objective(trial):
        params = {
            **LGB_FIXED,
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
        for tr_idx, val_idx in cv_splits:
            mask_tr  = strat_label[tr_idx]  == ttype
            mask_val = strat_label[val_idx] == ttype
            X_tr  = X_train.iloc[tr_idx][mask_tr]
            X_val = X_train.iloc[val_idx][mask_val]
            y_tr  = y_train[tr_idx][mask_tr]
            y_val = y_train[val_idx][mask_val]

            y_tr_fit, y_val_fit = y_tr, y_val

            model = LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr_fit,
                eval_set=[(X_val, y_val_fit)],
                callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
            )
            pred = model.predict(X_val)
            scores.append(r2_score(y_val, pred))
        return np.mean(scores)
    return objective

best_lgb_params = {}
for ttype in ["tg", "egc"]:
    print(f"\n  Tuning LGB {ttype.upper()}...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_lgb_objective(ttype), n_trials=N_TRIALS, show_progress_bar=False)
    best_lgb_params[ttype] = {**LGB_FIXED, **study.best_params}
    print(f"  Best R²({ttype.upper()}) = {study.best_value:+.4f}")
    print(f"  Best params: {study.best_params}")


# ── XGB hyperparameter tuning ─────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  XGB TUNING  ({N_XGB_TRIALS} trials per target)")
print(f"{'='*60}")

def make_xgb_objective(ttype):
    def objective(trial):
        params = {
            **XGB_FIXED,
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
        for tr_idx, val_idx in cv_splits:
            mask_tr  = strat_label[tr_idx]  == ttype
            mask_val = strat_label[val_idx] == ttype
            X_tr  = X_train.iloc[tr_idx][mask_tr]
            X_val = X_train.iloc[val_idx][mask_val]
            y_tr  = y_train[tr_idx][mask_tr]
            y_val = y_train[val_idx][mask_val]

            y_tr_fit, y_val_fit = y_tr, y_val

            model = XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(X_tr, y_tr_fit, eval_set=[(X_val, y_val_fit)], verbose=False)
            pred = model.predict(X_val)
            scores.append(r2_score(y_val, pred))
        return np.mean(scores)
    return objective

best_xgb_params = {}
for ttype in ["tg", "egc"]:
    print(f"\n  Tuning XGB {ttype.upper()}...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_xgb_objective(ttype), n_trials=N_XGB_TRIALS, show_progress_bar=False)
    best_xgb_params[ttype] = {**XGB_FIXED, **study.best_params}
    print(f"  Best R²({ttype.upper()}) = {study.best_value:+.4f}")
    print(f"  Best params: {study.best_params}")


# ── Final CV: score + accumulate fold-ensembled test predictions ───────────────

print(f"\n{'='*60}")
print(f"  FINAL CV  (LGB+XGB blend)  +  FOLD-ENSEMBLED TEST PREDICTIONS")
print(f"{'='*60}\n")

test_tg      = test[test["target_type"] == "tg"].copy()
test_egc     = test[test["target_type"] == "egc"].copy()
test_subsets = {"tg": test_tg, "egc": test_egc}

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

        y_tr_fit, y_val_fit = y_tr, y_val

        lgb_model = LGBMRegressor(**best_lgb_params[ttype])
        lgb_model.fit(
            X_tr, y_tr_fit,
            eval_set=[(X_val, y_val_fit)],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
        )

        xgb_model = XGBRegressor(**best_xgb_params[ttype], early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        xgb_model.fit(X_tr, y_tr_fit, eval_set=[(X_val, y_val_fit)], verbose=False)

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

# ── Blend weight search on out-of-fold predictions ────────────────────────────

print(f"\n{'='*60}")
print(f"  BLEND WEIGHT SEARCH  (grid w=0.1…0.9, step 0.1, all OOF)")
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
        f"  {ttype.upper()}: w(LGB)={best_w:.1f}  w(XGB)={(1.0-best_w):.1f}"
        f"  OOF R²={best_r2:+.4f}  naive 50/50: {naive_r2:+.4f}  Δ={best_r2-naive_r2:+.5f}"
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
print(f"\n  >>> CV score (competition metric) = {cv_r2:+.4f} <<<")

for ttype, subset in test_subsets.items():
    lgb_avg = test_pred_lgb[ttype] / N_FOLDS
    xgb_avg = test_pred_xgb[ttype] / N_FOLDS
    subset["target"] = blend_w[ttype] * lgb_avg + (1.0 - blend_w[ttype]) * xgb_avg


# ── Submission ─────────────────────────────────────────────────────────────────

submission = (
    pd.concat([test_tg, test_egc])[["id", "target"]]
    .sort_values("id")
)
submission.to_csv(OUTPUT_PATH, index=False)

print(f"\nSubmission saved -> {OUTPUT_PATH}")
print(f"  {len(submission):,} rows  |  id range: {submission['id'].min()}-{submission['id'].max()}")
print(f"\n  First 5 rows:")
print(submission.head(5).to_string(index=False))

elapsed = datetime.now() - start
print(f"\nDone. Total time: {elapsed}")
