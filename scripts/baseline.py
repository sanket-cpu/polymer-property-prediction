"""
Baseline model with proper validation harness.

Features  : RDKit descriptors (~210) + Morgan fingerprints (2048 bits)
            + 2 polymer-topology features (backbone span) = ~2260 total
Model     : LightGBM, hyperparameters tuned per target with Optuna
Validation: 5-fold CV grouped by SMILES, stratified by target_type, scored mean R²
Prediction: ensemble of the 5 fold-trained models (average), not a single retrain

Run from project root:
    polymer-property-prediction/Scripts/python.exe scripts/baseline.py
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
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_PATH  = "data/train.csv"
TEST_PATH   = "data/test.csv"
OUTPUT_PATH = "outputs/submission.csv"
N_FOLDS     = 5
SEED        = 42

EARLY_STOPPING_ROUNDS = 50   # stop if no improvement for 50 consecutive trees
N_TRIALS              = 50   # Optuna trials per target (50 x 2 targets x 5 folds = 500 fits)
TUNE_N_ESTIMATORS     = 1000 # lower ceiling during tuning — relative comparison only
FINAL_N_ESTIMATORS    = 3000 # higher ceiling for final CV — full model quality

LGB_FIXED = {
    "n_estimators" : FINAL_N_ESTIMATORS,  # overridden to TUNE_N_ESTIMATORS during search
    "random_state" : SEED,
    "n_jobs"       : -1,
    "verbose"      : -1,
}


# ── Feature computation ────────────────────────────────────────────────────────

def _desc_one(smi):
    return Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smi))

def _fp_one(smi):
    mol = Chem.MolFromSmiles(smi)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

def _maccs_one(smi):
    mol = Chem.MolFromSmiles(smi)
    return list(MACCSkeys.GenMACCSKeys(mol))

def _topo_one(smi):
    # Backbone span: shortest-path distance (in bonds) between the two `*`
    # attachment points. This is the repeat unit's "backbone length" — a
    # longer/more flexible backbone tends to lower Tg, a short rigid one
    # raises it. star_distance_frac normalizes that span against the whole
    # molecule's topological diameter, so we also capture whether the
    # backbone IS basically the whole molecule (frac near 1) or just a path
    # through a bulkier structure with side groups (frac well below 1).
    mol = Chem.MolFromSmiles(smi)
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    dmat = Chem.GetDistanceMatrix(mol)
    star_dist = dmat[star_idx[0], star_idx[1]]
    diameter = dmat.max()
    return {
        "star_distance": star_dist,
        "star_distance_frac": star_dist / diameter if diameter > 0 else 0.0,
    }

def compute_features(df):
    smiles = df["smiles"].tolist()
    descs = Parallel(n_jobs=-1, prefer="threads")(delayed(_desc_one)(s)  for s in smiles)
    fps   = Parallel(n_jobs=-1, prefer="threads")(delayed(_fp_one)(s)    for s in smiles)
    maccs = Parallel(n_jobs=-1, prefer="threads")(delayed(_maccs_one)(s) for s in smiles)
    topo  = Parallel(n_jobs=-1, prefer="threads")(delayed(_topo_one)(s)  for s in smiles)

    desc_df  = pd.DataFrame(descs, index=df.index)
    fp_df    = pd.DataFrame(fps,   index=df.index,
                             columns=[f"morgan_{i}" for i in range(2048)])
    maccs_df = pd.DataFrame(maccs, index=df.index,
                             columns=[f"maccs_{i}" for i in range(167)])
    topo_df  = pd.DataFrame(topo,  index=df.index)

    combined = pd.concat([desc_df, fp_df, maccs_df, topo_df], axis=1)
    # LightGBM handles NaN natively — only need to remove inf
    return combined.replace([np.inf, -np.inf], np.nan)


# ── Load ───────────────────────────────────────────────────────────────────────

start = datetime.now()
print(f"Started at {start.strftime('%H:%M:%S')}\n")

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  train: {len(train):,} rows  |  test: {len(test):,} rows")


# ── Features ───────────────────────────────────────────────────────────────────

print("\nComputing features (RDKit descriptors + Morgan fingerprints + topology)...")
X_train = compute_features(train)
X_test  = compute_features(test)
print(f"  Done.  {X_train.shape[1]} features per molecule")

y_train     = train["target"].values
groups      = train["smiles"].values
strat_label = train["target_type"].values

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_splits = list(sgkf.split(X_train, strat_label, groups))


# ── Hyperparameter tuning (Optuna, per target) ──────────────────────────────────

print(f"\n{'='*58}")
print(f"  HYPERPARAMETER TUNING  ({N_TRIALS} trials per target)")
print(f"{'='*58}")

def make_objective(ttype):
    def objective(trial):
        params = {
            **LGB_FIXED,
            "n_estimators"     : TUNE_N_ESTIMATORS,
            "num_leaves"       : trial.suggest_int("num_leaves", 15, 127),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        scores = []
        for tr_idx, val_idx in cv_splits:
            mask_tr  = strat_label[tr_idx]  == ttype
            mask_val = strat_label[val_idx] == ttype
            X_tr, X_val = X_train.iloc[tr_idx][mask_tr], X_train.iloc[val_idx][mask_val]
            y_tr, y_val = y_train[tr_idx][mask_tr], y_train[val_idx][mask_val]

            model = LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
            )
            scores.append(r2_score(y_val, model.predict(X_val)))
        return np.mean(scores)
    return objective

best_params = {}
for ttype in ["tg", "egc"]:
    print(f"\n  Tuning {ttype.upper()}...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(ttype), n_trials=N_TRIALS, show_progress_bar=False)
    best_params[ttype] = {**LGB_FIXED, **study.best_params}
    print(f"  Best R²({ttype.upper()}) = {study.best_value:+.4f}")
    print(f"  Best params: {study.best_params}")


# ── Final CV: score + accumulate ensembled test predictions ────────────────────

print(f"\n{'='*58}")
print(f"  FINAL CV  (tuned params)  +  FOLD-ENSEMBLED TEST PREDICTIONS")
print(f"{'='*58}\n")

test_tg  = test[test["target_type"] == "tg"].copy()
test_egc = test[test["target_type"] == "egc"].copy()
test_subsets = {"tg": test_tg, "egc": test_egc}

test_pred_sum = {ttype: np.zeros(len(subset)) for ttype, subset in test_subsets.items()}

fold_r2 = {"tg": [], "egc": []}

for fold, (tr_idx, val_idx) in enumerate(cv_splits, 1):
    types_tr  = strat_label[tr_idx]
    types_val = strat_label[val_idx]

    scores = {}
    for ttype in ["tg", "egc"]:
        mask_tr  = types_tr  == ttype
        mask_val = types_val == ttype
        X_tr, X_val = X_train.iloc[tr_idx][mask_tr], X_train.iloc[val_idx][mask_val]
        y_tr, y_val = y_train[tr_idx][mask_tr], y_train[val_idx][mask_val]

        model = LGBMRegressor(**best_params[ttype])
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
        )
        scores[ttype] = r2_score(y_val, model.predict(X_val))
        fold_r2[ttype].append(scores[ttype])

        # accumulate this fold's prediction on the real test set (ensemble)
        test_pred_sum[ttype] += model.predict(X_test.loc[test_subsets[ttype].index])

    mean = (scores["tg"] + scores["egc"]) / 2
    print(f"  Fold {fold}  R²(Tg)={scores['tg']:+.4f}  R²(Egc)={scores['egc']:+.4f}  mean={mean:+.4f}")

cv_tg  = np.mean(fold_r2["tg"])
cv_egc = np.mean(fold_r2["egc"])
cv_r2  = (cv_tg + cv_egc) / 2

print(f"\n  Mean R²(Tg)  = {cv_tg:+.4f}  (std {np.std(fold_r2['tg']):.4f})")
print(f"  Mean R²(Egc) = {cv_egc:+.4f}  (std {np.std(fold_r2['egc']):.4f})")
print(f"\n  >>> CV score (competition metric) = {cv_r2:+.4f} <<<")

for ttype, subset in test_subsets.items():
    subset["target"] = test_pred_sum[ttype] / N_FOLDS


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
