import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV

# Force UTF-8 on the terminal
sys.stdout = io.TextIOWrapper(sys.__stdout__.buffer, encoding="utf-8", line_buffering=True)

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

TRAIN_PATH  = "train.csv"
TEST_PATH   = "test.csv"
OUTPUT_PATH = "outputs/submission.csv"
N_FOLDS     = 5
SEED        = 20
SEEDS       = [42, 0, 123, 7, 99]

EARLY_STOPPING_ROUNDS = 50
FINAL_N_ESTIMATORS    = 3000
PROBE_N_ESTIMATORS    = 300
N_CHAIN_UNITS         = 3

# ── Optimized Search Budget for Colab Infrastructure ──────────────────────────
N_CANDIDATES      = 4     # Lightened from 16 to fit 2-vCPU limits
N_SURVIVORS       = 2     # Lightened from 4
FIRST_STAGE_FOLDS = 1
TUNE_N_ESTIMATORS = 300   # Fast tuning estimators down from 1000

# Feature-pruning knobs
LOW_VAR_THRESHOLD = 1e-4
TOP_K_FEATURES    = 500

# Core throttling handles
_N_CPU         = os.cpu_count() or 2
_TUNE_N_JOBS   = -1
_FINAL_N_JOBS  = -1

LGB_FIXED = {
    "n_estimators" : FINAL_N_ESTIMATORS,
    "random_state" : SEED,
    "n_jobs"       : -1,
    "verbose"      : -1,
}

XGB_FIXED = {
    "n_estimators" : FINAL_N_ESTIMATORS,
    "random_state" : SEED,
    "n_jobs"       : -1,
    "tree_method"  : "hist",
    "device"       : "cuda",  # Leverages Colab T4 GPU
}

CAT_FIXED = {
    "n_estimators"         : FINAL_N_ESTIMATORS,
    "random_seed"          : SEED,
    "verbose"              : 0,
    "task_type"            : "GPU",  # Leverages Colab T4 GPU
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}

# Hyperparameter Search Spaces
LGB_SPACE = {
    "num_leaves"       : list(range(15, 256, 8)),
    "learning_rate"    : np.logspace(np.log10(0.005), np.log10(0.1), 30).tolist(),
    "min_child_samples": list(range(5, 101, 5)),
    "min_split_gain"   : np.linspace(0.0, 1.0, 21).tolist(),
    "subsample"        : np.linspace(0.5, 1.0, 11).tolist(),
    "colsample_bytree" : np.linspace(0.4, 1.0, 13).tolist(),
    "reg_alpha"        : np.logspace(-4, 1, 20).tolist(),
    "reg_lambda"       : np.logspace(-4, 1, 20).tolist(),
}

XGB_SPACE = {
    "max_depth"       : list(range(3, 9)),
    "learning_rate"   : np.logspace(np.log10(0.005), np.log10(0.1), 30).tolist(),
    "subsample"       : np.linspace(0.5, 1.0, 11).tolist(),
    "colsample_bytree": np.linspace(0.4, 1.0, 13).tolist(),
    "reg_alpha"       : np.logspace(-4, 1, 20).tolist(),
    "reg_lambda"      : np.logspace(-4, 1, 20).tolist(),
    "min_child_weight": np.logspace(0, np.log10(20), 15).tolist(),
}

CAT_SPACE = {
    "depth"            : list(range(4, 9)),
    "learning_rate"    : np.logspace(np.log10(0.01), np.log10(0.2), 20).tolist(),
    "l2_leaf_reg"      : np.linspace(1.0, 10.0, 19).tolist(),
}

# ── Feature Extractors ────────────────────────────────────────────────────────

def _desc_one(mol):
    return Descriptors.CalcMolDescriptors(mol) if mol else {name: np.nan for name, _ in Descriptors._descList}

def _ecfp4_one(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)) if mol else [np.nan] * 2048

def _ecfp6_one(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)) if mol else [np.nan] * 2048

def _maccs_one(mol):
    return list(MACCSkeys.GenMACCSKeys(mol)) if mol else [np.nan] * 167

def _topo_one(mol):
    if mol is None: return {"star_distance": np.nan, "star_distance_frac": np.nan}
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(star_idx) != 2: return {"star_distance": np.nan, "star_distance_frac": np.nan}
    dmat = Chem.GetDistanceMatrix(mol)
    star_dist = dmat[star_idx[0], star_idx[1]]
    diameter  = dmat.max()
    return {"star_distance": star_dist, "star_distance_frac": star_dist / diameter if diameter > 0 else 0.0}

def _electronic_one(mol):
    if mol is None:
        return {"num_aromatic_rings": np.nan, "aromatic_atom_fraction": np.nan, "num_rotatable_bonds": np.nan, "sp2_atom_fraction": np.nan, "num_nonarom_double_bonds": np.nan}
    n_atoms = mol.GetNumAtoms()
    n_arom  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_sp2   = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2)
    n_dbl   = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0 and not b.GetIsAromatic())
    return {
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aromatic_atom_fraction": n_arom / n_atoms if n_atoms > 0 else 0.0,
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "sp2_atom_fraction": n_sp2 / n_atoms if n_atoms > 0 else 0.0,
        "num_nonarom_double_bonds": n_dbl,
    }

def _tg_specific_one(mol):
    if mol is None: return {"backbone_rotatable_bonds": np.nan}
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(star_idx) != 2: return {"backbone_rotatable_bonds": np.nan}
    path = Chem.GetShortestPath(mol, star_idx[0], star_idx[1])
    rot = sum(1 for i in range(len(path) - 1) if mol.GetBondBetweenAtoms(path[i], path[i + 1]).GetBondTypeAsDouble() == 1.0 and not mol.GetBondBetweenAtoms(path[i], path[i + 1]).IsInRing())
    return {"backbone_rotatable_bonds": rot}

def _conjugation_one(mol):
    if mol is None: return {"max_conjugation_path": np.nan}
    sp2 = {a.GetIdx() for a in mol.GetAtoms() if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2}
    if not sp2: return {"max_conjugation_path": 0}
    visited, max_comp = set(), 0
    for start in sp2:
        if start in visited: continue
        comp, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in comp: continue
            comp.add(node)
            for bond in mol.GetAtomWithIdx(node).GetBonds():
                nbr = bond.GetOtherAtomIdx(node)
                if nbr in sp2 and nbr not in comp: stack.append(nbr)
        visited |= comp
        max_comp = max(max_comp, len(comp))
    return {"max_conjugation_path": max_comp}

def _build_chain(mol, n_units=N_CHAIN_UNITS):
    if mol is None: return None
    base_stars = sorted(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    if len(base_stars) != 2 or mol.GetAtomWithIdx(base_stars[0]).GetDegree() != 1 or mol.GetAtomWithIdx(base_stars[1]).GetDegree() != 1:
        return None
    left_star_base, right_star_base = base_stars
    chain = Chem.RWMol(mol)
    chain_right_star = right_star_base

    for _ in range(n_units - 1):
        offset = chain.GetNumAtoms()
        right_nbr = chain.GetAtomWithIdx(chain_right_star).GetNeighbors()[0].GetIdx()
        bond_type = chain.GetBondBetweenAtoms(chain_right_star, right_nbr).GetBondType()
        new_left_star, new_right_star = left_star_base + offset, right_star_base + offset
        new_left_nbr = mol.GetAtomWithIdx(left_star_base).GetNeighbors()[0].GetIdx() + offset

        chain = Chem.RWMol(Chem.CombineMols(chain.GetMol(), mol))
        chain.AddBond(right_nbr, new_left_nbr, bond_type)
        for idx in sorted([chain_right_star, new_left_star], reverse=True): chain.RemoveAtom(idx)
        chain_right_star = new_right_star - sum(1 for idx in [chain_right_star, new_left_star] if idx < new_right_star)

    for idx in sorted((a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum() == 0), reverse=True): chain.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(chain)
        return chain.GetMol()
    except Exception:
        return None

# ── Deadlock-Proof Single-Pass Engine ─────────────────────────────────────────

def _compute_all_for_one(smi):
    mol = Chem.MolFromSmiles(smi)
    ch_mol = _build_chain(mol) if mol else None
    if ch_mol is None: ch_mol = mol

    d, e4, e6, ma, to, el, tg, co = _desc_one(mol), _ecfp4_one(mol), _ecfp6_one(mol), _maccs_one(mol), _topo_one(mol), _electronic_one(mol), _tg_specific_one(mol), _conjugation_one(mol)
    ch_d, ch_e4, ch_e6, ch_ma, ch_el, ch_co = _desc_one(ch_mol), _ecfp4_one(ch_mol), _ecfp6_one(ch_mol), _maccs_one(ch_mol), _electronic_one(ch_mol), _conjugation_one(ch_mol)

    res = {}
    res.update(d)
    for i, v in enumerate(e4): res[f"ecfp4_{i}"] = v
    for i, v in enumerate(e6): res[f"ecfp6_{i}"] = v
    for i, v in enumerate(ma): res[f"maccs_{i}"] = v
    res.update(to)
    res.update(el)
    res.update(tg)
    res.update(co)

    p = f"ch{N_CHAIN_UNITS}_"
    for k, v in ch_d.items(): res[f"{p}{k}"] = v
    for i, v in enumerate(ch_e4): res[f"{p}ecfp4_{i}"] = v
    for i, v in enumerate(ch_e6): res[f"{p}ecfp6_{i}"] = v
    for i, v in enumerate(ch_ma): res[f"{p}maccs_{i}"] = v
    for k, v in ch_el.items(): res[f"{p}{k}"] = v
    for k, v in ch_co.items(): res[f"{p}{k}"] = v
    return res

def compute_features(df):
    smiles = df["smiles"].tolist()
    results = Parallel(n_jobs=-1, prefer="processes")(delayed(_compute_all_for_one)(s) for s in smiles)
    combined = pd.DataFrame(results, index=df.index)

    # CRITICAL DEADLOCK FIX: Strip infinite values BEFORE casting to float32
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.astype(np.float32)
    return combined

# ── Pipeline Run ──────────────────────────────────────────────────────────────

start = datetime.now()
print(f"Started at {start.strftime('%H:%M:%S')}\n")

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  train: {len(train):,} rows  |  test: {len(test):,} rows")

def _to_canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi

train["canon_smiles"] = train["smiles"].apply(_to_canon)
n_before = len(train)
train = train.groupby(["canon_smiles", "target_type"], as_index=False).agg(smiles=("smiles", "first"), target=("target", "mean"))
print(f"Duplicate merge: {n_before:,} -> {len(train):,} rows")

print("\nComputing features via single-pass architecture...")
X_train = compute_features(train)
X_test  = compute_features(test)

print("\nApplying low-variance prefilter...")
_keep_mask = (X_train.var(axis=0, skipna=True).fillna(0) > LOW_VAR_THRESHOLD)
_prefilter_cols = _keep_mask[_keep_mask].index.tolist()
X_train, X_test = X_train[_prefilter_cols], X_test[_prefilter_cols]
print(f"  {len(_prefilter_cols):,} columns retained.")

y_train, strat_label, groups = train["target"].values, train["target_type"].values, train["canon_smiles"].values
X_train_full, X_test_full = X_train, X_test

test_tg, test_egc = test[test["target_type"] == "tg"].copy(), test[test["target_type"] == "egc"].copy()
test_subsets = {"tg": test_tg, "egc": test_egc}

all_seed_test_tg, all_seed_test_egc, seed_cv_scores = [], [], []

for run_idx, run_seed in enumerate(SEEDS):
    print(f"\nRUN {run_idx + 1}/{len(SEEDS)} (seed={run_seed})")

    _lgb_fixed = {**LGB_FIXED, "random_state": run_seed}
    _xgb_fixed = {**XGB_FIXED, "random_state": run_seed}
    _cat_fixed = {**CAT_FIXED, "random_seed":  run_seed}

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=run_seed)
    cv_splits = list(sgkf.split(X_train_full, strat_label, groups))

    # Feature Pruning
    probe_tr_idx, _ = cv_splits[0]
    def _prune_for_ttype(ttype):
        mask = strat_label[probe_tr_idx] == ttype
        probe = LGBMRegressor(n_estimators=PROBE_N_ESTIMATORS, num_leaves=63, random_state=run_seed, n_jobs=-1, verbose=-1, importance_type="gain")
        probe.fit(X_train_full.iloc[probe_tr_idx][mask], y_train[probe_tr_idx][mask])
        return set(pd.Series(probe.feature_importances_, index=X_train_full.columns).sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist())

    keep_cols = set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for cols in pool.map(_prune_for_ttype, ["tg", "egc"]): keep_cols.update(cols)

    keep_cols = sorted(keep_cols)
    X_train, X_test = X_train_full[keep_cols], X_test_full[keep_cols]

    # Optimization Step
    def _cv_score(model_name, params, ttype, fold_ids):
        scores = []
        for fi in fold_ids:
            tr_idx, val_idx = cv_splits[fi]
            mask_tr, mask_val = strat_label[tr_idx] == ttype, strat_label[val_idx] == ttype
            X_tr, X_val = X_train.iloc[tr_idx][mask_tr], X_train.iloc[val_idx][mask_val]
            y_tr, y_val = y_train[tr_idx][mask_tr], y_train[val_idx][mask_val]

            if model_name == "lgb":
                m = LGBMRegressor(**params)
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)])
            elif model_name == "xgb":
                m = XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                m = CatBoostRegressor(**params)
                m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            scores.append(r2_score(y_val, m.predict(X_val)))
        return np.mean(scores)

    _space_map, _fix_map = {"lgb": LGB_SPACE, "xgb": XGB_SPACE, "cat": CAT_SPACE}, {"lgb": _lgb_fixed, "xgb": _xgb_fixed, "cat": _cat_fixed}
    _extra_map = {"lgb": {"n_estimators": TUNE_N_ESTIMATORS, "n_jobs": _TUNE_N_JOBS}, "xgb": {"n_estimators": TUNE_N_ESTIMATORS, "n_jobs": _TUNE_N_JOBS}, "cat": {"n_estimators": TUNE_N_ESTIMATORS}}

    def _run_search(args):
        model_name, ttype = args
        rng = np.random.RandomState(run_seed)
        candidates = list(ParameterSampler(_space_map[model_name], n_iter=N_CANDIDATES, random_state=rng))

        stage1 = []
        for cand in candidates:
            params = {**_fix_map[model_name], **_extra_map[model_name], **cand}
            stage1.append((_cv_score(model_name, params, ttype, list(range(N_FOLDS))[:FIRST_STAGE_FOLDS]), cand))
        stage1.sort(key=lambda x: -x[0])

        best_score, best_params = -np.inf, None
        for cand in [c for _, c in stage1[:N_SURVIVORS]]:
            params = {**_fix_map[model_name], **_extra_map[model_name], **cand}
            score = _cv_score(model_name, params, ttype, list(range(N_FOLDS)))
            if score > best_score: best_score, best_params = score, params
        return model_name, ttype, best_score, best_params

    best_params_dict = {"lgb": {}, "xgb": {}, "cat": {}}
    # Changed max_workers to 1 to prevent concurrent GPU access issues with CatBoost/XGBoost
    with ThreadPoolExecutor(max_workers=1) as pool:
        for model_name, ttype, best_val, b_params in pool.map(_run_search, [(m, t) for m in ["lgb", "xgb", "cat"] for t in ["tg", "egc"]]):
            best_params_dict[model_name][ttype] = b_params

    # Final Stacking
    test_pred_lgb, test_pred_xgb, test_pred_cat = {t: np.zeros(len(s)) for t, s in test_subsets.items()}, {t: np.zeros(len(s)) for t, s in test_subsets.items()}, {t: np.zeros(len(s)) for t, s in test_subsets.items()}
    fold_lgb_preds, fold_xgb_preds, fold_cat_preds, fold_y_vals = {"tg": [], "egc": []}, {"tg": [], "egc": []}, {"tg": [], "egc": []}, {"tg": [], "egc": []}

    def _fit_fold_ttype(args):
        fold_idx, ttype = args
        tr_idx, val_idx = cv_splits[fold_idx]
        mask_tr, mask_val = strat_label[tr_idx] == ttype, strat_label[val_idx] == ttype
        X_tr, X_val, y_tr, y_val = X_train.iloc[tr_idx][mask_tr], X_train.iloc[val_idx][mask_val], y_train[tr_idx][mask_tr], y_train[val_idx][mask_val]
        X_test_sub = X_test.loc[test_subsets[ttype].index]

        lgb_model = LGBMRegressor(**{**best_params_dict["lgb"][ttype], "n_jobs": _FINAL_N_JOBS})
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)])

        xgb_model = XGBRegressor(**{**best_params_dict["xgb"][ttype], "n_jobs": _FINAL_N_JOBS}, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        cat_model = CatBoostRegressor(**best_params_dict["cat"][ttype])
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        return fold_idx, ttype, lgb_model.predict(X_val), xgb_model.predict(X_val), cat_model.predict(X_val), lgb_model.predict(X_test_sub), xgb_model.predict(X_test_sub), cat_model.predict(X_test_sub), y_val

    # Changed max_workers to 1 to prevent concurrent GPU access issues with CatBoost/XGBoost
    with ThreadPoolExecutor(max_workers=1) as pool:
        for fold_idx, ttype, lgb_val, xgb_val, cat_val, lgb_test, xgb_test, cat_test, y_val in pool.map(_fit_fold_ttype, [(fi, tt) for fi in range(N_FOLDS) for tt in ["tg", "egc"]]):
            fold_lgb_preds[ttype].append(lgb_val); fold_xgb_preds[ttype].append(xgb_val); fold_cat_preds[ttype].append(cat_val); fold_y_vals[ttype].append(y_val)
            test_pred_lgb[ttype] += lgb_test; test_pred_xgb[ttype] += xgb_test; test_pred_cat[ttype] += cat_test

    stacker, fold_r2_stacked = {}, {"tg": [], "egc": []}
    for ttype in ["tg", "egc"]:
        lgb_p, xgb_p, cat_p, y_p = fold_lgb_preds[ttype], fold_xgb_preds[ttype], fold_cat_preds[ttype], fold_y_vals[ttype]
        for h in range(N_FOLDS):
            tr_f = [i for i in range(N_FOLDS) if i != h]
            mx_tr = np.column_stack([np.concatenate([lgb_p[i] for i in tr_f]), np.concatenate([xgb_p[i] for i in tr_f]), np.concatenate([cat_p[i] for i in tr_f])])
            my_tr = np.concatenate([y_p[i] for i in tr_f])
            fold_r2_stacked[ttype].append(r2_score(y_p[h], RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(mx_tr, my_tr).predict(np.column_stack([lgb_p[h], xgb_p[h], cat_p[h]]))))

        ridge_final = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(np.column_stack([np.concatenate(lgb_p), np.concatenate(xgb_p), np.concatenate(cat_p)]), np.concatenate(y_p))
        stacker[ttype] = ridge_final

    cv_tg, cv_egc = np.mean(fold_r2_stacked["tg"]), np.mean(fold_r2_stacked["egc"])
    print(f"  >>> Seed {run_seed} Meta CV: Tg={cv_tg:+.4f} | Egc={cv_egc:+.4f} | Mean={(cv_tg + cv_egc)/2:+.4f} <<<")
    seed_cv_scores.append((cv_tg + cv_egc) / 2)

    all_seed_test_tg.append(stacker["tg"].predict(np.column_stack([test_pred_lgb["tg"]/N_FOLDS, test_pred_xgb["tg"]/N_FOLDS, test_pred_cat["tg"]/N_FOLDS])))
    all_seed_test_egc.append(stacker["egc"].predict(np.column_stack([test_pred_lgb["egc"]/N_FOLDS, test_pred_xgb["egc"]/N_FOLDS, test_pred_cat["egc"]/N_FOLDS])))

print(f"\nFinal Blended Multi-Seed Mean CV: {np.mean(seed_cv_scores):+.4f}")

# Submission Build
test_tg["target"] = np.mean(all_seed_test_tg, axis=0)
test_egc["target"] = np.mean(all_seed_test_egc, axis=0)
submission = pd.concat([test_tg, test_egc])[["id", "target"]].sort_values("id")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved submission package -> {OUTPUT_PATH}. Execution Time: {datetime.now() - start}")
