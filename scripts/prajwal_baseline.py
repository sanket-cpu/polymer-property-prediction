import io
import os
import sys
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.linear_model import RidgeCV, LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# --- Logging Fix (Colab Compatible) ---
os.makedirs("logs", exist_ok=True)
_log_path = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
_log_fh = open(_log_path, "w", encoding="utf-8", buffering=1)

class _Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile
    def write(self, data):
        self.terminal.write(data)
        self.logfile.write(data)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = _Tee(sys.stdout, _log_fh)
print(f"Logging to {_log_path}\n")

RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_PATH  = "train.csv"
TEST_PATH   = "test.csv"
OUTPUT_PATH = "outputs/submission.csv"
N_FOLDS     = 5
SEED        = 42
SEEDS       = [42, 0, 123]  # seeds for multi-seed prediction averaging; set to [42] for a single run

EARLY_STOPPING_ROUNDS = 50
N_TRIALS              = 20  # LGB Optuna trials per target; set to 2 for a quick smoke-test
N_XGB_TRIALS          = 10  # XGB Optuna trials per target
N_CAT_TRIALS          = 10  # CatBoost Optuna trials per target
TUNE_N_ESTIMATORS     = 1000
FINAL_N_ESTIMATORS    = 3000
PROBE_N_ESTIMATORS    = 300
N_CHAIN_UNITS         = 3   # repeat units to stitch for chain-extension features

# Physics-informed residual learning: only trust a physics baseline enough to
# subtract it if its own CV R^2 clears this bar. Below it, the raw physics
# quantities are still added as ordinary features (never actively harmful),
# but we don't risk training on a near-random residual.
PHYSICS_R2_GATE  = 0.10
HUCKEL_K_POINTS  = 200  # k-mesh density over the first Brillouin zone for the Huckel band structure

# Parallelism — no algorithm change, pure wall-clock speedup.
N_OPTUNA_JOBS  = 2
_N_CPU         = os.cpu_count() or 1
_N_TUNE_CONC   = 3 * 2 * N_OPTUNA_JOBS
_N_CV_CONC     = N_FOLDS * 2
_TUNE_N_JOBS   = max(1, _N_CPU // _N_TUNE_CONC)
_FINAL_N_JOBS  = max(1, _N_CPU // _N_CV_CONC)

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
    "device"       : "cpu",
}

CAT_FIXED = {
    "n_estimators"         : FINAL_N_ESTIMATORS,
    "random_seed"          : SEED,
    "verbose"              : 0,
    "thread_count"         : -1,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}


# ══════════════════════════════════════════════════════════════════════════
# PHYSICS SECTION
# ══════════════════════════════════════════════════════════════════════════

def _get_star_atoms(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _conjugated_subgraph(mol):
    sp2 = {
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2
    }
    if not sp2:
        return []
    visited, best = set(), []
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
        if len(comp) > len(best):
            best = list(comp)
    return sorted(best)


def compute_huckel_gap(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan

    star_idx = _get_star_atoms(mol)
    if len(star_idx) != 2:
        return np.nan

    conj = _conjugated_subgraph(mol)
    n = len(conj)
    if n < 2:
        return np.nan
    pos = {atom: i for i, atom in enumerate(conj)}

    H_intra = np.zeros((n, n))
    for atom_idx in conj:
        i = pos[atom_idx]
        for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
            j_atom = bond.GetOtherAtomIdx(atom_idx)
            if j_atom in pos:
                j = pos[j_atom]
                H_intra[i, j] = -1.0
                H_intra[j, i] = -1.0

    def _link_atom(star):
        nbrs = mol.GetAtomWithIdx(star).GetNeighbors()
        if not nbrs:
            return None
        nbr_idx = nbrs[0].GetIdx()
        return nbr_idx if nbr_idx in pos else None

    left_link  = _link_atom(star_idx[0])
    right_link = _link_atom(star_idx[1])
    if left_link is None or right_link is None:
        return np.nan

    i_left, i_right = pos[left_link], pos[right_link]

    n_occ = n // 2
    if n_occ < 1 or n_occ >= n:
        return np.nan

    k_mesh = np.linspace(0.0, 2 * np.pi, HUCKEL_K_POINTS, endpoint=False)
    homo_band = np.empty(HUCKEL_K_POINTS)
    lumo_band = np.empty(HUCKEL_K_POINTS)

    for kk, k in enumerate(k_mesh):
        Hk = H_intra.astype(complex).copy()
        Hk[i_right, i_left] += -1.0 * np.exp(-1j * k)
        Hk[i_left, i_right] += -1.0 * np.exp(1j * k)
        eigvals = np.linalg.eigvalsh(Hk)
        homo_band[kk] = eigvals[n_occ - 1]
        lumo_band[kk] = eigvals[n_occ]

    gap = lumo_band.min() - homo_band.max()
    return float(max(gap, 0.0))


_GROUP_SMARTS = {
    "amide":          "[NX3][CX3](=O)",
    "imide":          "[CX3](=O)[NX3][CX3](=O)",
    "ester":          "[CX3](=O)[OX2H0]",
    "ether":          "[OD2]([#6])[#6]",
    "sulfone":        "[SX4](=O)(=O)",
    "carbonate":      "[OX2][CX3](=O)[OX2]",
    "hbond_donor":    "[#7,#8;!H0]",
    "hbond_acceptor": "[#7,#8]",
    "aromatic_ring":  "c1ccccc1",
}
_GROUP_FEAT_COLS = list(_GROUP_SMARTS.keys()) + [
    "backbone_rotatable_bonds", "num_aromatic_rings", "molecular_weight_per_repeat",
]


def extract_group_counts(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in _GROUP_FEAT_COLS}

    counts = {}
    for name, smarts in _GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        counts[name] = len(mol.GetSubstructMatches(patt)) if patt is not None else np.nan

    counts["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)

    star_idx = _get_star_atoms(mol)
    if len(star_idx) == 2:
        path = Chem.GetShortestPath(mol, star_idx[0], star_idx[1])
        rot = 0
        for i in range(len(path) - 1):
            bond = mol.GetBondBetweenAtoms(path[i], path[i + 1])
            if bond.GetBondTypeAsDouble() == 1.0 and not bond.IsInRing():
                rot += 1
        counts["backbone_rotatable_bonds"] = rot
    else:
        counts["backbone_rotatable_bonds"] = np.nan

    counts["molecular_weight_per_repeat"] = rdMolDescriptors.CalcExactMolWt(mol)
    return counts


class PhysicsBaseline:
    def __init__(self):
        self.egc_model = None
        self.tg_model  = None
        self.egc_fallback_mean = None
        self.tg_fallback_mean  = None

    @staticmethod
    def compute_physics_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["huckel_gap"] = out["smiles"].apply(compute_huckel_gap)
        group_feats = out["smiles"].apply(extract_group_counts).apply(pd.Series)
        out = pd.concat([out.reset_index(drop=True), group_feats.reset_index(drop=True)], axis=1)
        return out

    def fit(self, train_df: pd.DataFrame):
        df = self.compute_physics_columns(train_df)

        egc = df[df["target_type"] == "egc"].dropna(subset=["huckel_gap"])
        if len(egc) >= 10:
            self.egc_model = LinearRegression().fit(egc[["huckel_gap"]].values, egc["target"].values)
        self.egc_fallback_mean = df.loc[df["target_type"] == "egc", "target"].mean()

        tg = df[df["target_type"] == "tg"]
        tg_valid = tg.dropna(subset=_GROUP_FEAT_COLS)
        if len(tg_valid) >= 10:
            self.tg_model = LinearRegression().fit(
                tg_valid[_GROUP_FEAT_COLS].values, tg_valid["target"].values
            )
        self.tg_fallback_mean = df.loc[df["target_type"] == "tg", "target"].mean()
        return self

    def predict_baseline(self, df: pd.DataFrame) -> np.ndarray:
        phys = self.compute_physics_columns(df)
        baseline = np.empty(len(phys))
        for i, (_, row) in enumerate(phys.iterrows()):
            ttype = row["target_type"]
            if ttype == "egc":
                if self.egc_model is not None and pd.notna(row["huckel_gap"]):
                    baseline[i] = self.egc_model.predict([[row["huckel_gap"]]])[0]
                else:
                    baseline[i] = self.egc_fallback_mean
            elif ttype == "tg":
                feat_vals = row[_GROUP_FEAT_COLS].values.astype(float)
                if self.tg_model is not None and not np.isnan(feat_vals).any():
                    baseline[i] = self.tg_model.predict([feat_vals])[0]
                else:
                    baseline[i] = self.tg_fallback_mean
            else:
                baseline[i] = 0.0
        return baseline

    def residual(self, df: pd.DataFrame) -> np.ndarray:
        return df["target"].values - self.predict_baseline(df)

    def inverse(self, df: pd.DataFrame, residual_pred: np.ndarray) -> np.ndarray:
        return self.predict_baseline(df) + residual_pred


def physics_sanity_check(train: pd.DataFrame) -> dict:
    print(f"\n{'='*60}\n  PHYSICS SANITY CHECK\n{'='*60}")
    results = {}

    egc = train[train["target_type"] == "egc"].copy()
    egc["huckel_gap"] = egc["smiles"].apply(compute_huckel_gap)
    coverage = egc["huckel_gap"].notna().mean() if len(egc) else 0.0
    egc_valid = egc.dropna(subset=["huckel_gap"])
    print(f"  EGC Huckel gap coverage: {coverage:.1%}  ({len(egc_valid)}/{len(egc)} usable rows)")
    if len(egc_valid) >= 20 and egc_valid["canon_smiles"].nunique() >= 5:
        gkf = GroupKFold(n_splits=min(N_FOLDS, egc_valid["canon_smiles"].nunique()))
        X, y, groups = egc_valid[["huckel_gap"]].values, egc_valid["target"].values, egc_valid["canon_smiles"].values
        scores = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            lr = LinearRegression().fit(X[tr_idx], y[tr_idx])
            scores.append(r2_score(y[val_idx], lr.predict(X[val_idx])))
        results["egc"] = float(np.mean(scores))
        print(f"  EGC physics-baseline CV R^2: {results['egc']:+.4f} (std {np.std(scores):.4f})")
    else:
        results["egc"] = -np.inf
        print("  EGC: not enough valid rows for a reliable CV estimate -> treating as ungated (features only).")

    tg = train[train["target_type"] == "tg"].copy()
    group_feats = tg["smiles"].apply(extract_group_counts).apply(pd.Series)
    tg = pd.concat([tg.reset_index(drop=True), group_feats.reset_index(drop=True)], axis=1)
    tg_valid = tg.dropna(subset=_GROUP_FEAT_COLS)
    print(f"  TG group-count coverage: {len(tg_valid)}/{len(tg)} usable rows")
    if len(tg_valid) >= 20 and tg_valid["canon_smiles"].nunique() >= 5:
        gkf = GroupKFold(n_splits=min(N_FOLDS, tg_valid["canon_smiles"].nunique()))
        X, y, groups = tg_valid[_GROUP_FEAT_COLS].values, tg_valid["target"].values, tg_valid["canon_smiles"].values
        scores = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            lr = LinearRegression().fit(X[tr_idx], y[tr_idx])
            scores.append(r2_score(y[val_idx], lr.predict(X[val_idx])))
        results["tg"] = float(np.mean(scores))
        print(f"  TG physics-baseline CV R^2: {results['tg']:+.4f} (std {np.std(scores):.4f})")
    else:
        results["tg"] = -np.inf
        print("  TG: not enough valid rows for a reliable CV estimate -> treating as ungated (features only).")

    for ttype in ["tg", "egc"]:
        mode = "RESIDUAL LEARNING (subtract physics baseline)" if results[ttype] > PHYSICS_R2_GATE \
            else "FEATURES ONLY (physics added as columns, no target transform)"
        print(f"  >>> {ttype.upper()}: R^2={results[ttype]:+.4f}  ->  {mode}")

    return results


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
    n_sp2   = sum(
        1 for a in mol.GetAtoms()
        if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2
    )
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

        for idx in sorted([chain_right_star, new_left_star], reverse=True):
            chain.RemoveAtom(idx)

        n_below = sum(1 for idx in [chain_right_star, new_left_star] if idx < new_right_star)
        chain_right_star = new_right_star - n_below

    for idx in sorted((a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
        chain.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(chain)
        return Chem.MolToSmiles(chain.GetMol())
    except Exception:
        return None


def compute_features(df):
    smiles = df["smiles"].tolist()

    chain_raw = [_build_chain(s) for s in smiles]
    n_fail    = sum(1 for c in chain_raw if c is None)
    chain_smi = [c if c is not None else s for c, s in zip(chain_raw, smiles)]
    if n_fail:
        print(f"    Chain build failures: {n_fail}/{len(smiles)} (fell back to monomer)")

    descs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_desc_one)(s)        for s in smiles)
    ecfp4  = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp4_one)(s)       for s in smiles)
    ecfp6  = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp6_one)(s)       for s in smiles)
    maccs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_maccs_one)(s)       for s in smiles)
    topo   = Parallel(n_jobs=-1, prefer="threads")(delayed(_topo_one)(s)        for s in smiles)
    elec   = Parallel(n_jobs=-1, prefer="threads")(delayed(_electronic_one)(s)  for s in smiles)
    tgfeat = Parallel(n_jobs=-1, prefer="threads")(delayed(_tg_specific_one)(s) for s in smiles)
    conj   = Parallel(n_jobs=-1, prefer="threads")(delayed(_conjugation_one)(s) for s in smiles)
    huckel = Parallel(n_jobs=-1, prefer="threads")(delayed(compute_huckel_gap)(s)   for s in smiles)
    groups = Parallel(n_jobs=-1, prefer="threads")(delayed(extract_group_counts)(s) for s in smiles)

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
    huckel_df = pd.DataFrame({"huckel_gap": huckel}, index=df.index)
    groups_df = pd.DataFrame(groups,  index=df.index).add_prefix("grp_")

    ch_desc_df  = pd.DataFrame(ch_descs, index=df.index).add_prefix(p)
    ch_ecfp4_df = pd.DataFrame(ch_ecfp4, index=df.index, columns=[f"{p}ecfp4_{i}" for i in range(2048)])
    ch_ecfp6_df = pd.DataFrame(ch_ecfp6, index=df.index, columns=[f"{p}ecfp6_{i}" for i in range(2048)])
    ch_maccs_df = pd.DataFrame(ch_maccs, index=df.index, columns=[f"{p}maccs_{i}" for i in range(167)])
    ch_elec_df  = pd.DataFrame(ch_elec,  index=df.index).add_prefix(p)
    ch_conj_df  = pd.DataFrame(ch_conj,  index=df.index).add_prefix(p)

    combined = pd.concat([
        desc_df, ecfp4_df, ecfp6_df, maccs_df, topo_df, elec_df, tgfeat_df, conj_df,
        huckel_df, groups_df,
        ch_desc_df, ch_ecfp4_df, ch_ecfp6_df, ch_maccs_df, ch_elec_df, ch_conj_df,
    ], axis=1)
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

n_before = len(train)
train = (
    train.groupby(["canon_smiles", "target_type"], as_index=False)
    .agg(smiles=("smiles", "first"), target=("target", "mean"))
)
n_after = len(train)
print(f"\nDuplicate merge: {n_before:,} -> {n_after:,} rows  ({n_before - n_after} groups collapsed to mean)")


# ── Physics sanity check (decides residual-learning mode per target) ──────────

physics_r2    = physics_sanity_check(train)
PHYSICS_MODE  = {t: (physics_r2[t] > PHYSICS_R2_GATE) for t in ["tg", "egc"]}
physics       = PhysicsBaseline().fit(train)

print(f"\nPhysics residual-learning mode: {PHYSICS_MODE}")
print("  (targets marked False still get huckel_gap / grp_* as ordinary GBM features,")
print("   they just aren't used to transform the training target.)\n")

physics_baseline_train = physics.predict_baseline(train)
y_train_effective = train["target"].values.copy()
for ttype, use_residual in PHYSICS_MODE.items():
    if use_residual:
        mask = (train["target_type"] == ttype).values
        y_train_effective[mask] = train["target"].values[mask] - physics_baseline_train[mask]


# ── Features ───────────────────────────────────────────────────────────────────

print("\nComputing features...")
X_train = compute_features(train)
X_test  = compute_features(test)
print(f"  Done.  {X_train.shape[1]} features per molecule")

y_train     = y_train_effective  # residual-or-raw, per PHYSICS_MODE above
strat_label = train["target_type"].values
groups      = train["canon_smiles"].values

X_train_full = X_train
X_test_full  = X_test

test_tg      = test[test["target_type"] == "tg"].copy()
test_egc     = test[test["target_type"] == "egc"].copy()
test_subsets = {"tg": test_tg, "egc": test_egc}

physics_baseline_test = {
    ttype: physics.predict_baseline(subset) for ttype, subset in test_subsets.items()
}

all_seed_test_tg  = []
all_seed_test_egc = []
seed_cv_scores    = []

print(f"\nMulti-seed run: SEEDS={SEEDS}  N_TRIALS={N_TRIALS}  N_XGB_TRIALS={N_XGB_TRIALS}  N_CAT_TRIALS={N_CAT_TRIALS}")
_tune_factor = 2 * N_OPTUNA_JOBS
_per_seed_hr = (70 + (N_TRIALS * 5 + N_XGB_TRIALS * 5 + N_CAT_TRIALS * 5) / _tune_factor) / 60
_est_hr      = 0.33 + len(SEEDS) * _per_seed_hr
print(f"Estimated runtime: ~{_est_hr:.1f} hours\n")


# ── Per-seed loop: prune -> tune -> CV -> collect test predictions ─────────────

for run_idx, run_seed in enumerate(SEEDS):
    print(f"\n{'#'*60}")
    print(f"  RUN {run_idx + 1}/{len(SEEDS)}  (seed={run_seed})")
    print(f"{'#'*60}")

    _lgb_fixed = {**LGB_FIXED, "random_state": run_seed}
    _xgb_fixed = {**XGB_FIXED, "random_state": run_seed}
    _cat_fixed = {**CAT_FIXED, "random_seed":  run_seed}

    sgkf      = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=run_seed)
    cv_splits = list(sgkf.split(X_train_full, strat_label, groups))

    print(f"\n{'='*60}")
    print(f"  FEATURE PRUNING  ({PROBE_N_ESTIMATORS}-tree probe, permutation importance)")
    print(f"{'='*60}\n")

    n_orig = X_train_full.shape[1]
    probe_tr_idx, probe_val_idx = cv_splits[0]
    _probe_jobs = max(1, _N_CPU // 2)

    def _prune_for_ttype(ttype):
        mask_tr  = strat_label[probe_tr_idx]  == ttype
        mask_val = strat_label[probe_val_idx] == ttype
        X_probe_tr  = X_train_full.iloc[probe_tr_idx][mask_tr]
        X_probe_val = X_train_full.iloc[probe_val_idx][mask_val]
        y_probe_tr  = y_train[probe_tr_idx][mask_tr]
        y_probe_val = y_train[probe_val_idx][mask_val]
        probe = LGBMRegressor(
            n_estimators=PROBE_N_ESTIMATORS, num_leaves=63,
            random_state=run_seed, n_jobs=_probe_jobs, verbose=-1,
        )
        probe.fit(X_probe_tr, y_probe_tr)
        result = permutation_importance(
            probe, X_probe_val, y_probe_val,
            n_repeats=3, random_state=run_seed, n_jobs=_probe_jobs,
        )
        imp      = pd.Series(result.importances_mean, index=X_train_full.columns)
        positive = imp[imp > 0].index
        print(f"  {ttype.upper()}: {len(positive):,} / {n_orig:,} features with positive permutation importance")
        return set(positive.tolist())

    keep_cols = set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for cols in pool.map(_prune_for_ttype, ["tg", "egc"]):
            keep_cols.update(cols)

    keep_cols = sorted(keep_cols)
    X_train   = X_train_full[keep_cols]
    X_test    = X_test_full[keep_cols]
    print(f"\n  Kept {len(keep_cols):,} features, dropped {n_orig - len(keep_cols):,} zero-importance")

    print(f"\n{'='*60}")
    print(f"  TUNING  LGB({N_TRIALS}) + XGB({N_XGB_TRIALS}) + CAT({N_CAT_TRIALS}) trials  |  6 studies concurrently")
    print(f"{'='*60}")

    def make_lgb_objective(ttype, _cv=cv_splits, _X=X_train, _y=y_train,
                           _sl=strat_label, _fixed=_lgb_fixed):
        def objective(trial):
            params = {
                **_fixed,
                "n_estimators"     : TUNE_N_ESTIMATORS,
                "n_jobs"           : _TUNE_N_JOBS,
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

    def make_xgb_objective(ttype, _cv=cv_splits, _X=X_train, _y=y_train,
                           _sl=strat_label, _fixed=_xgb_fixed):
        def objective(trial):
            params = {
                **_fixed,
                "n_estimators"    : TUNE_N_ESTIMATORS,
                "n_jobs"          : _TUNE_N_JOBS,
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

    def make_cat_objective(ttype, _cv=cv_splits, _X=X_train, _y=y_train,
                           _sl=strat_label, _fixed=_cat_fixed):
        def objective(trial):
            params = {
                **_fixed,
                "n_estimators"     : TUNE_N_ESTIMATORS,
                "thread_count"     : _TUNE_N_JOBS,
                "depth"            : trial.suggest_int("depth", 4, 8),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg"      : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "rsm"              : trial.suggest_float("rsm", 0.4, 1.0),
                "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 50),
            }
            scores = []
            for tr_idx, val_idx in _cv:
                mask_tr  = _sl[tr_idx]  == ttype
                mask_val = _sl[val_idx] == ttype
                X_tr  = _X.iloc[tr_idx][mask_tr]
                X_val = _X.iloc[val_idx][mask_val]
                y_tr  = _y[tr_idx][mask_tr]
                y_val = _y[val_idx][mask_val]
                model = CatBoostRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                pred = model.predict(X_val)
                scores.append(r2_score(y_val, pred))
            return np.mean(scores)
        return objective

    _make_obj  = {"lgb": make_lgb_objective, "xgb": make_xgb_objective, "cat": make_cat_objective}
    _n_trials  = {"lgb": N_TRIALS, "xgb": N_XGB_TRIALS, "cat": N_CAT_TRIALS}
    _fix_map   = {"lgb": _lgb_fixed, "xgb": _xgb_fixed, "cat": _cat_fixed}

    def _run_study(args):
        model, ttype = args
        s = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=run_seed))
        s.optimize(_make_obj[model](ttype), n_trials=_n_trials[model],
                   n_jobs=N_OPTUNA_JOBS, show_progress_bar=False)
        return model, ttype, s.best_value, {**_fix_map[model], **s.best_params}

    print(f"\n  Running 6 studies concurrently ({N_OPTUNA_JOBS} trial threads each)...")
    best_lgb_params: dict = {}
    best_xgb_params: dict = {}
    best_cat_params: dict = {}
    _dest = {"lgb": best_lgb_params, "xgb": best_xgb_params, "cat": best_cat_params}
    tune_tasks = [(m, t) for m in ["lgb", "xgb", "cat"] for t in ["tg", "egc"]]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for model, ttype, best_val, best_params in pool.map(_run_study, tune_tasks):
            _dest[model][ttype] = best_params
            print(f"  {model.upper()} {ttype.upper()}: Best R²={best_val:+.4f}  params={best_params}")

    print(f"\n{'='*60}")
    print(f"  FINAL CV  (LGB+XGB+CAT)  +  FOLD-ENSEMBLED TEST PREDICTIONS")
    print(f"{'='*60}\n")

    test_pred_lgb  = {ttype: np.zeros(len(subset)) for ttype, subset in test_subsets.items()}
    test_pred_xgb  = {ttype: np.zeros(len(subset)) for ttype, subset in test_subsets.items()}
    test_pred_cat  = {ttype: np.zeros(len(subset)) for ttype, subset in test_subsets.items()}
    fold_lgb_preds = {"tg": [], "egc": []}
    fold_xgb_preds = {"tg": [], "egc": []}
    fold_cat_preds = {"tg": [], "egc": []}
    fold_y_vals    = {"tg": [], "egc": []}

    def _fit_fold_ttype(args):
        fold_idx, ttype = args
        tr_idx, val_idx = cv_splits[fold_idx]
        mask_tr  = strat_label[tr_idx]  == ttype
        mask_val = strat_label[val_idx] == ttype
        X_tr  = X_train.iloc[tr_idx][mask_tr]
        X_val = X_train.iloc[val_idx][mask_val]
        y_tr  = y_train[tr_idx][mask_tr]
        y_val = y_train[val_idx][mask_val]
        X_test_sub = X_test.loc[test_subsets[ttype].index]

        lgb_model = LGBMRegressor(**{**best_lgb_params[ttype], "n_jobs": _FINAL_N_JOBS})
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)])

        xgb_model = XGBRegressor(**{**best_xgb_params[ttype], "n_jobs": _FINAL_N_JOBS},
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        cat_model = CatBoostRegressor(**{**best_cat_params[ttype], "thread_count": _FINAL_N_JOBS})
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        lgb_val  = lgb_model.predict(X_val)
        xgb_val  = xgb_model.predict(X_val)
        cat_val  = cat_model.predict(X_val)
        lgb_test = lgb_model.predict(X_test_sub)
        xgb_test = xgb_model.predict(X_test_sub)
        cat_test = cat_model.predict(X_test_sub)
        return fold_idx, ttype, lgb_val, xgb_val, cat_val, lgb_test, xgb_test, cat_test, y_val

    print(f"  Running all {N_FOLDS * 2} (fold, target) tasks concurrently...")
    cv_tasks    = [(fi, tt) for fi in range(N_FOLDS) for tt in ["tg", "egc"]]
    fold_r2_log = {}

    with ThreadPoolExecutor(max_workers=N_FOLDS * 2) as pool:
        for fold_idx, ttype, lgb_val, xgb_val, cat_val, lgb_test, xgb_test, cat_test, y_val in pool.map(_fit_fold_ttype, cv_tasks):
            r2 = r2_score(y_val, (lgb_val + xgb_val + cat_val) / 3)
            fold_r2_log.setdefault(fold_idx, {})[ttype] = r2
            fold_lgb_preds[ttype].append(lgb_val)
            fold_xgb_preds[ttype].append(xgb_val)
            fold_cat_preds[ttype].append(cat_val)
            fold_y_vals[ttype].append(y_val)
            test_pred_lgb[ttype] += lgb_test
            test_pred_xgb[ttype] += xgb_test
            test_pred_cat[ttype] += cat_test

    for fi in sorted(fold_r2_log):
        s = fold_r2_log[fi]
        mean = (s["tg"] + s["egc"]) / 2
        print(f"  Fold {fi+1}  R²(Tg)={s['tg']:+.4f}  R²(Egc)={s['egc']:+.4f}  mean={mean:+.4f}  (naive equal-weight, on effective/residual target)")

    print(f"\n{'='*60}")
    print(f"  OOF STACKING  (RidgeCV meta-learner, nested-fold evaluation)")
    print(f"{'='*60}\n")

    _RIDGE_ALPHAS = np.logspace(-3, 3, 25)
    stacker    = {}
    nest_preds = {"tg": [], "egc": []}
    nest_true  = {"tg": [], "egc": []}

    for ttype in ["tg", "egc"]:
        lgb_parts = fold_lgb_preds[ttype]
        xgb_parts = fold_xgb_preds[ttype]
        cat_parts = fold_cat_preds[ttype]
        y_parts   = fold_y_vals[ttype]

        for hold in range(N_FOLDS):
            tr_f = [i for i in range(N_FOLDS) if i != hold]
            meta_X_tr = np.column_stack([
                np.concatenate([lgb_parts[i] for i in tr_f]),
                np.concatenate([xgb_parts[i] for i in tr_f]),
                np.concatenate([cat_parts[i] for i in tr_f]),
            ])
            meta_y_tr  = np.concatenate([y_parts[i] for i in tr_f])
            meta_X_val = np.column_stack([lgb_parts[hold], xgb_parts[hold], cat_parts[hold]])

            ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
            ridge.fit(meta_X_tr, meta_y_tr)
            nest_preds[ttype].append(ridge.predict(meta_X_val))
            nest_true[ttype].append(y_parts[hold])

        all_X = np.column_stack([
            np.concatenate(lgb_parts), np.concatenate(xgb_parts), np.concatenate(cat_parts),
        ])
        all_y = np.concatenate(y_parts)
        ridge_final = RidgeCV(alphas=_RIDGE_ALPHAS)
        ridge_final.fit(all_X, all_y)
        stacker[ttype] = ridge_final
        coef = ridge_final.coef_
        print(
            f"  {ttype.upper()}: stacker coefs  LGB={coef[0]:.3f}  XGB={coef[1]:.3f}"
            f"  CAT={coef[2]:.3f}  alpha={ridge_final.alpha_:.4g}"
        )

    fold_r2_stacked = {"tg": [], "egc": []}
    for ttype in ["tg", "egc"]:
        for hold in range(N_FOLDS):
            fold_r2_stacked[ttype].append(r2_score(nest_true[ttype][hold], nest_preds[ttype][hold]))

    cv_tg  = np.mean(fold_r2_stacked["tg"])
    cv_egc = np.mean(fold_r2_stacked["egc"])
    cv_r2  = (cv_tg + cv_egc) / 2

    print(f"\n  Mean R²(Tg)  [on effective target] = {cv_tg:+.4f}  (std {np.std(fold_r2_stacked['tg']):.4f})")
    print(f"  Mean R²(Egc) [on effective target] = {cv_egc:+.4f}  (std {np.std(fold_r2_stacked['egc']):.4f})")

    # Also report R^2 in RAW target units (inverting residual mode) so this
    # run is directly comparable to your original 0.90 baseline.
    raw_r2 = {}
    for ttype in ["tg", "egc"]:
        raw_true, raw_pred = [], []
        for hold in range(N_FOLDS):
            tr_idx, val_idx = cv_splits[hold]
            mask_val = strat_label[val_idx] == ttype
            val_rows = train.iloc[val_idx][mask_val]
            true_raw = val_rows["target"].values
            pred_eff = nest_preds[ttype][hold]
            if PHYSICS_MODE[ttype]:
                pred_raw = physics.predict_baseline(val_rows) + pred_eff
            else:
                pred_raw = pred_eff
            raw_true.append(true_raw)
            raw_pred.append(pred_raw)
        raw_r2[ttype] = r2_score(np.concatenate(raw_true), np.concatenate(raw_pred))

    cv_r2_raw = (raw_r2["tg"] + raw_r2["egc"]) / 2
    print(f"\n  Mean R²(Tg)  [RAW target units, comparable to original baseline] = {raw_r2['tg']:+.4f}")
    print(f"  Mean R²(Egc) [RAW target units, comparable to original baseline] = {raw_r2['egc']:+.4f}")
    print(f"\n  >>> CV score (seed={run_seed}), RAW units = {cv_r2_raw:+.4f} <<<")

    seed_tg_pred_eff  = stacker["tg"].predict(np.column_stack([
        test_pred_lgb["tg"]  / N_FOLDS,
        test_pred_xgb["tg"]  / N_FOLDS,
        test_pred_cat["tg"]  / N_FOLDS,
    ]))
    seed_egc_pred_eff = stacker["egc"].predict(np.column_stack([
        test_pred_lgb["egc"] / N_FOLDS,
        test_pred_xgb["egc"] / N_FOLDS,
        test_pred_cat["egc"] / N_FOLDS,
    ]))

    seed_tg_pred  = physics_baseline_test["tg"]  + seed_tg_pred_eff  if PHYSICS_MODE["tg"]  else seed_tg_pred_eff
    seed_egc_pred = physics_baseline_test["egc"] + seed_egc_pred_eff if PHYSICS_MODE["egc"] else seed_egc_pred_eff

    all_seed_test_tg.append(seed_tg_pred)
    all_seed_test_egc.append(seed_egc_pred)
    seed_cv_scores.append(cv_r2_raw)


# ── Multi-seed summary ─────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  MULTI-SEED SUMMARY  (RAW target units)")
print(f"{'='*60}")
for seed, score in zip(SEEDS, seed_cv_scores):
    print(f"  Seed {seed:3d}: CV = {score:+.4f}")
print(f"  Mean CV  : {np.mean(seed_cv_scores):+.4f}  (std {np.std(seed_cv_scores):.4f})")
print(f"\n  Physics residual-learning mode used: {PHYSICS_MODE}")
print(f"  Physics-baseline standalone CV R^2 : {physics_r2}")

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
