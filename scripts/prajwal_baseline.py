"""
Physics-informed pipeline v2.
Changes vs the 0.9054-plateau version, and why:

1. NOISE-CEILING DIAGNOSTIC (new)
   Duplicate (canon_smiles, target_type) groups give a direct empirical
   estimate of label noise variance. We report 1 - noise_var/target_var as
   the theoretical max R^2 any model can achieve. If your score sits near
   this number, more model complexity cannot help -- only better labels or
   noise-aware training can. We print this BEFORE doing anything else, so
   effort isn't wasted chasing an unreachable target.

2. INVERSE-VARIANCE SAMPLE WEIGHTING (new, replaces naive mean-collapse)
   Previously duplicate groups were collapsed to a plain mean and every row
   trained with equal weight. Now each collapsed row keeps a weight
   proportional to 1/within-group-variance (capped), so noisy/conflicting
   labels count less and tight, reliable groups count more. This uses
   information that was already being computed (the spread printout) but
   previously thrown away after printing.

3. RICHARDSON CHAIN EXTRAPOLATION (new hand-built feature block)
   Bulk/periodic polymer descriptors converge to their infinite-chain value
   like a + b/n as the oligomer length n grows (standard result for
   periodic 1-D systems, same reasoning behind the existing Huckel k-mesh
   band structure). Instead of using a single n=3 chain snapshot as a
   feature (previous version), we build chains at n=1,2,3, compute the
   same descriptors at each length, and fit a per-molecule least-squares
   extrapolation to n->infinity. The extrapolated value is a MUCH better
   estimate of the true bulk descriptor than any single finite n, and it
   is derived by hand (closed-form regression), not learned or pretrained.

4. PLS-ON-FINGERPRINTS AS A FOURTH STACK MEMBER (new)
   LGB/XGB/CAT are all boosted trees on the same feature matrix and are
   highly correlated learners -- stacking them mostly cancels variance, not
   bias. Partial Least Squares on the fingerprint block has a structurally
   different inductive bias (linear, latent-factor, dense) and gives the
   Ridge stacker something genuinely different to combine.

5. TUNE ONCE, REUSE ACROSS SEEDS (perf, not accuracy)
   Optuna hyperparameter optima are a property of the feature space / model
   family, not of the CV shuffling seed. Retuning per seed (previous
   version) was ~3x the compute for no expected accuracy gain. We tune once
   at SEEDS[0], freeze params, and reuse them for every seed's final fit +
   bagging pass.

6. TRIMMED FINGERPRINTS (perf + noise reduction)
   ECFP6 substantially subsumes ECFP4 substructures at this radius/bit
   budget and roughly doubles fingerprint dimensionality for little unique
   signal. Dropped. ECFP4 shrunk 2048->1024 bits. This cuts feature-matrix
   width (and therefore tuning-probe time, permutation-importance time, and
   overfitting surface) by roughly 55% with negligible expected R^2 cost --
   verify this with the ablation flag below before trusting it blindly.
"""

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
from sklearn.cross_decomposition import PLSRegression
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
TUNE_FOLDS  = 3          # fewer folds during hyperparameter search (perf; final CV still uses N_FOLDS)
SEED        = 42
SEEDS       = [42, 0, 123]

EARLY_STOPPING_ROUNDS = 50
N_TRIALS              = 20
N_XGB_TRIALS          = 10
N_CAT_TRIALS          = 10
TUNE_N_ESTIMATORS     = 1000
FINAL_N_ESTIMATORS    = 3000
PROBE_N_ESTIMATORS    = 300
N_CHAIN_UNITS_MAX     = 3     # build oligomers of length 1..N_CHAIN_UNITS_MAX for Richardson extrapolation

PHYSICS_R2_GATE  = 0.10
HUCKEL_K_POINTS  = 200

# Fingerprint sizing -- see rationale item 6 above. Flip ABLATE_FP_TRIM to
# False to reproduce the original 2048/2048 ECFP4+ECFP6 setup and confirm
# the trim doesn't cost meaningful R^2 on your data before trusting it.
ABLATE_FP_TRIM = True
ECFP4_BITS     = 1024 if ABLATE_FP_TRIM else 2048
USE_ECFP6      = not ABLATE_FP_TRIM

# Noise-aware training: cap the inverse-variance weight so a single
# near-zero-spread duplicate group can't dominate the loss.
MAX_SAMPLE_WEIGHT = 5.0

N_OPTUNA_JOBS  = 2
_N_CPU         = os.cpu_count() or 1
_N_TUNE_CONC   = 3 * 2 * N_OPTUNA_JOBS
_N_CV_CONC     = N_FOLDS * 2
_TUNE_N_JOBS   = max(1, _N_CPU // _N_TUNE_CONC)
_FINAL_N_JOBS  = max(1, _N_CPU // _N_CV_CONC)

LGB_FIXED = {"n_estimators": FINAL_N_ESTIMATORS, "random_state": SEED, "n_jobs": -1, "verbose": -1}
XGB_FIXED = {"n_estimators": FINAL_N_ESTIMATORS, "random_state": SEED, "n_jobs": -1,
             "tree_method": "hist", "device": "cpu"}
CAT_FIXED = {"n_estimators": FINAL_N_ESTIMATORS, "random_seed": SEED, "verbose": 0,
             "thread_count": -1, "early_stopping_rounds": EARLY_STOPPING_ROUNDS}


# ══════════════════════════════════════════════════════════════════════════
# PHYSICS SECTION (unchanged from v1 -- Huckel gap + functional-group counts)
# ══════════════════════════════════════════════════════════════════════════

def _get_star_atoms(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _conjugated_subgraph(mol):
    sp2 = {a.GetIdx() for a in mol.GetAtoms()
           if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2}
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

    left_link, right_link = _link_atom(star_idx[0]), _link_atom(star_idx[1])
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
    "amide": "[NX3][CX3](=O)", "imide": "[CX3](=O)[NX3][CX3](=O)",
    "ester": "[CX3](=O)[OX2H0]", "ether": "[OD2]([#6])[#6]",
    "sulfone": "[SX4](=O)(=O)", "carbonate": "[OX2][CX3](=O)[OX2]",
    "hbond_donor": "[#7,#8;!H0]", "hbond_acceptor": "[#7,#8]",
    "aromatic_ring": "c1ccccc1",
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
        print("  EGC: not enough valid rows -> treating as ungated (features only).")

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
        print("  TG: not enough valid rows -> treating as ungated (features only).")

    for ttype in ["tg", "egc"]:
        mode = "RESIDUAL LEARNING" if results[ttype] > PHYSICS_R2_GATE else "FEATURES ONLY"
        print(f"  >>> {ttype.upper()}: R^2={results[ttype]:+.4f}  ->  {mode}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# NOISE-CEILING DIAGNOSTIC (new)
# ══════════════════════════════════════════════════════════════════════════

def noise_ceiling_diagnostic(raw_train: pd.DataFrame) -> dict:
    """
    Empirical upper bound on achievable R^2 per target type, estimated from
    duplicate (canon_smiles, target_type) label disagreement. If your
    model's CV R^2 is already close to this ceiling, further modeling effort
    has low expected return -- the gap is measurement noise, not model bias.
    """
    print(f"\n{'='*60}\n  NOISE-CEILING DIAGNOSTIC\n{'='*60}")
    ceilings = {}
    for ttype in ["tg", "egc"]:
        sub = raw_train[raw_train["target_type"] == ttype]
        groups = sub.groupby("canon_smiles")["target"].agg(list)
        dup_groups = groups[groups.apply(len) > 1]
        if len(dup_groups) < 3:
            print(f"  {ttype.upper()}: too few duplicate groups ({len(dup_groups)}) for a reliable estimate.")
            ceilings[ttype] = None
            continue
        # Pooled within-group variance = noise variance estimate.
        within_var = dup_groups.apply(lambda v: np.var(v, ddof=1)).mean()
        total_var = sub["target"].var(ddof=1)
        ceiling = 1.0 - within_var / total_var if total_var > 0 else None
        ceilings[ttype] = ceiling
        print(f"  {ttype.upper()}: {len(dup_groups)} duplicate groups | "
              f"within-group var={within_var:.4f} | total var={total_var:.4f}")
        if ceiling is not None:
            print(f"  {ttype.upper()}: estimated max achievable R^2 (label-noise ceiling) = {ceiling:.4f}")
    print()
    return ceilings


def compute_sample_weights(raw_train: pd.DataFrame, collapsed: pd.DataFrame) -> np.ndarray:
    """
    Inverse-variance weighting for the collapsed (canon_smiles, target_type)
    rows. Groups with tight label agreement get weight up to
    MAX_SAMPLE_WEIGHT; groups with only one observation (no variance info)
    get weight 1.0; noisy/conflicting groups get down-weighted toward the
    weight floor. This replaces "collapse to mean, treat all rows equally."
    """
    var_lookup = (
        raw_train.groupby(["canon_smiles", "target_type"])["target"]
        .agg(lambda v: np.var(v, ddof=1) if len(v) > 1 else np.nan)
    )
    # Median within-group variance as a reference scale.
    ref_var = var_lookup.dropna().median()
    if pd.isna(ref_var) or ref_var <= 0:
        return np.ones(len(collapsed))

    weights = np.ones(len(collapsed))
    for i, row in collapsed.iterrows():
        key = (row["canon_smiles"], row["target_type"])
        v = var_lookup.get(key, np.nan)
        if pd.isna(v):
            weights[i] = 1.0
        else:
            w = ref_var / max(v, 1e-8)
            weights[i] = float(np.clip(w, 1.0 / MAX_SAMPLE_WEIGHT, MAX_SAMPLE_WEIGHT))
    return weights


# ── Feature computation ──────────────────────────────────────────────────────

def _desc_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {name: np.nan for name, _ in Descriptors._descList}
    return Descriptors.CalcMolDescriptors(mol)

def _ecfp4_one(smi, n_bits=ECFP4_BITS):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * n_bits
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))

def _ecfp6_one(smi, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * n_bits
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits))

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
    return {"star_distance": star_dist,
            "star_distance_frac": star_dist / diameter if diameter > 0 else 0.0}

def _electronic_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"num_aromatic_rings": np.nan, "aromatic_atom_fraction": np.nan,
                "num_rotatable_bonds": np.nan, "sp2_atom_fraction": np.nan,
                "num_nonarom_double_bonds": np.nan}
    n_atoms = mol.GetNumAtoms()
    n_arom  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_sp2   = sum(1 for a in mol.GetAtoms()
                  if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2)
    n_dbl   = sum(1 for b in mol.GetBonds()
                  if b.GetBondTypeAsDouble() == 2.0 and not b.GetIsAromatic())
    return {"num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "aromatic_atom_fraction": n_arom / n_atoms if n_atoms > 0 else 0.0,
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "sp2_atom_fraction": n_sp2 / n_atoms if n_atoms > 0 else 0.0,
            "num_nonarom_double_bonds": n_dbl}

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
    sp2 = {a.GetIdx() for a in mol.GetAtoms()
           if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2}
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


def _build_chain(smi, n_units):
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
        offset    = chain.GetNumAtoms()
        right_nbr = chain.GetAtomWithIdx(chain_right_star).GetNeighbors()[0].GetIdx()
        bond_type = chain.GetBondBetweenAtoms(chain_right_star, right_nbr).GetBondType()

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


# Descriptors tracked across chain lengths for Richardson extrapolation.
# Kept to a focused, physically-motivated subset (not the full ~200-column
# RDKit descriptor list) so the extrapolation fit is well-conditioned per
# molecule and the extra compute stays bounded.
_EXTRAP_DESC_NAMES = [
    "MolWt", "TPSA", "MolLogP", "NumRotatableBonds", "NumAromaticRings",
    "FractionCSP3", "NumHAcceptors", "NumHDonors", "LabuteASA",
]

def _extrap_desc_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {name: np.nan for name in _EXTRAP_DESC_NAMES}
    all_desc = dict(Descriptors.CalcMolDescriptors(mol))
    return {name: all_desc.get(name, np.nan) for name in _EXTRAP_DESC_NAMES}


def _richardson_extrapolate_one(smi, n_max=N_CHAIN_UNITS_MAX):
    """
    Build oligomer chains of length 1..n_max, compute a small physically
    interpretable descriptor set at each length, then fit
        descriptor(n) = a + b / n
    per descriptor via least squares (2+ points -> exact/near-exact fit;
    the closed-form periodic-system convergence law, same physics as the
    Huckel k-mesh band structure elsewhere in this pipeline). `a` is the
    extrapolated infinite-chain (bulk) value; `b` is the finite-size
    correction magnitude, itself informative (large |b| => descriptor is
    strongly finite-size-sensitive, e.g. end-group-dominated).
    Falls back to the raw n_max value if chain-building fails at any length.
    """
    lengths, rows = [], []
    for n in range(1, n_max + 1):
        chain_smi = _build_chain(smi, n) if n > 1 else smi
        if chain_smi is None:
            continue
        rows.append(_extrap_desc_one(chain_smi))
        lengths.append(n)

    out = {}
    if len(lengths) < 2:
        # Fall back: no extrapolation possible, use raw descriptor as "a", b=0.
        single = rows[0] if rows else {name: np.nan for name in _EXTRAP_DESC_NAMES}
        for name in _EXTRAP_DESC_NAMES:
            out[f"extrap_a_{name}"] = single.get(name, np.nan)
            out[f"extrap_b_{name}"] = 0.0
        return out

    inv_n = np.array([1.0 / n for n in lengths])
    A = np.column_stack([np.ones_like(inv_n), inv_n])
    for name in _EXTRAP_DESC_NAMES:
        y = np.array([r.get(name, np.nan) for r in rows], dtype=float)
        if np.isnan(y).any():
            out[f"extrap_a_{name}"] = np.nan
            out[f"extrap_b_{name}"] = np.nan
            continue
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[f"extrap_a_{name}"] = float(coef[0])
        out[f"extrap_b_{name}"] = float(coef[1])
    return out


def compute_features(df):
    smiles = df["smiles"].tolist()

    descs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_desc_one)(s)        for s in smiles)
    ecfp4  = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp4_one)(s)       for s in smiles)
    maccs  = Parallel(n_jobs=-1, prefer="threads")(delayed(_maccs_one)(s)       for s in smiles)
    topo   = Parallel(n_jobs=-1, prefer="threads")(delayed(_topo_one)(s)        for s in smiles)
    elec   = Parallel(n_jobs=-1, prefer="threads")(delayed(_electronic_one)(s)  for s in smiles)
    tgfeat = Parallel(n_jobs=-1, prefer="threads")(delayed(_tg_specific_one)(s) for s in smiles)
    conj   = Parallel(n_jobs=-1, prefer="threads")(delayed(_conjugation_one)(s) for s in smiles)
    huckel = Parallel(n_jobs=-1, prefer="threads")(delayed(compute_huckel_gap)(s)   for s in smiles)
    groups = Parallel(n_jobs=-1, prefer="threads")(delayed(extract_group_counts)(s) for s in smiles)
    extrap = Parallel(n_jobs=-1, prefer="threads")(delayed(_richardson_extrapolate_one)(s) for s in smiles)

    desc_df   = pd.DataFrame(descs,   index=df.index)
    ecfp4_df  = pd.DataFrame(ecfp4,   index=df.index, columns=[f"ecfp4_{i}" for i in range(ECFP4_BITS)])
    maccs_df  = pd.DataFrame(maccs,   index=df.index, columns=[f"maccs_{i}" for i in range(167)])
    topo_df   = pd.DataFrame(topo,    index=df.index)
    elec_df   = pd.DataFrame(elec,    index=df.index)
    tgfeat_df = pd.DataFrame(tgfeat,  index=df.index)
    conj_df   = pd.DataFrame(conj,    index=df.index)
    huckel_df = pd.DataFrame({"huckel_gap": huckel}, index=df.index)
    groups_df = pd.DataFrame(groups,  index=df.index).add_prefix("grp_")
    extrap_df = pd.DataFrame(extrap,  index=df.index)

    blocks = [desc_df, ecfp4_df, maccs_df, topo_df, elec_df, tgfeat_df, conj_df,
              huckel_df, groups_df, extrap_df]

    if USE_ECFP6:
        ecfp6 = Parallel(n_jobs=-1, prefer="threads")(delayed(_ecfp6_one)(s) for s in smiles)
        ecfp6_df = pd.DataFrame(ecfp6, index=df.index, columns=[f"ecfp6_{i}" for i in range(2048)])
        blocks.insert(2, ecfp6_df)

    combined = pd.concat(blocks, axis=1)
    combined = combined.astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return combined


# ── Load ──────────────────────────────────────────────────────────────────────

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
    train.groupby(["canon_smiles", "target_type"])["target"].agg(list).reset_index()
)
dup_groups = dup_groups[dup_groups["target"].apply(len) > 1]
print(f"\nDuplicate (canon SMILES, target_type) groups: {len(dup_groups)}")
if len(dup_groups) > 0:
    print("  [spread = max - min; large spread = labeling conflict]")
    for _, row in dup_groups.iterrows():
        vals = row["target"]
        spread = max(vals) - min(vals)
        print(f"  {row['target_type'].upper()}  spread={spread:.2f}"
              f"  values={[round(v, 2) for v in vals]}"
              f"  smiles={row['canon_smiles'][:50]}")

# --- Noise-ceiling diagnostic runs on the RAW (pre-collapse) data ---
noise_ceilings = noise_ceiling_diagnostic(train)

n_before = len(train)
train_collapsed = (
    train.groupby(["canon_smiles", "target_type"], as_index=False)
    .agg(smiles=("smiles", "first"), target=("target", "mean"))
)
n_after = len(train_collapsed)
print(f"\nDuplicate merge: {n_before:,} -> {n_after:,} rows  ({n_before - n_after} groups collapsed to mean)")

# --- Inverse-variance sample weights (new) ---
sample_weights_raw = compute_sample_weights(train, train_collapsed)
print(f"Sample weight range: [{sample_weights_raw.min():.3f}, {sample_weights_raw.max():.3f}]  "
      f"(1.0 = no duplicate info; >1 = tight-agreement group; <1 = noisy group)")

train = train_collapsed


# ── Physics sanity check (decides residual-learning mode per target) ─────────

physics_r2   = physics_sanity_check(train)
PHYSICS_MODE = {t: (physics_r2[t] > PHYSICS_R2_GATE) for t in ["tg", "egc"]}
physics      = PhysicsBaseline().fit(train)

print(f"\nPhysics residual-learning mode: {PHYSICS_MODE}\n")

physics_baseline_train = physics.predict_baseline(train)
y_train_effective = train["target"].values.copy()
for ttype, use_residual in PHYSICS_MODE.items():
    if use_residual:
        mask = (train["target_type"] == ttype).values
        y_train_effective[mask] = train["target"].values[mask] - physics_baseline_train[mask]


# ── Features ──────────────────────────────────────────────────────────────────

print("\nComputing features (includes Richardson chain-extrapolation block)...")
X_train_full = compute_features(train)
X_test_full  = compute_features(test)
print(f"  Done.  {X_train_full.shape[1]} features per molecule")

y_train        = y_train_effective
strat_label    = train["target_type"].values
groups         = train["canon_smiles"].values
sample_weights = sample_weights_raw

test_tg  = test[test["target_type"] == "tg"].copy()
test_egc = test[test["target_type"] == "egc"].copy()
test_subsets = {"tg": test_tg, "egc": test_egc}

physics_baseline_test = {
    ttype: physics.predict_baseline(subset) for ttype, subset in test_subsets.items()
}

all_seed_test_tg, all_seed_test_egc, seed_cv_scores = [], [], []

print(f"\nMulti-seed run: SEEDS={SEEDS}  N_TRIALS={N_TRIALS}  N_XGB_TRIALS={N_XGB_TRIALS}  N_CAT_TRIALS={N_CAT_TRIALS}")
print("Hyperparameters are tuned ONCE (at SEEDS[0]) and reused for every seed's final fit -- see rationale item 5.\n")

# Params populated on the first seed's tuning pass, then frozen and reused.
FROZEN_LGB_PARAMS: dict = {}
FROZEN_XGB_PARAMS: dict = {}
FROZEN_CAT_PARAMS: dict = {}
FROZEN_KEEP_COLS: list = []


for run_idx, run_seed in enumerate(SEEDS):
    print(f"\n{'#'*60}\n  RUN {run_idx + 1}/{len(SEEDS)}  (seed={run_seed})\n{'#'*60}")

    _lgb_fixed = {**LGB_FIXED, "random_state": run_seed}
    _xgb_fixed = {**XGB_FIXED, "random_state": run_seed}
    _cat_fixed = {**CAT_FIXED, "random_seed":  run_seed}

    sgkf      = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=run_seed)
    cv_splits = list(sgkf.split(X_train_full, strat_label, groups))

    is_tuning_run = (run_idx == 0)

    if is_tuning_run:
        print(f"\n{'='*60}\n  FEATURE PRUNING  ({PROBE_N_ESTIMATORS}-tree probe, permutation importance)\n{'='*60}\n")
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
            w_probe_tr  = sample_weights[probe_tr_idx][mask_tr]
            probe = LGBMRegressor(n_estimators=PROBE_N_ESTIMATORS, num_leaves=63,
                                   random_state=run_seed, n_jobs=_probe_jobs, verbose=-1)
            probe.fit(X_probe_tr, y_probe_tr, sample_weight=w_probe_tr)
            result = permutation_importance(probe, X_probe_val, y_probe_val,
                                             n_repeats=3, random_state=run_seed, n_jobs=_probe_jobs)
            imp = pd.Series(result.importances_mean, index=X_train_full.columns)
            positive = imp[imp > 0].index
            print(f"  {ttype.upper()}: {len(positive):,} / {n_orig:,} features with positive permutation importance")
            return set(positive.tolist())

        keep_cols = set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for cols in pool.map(_prune_for_ttype, ["tg", "egc"]):
                keep_cols.update(cols)
        FROZEN_KEEP_COLS[:] = sorted(keep_cols)
        print(f"\n  Kept {len(FROZEN_KEEP_COLS):,} features, dropped {n_orig - len(FROZEN_KEEP_COLS):,} zero-importance")

    keep_cols = FROZEN_KEEP_COLS
    X_train = X_train_full[keep_cols]
    X_test  = X_test_full[keep_cols]

    if is_tuning_run:
        print(f"\n{'='*60}\n  TUNING (once)  LGB({N_TRIALS}) + XGB({N_XGB_TRIALS}) + CAT({N_CAT_TRIALS}) trials, "
              f"{TUNE_FOLDS}-fold  |  6 studies concurrently\n{'='*60}")

        tune_splits = cv_splits[:TUNE_FOLDS]

        def make_lgb_objective(ttype, _cv=tune_splits, _X=X_train, _y=y_train,
                               _sl=strat_label, _w=sample_weights, _fixed=_lgb_fixed):
            def objective(trial):
                params = {**_fixed, "n_estimators": TUNE_N_ESTIMATORS, "n_jobs": _TUNE_N_JOBS,
                          "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                          "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                          "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                          "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                          "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                          "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                          "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                          "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)}
                scores = []
                for tr_idx, val_idx in _cv:
                    mask_tr, mask_val = _sl[tr_idx] == ttype, _sl[val_idx] == ttype
                    X_tr, X_val = _X.iloc[tr_idx][mask_tr], _X.iloc[val_idx][mask_val]
                    y_tr, y_val = _y[tr_idx][mask_tr], _y[val_idx][mask_val]
                    w_tr = _w[tr_idx][mask_tr]
                    model = LGBMRegressor(**params)
                    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
                              callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)])
                    scores.append(r2_score(y_val, model.predict(X_val)))
                return np.mean(scores)
            return objective

        def make_xgb_objective(ttype, _cv=tune_splits, _X=X_train, _y=y_train,
                               _sl=strat_label, _w=sample_weights, _fixed=_xgb_fixed):
            def objective(trial):
                params = {**_fixed, "n_estimators": TUNE_N_ESTIMATORS, "n_jobs": _TUNE_N_JOBS,
                          "max_depth": trial.suggest_int("max_depth", 3, 8),
                          "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                          "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                          "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                          "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                          "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                          "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True)}
                scores = []
                for tr_idx, val_idx in _cv:
                    mask_tr, mask_val = _sl[tr_idx] == ttype, _sl[val_idx] == ttype
                    X_tr, X_val = _X.iloc[tr_idx][mask_tr], _X.iloc[val_idx][mask_val]
                    y_tr, y_val = _y[tr_idx][mask_tr], _y[val_idx][mask_val]
                    w_tr = _w[tr_idx][mask_tr]
                    model = XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
                    scores.append(r2_score(y_val, model.predict(X_val)))
                return np.mean(scores)
            return objective

        def make_cat_objective(ttype, _cv=tune_splits, _X=X_train, _y=y_train,
                               _sl=strat_label, _w=sample_weights, _fixed=_cat_fixed):
            def objective(trial):
                params = {**_fixed, "n_estimators": TUNE_N_ESTIMATORS, "thread_count": _TUNE_N_JOBS,
                          "depth": trial.suggest_int("depth", 4, 8),
                          "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                          "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                          "rsm": trial.suggest_float("rsm", 0.4, 1.0),
                          "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50)}
                scores = []
                for tr_idx, val_idx in _cv:
                    mask_tr, mask_val = _sl[tr_idx] == ttype, _sl[val_idx] == ttype
                    X_tr, X_val = _X.iloc[tr_idx][mask_tr], _X.iloc[val_idx][mask_val]
                    y_tr, y_val = _y[tr_idx][mask_tr], _y[val_idx][mask_val]
                    w_tr = _w[tr_idx][mask_tr]
                    model = CatBoostRegressor(**params)
                    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_val, y_val))
                    scores.append(r2_score(y_val, model.predict(X_val)))
                return np.mean(scores)
            return objective

        _make_obj = {"lgb": make_lgb_objective, "xgb": make_xgb_objective, "cat": make_cat_objective}
        _n_trials = {"lgb": N_TRIALS, "xgb": N_XGB_TRIALS, "cat": N_CAT_TRIALS}
        _fix_map  = {"lgb": _lgb_fixed, "xgb": _xgb_fixed, "cat": _cat_fixed}

        def _run_study(args):
            model, ttype = args
            s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=run_seed))
            s.optimize(_make_obj[model](ttype), n_trials=_n_trials[model],
                       n_jobs=N_OPTUNA_JOBS, show_progress_bar=False)
            return model, ttype, s.best_value, {**_fix_map[model], **s.best_params}

        print(f"\n  Running 6 studies concurrently ({N_OPTUNA_JOBS} trial threads each)...")
        _dest = {"lgb": FROZEN_LGB_PARAMS, "xgb": FROZEN_XGB_PARAMS, "cat": FROZEN_CAT_PARAMS}
        tune_tasks = [(m, t) for m in ["lgb", "xgb", "cat"] for t in ["tg", "egc"]]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for model, ttype, best_val, best_params in pool.map(_run_study, tune_tasks):
                _dest[model][ttype] = best_params
                print(f"  {model.upper()} {ttype.upper()}: Best R²={best_val:+.4f}  params={best_params}")
    else:
        print("  (reusing frozen hyperparameters and pruned feature set from RUN 1 -- see rationale item 5)")

    best_lgb_params = {t: {**FROZEN_LGB_PARAMS[t], "random_state": run_seed} for t in ["tg", "egc"]}
    best_xgb_params = {t: {**FROZEN_XGB_PARAMS[t], "random_state": run_seed} for t in ["tg", "egc"]}
    best_cat_params = {t: {**FROZEN_CAT_PARAMS[t], "random_seed":  run_seed} for t in ["tg", "egc"]}

    print(f"\n{'='*60}\n  FINAL CV  (LGB+XGB+CAT+PLS)  +  FOLD-ENSEMBLED TEST PREDICTIONS\n{'='*60}\n")

    test_pred_lgb = {t: np.zeros(len(s)) for t, s in test_subsets.items()}
    test_pred_xgb = {t: np.zeros(len(s)) for t, s in test_subsets.items()}
    test_pred_cat = {t: np.zeros(len(s)) for t, s in test_subsets.items()}
    test_pred_pls = {t: np.zeros(len(s)) for t, s in test_subsets.items()}
    fold_lgb_preds = {"tg": [], "egc": []}
    fold_xgb_preds = {"tg": [], "egc": []}
    fold_cat_preds = {"tg": [], "egc": []}
    fold_pls_preds = {"tg": [], "egc": []}
    fold_y_vals    = {"tg": [], "egc": []}

    # PLS operates on fingerprint columns only (dense, linear-friendly),
    # not the full mixed descriptor/fingerprint matrix.
    fp_cols = [c for c in keep_cols if c.startswith(("ecfp4_", "ecfp6_", "maccs_"))]
    if len(fp_cols) < 5:
        fp_cols = keep_cols  # fallback if pruning removed almost all fp columns

    def _fit_fold_ttype(args):
        fold_idx, ttype = args
        tr_idx, val_idx = cv_splits[fold_idx]
        mask_tr, mask_val = strat_label[tr_idx] == ttype, strat_label[val_idx] == ttype
        X_tr, X_val = X_train.iloc[tr_idx][mask_tr], X_train.iloc[val_idx][mask_val]
        y_tr, y_val = y_train[tr_idx][mask_tr], y_train[val_idx][mask_val]
        w_tr = sample_weights[tr_idx][mask_tr]
        X_test_sub = X_test.loc[test_subsets[ttype].index]

        lgb_model = LGBMRegressor(**{**best_lgb_params[ttype], "n_jobs": _FINAL_N_JOBS})
        lgb_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
                      callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)])

        xgb_model = XGBRegressor(**{**best_xgb_params[ttype], "n_jobs": _FINAL_N_JOBS},
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        xgb_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)

        cat_model = CatBoostRegressor(**{**best_cat_params[ttype], "thread_count": _FINAL_N_JOBS})
        cat_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_val, y_val))

        # PLS: structurally different learner (linear, latent-factor) on the
        # fingerprint block -- real ensemble diversity, see rationale item 4.
        X_tr_fp  = X_tr[fp_cols].fillna(0.0)
        X_val_fp = X_val[fp_cols].fillna(0.0)
        X_test_fp = X_test_sub[fp_cols].fillna(0.0)
        n_comp = min(20, X_tr_fp.shape[1] - 1, X_tr_fp.shape[0] - 1)
        n_comp = max(n_comp, 1)
        pls_model = PLSRegression(n_components=n_comp)
        pls_model.fit(X_tr_fp, y_tr)

        lgb_val, xgb_val, cat_val = lgb_model.predict(X_val), xgb_model.predict(X_val), cat_model.predict(X_val)
        pls_val = pls_model.predict(X_val_fp).ravel()
        lgb_test, xgb_test, cat_test = lgb_model.predict(X_test_sub), xgb_model.predict(X_test_sub), cat_model.predict(X_test_sub)
        pls_test = pls_model.predict(X_test_fp).ravel()

        return (fold_idx, ttype, lgb_val, xgb_val, cat_val, pls_val,
                lgb_test, xgb_test, cat_test, pls_test, y_val)

    print(f"  Running all {N_FOLDS * 2} (fold, target) tasks concurrently...")
    cv_tasks = [(fi, tt) for fi in range(N_FOLDS) for tt in ["tg", "egc"]]
    fold_r2_log = {}

    with ThreadPoolExecutor(max_workers=N_FOLDS * 2) as pool:
        for (fold_idx, ttype, lgb_val, xgb_val, cat_val, pls_val,
             lgb_test, xgb_test, cat_test, pls_test, y_val) in pool.map(_fit_fold_ttype, cv_tasks):
            r2 = r2_score(y_val, (lgb_val + xgb_val + cat_val + pls_val) / 4)
            fold_r2_log.setdefault(fold_idx, {})[ttype] = r2
            fold_lgb_preds[ttype].append(lgb_val)
            fold_xgb_preds[ttype].append(xgb_val)
            fold_cat_preds[ttype].append(cat_val)
            fold_pls_preds[ttype].append(pls_val)
            fold_y_vals[ttype].append(y_val)
            test_pred_lgb[ttype] += lgb_test
            test_pred_xgb[ttype] += xgb_test
            test_pred_cat[ttype] += cat_test
            test_pred_pls[ttype] += pls_test

    for fi in sorted(fold_r2_log):
        s = fold_r2_log[fi]
        mean = (s["tg"] + s["egc"]) / 2
        print(f"  Fold {fi+1}  R²(Tg)={s['tg']:+.4f}  R²(Egc)={s['egc']:+.4f}  mean={mean:+.4f}  (naive equal-weight, effective target)")

    print(f"\n{'='*60}\n  OOF STACKING  (RidgeCV meta-learner, nested-fold evaluation)\n{'='*60}\n")

    _RIDGE_ALPHAS = np.logspace(-3, 3, 25)
    stacker, nest_preds, nest_true = {}, {"tg": [], "egc": []}, {"tg": [], "egc": []}

    for ttype in ["tg", "egc"]:
        lgb_parts, xgb_parts, cat_parts, pls_parts, y_parts = (
            fold_lgb_preds[ttype], fold_xgb_preds[ttype], fold_cat_preds[ttype],
            fold_pls_preds[ttype], fold_y_vals[ttype],
        )
        for hold in range(N_FOLDS):
            tr_f = [i for i in range(N_FOLDS) if i != hold]
            meta_X_tr = np.column_stack([
                np.concatenate([lgb_parts[i] for i in tr_f]),
                np.concatenate([xgb_parts[i] for i in tr_f]),
                np.concatenate([cat_parts[i] for i in tr_f]),
                np.concatenate([pls_parts[i] for i in tr_f]),
            ])
            meta_y_tr  = np.concatenate([y_parts[i] for i in tr_f])
            meta_X_val = np.column_stack([lgb_parts[hold], xgb_parts[hold], cat_parts[hold], pls_parts[hold]])
            ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
            ridge.fit(meta_X_tr, meta_y_tr)
            nest_preds[ttype].append(ridge.predict(meta_X_val))
            nest_true[ttype].append(y_parts[hold])

        all_X = np.column_stack([np.concatenate(lgb_parts), np.concatenate(xgb_parts),
                                  np.concatenate(cat_parts), np.concatenate(pls_parts)])
        all_y = np.concatenate(y_parts)
        ridge_final = RidgeCV(alphas=_RIDGE_ALPHAS)
        ridge_final.fit(all_X, all_y)
        stacker[ttype] = ridge_final
        coef = ridge_final.coef_
        print(f"  {ttype.upper()}: stacker coefs  LGB={coef[0]:.3f}  XGB={coef[1]:.3f}  "
              f"CAT={coef[2]:.3f}  PLS={coef[3]:.3f}  alpha={ridge_final.alpha_:.4g}")

    fold_r2_stacked = {"tg": [], "egc": []}
    for ttype in ["tg", "egc"]:
        for hold in range(N_FOLDS):
            fold_r2_stacked[ttype].append(r2_score(nest_true[ttype][hold], nest_preds[ttype][hold]))

    cv_tg, cv_egc = np.mean(fold_r2_stacked["tg"]), np.mean(fold_r2_stacked["egc"])
    print(f"\n  Mean R²(Tg)  [effective target] = {cv_tg:+.4f}  (std {np.std(fold_r2_stacked['tg']):.4f})")
    print(f"  Mean R²(Egc) [effective target] = {cv_egc:+.4f}  (std {np.std(fold_r2_stacked['egc']):.4f})")

    raw_r2 = {}
    for ttype in ["tg", "egc"]:
        raw_true, raw_pred = [], []
        for hold in range(N_FOLDS):
            tr_idx, val_idx = cv_splits[hold]
            mask_val = strat_label[val_idx] == ttype
            val_rows = train.iloc[val_idx][mask_val]
            true_raw = val_rows["target"].values
            pred_eff = nest_preds[ttype][hold]
            pred_raw = physics.predict_baseline(val_rows) + pred_eff if PHYSICS_MODE[ttype] else pred_eff
            raw_true.append(true_raw)
            raw_pred.append(pred_raw)
        raw_r2[ttype] = r2_score(np.concatenate(raw_true), np.concatenate(raw_pred))

    cv_r2_raw = (raw_r2["tg"] + raw_r2["egc"]) / 2
    print(f"\n  Mean R²(Tg)  [RAW units] = {raw_r2['tg']:+.4f}"
          + (f"  (noise ceiling {noise_ceilings['tg']:.4f})" if noise_ceilings.get("tg") else ""))
    print(f"  Mean R²(Egc) [RAW units] = {raw_r2['egc']:+.4f}"
          + (f"  (noise ceiling {noise_ceilings['egc']:.4f})" if noise_ceilings.get("egc") else ""))
    print(f"\n  >>> CV score (seed={run_seed}), RAW units = {cv_r2_raw:+.4f} <<<")

    seed_tg_pred_eff = stacker["tg"].predict(np.column_stack([
        test_pred_lgb["tg"] / N_FOLDS, test_pred_xgb["tg"] / N_FOLDS,
        test_pred_cat["tg"] / N_FOLDS, test_pred_pls["tg"] / N_FOLDS,
    ]))
    seed_egc_pred_eff = stacker["egc"].predict(np.column_stack([
        test_pred_lgb["egc"] / N_FOLDS, test_pred_xgb["egc"] / N_FOLDS,
        test_pred_cat["egc"] / N_FOLDS, test_pred_pls["egc"] / N_FOLDS,
    ]))

    seed_tg_pred  = physics_baseline_test["tg"]  + seed_tg_pred_eff  if PHYSICS_MODE["tg"]  else seed_tg_pred_eff
    seed_egc_pred = physics_baseline_test["egc"] + seed_egc_pred_eff if PHYSICS_MODE["egc"] else seed_egc_pred_eff

    all_seed_test_tg.append(seed_tg_pred)
    all_seed_test_egc.append(seed_egc_pred)
    seed_cv_scores.append(cv_r2_raw)


# ── Multi-seed summary ────────────────────────────────────────────────────────

print(f"\n{'='*60}\n  MULTI-SEED SUMMARY  (RAW target units)\n{'='*60}")
for seed, score in zip(SEEDS, seed_cv_scores):
    print(f"  Seed {seed:3d}: CV = {score:+.4f}")
print(f"  Mean CV  : {np.mean(seed_cv_scores):+.4f}  (std {np.std(seed_cv_scores):.4f})")
print(f"\n  Physics residual-learning mode used: {PHYSICS_MODE}")
print(f"  Physics-baseline standalone CV R^2 : {physics_r2}")
print(f"  Label-noise R^2 ceiling (empirical): {noise_ceilings}")

test_tg["target"]  = np.mean(all_seed_test_tg,  axis=0)
test_egc["target"] = np.mean(all_seed_test_egc, axis=0)


# ── Submission ────────────────────────────────────────────────────────────────

submission = pd.concat([test_tg, test_egc])[["id", "target"]].sort_values("id")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)

print(f"\nSubmission saved -> {OUTPUT_PATH}")
print(f"  {len(submission):,} rows  |  id range: {submission['id'].min()}-{submission['id'].max()}")
print(f"\n  First 5 rows:")
print(submission.head(5).to_string(index=False))

elapsed = datetime.now() - start
print(f"\nDone. Total time: {elapsed}")
