"""
ANRF AISEHack 2.0 -- Polymer Property Prediction. Single self-contained
script for a Kaggle Script/Notebook kernel.
"""

import hashlib
import inspect
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem, Descriptors, Descriptors3D, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys,
    rdPartialCharges, Fragments,
)

RDLogger.DisableLog('rdApp.*')

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_JOBS = -1

# ============================================================================
# FAST MODE: Toggle for rapid iteration (3-4x faster). Set False for final LB sub.
# ============================================================================
FAST_MODE = False   # <--- flip this to False for the final submission run

# ============================================================================
# GPU: Toggle GPU acceleration for XGBoost / CatBoost / LightGBM. These three
# are called inside every Optuna trial, every bagged refit, and every
# pseudo-labeling round, so they dominate wall-clock time -- this is the
# highest-leverage place to add GPU. Set False to fall back to CPU (e.g. if
# no CUDA device is available, or if your LightGBM build is CPU-only).
# ============================================================================
USE_GPU = True
LIGHTGBM_GPU = False  # most pip LightGBM wheels are CPU-only; flip on only if
                       # you've built/installed a GPU-enabled lightgbm

N_OPTUNA_TRIALS = 35
BAG_SEEDS = list(range(10))

# Upper bound on n_estimators/iterations offered to Optuna during the search
# phase. The full range (up to 800) is reserved for the final, untuned-range
# search on a FAST_MODE=False run -- during iteration, capping this avoids
# burning minutes per trial on trials that were never going to be picked.
BOOSTING_SEARCH_MAX_ESTIMATORS = 800

if FAST_MODE:
    N_OPTUNA_TRIALS = 12          # was 35; 12 still finds the big wins
    BAG_SEEDS = list(range(3))    # was 10; 3 captures most variance reduction
    BOOSTING_SEARCH_MAX_ESTIMATORS = 300   # was 800; trims per-trial fit cost

PL_CONF_FRACTION = 0.5
PL_SAMPLE_WEIGHT = 0.5
PL_MIN_ROWS = 5
PL_BAG_SEEDS = BAG_SEEDS[:len(BAG_SEEDS) // 2]
PL_CONF_CANDIDATES = [0.65, 0.8]
PL_CONF_SEARCH_TARGETS = {'eps', 'ei', 'nc'}

GNN_STACK_TARGETS = {'egc', 'tg', 'egb', 'eea'}
GNN_BAG_SEEDS = 3
GNN_BAG_SEEDS_BY_TARGET = {'egb': 8, 'eea': 8}
XPROP_TARGETS = {'egb', 'ei', 'eea', 'eps', 'nc'}
MLP_TARGETS = {'tg', 'egc'}
GNN_D_H = 300
GNN_DEPTH = 3
GNN_DROPOUT = 0.0
GNN_FFN_HIDDEN = 300
GNN_FFN_LAYERS = 1
GNN_INNER_VAL_SPLITS = 9
GNN_TARGET_CONFIG = {
    'egc': dict(batch_size=64, max_epochs=250, patience=20),
    'tg':  dict(batch_size=64, max_epochs=250, patience=20),
    'egb': dict(batch_size=32, max_epochs=300, patience=40),
    'eea': dict(batch_size=16, max_epochs=300, patience=40),
}
GNN_DEFAULT_CONFIG = dict(batch_size=64, max_epochs=250, patience=20)

# Fast-mode gating for the in-notebook GNN / 3D / PL-search / MLP work --
# these are real time sinks that are safe to skip during iteration.
if FAST_MODE:
    GNN_STACK_TARGETS = set()
    D3D_TARGETS_OVERRIDE = set()
    PL_CONF_SEARCH_TARGETS = set()
    MLP_TARGETS = set()
    GNN_BAG_SEEDS = 1
else:
    D3D_TARGETS_OVERRIDE = None


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
    return here / ""


INPUT_DIR = _find_input_dir()
TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"
SAMPLE_SUB_PATH = INPUT_DIR / "sample_submission.csv"
OUT_PATH = Path("submission.csv")

FP_BITS = 256
FP_RADIUS = 2
VAR_THRESH = 1e-6
CORR_THRESH = 0.98
SELECT_K = 100

# New descriptor block sizes (Atom Pair + Topological Torsion fingerprints)
AP_BITS = 512
TT_BITS = 256

_SLOW_OR_UNSTABLE = {'Ipc'}
_DESC_LIST = [(n, f) for n, f in Descriptors._descList if n not in _SLOW_OR_UNSTABLE]
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
_FRAGMENT_FUNCS = [(n, getattr(Fragments, n)) for n in dir(Fragments) if n.startswith('fr_')]

SMALL_TARGETS = {'egb', 'ei', 'eea', 'eps', 'nc'}
LARGE_TARGETS = {'tg', 'egc'}
D3D_TARGETS = {'eps', 'ei', 'nc'}
if D3D_TARGETS_OVERRIDE is not None:
    D3D_TARGETS = D3D_TARGETS_OVERRIDE
N_SPLITS = 5
N_REPEATS_SMALL = 3

# nc gets a sqrt transform: nc is coupled to eps via eps ~= nc^2, so nc has
# a mild quadratic tail that a sqrt stabilizes -- compresses the upper tail
# where the eps linkage lives.
TARGET_TRANSFORMS = {
    'eps': (np.log, np.exp),
    'ei': (np.log, np.exp),
    'nc': (np.sqrt, np.square),
}

MARGIN_FRACTION = 0.10
PHYSICAL_FLOORS = {'egc': 0.0, 'egb': 0.0, 'nc': 1.0, 'eps': 1.0, 'ei': 0.0}
MODEL_NAMES = ['Ridge', 'KNN', 'RF', 'ExtraTrees', 'GBM', 'XGB', 'CatBoost', 'LightGBM']
META_ALPHA = 1.0


# ---------------------------------------------------------------------------
# 1. Featurization
# ---------------------------------------------------------------------------
def _parse_mol(smiles):
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    return mol


def _make_dimer_mol(smiles):
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

        combo.GetAtomWithIdx(a2).SetAtomMapNum(1)
        combo.GetAtomWithIdx(b1).SetAtomMapNum(2)

        combo.AddBond(a2, b1, Chem.BondType.SINGLE)
        for idx in sorted(dummy_idx, reverse=True):
            combo.RemoveAtom(idx)

        dimer = combo.GetMol()
        try:
            Chem.SanitizeMol(dimer)
        except Exception:
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


def junction_bond_feats(dimer):
    defaults = {'junction_bond_conjugated': 0.0, 'junction_bond_aromatic': 0.0,
                'junction_bond_in_ring': 0.0}
    if dimer is None:
        return defaults
    try:
        a2_atom = next((a for a in dimer.GetAtoms() if a.GetAtomMapNum() == 1), None)
        b1_atom = next((a for a in dimer.GetAtoms() if a.GetAtomMapNum() == 2), None)
        if a2_atom is None or b1_atom is None:
            return defaults
        bond = dimer.GetBondBetweenAtoms(a2_atom.GetIdx(), b1_atom.GetIdx())
        if bond is None:
            return defaults
        return {
            'junction_bond_conjugated': float(bond.GetIsConjugated()),
            'junction_bond_aromatic': float(bond.GetIsAromatic()),
            'junction_bond_in_ring': float(bond.IsInRing()),
        }
    except Exception:
        return defaults


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


# ---------------------------------------------------------------------------
# NEW: Atom Pair fingerprints -- complementary path-based topology to Morgan.
# ---------------------------------------------------------------------------
def _atom_pair_bits(mol):
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=AP_BITS)
    arr = np.zeros((AP_BITS,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return {f'ap_{i}': int(arr[i]) for i in range(AP_BITS)}


# ---------------------------------------------------------------------------
# NEW: Topological Torsion fingerprints -- 4-atom dihedral signatures.
# ---------------------------------------------------------------------------
def _torsion_bits(mol):
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=TT_BITS)
    arr = np.zeros((TT_BITS,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return {f'tt_{i}': int(arr[i]) for i in range(TT_BITS)}


# ---------------------------------------------------------------------------
# NEW: Backbone conjugation -- what fraction of the *-to-* path is
# conjugated/aromatic. Different from global ConjugationRatio or
# max_conj_component_size: this looks specifically along the backbone path.
# ---------------------------------------------------------------------------
def backbone_conjugation_feats(mol):
    defaults = {'backbone_conj_frac': 0.0, 'backbone_arom_frac': 0.0}
    dummy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    if len(dummy_idx) != 2:
        return defaults
    try:
        path = Chem.GetShortestPath(mol, dummy_idx[0], dummy_idx[1])
        if not path or len(path) < 2:
            return defaults
        n_bonds, n_conj, n_arom = 0, 0, 0
        for i in range(len(path) - 1):
            b = mol.GetBondBetweenAtoms(path[i], path[i + 1])
            if b is None:
                continue
            n_bonds += 1
            if b.GetIsConjugated():
                n_conj += 1
            if b.GetIsAromatic():
                n_arom += 1
        return {
            'backbone_conj_frac': n_conj / n_bonds if n_bonds else 0.0,
            'backbone_arom_frac': n_arom / n_bonds if n_bonds else 0.0,
        }
    except Exception:
        return defaults


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
    """RDKit featurization for one molecule: monomer descriptors + physics
    ratios + Morgan fingerprint + full monomer->dimer descriptor deltas
    (+ NEW: monomer->dimer ratio features, junction-bond features)."""
    mol = _parse_mol(smiles)
    if mol is None:
        return None

    feats = {}
    try:
        mono_desc = _safe_descriptors(mol)
        feats.update(mono_desc)
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

            dimer_desc = _safe_descriptors(dimer)
            for key, mono_val in mono_desc.items():
                dimer_val = dimer_desc.get(key, np.nan)
                if np.isfinite(mono_val) and np.isfinite(dimer_val):
                    feats[f'dimer_delta_{key}'] = dimer_val - mono_val
                else:
                    feats[f'dimer_delta_{key}'] = np.nan

                # NEW: multiplicative (ratio) dimer features -- "how much does
                # this property scale when the chain doubles?", a signal pure
                # deltas alone miss.
                if np.isfinite(mono_val) and np.isfinite(dimer_val) and abs(mono_val) > 1e-9:
                    feats[f'dimer_ratio_{key}'] = dimer_val / mono_val
                else:
                    feats[f'dimer_ratio_{key}'] = np.nan
        else:
            feats['dimer_delta_AromaticRatio'] = 0.0
            feats['dimer_delta_ConjugationRatio'] = 0.0
            feats['dimer_delta_RotBondsPerHeavy'] = 0.0
            for key in mono_desc:
                feats[f'dimer_delta_{key}'] = 0.0
                feats[f'dimer_ratio_{key}'] = np.nan
        feats.update(junction_bond_feats(dimer))
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
    mol = Chem.Mol(mol)
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


def fragment_counts(mol):
    out = {}
    for name, func in _FRAGMENT_FUNCS:
        try:
            out[f'frag_{name}'] = func(mol)
        except Exception:
            out[f'frag_{name}'] = 0
    return out


def attachment_point_feats(mol):
    defaults = {
        'attach_path_len': 0.0, 'attach_path_len_frac': 0.0,
        'attach_backbone_frac': 0.0, 'attach_same_ring': 0.0,
        'attach_neighbor_aromatic_frac': 0.0, 'attach_neighbor_sp3_frac': 0.0,
    }
    dummy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    if len(dummy_idx) != 2:
        return defaults
    try:
        path = Chem.GetShortestPath(mol, dummy_idx[0], dummy_idx[1])
        if not path:
            return defaults
        heavy = mol.GetNumHeavyAtoms() or 1
        path_len = len(path) - 1
        backbone_atoms = len(path) - 2

        nbrs = []
        for idx in dummy_idx:
            atom_nbrs = mol.GetAtomWithIdx(idx).GetNeighbors()
            if atom_nbrs:
                nbrs.append(atom_nbrs[0])

        same_ring = 0.0
        if len(nbrs) == 2:
            ring_info = mol.GetRingInfo()
            same_ring = float(any(
                nbrs[0].GetIdx() in ring and nbrs[1].GetIdx() in ring
                for ring in ring_info.AtomRings()
            ))

        arom_frac = float(np.mean([n.GetIsAromatic() for n in nbrs])) if nbrs else 0.0
        sp3_frac = float(np.mean(
            [n.GetHybridization() == Chem.HybridizationType.SP3 for n in nbrs]
        )) if nbrs else 0.0

        return {
            'attach_path_len': float(path_len),
            'attach_path_len_frac': path_len / heavy,
            'attach_backbone_frac': backbone_atoms / heavy,
            'attach_same_ring': same_ring,
            'attach_neighbor_aromatic_frac': arom_frac,
            'attach_neighbor_sp3_frac': sp3_frac,
        }
    except Exception:
        return defaults


def conjugation_extent_feats(mol):
    heavy = mol.GetNumHeavyAtoms() or 1
    conj_bonds = [b for b in mol.GetBonds() if b.GetIsConjugated()]
    if not conj_bonds:
        return {'max_conj_component_size': 0.0, 'max_conj_component_frac': 0.0,
                'n_conj_components': 0.0}

    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for b in conj_bonds:
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        parent.setdefault(a1, a1)
        parent.setdefault(a2, a2)
        r1, r2 = find(a1), find(a2)
        if r1 != r2:
            parent[r1] = r2

    component_sizes = {}
    for atom_idx in parent:
        root = find(atom_idx)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    max_size = max(component_sizes.values())
    return {
        'max_conj_component_size': float(max_size),
        'max_conj_component_frac': max_size / heavy,
        'n_conj_components': float(len(component_sizes)),
    }


_D3D_DEFAULTS = {
    'd3d_dipole': np.nan, 'd3d_radius_of_gyration': np.nan, 'd3d_asphericity': np.nan,
    'd3d_eccentricity': np.nan, 'd3d_spherocity': np.nan, 'd3d_npr1': np.nan, 'd3d_npr2': np.nan,
}


def conformer_3d_feats(smiles):
    dimer = _make_dimer_mol(smiles)
    if dimer is None:
        return dict(_D3D_DEFAULTS)
    try:
        mol_h = Chem.AddHs(dimer)
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_STATE
        if AllChem.EmbedMolecule(mol_h, params) == -1:
            return dict(_D3D_DEFAULTS)
        if AllChem.MMFFOptimizeMolecule(mol_h) == -1:
            return dict(_D3D_DEFAULTS)

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h)
        if mmff_props is None:
            return dict(_D3D_DEFAULTS)
        conf = mol_h.GetConformer()
        dipole_vec = np.zeros(3)
        for atom in mol_h.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            dipole_vec += mmff_props.GetMMFFPartialCharge(idx) * np.array([pos.x, pos.y, pos.z])

        return {
            'd3d_dipole': float(np.linalg.norm(dipole_vec)),
            'd3d_radius_of_gyration': Descriptors3D.RadiusOfGyration(mol_h),
            'd3d_asphericity': Descriptors3D.Asphericity(mol_h),
            'd3d_eccentricity': Descriptors3D.Eccentricity(mol_h),
            'd3d_spherocity': Descriptors3D.SpherocityIndex(mol_h),
            'd3d_npr1': Descriptors3D.NPR1(mol_h),
            'd3d_npr2': Descriptors3D.NPR2(mol_h),
        }
    except Exception:
        return dict(_D3D_DEFAULTS)


# ---------------------------------------------------------------------------
# NEW: fast, numpy-based feature pruner (~10x faster than pandas .corr() for
# >700 columns). Same semantics: drop near-zero variance columns, then drop
# one column from each |r|>CORR_THRESH pair.
# ---------------------------------------------------------------------------
def fit_feature_pruner(df):
    arr = df.to_numpy(dtype=float)
    variances = np.nanvar(arr, axis=0)
    keep_mask = variances > VAR_THRESH
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        return df.columns.tolist()

    sub = arr[:, keep_idx].copy()
    col_means = np.nanmean(sub, axis=0)
    inds = np.where(np.isnan(sub))
    sub[inds] = np.take(col_means, inds[1])

    corr = np.corrcoef(sub, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    bad = (np.abs(corr) > CORR_THRESH) & upper
    drop_within = set(np.where(bad.any(axis=0))[0].tolist())

    final_idx = [keep_idx[i] for i in range(len(keep_idx)) if i not in drop_within]
    return [df.columns[i] for i in final_idx]


# ---------------------------------------------------------------------------
# NEW: parallel RDKit featurization + disk cache. compute_raw_features is
# the #1 wall-clock bottleneck before any model trains -- this parallelizes
# both the baseline featurize() pass and the auxiliary-feature pass, and
# caches the (raw_df, valid_df) result to disk keyed on a hash of the
# smiles column, so repeat runs on unchanged data skip RDKit entirely.
# ---------------------------------------------------------------------------
CACHE_DIR = Path("feature_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _hash_smiles_series(s):
    return hashlib.sha256(pd.util.hash_pandas_object(s).values.tobytes()).hexdigest()[:16]


def _featurize_safe(smiles):
    try:
        return featurize(smiles)
    except Exception:
        return None


def _aux_all(mol):
    """Bundle every auxiliary (non-featurize()) feature block so it can be
    parallelized in one shot per molecule."""
    return {
        **maccs_keys(mol),
        **gasteiger_features(mol),
        **fragment_counts(mol),
        **attachment_point_feats(mol),
        **conjugation_extent_feats(mol),
        **_atom_pair_bits(mol),
        **_torsion_bits(mol),
        **backbone_conjugation_feats(mol),
    }


def compute_raw_features(df, n_jobs=-1):
    """Parallel RDKit featurization (baseline + MACCS + Gasteiger +
    Fragments + attachment-point + conjugation-extent + NEW: Atom Pair /
    Topological Torsion fingerprints + backbone-conjugation features),
    unpruned, with an automatic disk cache keyed on the smiles column.
    Takes a df with a 'smiles' column, handles the featurize()-can-return-
    None filtering itself, and returns (raw_df, valid_df)."""
    cache_key = _hash_smiles_series(df['smiles'])
    cache_path = CACHE_DIR / f"raw_{cache_key}.pkl"
    if cache_path.exists():
        return pickle.load(open(cache_path, "rb"))

    raw_feats = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_featurize_safe)(s) for s in df['smiles']
    )
    raw_feats = pd.Series(raw_feats, index=df.index)
    valid_mask = raw_feats.notna()
    if (~valid_mask).sum():
        print(f"  dropping {(~valid_mask).sum()} rows that failed to featurize")
    valid_df = df[valid_mask].reset_index(drop=True)
    raw_baseline_df = pd.DataFrame(list(raw_feats[valid_mask])).reset_index(drop=True)

    mols = valid_df['smiles'].apply(_parse_mol)
    aux_list = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_aux_all)(m) for m in mols
    )
    aux_df = pd.DataFrame(aux_list).reset_index(drop=True)

    raw_df = pd.concat([raw_baseline_df, aux_df], axis=1)
    result = (raw_df, valid_df)
    pickle.dump(result, open(cache_path, "wb"))
    return result


def build_cross_target_lookup(train_valid):
    all_types = sorted(train_valid['target_type'].unique())
    wide = (train_valid.groupby(['canon', 'target_type'])['target']
            .mean().unstack('target_type').reindex(columns=all_types))
    return wide


def cross_target_feats(df, lookup):
    all_types = lookup.columns
    vals = lookup.reindex(df['canon'].values)
    vals.index = df.index
    for tt in all_types:
        own_mask = (df['target_type'] == tt).values
        if own_mask.any():
            vals.loc[own_mask, tt] = np.nan
    known = vals.notna().astype(int).add_prefix('known_')
    phys = physics_cross_feats(vals)
    vals = vals.add_prefix('xtarget_')
    return pd.concat([vals, known, phys], axis=1)


def physics_cross_feats(vals):
    cols = vals.columns
    out = {}
    if 'nc' in cols:
        out['phys_nc_sq'] = vals['nc'] ** 2
    if 'eps' in cols:
        out['phys_eps_sqrt'] = np.sqrt(vals['eps'].clip(lower=0))
    if 'ei' in cols and 'eea' in cols:
        out['phys_gap_from_ie_ea'] = vals['ei'] - vals['eea']
    if 'egb' in cols and 'eea' in cols:
        out['phys_ie_from_egb_ea'] = vals['egb'] + vals['eea']
    if 'egc' in cols and 'eea' in cols:
        out['phys_ie_from_egc_ea'] = vals['egc'] + vals['eea']
    if 'ei' in cols and 'egb' in cols:
        out['phys_ea_from_ie_egb'] = vals['ei'] - vals['egb']
    return pd.DataFrame(out, index=vals.index)


def get_xtarget_eval_splits(df, target_type, n_repeats=N_REPEATS_SMALL, base_seed=RANDOM_STATE):
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values
    seed_folds = []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=base_seed + r)
        seed_folds.append(list(gkf.split(np.zeros(len(sub_index)), groups=groups)))
    return sub_index, seed_folds


def paired_delta_verdict(tt, X_a, X_b, y_all, train_valid, t0, label='xtarget'):
    sub_index, seed_folds = get_xtarget_eval_splits(train_valid, tt)
    factory = get_zoo_factories(tt, {})['LightGBM']

    a_fold_scores, b_fold_scores = [], []
    for folds in seed_folds:
        a_fold_scores.append(score_model_folds(factory, X_a, y_all, sub_index, [folds]))
        b_fold_scores.append(score_model_folds(factory, X_b, y_all, sub_index, [folds]))
    a_fold_scores = np.concatenate(a_fold_scores)
    b_fold_scores = np.concatenate(b_fold_scores)

    a_mean = float(a_fold_scores.mean())
    b_mean = float(b_fold_scores.mean())
    fold_deltas = b_fold_scores - a_fold_scores
    delta = float(fold_deltas.mean())
    noise_floor = float(fold_deltas.std())
    accepted = delta > noise_floor
    print(f"  {tt:5s}: baseline={a_mean:.4f}  +{label}={b_mean:.4f}  "
          f"paired delta={delta:+.4f} (+/-{noise_floor:.3f})  -> "
          f"{'ACCEPT' if accepted else 'reject'}  [{time.time()-t0:.0f}s]")
    return accepted, a_mean, b_mean, delta, noise_floor


def evaluate_cross_target_features(X_base, X_xt, y_all, train_valid, target_types, t0):
    print("\n" + "=" * 100)
    print("Step 12 re-open -- cross-target feature block, accept/reject vs. noise floor "
          "(LightGBM proxy, real harness, 3-seed paired delta)")
    print("=" * 100)

    base_means, xt_means = [], []
    verdicts = {}
    for tt in target_types:
        accepted, base_mean, xt_mean, _, _ = paired_delta_verdict(
            tt, X_base, X_xt, y_all, train_valid, t0, label='xtarget')
        verdicts[tt] = accepted
        base_means.append(base_mean)
        xt_means.append(xt_mean)

    shipped = [tt for tt in target_types if verdicts[tt]]
    print(f"\n  MEAN across {len(target_types)} targets (informational only -- ship/no-ship is "
          f"now decided per-target above, not pooled): baseline={float(np.mean(base_means)):.4f}  "
          f"+xtarget={float(np.mean(xt_means)):.4f}")
    print(f"  Cross-target features SHIP for: {shipped if shipped else '(none)'}")
    return verdicts


def build_feature_matrix(df, cross_target_lookup=None):
    raw_df, valid_df = compute_raw_features(df)
    if cross_target_lookup is not None:
        raw_df = pd.concat([raw_df, cross_target_feats(valid_df, cross_target_lookup)], axis=1)
    feature_cols = fit_feature_pruner(raw_df)
    X = raw_df[feature_cols]
    return X, feature_cols, valid_df


# ---------------------------------------------------------------------------
# 2. CV harness
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
    train['orig_row'] = np.arange(len(train))
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
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    return sub_index, list(gkf.split(np.zeros(len(sub_index)), groups=groups))


def constant_column_mask(Xtr):
    with np.errstate(invalid='ignore'):
        stds = np.nanstd(Xtr, axis=0)
    return (stds > 0) & ~np.isnan(stds)


def drop_constant_columns(Xtr, Xva):
    keep = constant_column_mask(Xtr)
    return Xtr[:, keep], Xva[:, keep]


def wrap_for_target(model_factory, target_type):
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
def linear_models(seed_offset=0):
    rs = RANDOM_STATE + seed_offset
    return {
        'Ridge': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('kbest', SelectKBest(f_regression, k=SELECT_K)),
            ('m', Ridge(alpha=5.0, random_state=rs)),
        ]),
        'KNN': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('kbest', SelectKBest(f_regression, k=SELECT_K)),
            ('m', KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=N_JOBS)),
        ]),
    }


def tree_models(seed_offset=0):
    rs = RANDOM_STATE + seed_offset
    return {
        'RF': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                n_jobs=N_JOBS, random_state=rs)),
        ]),
        'ExtraTrees': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', ExtraTreesRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                n_jobs=N_JOBS, random_state=rs)),
        ]),
        'GBM': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=rs)),
        ]),
    }


# NEW (Fast Mode): for small targets in FAST_MODE, drop the slow/redundant
# tree models that ablations show correlate at >0.95 with the boosting trio
# on small N -- cuts iteration time with minimal CV impact.
_SMALL_TARGET_MODEL_NAMES = ['Ridge', 'KNN', 'LightGBM', 'XGB', 'CatBoost'] if FAST_MODE else None


def model_names_for(tt):
    if _SMALL_TARGET_MODEL_NAMES is not None and tt in SMALL_TARGETS:
        return _SMALL_TARGET_MODEL_NAMES
    return MODEL_NAMES


def boosting_default_params():
    """GPU: XGB uses tree_method='hist' + device='cuda' (XGBoost>=2.0 unified
    GPU API). CatBoost uses task_type='GPU'. LightGBM only switches to
    device='gpu' if LIGHTGBM_GPU is True, since most pip wheels are CPU-only
    and will error (or silently ignore the flag) without a GPU-enabled build.
    """
    xgb_params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                       random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
    cat_params = dict(iterations=300, depth=6, learning_rate=0.05,
                       l2_leaf_reg=3.0, random_state=RANDOM_STATE, verbose=False)
    lgb_params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.8,
                       random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)

    if USE_GPU:
        xgb_params.update(tree_method='hist', device='cuda')
        cat_params.update(task_type='GPU', devices='0')
    else:
        cat_params.update(thread_count=N_JOBS)

    if USE_GPU and LIGHTGBM_GPU:
        lgb_params.update(device='gpu')

    return {'XGB': xgb_params, 'CatBoost': cat_params, 'LightGBM': lgb_params}


BOOSTING_CTORS = {'XGB': XGBRegressor, 'CatBoost': CatBoostRegressor, 'LightGBM': LGBMRegressor}

TUNABLE_TREE_CTORS = {'RF': RandomForestRegressor, 'ExtraTrees': ExtraTreesRegressor,
                       'GBM': GradientBoostingRegressor}
TUNABLE_LINEAR_CTORS = {'Ridge': Ridge, 'KNN': KNeighborsRegressor}

# NEW: in FAST_MODE, skip tuning RF/ExtraTrees/GBM entirely -- these are
# CPU-only (no GPU path without a cuML rewrite), Optuna's trial loop can't
# parallelize across trials the way it can within one RF fit, and in
# practice CatBoost/XGB/LightGBM tuning tends to find the real wins (see
# your own run: only CatBoost accepted a tuned config across both rounds).
# Ridge/KNN stay tunable since they're cheap (SelectKBest caps k<=200).
# Flip SKIP_TREE_TUNING_IN_FAST_MODE off if you want RF/ExtraTrees/GBM
# tuning back during iteration.
SKIP_TREE_TUNING_IN_FAST_MODE = True
if FAST_MODE and SKIP_TREE_TUNING_IN_FAST_MODE:
    EXTENDED_TUNABLE = list(TUNABLE_LINEAR_CTORS)
else:
    EXTENDED_TUNABLE = list(TUNABLE_TREE_CTORS) + list(TUNABLE_LINEAR_CTORS)


def extended_default_params():
    return {
        'RF': dict(n_estimators=400, max_depth=8, min_samples_leaf=2, n_jobs=N_JOBS),
        'ExtraTrees': dict(n_estimators=400, max_depth=8, min_samples_leaf=2, n_jobs=N_JOBS),
        'GBM': dict(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9,
                    min_samples_leaf=1),
        'Ridge': dict(alpha=5.0, select_k=SELECT_K),
        'KNN': dict(n_neighbors=10, select_k=SELECT_K, weights='distance'),
    }


def extended_search_space(trial, name):
    """CPU-only sklearn models (no GPU path exists for RF/ExtraTrees/GBM
    without switching to cuML). n_estimators upper bound is capped by
    BOOSTING_SEARCH_MAX_ESTIMATORS the same way boosting_search_space is,
    since these are exactly as expensive per-tree as the boosting models
    and were previously left uncapped -- this was the FAST_MODE=True
    bottleneck after GPU-accelerating XGB/CatBoost/LightGBM."""
    max_est_rf = min(BOOSTING_SEARCH_MAX_ESTIMATORS, 600)
    max_est_gbm = min(BOOSTING_SEARCH_MAX_ESTIMATORS, 300)
    max_depth_rf = 10 if FAST_MODE else 15
    if name in ('RF', 'ExtraTrees'):
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, max_est_rf),
            max_depth=trial.suggest_int('max_depth', 4, max_depth_rf),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            n_jobs=N_JOBS,
        )
    if name == 'GBM':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, max_est_gbm),
            max_depth=trial.suggest_int('max_depth', 2, 6),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        )
    if name == 'Ridge':
        return dict(
            alpha=trial.suggest_float('alpha', 0.1, 50.0, log=True),
            select_k=trial.suggest_int('select_k', 30, 200),
        )
    if name == 'KNN':
        return dict(
            n_neighbors=trial.suggest_int('n_neighbors', 3, 30),
            select_k=trial.suggest_int('select_k', 30, 200),
            weights='distance',
        )
    raise ValueError(name)


def build_extended_pipeline(name, params, seed_offset=0):
    rs = RANDOM_STATE + seed_offset
    if name in TUNABLE_TREE_CTORS:
        ctor = TUNABLE_TREE_CTORS[name]
        return Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', ctor(**params, random_state=rs)),
        ])
    ctor = TUNABLE_LINEAR_CTORS[name]
    params = dict(params)
    k = params.pop('select_k')
    if name != 'KNN':
        params['random_state'] = rs
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('kbest', SelectKBest(f_regression, k=k)),
        ('m', ctor(**params)),
    ])


def tune_extended_model(name, target_type, X_df, y_series, search_sub_index, search_folds,
                         n_trials=N_OPTUNA_TRIALS):
    def objective(trial):
        params = extended_search_space(trial, name)
        factory = wrap_for_target(lambda: build_extended_pipeline(name, params), target_type)
        mean_r2, _ = score_model(factory, X_df, y_series, search_sub_index, [search_folds])
        return mean_r2

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_all_extended_models(X_by_target, y_all, train_valid, target_types, t0):
    print("\n" + "=" * 100)
    print(f"Live Optuna tuning (Task 3 extension) -- {N_OPTUNA_TRIALS} trials, 3-fold search, "
          f"per newly-tunable model per target (LARGE_TARGETS only -- see docstring)")
    print("=" * 100)

    search_targets = [tt for tt in target_types if tt in LARGE_TARGETS]
    search_cache = {tt: cached_search(tt) for tt in search_targets}
    harness_cache = {tt: cached_harness(tt) for tt in search_targets}

    defaults = extended_default_params()
    accepted_tuned_configs = {}
    for name in EXTENDED_TUNABLE:
        for tt in search_targets:
            X = X_by_target[tt]
            search_sub_index, search_folds = search_cache[tt]
            best_params = tune_extended_model(name, tt, X, y_all, search_sub_index, search_folds)
            full_params = {**defaults[name], **best_params}

            harness_sub_index, harness_repeats = harness_cache[tt]
            default_factory = wrap_for_target(
                lambda p=defaults[name]: build_extended_pipeline(name, p), tt)
            default_mean, default_std = score_model(
                default_factory, X, y_all, harness_sub_index, harness_repeats)
            tuned_factory = wrap_for_target(
                lambda p=full_params: build_extended_pipeline(name, p), tt)
            tuned_mean, tuned_std = score_model(
                tuned_factory, X, y_all, harness_sub_index, harness_repeats)

            noise_floor = max(tuned_std, default_std)
            if tuned_mean - default_mean > noise_floor:
                accepted_tuned_configs[(name, tt)] = full_params
                verdict = f"ACCEPT tuned ({default_mean:.4f} -> {tuned_mean:.4f})"
            else:
                verdict = f"keep default (tuned {tuned_mean:.4f} vs default {default_mean:.4f}, " \
                          f"didn't clear noise floor {noise_floor:.4f})"
            print(f"  {name:10s} {tt:5s}: {verdict}  [{time.time()-t0:.0f}s]")

    return accepted_tuned_configs


def get_zoo_factories(tt, accepted_tuned_configs, seed_offset=0):
    factories = {}
    for name, pipeline_template in {**linear_models(seed_offset), **tree_models(seed_offset)}.items():
        tuned = accepted_tuned_configs.get((name, tt)) if name in EXTENDED_TUNABLE else None
        if tuned is not None:
            factories[name] = wrap_for_target(
                lambda p=tuned, n=name, so=seed_offset: build_extended_pipeline(n, p, so), tt)
        else:
            factories[name] = wrap_for_target(lambda p=pipeline_template: clone(p), tt)

    defaults = boosting_default_params()
    for name, ctor in BOOSTING_CTORS.items():
        base_params = accepted_tuned_configs.get((name, tt), defaults[name])
        params = {**base_params, 'random_state': RANDOM_STATE + seed_offset}
        factories[name] = wrap_for_target(lambda p=params, c=ctor: c(**p), tt)

    return factories


def boosting_search_space(trial, name):
    """GPU flags are attached here too (not just in boosting_default_params)
    since Optuna calls BOOSTING_CTORS[name](**params) directly on whatever
    this returns. n_estimators/iterations upper bound is capped by
    BOOSTING_SEARCH_MAX_ESTIMATORS (lower in FAST_MODE) to keep each trial
    cheap during iteration; the full range is only used in the final,
    FAST_MODE=False run.
    """
    max_est = BOOSTING_SEARCH_MAX_ESTIMATORS
    if name == 'XGB':
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, max_est),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.01, 20.0, log=True),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
        )
        if USE_GPU:
            params.update(tree_method='hist', device='cuda')
        return params
    if name == 'CatBoost':
        params = dict(
            iterations=trial.suggest_int('iterations', 100, min(max_est, 400)),
            depth=trial.suggest_int('depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 20.0, log=True),
            random_state=RANDOM_STATE, verbose=False,
        )
        if USE_GPU:
            params.update(task_type='GPU', devices='0')
        else:
            params.update(thread_count=N_JOBS)
        return params
    if name == 'LightGBM':
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, max_est),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            num_leaves=trial.suggest_int('num_leaves', 15, 255),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1,
        )
        if USE_GPU and LIGHTGBM_GPU:
            params.update(device='gpu')
        return params
    raise ValueError(name)


def tune_boosting_model(name, target_type, X_df, y_series, search_sub_index, search_folds,
                         n_trials=N_OPTUNA_TRIALS):
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


def tune_all_boosting_models(X_by_target, y_all, train_valid, target_types, t0):
    print("\n" + "=" * 100)
    print(f"Live Optuna tuning -- {N_OPTUNA_TRIALS} trials, 3-fold search, "
          f"per boosting model per target (LARGE_TARGETS only -- see docstring)")
    print("=" * 100)

    search_targets = [tt for tt in target_types if tt in LARGE_TARGETS]
    search_cache = {tt: cached_search(tt) for tt in search_targets}
    harness_cache = {tt: cached_harness(tt) for tt in search_targets}

    defaults = boosting_default_params()
    accepted_tuned_configs = {}
    for name in BOOSTING_CTORS:
        ctor = BOOSTING_CTORS[name]
        for tt in search_targets:
            X = X_by_target[tt]
            search_sub_index, search_folds = search_cache[tt]
            best_params = tune_boosting_model(name, tt, X, y_all, search_sub_index, search_folds)
            full_params = {**defaults[name], **best_params}

            harness_sub_index, harness_repeats = harness_cache[tt]
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


# ---------------------------------------------------------------------------
# NEW: cache GroupKFold splits so we never recompute the same split twice.
# Optuna's inner loop used to rebuild the splitter on every trial; this is
# a 5-10 min saver on the full run. Shared by tune_all_boosting_models and
# tune_all_extended_models (both scan the same LARGE_TARGETS splits).
# ---------------------------------------------------------------------------
_SPLIT_CACHE = {}
# Set once at the top of main() to train_valid -- lets cached_harness/
# cached_search (called from tune_all_boosting_models/tune_all_extended_models,
# which only receive train_valid as a parameter like every other call site)
# reach it without threading a new parameter through every caller.
_TRAIN_VALID_REF = [None]


def cached_harness(tt):
    """Cache GroupKFold harness splits so we never recompute the same split
    twice -- shared across tune_all_boosting_models and
    tune_all_extended_models, which otherwise each rebuild the identical
    per-target splits."""
    if tt not in _SPLIT_CACHE:
        _SPLIT_CACHE[tt] = get_harness_splits(_TRAIN_VALID_REF[0], tt)
    return _SPLIT_CACHE[tt]


def cached_search(tt):
    key = ('search', tt)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = get_search_splits(_TRAIN_VALID_REF[0], tt)
    return _SPLIT_CACHE[key]


def score_model_folds(model_factory, X_df, y_series, sub_index, repeats):
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
    return np.array(scores)


def score_model(model_factory, X_df, y_series, sub_index, repeats):
    scores = score_model_folds(model_factory, X_df, y_series, sub_index, repeats)
    return float(scores.mean()), float(scores.std())


def generate_oof(model_factory, X_df, y_series, sub_index, repeats):
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


def generate_oof_with_fold_models(model_factory, X_df, y_series, sub_index, repeats):
    n = len(sub_index)
    accum = np.zeros(n)
    fold_models = []
    for folds in repeats:
        fold_pred = np.full(n, np.nan)
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            Xtr_full = X_df.loc[tr_idx].values.astype(float)
            Xva_full = X_df.loc[va_idx].values.astype(float)
            col_mask = constant_column_mask(Xtr_full)
            Xtr, Xva = Xtr_full[:, col_mask], Xva_full[:, col_mask]
            ytr = y_series.loc[tr_idx].values
            model = model_factory()
            model.fit(Xtr, ytr)
            fold_pred[va_pos] = model.predict(Xva)
            fold_models.append((model, col_mask))
        accum += fold_pred
    oof = accum / len(repeats)
    return pd.Series(oof, index=sub_index), fold_models


def apply_fold_ensemble(fold_models, X_ext_df):
    Xext = X_ext_df.values.astype(float)
    preds = np.zeros(len(Xext))
    for model, col_mask in fold_models:
        preds += model.predict(Xext[:, col_mask])
    return preds / len(fold_models)


def fit_predict_multi(model_factory, X_tr_df, y_tr_series, X_te_dfs, sample_weight=None):
    Xtr = X_tr_df.values.astype(float)
    keep = constant_column_mask(Xtr)
    model = model_factory()

    fit_kwargs = {}
    if sample_weight is not None:
        inner = getattr(model, 'regressor', model)
        final_est = inner.named_steps['m'] if hasattr(inner, 'named_steps') else inner
        if 'sample_weight' in inspect.signature(final_est.fit).parameters:
            kwarg = 'm__sample_weight' if hasattr(inner, 'named_steps') else 'sample_weight'
            fit_kwargs[kwarg] = sample_weight

    model.fit(Xtr[:, keep], y_tr_series.values, **fit_kwargs)
    return [model.predict(X.values.astype(float)[:, keep]) for X in X_te_dfs]


def fit_predict_full(model_factory, X_tr_df, y_tr_series, X_te_df, sample_weight=None):
    return fit_predict_multi(
        model_factory, X_tr_df, y_tr_series, [X_te_df], sample_weight)[0]


PHYS_ANCHORS = {
    'eps': lambda xt: xt['xtarget_nc'] ** 2,
    'nc':  lambda xt: np.sqrt(xt['xtarget_eps'].clip(lower=0)),
    'ei':  lambda xt: xt['xtarget_egb'] + xt['xtarget_eea'],
}

PHYS_IMPUTE_ANCHORS = {}


def _impute_anchor(tt, anchor_tr_all, anchor_te_all, pred_partners):
    if not pred_partners or tt not in PHYS_IMPUTE_ANCHORS:
        return anchor_tr_all, anchor_te_all
    partner_props, combine = PHYS_IMPUTE_ANCHORS[tt]
    if not all(pp in pred_partners for pp in partner_props):
        return anchor_tr_all, anchor_te_all
    ptr = pd.DataFrame({pp: pred_partners[pp][0] for pp in partner_props})
    pte = pd.DataFrame({pp: pred_partners[pp][1] for pp in partner_props})
    imp_tr = combine(ptr).reindex(anchor_tr_all.index)
    imp_te = combine(pte).reindex(anchor_te_all.index)
    anchor_tr_all = anchor_tr_all.fillna(imp_tr)
    anchor_te_all = anchor_te_all.fillna(imp_te)
    return anchor_tr_all, anchor_te_all


def compute_physdelta_predictions(tt, X_feat_train, X_feat_test, xprop_train, xprop_test,
                                  y_all, train_valid, test, pred_partners=None):
    if tt not in PHYS_ANCHORS:
        return None
    X_feat_test = X_feat_test.reindex(columns=X_feat_train.columns)
    anchor_tr_all = PHYS_ANCHORS[tt](xprop_train)
    anchor_te_all = PHYS_ANCHORS[tt](xprop_test)
    sub_index, repeats = get_harness_splits(train_valid, tt)
    n_measured = int(anchor_tr_all.loc[sub_index].notna().sum())
    anchor_tr_all, anchor_te_all = _impute_anchor(
        tt, anchor_tr_all, anchor_te_all, pred_partners)
    n_imputed = int(anchor_tr_all.loc[sub_index].notna().sum()) - n_measured
    if anchor_tr_all.loc[sub_index].notna().sum() < PL_MIN_ROWS:
        return None

    def lgb():
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.9, colsample_bytree=0.8, verbose=-1,
                       random_state=RANDOM_STATE, n_jobs=N_JOBS)
        if USE_GPU and LIGHTGBM_GPU:
            params['device'] = 'gpu'
        return LGBMRegressor(**params)

    y = y_all
    accum = np.zeros(len(sub_index), dtype=float)
    for folds in repeats:
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            Xtr = X_feat_train.loc[tr_idx].values.astype(float)
            Xva = X_feat_train.loc[va_idx].values.astype(float)
            keep = constant_column_mask(Xtr)
            Xtr, Xva = Xtr[:, keep], Xva[:, keep]
            ytr = y.loc[tr_idx].values
            a_tr = anchor_tr_all.loc[tr_idx].values
            a_va = anchor_tr_all.loc[va_idx].values
            has_tr, has_va = ~np.isnan(a_tr), ~np.isnan(a_va)

            direct = lgb().fit(Xtr, ytr)
            pred = direct.predict(Xva)
            if has_tr.sum() >= PL_MIN_ROWS and has_va.any():
                resid = lgb().fit(Xtr[has_tr], ytr[has_tr] - a_tr[has_tr])
                pred[has_va] = a_va[has_va] + resid.predict(Xva[has_va])
            accum[va_pos] += pred
    oof = pd.Series(accum / len(repeats), index=sub_index)

    test_series = pd.Series(np.nan, index=test.index, dtype=float)
    te_rows = test.index[(test['target_type'] == tt).values]
    if len(te_rows) > 0:
        Xtr = X_feat_train.loc[sub_index].values.astype(float)
        keep = constant_column_mask(Xtr)
        Xtr = Xtr[:, keep]
        Xte = X_feat_test.loc[te_rows].values.astype(float)[:, keep]
        ytr = y.loc[sub_index].values
        a_tr = anchor_tr_all.loc[sub_index].values
        a_te = anchor_te_all.loc[te_rows].values
        has_tr, has_te = ~np.isnan(a_tr), ~np.isnan(a_te)
        direct = lgb().fit(Xtr, ytr)
        preds = direct.predict(Xte)
        if has_tr.sum() >= PL_MIN_ROWS and has_te.any():
            resid = lgb().fit(Xtr[has_tr], ytr[has_tr] - a_tr[has_tr])
            preds[has_te] = a_te[has_te] + resid.predict(Xte[has_te])
        test_series.loc[te_rows] = preds

    n_anchor = int(anchor_tr_all.loc[sub_index].notna().sum())
    imp_note = f", {n_imputed} imputed" if n_imputed else ""
    print(f"  {tt:5s}: physics-delta model built (OOF R2="
          f"{r2_score(y.loc[sub_index].values, oof.loc[sub_index].values):.4f}, "
          f"{n_anchor}/{len(sub_index)} anchor rows [{n_measured} measured{imp_note}])")
    return oof, test_series


def _find_gnn_file(name):
    candidates = [INPUT_DIR]
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        candidates.append(Path.cwd())
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return None


def load_gnn_predictions(tt, train_valid, test):
    oof_path = _find_gnn_file(f"gnn_oof_{tt}.csv")
    test_path = _find_gnn_file(f"gnn_test_{tt}.csv")
    if oof_path is None or test_path is None:
        return None

    oof_raw = pd.read_csv(oof_path)
    test_raw = pd.read_csv(test_path)

    row_to_index = pd.Series(train_valid.index.values,
                             index=train_valid['orig_row'].to_numpy())
    mapped = oof_raw['orig_row'].map(row_to_index)
    keep = mapped.notna()
    oof = pd.Series(np.nan, index=train_valid.index, dtype=float)
    oof.loc[mapped[keep].astype(int).to_numpy()] = oof_raw.loc[keep, 'gnn_pred'].to_numpy()

    test_series = pd.Series(np.nan, index=test.index, dtype=float)
    in_range = test_raw['test_row'].isin(test.index)
    test_series.loc[test_raw.loc[in_range, 'test_row'].to_numpy()] = \
        test_raw.loc[in_range, 'gnn_pred'].to_numpy()

    own_rows = train_valid.index[train_valid['target_type'] == tt]
    if oof.loc[own_rows].isna().any():
        n_missing = int(oof.loc[own_rows].isna().sum())
        print(f"  {tt:5s}: GNN OOF file covers only "
              f"{len(own_rows) - n_missing}/{len(own_rows)} rows -- skipping GNN column")
        return None
    return oof, test_series


_GNN = {}


def _gnn_setup():
    if _GNN:
        return _GNN
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from chemprop import data as cp_data, models as cp_models, nn as cp_nn
    from chemprop.featurizers import (MultiHotAtomFeaturizer,
                                      SimpleMoleculeMolGraphFeaturizer)
    import torch
    torch.set_float32_matmul_precision('medium')

    base = MultiHotAtomFeaturizer.v2()
    seq = lambda x: list(x.keys()) if isinstance(x, dict) else list(x)
    nums = seq(base.atomic_nums)
    for z in (0, 48, 50, 52, 82):
        if z not in nums:
            nums.append(z)
    atom_f = MultiHotAtomFeaturizer(
        atomic_nums=nums, degrees=seq(base.degrees),
        formal_charges=seq(base.formal_charges), chiral_tags=seq(base.chiral_tags),
        num_Hs=seq(base.num_Hs), hybridizations=seq(base.hybridizations))
    _GNN.update(
        pl=pl, ES=EarlyStopping, MC=ModelCheckpoint, cp_data=cp_data,
        cp_models=cp_models, cp_nn=cp_nn, torch=torch,
        featurizer=SimpleMoleculeMolGraphFeaturizer(atom_featurizer=atom_f),
        accel=('gpu' if torch.cuda.is_available() else 'cpu'))
    return _GNN


def _gnn_fit_predict(mols, y_model, canon, tr_pos, eval_mol_lists, cfg, seed, work_dir):
    g = _gnn_setup()
    pl, cp_data, cp_models, cp_nn = g['pl'], g['cp_data'], g['cp_models'], g['cp_nn']
    featurizer = g['featurizer']
    pl.seed_everything(seed, workers=True, verbose=False)

    inner_groups = canon[tr_pos]
    n_inner = min(GNN_INNER_VAL_SPLITS, len(np.unique(inner_groups)))
    gkf = GroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    itr, iva = next(iter(gkf.split(np.zeros(len(tr_pos)), groups=inner_groups)))
    abs_tr, abs_iva = tr_pos[itr], tr_pos[iva]

    def dset(idx):
        dps = [cp_data.MoleculeDatapoint(mol=mols[i], y=np.array([y_model[i]], dtype=float))
               for i in idx]
        return cp_data.MoleculeDataset(dps, featurizer)

    train_dset = dset(abs_tr)
    scaler = train_dset.normalize_targets()
    ival_dset = dset(abs_iva)
    ival_dset.normalize_targets(scaler)
    bs = min(cfg['batch_size'], max(4, len(abs_tr) // 4))
    train_loader = cp_data.build_dataloader(train_dset, batch_size=bs, num_workers=0,
                                            shuffle=True, seed=seed)
    ival_loader = cp_data.build_dataloader(ival_dset, batch_size=bs, num_workers=0,
                                           shuffle=False)
    eval_loaders = []
    for ml in eval_mol_lists:
        dps = [cp_data.MoleculeDatapoint(mol=m, y=np.array([0.0])) for m in ml]
        eval_loaders.append(cp_data.build_dataloader(
            cp_data.MoleculeDataset(dps, featurizer), batch_size=64, num_workers=0,
            shuffle=False))

    mp = cp_nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim,
                                  d_h=GNN_D_H, depth=GNN_DEPTH, dropout=GNN_DROPOUT)
    ffn = cp_nn.RegressionFFN(
        input_dim=GNN_D_H, hidden_dim=GNN_FFN_HIDDEN, n_layers=GNN_FFN_LAYERS,
        dropout=GNN_DROPOUT,
        output_transform=cp_nn.UnscaleTransform.from_standard_scaler(scaler))
    model = cp_models.MPNN(mp, cp_nn.MeanAggregation(), ffn, batch_norm=True,
                           metrics=[cp_nn.RMSE()])
    ckpt = g['MC'](dirpath=work_dir, filename="best", save_top_k=1,
                   monitor="val_loss", mode="min", save_last=False)
    trainer = pl.Trainer(
        accelerator=g['accel'], devices=1, max_epochs=cfg['max_epochs'],
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=True,
        callbacks=[ckpt, g['ES'](monitor="val_loss", mode="min", patience=cfg['patience'])],
        num_sanity_val_steps=0, deterministic=False)
    trainer.fit(model, train_loader, ival_loader)
    best = (cp_models.MPNN.load_from_checkpoint(ckpt.best_model_path)
            if ckpt.best_model_path else model)
    outs = []
    for loader in eval_loaders:
        raw = trainer.predict(best, loader)
        outs.append(np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in raw]))
    return outs


def compute_gnn_predictions(tt, train_valid, test):
    import shutil
    import tempfile

    _gnn_setup()
    cfg = GNN_TARGET_CONFIG.get(tt, GNN_DEFAULT_CONFIG)
    n_seeds = GNN_BAG_SEEDS_BY_TARGET.get(tt, GNN_BAG_SEEDS)
    sub_index, repeats = get_harness_splits(train_valid, tt)
    sub = train_valid.loc[sub_index]
    y_true = sub['target'].to_numpy(dtype=float)
    canon = sub['canon'].to_numpy()
    mols = [_parse_mol(s) for s in sub['smiles']]
    transform = TARGET_TRANSFORMS.get(tt)
    y_model = transform[0](y_true) if transform is not None else y_true

    te_mask = (test['target_type'] == tt).to_numpy()
    te_rows = test.index[te_mask]
    te_mols_all = [_parse_mol(s) for s in test.loc[te_rows, 'smiles']]
    keep = [i for i, m in enumerate(te_mols_all) if m is not None]
    te_rows_valid = te_rows[np.array(keep, dtype=int)] if keep else te_rows[:0]
    te_mols = [te_mols_all[i] for i in keep]

    work = Path(tempfile.mkdtemp(prefix=f"gnn_{tt}_"))
    try:
        oof_accum = np.zeros(len(sub), dtype=float)
        for r, folds in enumerate(repeats):
            for k, (tr_pos, va_pos) in enumerate(folds):
                seed_preds = []
                for s in range(n_seeds):
                    seed = RANDOM_STATE + 1000 * r + 10 * k + s
                    outs = _gnn_fit_predict(
                        mols, y_model, canon, tr_pos,
                        [[mols[i] for i in va_pos]], cfg, seed, work / f"o{r}_{k}_{s}")
                    seed_preds.append(outs[0])
                p = np.mean(seed_preds, axis=0)
                if transform is not None:
                    p = transform[1](p)
                oof_accum[va_pos] += p
        oof = oof_accum / len(repeats)
        oof_series = pd.Series(np.nan, index=train_valid.index, dtype=float)
        oof_series.loc[sub_index] = oof

        test_series = pd.Series(np.nan, index=test.index, dtype=float)
        if len(te_mols) > 0:
            all_pos = np.arange(len(sub))
            te_preds = []
            for s in range(n_seeds):
                outs = _gnn_fit_predict(mols, y_model, canon, all_pos, [te_mols], cfg,
                                        RANDOM_STATE + 500 + s, work / f"refit_{s}")
                te_preds.append(outs[0])
            te_pred = np.mean(te_preds, axis=0)
            if transform is not None:
                te_pred = transform[1](te_pred)
            test_series.loc[te_rows_valid] = te_pred
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"  {tt:5s}: trained GNN in-process (OOF R2={r2_score(y_true, oof):.4f}, "
          f"{n_seeds} seeds/fit)")
    return oof_series, test_series


def get_gnn_predictions(tt, train_valid, test):
    cached = load_gnn_predictions(tt, train_valid, test)
    if cached is not None:
        print(f"  {tt:5s}: using cached GNN predictions (gnn_oof_{tt}.csv)")
        return cached
    try:
        return compute_gnn_predictions(tt, train_valid, test)
    except ImportError:
        print(f"  {tt:5s}: chemprop unavailable -- GNN column skipped fail-soft "
              f"(add chemprop to the kernel to enable in-notebook GNN training)")
        return None


def compute_xprop_predictions(tt, xprop_train, xprop_test, y_all, train_valid, test):
    sub_index, repeats = get_harness_splits(train_valid, tt)
    Xtt = xprop_train.loc[sub_index]
    if Xtt.notna().to_numpy().sum() == 0:
        return None

    lgb_params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.9, colsample_bytree=0.8, verbose=-1,
                       random_state=RANDOM_STATE, n_jobs=N_JOBS)
    if USE_GPU and LIGHTGBM_GPU:
        lgb_params['device'] = 'gpu'
    factory = wrap_for_target(lambda: LGBMRegressor(**lgb_params), tt)
    oof = generate_oof(factory, xprop_train, y_all, sub_index, repeats)

    test_series = pd.Series(np.nan, index=test.index, dtype=float)
    te_rows = test.index[(test['target_type'] == tt).values]
    if len(te_rows) > 0:
        preds = fit_predict_full(factory, xprop_train.loc[sub_index],
                                 y_all.loc[sub_index], xprop_test.loc[te_rows])
        test_series.loc[te_rows] = preds
    print(f"  {tt:5s}: cross-property model built (OOF R2="
          f"{r2_score(y_all.loc[sub_index].values, oof.loc[sub_index].values):.4f})")
    return oof, test_series


def mlp_factory(tt):
    def make():
        return Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('kbest', SelectKBest(f_regression, k=SELECT_K)),
            ('m', MLPRegressor(
                hidden_layer_sizes=(128, 64), activation='relu', alpha=1e-3,
                learning_rate_init=1e-3, max_iter=500, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=15,
                random_state=RANDOM_STATE)),
        ])
    return wrap_for_target(make, tt)


def compute_mlp_predictions(tt, X_feat_train, X_feat_test, y_all, train_valid, test):
    sub_index, repeats = get_harness_splits(train_valid, tt)
    factory = mlp_factory(tt)
    oof = generate_oof(factory, X_feat_train, y_all, sub_index, repeats)

    test_series = pd.Series(np.nan, index=test.index, dtype=float)
    te_rows = test.index[(test['target_type'] == tt).values]
    if len(te_rows) > 0:
        Xte = X_feat_test.reindex(columns=X_feat_train.columns).loc[te_rows]
        preds = fit_predict_full(factory, X_feat_train.loc[sub_index],
                                 y_all.loc[sub_index], Xte)
        test_series.loc[te_rows] = preds
    print(f"  {tt:5s}: MLP model built (OOF R2="
          f"{r2_score(y_all.loc[sub_index].values, oof.loc[sub_index].values):.4f})")
    return oof, test_series


def evaluate_gnn_stack_column(tt, oof_df, gnn_oof, y_all, train_valid, t0, col_name='GNN'):
    sub_index, seed_folds = get_xtarget_eval_splits(train_valid, tt)
    cand = oof_df.copy()
    cand[col_name] = gnn_oof.loc[sub_index].to_numpy()
    meta = META_LEARNER_CANDIDATES['Ridge']

    a_scores, b_scores = [], []
    for folds in seed_folds:
        a_scores.append(score_model_folds(meta, oof_df, y_all, sub_index, [folds]))
        b_scores.append(score_model_folds(meta, cand, y_all, sub_index, [folds]))
    a = np.concatenate(a_scores)
    b = np.concatenate(b_scores)
    deltas = b - a
    delta, noise_floor = float(deltas.mean()), float(deltas.std())
    accepted = delta > noise_floor

    print(f"  {tt:5s}: {col_name} +column test ({len(a)} folds, 3-seed paired) -- "
          f"stack={a.mean():.4f}, +{col_name}={b.mean():.4f}, "
          f"delta={delta:+.4f} vs noise floor {noise_floor:.4f} "
          f"({'ACCEPT' if accepted else 'reject'}) [{time.time()-t0:.0f}s]")
    return accepted, cand


def evaluate_pl_conf_fraction(tt, X_tt, y_all, train_valid, X_te_tt,
                               accepted_tuned_configs, cv_r2, t0):
    sub_index, repeats = get_harness_splits(train_valid, tt)
    names = model_names_for(tt)
    factories = get_zoo_factories(tt, accepted_tuned_configs)
    clip_lo, clip_hi = clip_bounds_for(tt, y_all, train_valid)
    fracs = [PL_CONF_FRACTION] + PL_CONF_CANDIDATES
    per_frac = {f: [] for f in fracs}

    for folds in repeats:
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            X_tr, X_va = X_tt.loc[tr_idx], X_tt.loc[va_idx]
            y_tr, y_va = y_all.loc[tr_idx], y_all.loc[va_idx].values

            va_cols, te_cols = [], []
            for n in names:
                p_va, p_te = fit_predict_multi(factories[n], X_tr, y_tr, [X_va, X_te_tt])
                va_cols.append(p_va)
                te_cols.append(p_te)
            base_te = np.column_stack(te_cols)
            pred_std = base_te.std(axis=1)
            first_pass = base_te.mean(axis=1)
            unaug_score = r2_score(y_va, np.column_stack(va_cols).mean(axis=1))
            expected_resid = y_tr.std() * np.sqrt(max(1 - cv_r2, 0.01))

            seen = {}
            for f in fracs:
                conf_mask = pred_std < f * expected_resid
                key = conf_mask.tobytes()
                if key in seen:
                    per_frac[f].append(seen[key])
                    continue
                if conf_mask.sum() < PL_MIN_ROWS:
                    score = unaug_score
                else:
                    pseudo_y = np.clip(first_pass[conf_mask], clip_lo, clip_hi)
                    X_aug = pd.concat([X_tr, X_te_tt.loc[conf_mask]], ignore_index=True)
                    y_aug = pd.concat([y_tr, pd.Series(pseudo_y)], ignore_index=True)
                    w_aug = np.concatenate([
                        np.ones(len(y_tr)),
                        np.full(int(conf_mask.sum()), PL_SAMPLE_WEIGHT)])
                    aug_va = np.column_stack([
                        fit_predict_multi(factories[n], X_aug, y_aug, [X_va], w_aug)[0]
                        for n in names])
                    score = r2_score(y_va, aug_va.mean(axis=1))
                seen[key] = score
                per_frac[f].append(score)

    default_scores = np.array(per_frac[PL_CONF_FRACTION])
    best_frac, best_delta = PL_CONF_FRACTION, 0.0
    print(f"  {tt:5s}: PL_CONF_FRACTION search over {fracs} "
          f"({len(default_scores)} folds, simulated PL round per fold) "
          f"[{time.time()-t0:.0f}s]")
    print(f"    default {PL_CONF_FRACTION:.2f}: sim mean R2={default_scores.mean():.4f}")
    for f in PL_CONF_CANDIDATES:
        cand = np.array(per_frac[f])
        deltas = cand - default_scores
        delta, noise_floor = float(deltas.mean()), float(deltas.std())
        cleared = delta > noise_floor
        print(f"    cand    {f:.2f}: sim mean R2={cand.mean():.4f}  "
              f"delta={delta:+.4f} vs noise floor {noise_floor:.4f} "
              f"({'ACCEPT' if cleared else 'reject'})")
        if cleared and delta > best_delta:
            best_frac, best_delta = f, delta

    print(f"  {tt:5s}: PL_CONF_FRACTION -> {best_frac:.2f}"
          f"{' (default kept)' if best_frac == PL_CONF_FRACTION else ' (ACCEPTED)'}")
    return best_frac


def bagged_refit_predict(tt, X_tr, y_tr, X_te, accepted_tuned_configs, seeds, sample_weight=None):
    names = model_names_for(tt)
    preds = np.zeros((len(X_te), len(names)))
    for seed_offset in seeds:
        factories = get_zoo_factories(tt, accepted_tuned_configs, seed_offset)
        preds += np.column_stack([
            fit_predict_full(factories[name], X_tr, y_tr, X_te, sample_weight)
            for name in names
        ])
    return preds / len(seeds)


# ---------------------------------------------------------------------------
# NEW: meta-learner candidates now include RidgeCV, which finds a better
# per-target alpha via efficient GCV in milliseconds instead of the fixed
# META_ALPHA=1.0 -- usually a small but consistent lift.
# ---------------------------------------------------------------------------
META_LEARNER_CANDIDATES = {
    'Ridge': lambda: Ridge(alpha=META_ALPHA),
    'RidgeCV': lambda: RidgeCV(alphas=np.logspace(-2, 2, 20), cv=None, scoring='r2'),
    'Ridge_positive': lambda: Ridge(alpha=META_ALPHA, positive=True),
}


def select_meta_learner(oof_df, y_all, sub_index, repeats):
    """Tests RidgeCV and Ridge(positive=True) against the default Ridge, on
    the OOF matrix via the real harness -- same accept/reject-against-
    noise-floor pattern used for every tuned hyperparameter in this script.
    An alternative only replaces Ridge if it beats Ridge's mean by more than
    the noise floor (larger of the two configs' fold-to-fold std)."""
    default_mean, default_std = score_model(
        META_LEARNER_CANDIDATES['Ridge'], oof_df, y_all, sub_index, repeats)
    best_name, best_mean, best_std = 'Ridge', default_mean, default_std

    for name in ('RidgeCV', 'Ridge_positive'):
        mean, std = score_model(META_LEARNER_CANDIDATES[name], oof_df, y_all, sub_index, repeats)
        noise_floor = max(std, default_std)
        cleared = mean - default_mean > noise_floor
        print(f"    meta-learner {name:14s}: mean={mean:.4f} vs Ridge {default_mean:.4f} "
              f"({'ACCEPT' if cleared else 'reject'}, noise floor {noise_floor:.4f})")
        if cleared and mean > best_mean:
            best_name, best_mean, best_std = name, mean, std

    return best_name, META_LEARNER_CANDIDATES[best_name], best_mean, best_std


def clip_bounds_for(tt, y_all, train_valid):
    y = y_all.loc[train_valid['target_type'] == tt]
    lo, hi = y.min(), y.max()
    margin = MARGIN_FRACTION * (hi - lo)
    clip_lo, clip_hi = lo - margin, hi + margin
    if tt in PHYSICAL_FLOORS:
        clip_lo = max(clip_lo, PHYSICAL_FLOORS[tt])
    return clip_lo, clip_hi


def process_target(tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
                    test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
                    oof_meta, cv_scores, test_predictions, t0,
                    track_fold_models=False, xprop=None, dense_preds=None):
    sub_index, repeats = get_harness_splits(train_valid, tt)
    names = model_names_for(tt)
    factories = get_zoo_factories(tt, accepted_tuned_configs)
    X_tt = X_by_target[tt]

    fold_models_by_name = {} if track_fold_models else None
    oof_cols = {}
    for name in names:
        if track_fold_models:
            oof_series, fold_models = generate_oof_with_fold_models(
                factories[name], X_tt, y_all, sub_index, repeats)
            fold_models_by_name[name] = fold_models
        else:
            oof_series = generate_oof(factories[name], X_tt, y_all, sub_index, repeats)
        oof_cols[name] = oof_series
    oof_df = pd.DataFrame(oof_cols)[names]
    print(f"  {tt:5s}: OOF generated for all {len(names)} models [{time.time()-t0:.0f}s]")

    base_names = list(oof_df.columns)
    extra_test_cols = []

    if tt in GNN_STACK_TARGETS:
        loaded = get_gnn_predictions(tt, train_valid, test)
        if loaded is None:
            print(f"  {tt:5s}: no GNN predictions available -- stack unchanged")
        else:
            gnn_oof, gnn_test = loaded
            accepted, cand_oof = evaluate_gnn_stack_column(
                tt, oof_df, gnn_oof, y_all, train_valid, t0, col_name='GNN')
            if accepted:
                oof_df = cand_oof
                extra_test_cols.append(('GNN', gnn_test))

    if tt in XPROP_TARGETS and xprop is not None:
        xres = compute_xprop_predictions(tt, xprop[0], xprop[1], y_all, train_valid, test)
        if xres is not None:
            xoof, xtest = xres
            accepted, cand_oof = evaluate_gnn_stack_column(
                tt, oof_df, xoof, y_all, train_valid, t0, col_name='XProp')
            if accepted:
                oof_df = cand_oof
                extra_test_cols.append(('XProp', xtest))

    if tt in PHYS_ANCHORS and xprop is not None and not FAST_MODE:
        pres = compute_physdelta_predictions(
            tt, X_tt, test_feat_df_by_target[tt], xprop[0], xprop[1],
            y_all, train_valid, test, pred_partners=dense_preds)
        if pres is not None:
            poof, ptest = pres
            accepted, cand_oof = evaluate_gnn_stack_column(
                tt, oof_df, poof, y_all, train_valid, t0, col_name='PhysDelta')
            if accepted:
                oof_df = cand_oof
                extra_test_cols.append(('PhysDelta', ptest))

    if tt in MLP_TARGETS:
        mres = compute_mlp_predictions(
            tt, X_tt, test_feat_df_by_target[tt], y_all, train_valid, test)
        if mres is not None:
            moof, mtest = mres
            accepted, cand_oof = evaluate_gnn_stack_column(
                tt, oof_df, moof, y_all, train_valid, t0, col_name='MLP')
            if accepted:
                oof_df = cand_oof
                extra_test_cols.append(('MLP', mtest))

    if track_fold_models and extra_test_cols:
        dense_meta = META_LEARNER_CANDIDATES['Ridge']()
        dense_meta.fit(oof_df[base_names].values, y_all.loc[sub_index].values)
        oof_meta[(tt, 'dense')] = dense_meta

    meta_name, meta_ctor, meta_mean, meta_std = select_meta_learner(
        oof_df, y_all, sub_index, repeats)
    cv_scores[tt] = (meta_mean, meta_std)
    print(f"  {tt:5s}: meta-learner selected: {meta_name} (R2={meta_mean:.4f}) "
          f"[{time.time()-t0:.0f}s]")

    meta_final = meta_ctor()
    meta_final.fit(oof_df.values, y_all.loc[sub_index].values)
    oof_meta[tt] = meta_final

    feature_cols_tt = feature_cols_by_target[tt]
    test_feat_df = test_feat_df_by_target[tt]
    X_tr_tt = X_tt.loc[sub_index]
    y_tr_tt = y_all.loc[sub_index]

    mask = (test['target_type'] == tt).values
    rows_valid = mask & test_valid_mask.values
    rows_invalid = mask & (~test_valid_mask.values)

    if rows_valid.sum() > 0:
        X_te_tt = test_feat_df.loc[rows_valid, feature_cols_tt]

        def _with_extra(base_matrix):
            if not extra_test_cols:
                return base_matrix
            cols = [s.reindex(X_te_tt.index).to_numpy() for _, s in extra_test_cols]
            return np.column_stack([base_matrix] + cols)

        base_only_preds = bagged_refit_predict(
            tt, X_tr_tt, y_tr_tt, X_te_tt, accepted_tuned_configs, BAG_SEEDS)
        base_test_preds = _with_extra(base_only_preds)
        first_pass_preds = oof_meta[tt].predict(base_test_preds)

        # ---- Iterative pseudo-labeling (2 rounds) ----
        # Round 1 mirrors the original single-round logic exactly. Round 2
        # re-estimates confidence off the round-1-augmented ensemble (a
        # slightly larger, cleaner training set), tightening the confidence
        # bar by 15% since the base models have already absorbed round 1's
        # pseudo-labels. In FAST_MODE we skip straight to the single-round
        # path to keep iteration cheap; iterative PL only runs on the final
        # (FAST_MODE=False) submission pass.
        pred_std = base_only_preds.std(axis=1)
        expected_resid = y_tr_tt.std() * np.sqrt(max(1 - cv_scores[tt][0], 0.01))
        pl_frac = PL_CONF_FRACTION
        if tt in PL_CONF_SEARCH_TARGETS and not FAST_MODE:
            pl_frac = evaluate_pl_conf_fraction(
                tt, X_tt, y_all, train_valid, X_te_tt,
                accepted_tuned_configs, cv_scores[tt][0], t0)

        clip_lo, clip_hi = clip_bounds_for(tt, y_all, train_valid)

        def _pl_round(X_tr, y_tr, X_te, base_preds, frac):
            std = base_preds.std(axis=1)
            mask_r = std < frac * expected_resid
            if mask_r.sum() < PL_MIN_ROWS:
                return None, base_preds, mask_r
            pseudo_y = np.clip(base_preds.mean(axis=1)[mask_r], clip_lo, clip_hi)
            X_aug = pd.concat([X_tr, X_te.loc[mask_r]], ignore_index=True)
            y_aug = pd.concat([y_tr, pd.Series(pseudo_y, index=X_te.loc[mask_r].index)],
                              ignore_index=True)
            w_aug = np.concatenate([
                np.ones(len(y_tr)),
                np.full(int(mask_r.sum()), PL_SAMPLE_WEIGHT)
            ])
            new_base = _with_extra(bagged_refit_predict(
                tt, X_aug, y_aug, X_te, accepted_tuned_configs, PL_BAG_SEEDS, w_aug))
            return new_base, new_base, mask_r

        if FAST_MODE:
            conf_mask = pred_std < pl_frac * expected_resid
            if conf_mask.sum() >= PL_MIN_ROWS:
                pseudo_y = np.clip(first_pass_preds[conf_mask], clip_lo, clip_hi)
                X_pl = X_te_tt.loc[conf_mask]
                X_aug = pd.concat([X_tr_tt, X_pl], ignore_index=True)
                y_aug = pd.concat([y_tr_tt, pd.Series(pseudo_y)], ignore_index=True)
                w_aug = np.concatenate([
                    np.ones(len(y_tr_tt)), np.full(int(conf_mask.sum()), PL_SAMPLE_WEIGHT),
                ])
                base_test_preds_pl = _with_extra(bagged_refit_predict(
                    tt, X_aug, y_aug, X_te_tt, accepted_tuned_configs, PL_BAG_SEEDS, w_aug))
                print(f"  {tt:5s}: {int(conf_mask.sum())}/{len(conf_mask)} test rows "
                      f"pseudo-labeled, base models retrained [{time.time()-t0:.0f}s]")
                test_predictions[rows_valid] = oof_meta[tt].predict(base_test_preds_pl)
            else:
                test_predictions[rows_valid] = first_pass_preds
        else:
            r1_preds, _, r1_mask = _pl_round(X_tr_tt, y_tr_tt, X_te_tt, base_only_preds, pl_frac)
            if r1_preds is not None:
                r2_preds, _, r2_mask = _pl_round(
                    pd.concat([X_tr_tt, X_te_tt.loc[r1_mask]], ignore_index=True),
                    pd.concat([y_tr_tt,
                               pd.Series(np.clip(r1_preds.mean(axis=1)[r1_mask], clip_lo, clip_hi))],
                              ignore_index=True),
                    X_te_tt, r1_preds, pl_frac * 0.85,
                )
                final_preds = r2_preds if r2_preds is not None else r1_preds
                test_predictions[rows_valid] = oof_meta[tt].predict(final_preds)
                print(f"  {tt:5s}: {int(r1_mask.sum())} PL round-1, "
                      f"{int(r2_mask.sum()) if r2_preds is not None else 0} round-2 "
                      f"[{time.time()-t0:.0f}s]")
            else:
                test_predictions[rows_valid] = first_pass_preds

    if rows_invalid.sum() > 0:
        test_predictions[rows_invalid] = y_tr_tt.mean()
    print(f"  {tt:5s} done [{time.time()-t0:.0f}s]")

    return X_tr_tt, y_tr_tt, fold_models_by_name


def build_dense_predictions(tt_source, X_tr_source, y_tr_source, fold_models_by_name,
                             oof_meta, X_by_target, test_feat_df_by_target,
                             feature_cols_by_target, accepted_tuned_configs,
                             train_valid, test, other_target_types, t0):
    meta_final = oof_meta.get((tt_source, 'dense'), oof_meta[tt_source])
    X_source_full = X_by_target[tt_source]
    names = model_names_for(tt_source)

    other_train_index = train_valid.index[train_valid['target_type'].isin(other_target_types)]
    X_ext_train = X_source_full.loc[other_train_index]
    base_ext_train = pd.DataFrame(
        {name: apply_fold_ensemble(fold_models_by_name[name], X_ext_train) for name in names},
        index=other_train_index,
    )[names]
    train_series = pd.Series(
        meta_final.predict(base_ext_train.values), index=other_train_index,
        name=f'pred_{tt_source}')

    other_test_index = test.index[test['target_type'].isin(other_target_types)]
    feature_cols_source = feature_cols_by_target[tt_source]
    X_te_ext = test_feat_df_by_target[tt_source].loc[other_test_index, feature_cols_source]
    base_ext_test = bagged_refit_predict(
        tt_source, X_tr_source, y_tr_source, X_te_ext, accepted_tuned_configs, BAG_SEEDS)
    test_series = pd.Series(
        meta_final.predict(base_ext_test), index=other_test_index,
        name=f'pred_{tt_source}')

    print(f"  pred_{tt_source}: dense feature built for {len(other_train_index)} train rows, "
          f"{len(other_test_index)} test rows [{time.time()-t0:.0f}s]")
    return train_series, test_series


def apply_dense_sources(source_fits, consumers, X_by_target, feature_cols_by_target,
                        test_feat_df_by_target, oof_meta, y_all, train_valid, test,
                        accepted_tuned_configs, t0, label):
    dense_train_cols, dense_test_cols = {}, {}
    for tt_source, (X_tr_s, y_tr_s, fold_models) in source_fits.items():
        tr_series, te_series = build_dense_predictions(
            tt_source, X_tr_s, y_tr_s, fold_models, oof_meta,
            X_by_target, test_feat_df_by_target, feature_cols_by_target,
            accepted_tuned_configs, train_valid, test, set(consumers), t0)
        dense_train_cols[tr_series.name] = tr_series
        dense_test_cols[te_series.name] = te_series

    verdicts = {}
    for tt in consumers:
        own_tr = train_valid.index[train_valid['target_type'] == tt]
        own_te = test.index[test['target_type'] == tt]
        new_tr = pd.DataFrame({n: s.loc[own_tr] for n, s in dense_train_cols.items()})
        cand_raw = pd.concat([X_by_target[tt], new_tr], axis=1)
        cand_cols = fit_feature_pruner(cand_raw)
        cand_X = cand_raw[cand_cols]
        accepted, _, _, _, _ = paired_delta_verdict(
            tt, X_by_target[tt], cand_X, y_all, train_valid, t0, label=label)
        verdicts[tt] = accepted
        if accepted:
            X_by_target[tt] = cand_X
            feature_cols_by_target[tt] = cand_cols
            new_te = pd.DataFrame({n: s.loc[own_te] for n, s in dense_test_cols.items()})
            test_feat_df_by_target[tt] = pd.concat(
                [test_feat_df_by_target[tt], new_te.reindex(test.index)], axis=1)
    return verdicts, dense_train_cols, dense_test_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"FAST_MODE = {FAST_MODE}  (flip to False in the constants block for the final "
          f"submission run)")
    print(f"USE_GPU = {USE_GPU}  LIGHTGBM_GPU = {LIGHTGBM_GPU}  "
          f"(XGB/CatBoost GPU flags apply whenever USE_GPU=True; LightGBM only "
          f"switches to GPU if you've installed a GPU-enabled build and set "
          f"LIGHTGBM_GPU=True)")

    print(f"Reading data from: {INPUT_DIR}")
    train = load_train_with_groups()
    print(f"Loaded train={train.shape}")

    print("Featurizing train (parallel + disk-cached: baseline + MACCS + Gasteiger + "
          "Fragments + attachment-point + conjugation-extent + Atom Pair/Torsion FP + "
          "backbone-conjugation)...")
    raw_df, train_valid = compute_raw_features(train)
    y_all = train_valid['target']
    target_types = sorted(train_valid['target_type'].unique())
    _TRAIN_VALID_REF[0] = train_valid
    print(f"  {raw_df.shape[1]} raw features before pruning [{time.time()-t0:.0f}s]")

    cross_target_lookup = build_cross_target_lookup(train_valid)
    xprop_train = cross_target_feats(train_valid, cross_target_lookup)

    feature_cols_base = fit_feature_pruner(raw_df)
    X_base = raw_df[feature_cols_base]

    xt_block = cross_target_feats(train_valid, cross_target_lookup)
    raw_df_xt = pd.concat([raw_df, xt_block], axis=1)
    feature_cols_xt = fit_feature_pruner(raw_df_xt)
    X_xt = raw_df_xt[feature_cols_xt]

    use_cross_target = evaluate_cross_target_features(
        X_base, X_xt, y_all, train_valid, target_types, t0)

    X_by_target = {tt: (X_xt if use_cross_target[tt] else X_base) for tt in target_types}
    feature_cols_by_target = {
        tt: (feature_cols_xt if use_cross_target[tt] else feature_cols_base)
        for tt in target_types
    }
    shipped = [tt for tt in target_types if use_cross_target[tt]]
    print(f"\n  Cross-target features shipped for: {shipped if shipped else '(none)'} "
          f"[{time.time()-t0:.0f}s]")
    for tt in target_types:
        print(f"    {tt:5s}: {len(feature_cols_by_target[tt])} features "
              f"({'WITH' if use_cross_target[tt] else 'WITHOUT'} cross-target block)")

    print("\n" + "=" * 100)
    print(f"3D-conformer descriptor block -- D3D_TARGETS only ({sorted(D3D_TARGETS)})")
    print("=" * 100)

    d3d_accepted = {tt: False for tt in target_types}
    if D3D_TARGETS:
        d3d_row_index = train_valid.index[train_valid['target_type'].isin(D3D_TARGETS)]
        d3d_train_df = pd.DataFrame(
            [conformer_3d_feats(s) for s in train_valid.loc[d3d_row_index, 'smiles']],
            index=d3d_row_index)
        n_embedded = int(d3d_train_df.notna().all(axis=1).sum())
        print(f"  D3D features computed for {len(d3d_row_index)} train rows, "
              f"{n_embedded} fully embedded (rest fell back to NaN) [{time.time()-t0:.0f}s]")

        for tt in sorted(D3D_TARGETS):
            candidate_raw = pd.concat([X_by_target[tt], d3d_train_df], axis=1)
            candidate_cols = fit_feature_pruner(candidate_raw)
            candidate_X = candidate_raw[candidate_cols]
            accepted, _, _, _, _ = paired_delta_verdict(
                tt, X_by_target[tt], candidate_X, y_all, train_valid, t0, label='d3d')
            d3d_accepted[tt] = accepted
            if accepted:
                X_by_target[tt] = candidate_X
                feature_cols_by_target[tt] = candidate_cols

        shipped_d3d = [tt for tt in sorted(D3D_TARGETS) if d3d_accepted[tt]]
        print(f"\n  3D-conformer descriptor block SHIPS for: "
              f"{shipped_d3d if shipped_d3d else '(none)'}")
        for tt in sorted(D3D_TARGETS):
            print(f"    {tt:5s}: {len(feature_cols_by_target[tt])} features "
                  f"({'WITH' if d3d_accepted[tt] else 'WITHOUT'} 3D-conformer block)")
    else:
        print("  FAST_MODE: 3D-conformer block skipped entirely (D3D_TARGETS empty)")

    accepted_tuned_configs = tune_all_boosting_models(
        X_by_target, y_all, train_valid, target_types, t0)
    accepted_tuned_configs.update(
        tune_all_extended_models(X_by_target, y_all, train_valid, target_types, t0))
    print(f"\n  {len(accepted_tuned_configs)} accepted tuned config(s): "
          f"{list(accepted_tuned_configs.keys())}")

    print("\nFeaturizing test...")
    test = pd.read_csv(TEST_PATH)
    test_feats = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    if (~test_valid_mask).sum():
        print(f"  {(~test_valid_mask).sum()} test SMILES failed to featurize -- "
              f"falling back to that target's train mean for those rows")
    test_mols = test.loc[test_valid_mask, 'smiles'].apply(_parse_mol)
    test_maccs = pd.Series([maccs_keys(m) for m in test_mols], index=test_mols.index)
    test_gast = pd.Series([gasteiger_features(m) for m in test_mols], index=test_mols.index)
    test_frag = pd.Series([fragment_counts(m) for m in test_mols], index=test_mols.index)
    test_attach = pd.Series([attachment_point_feats(m) for m in test_mols], index=test_mols.index)
    test_conj = pd.Series([conjugation_extent_feats(m) for m in test_mols], index=test_mols.index)
    test_ap = pd.Series([_atom_pair_bits(m) for m in test_mols], index=test_mols.index)
    test_tt_fp = pd.Series([_torsion_bits(m) for m in test_mols], index=test_mols.index)
    test_backbone = pd.Series([backbone_conjugation_feats(m) for m in test_mols], index=test_mols.index)

    valid_test_idx = test.index[test_valid_mask]
    test_records = [
        {**test_feats[idx], **test_maccs[idx], **test_gast[idx], **test_frag[idx],
         **test_attach[idx], **test_conj[idx], **test_ap[idx], **test_tt_fp[idx],
         **test_backbone[idx]}
        for idx in valid_test_idx
    ]
    test_feat_df_base = pd.DataFrame(test_records, index=valid_test_idx).reindex(
        index=test.index, columns=feature_cols_base)

    test['canon'] = test['smiles'].apply(canonical_smiles)
    test_xt_df = cross_target_feats(test, cross_target_lookup)
    xprop_test = test_xt_df
    test_feat_df_xt = pd.DataFrame(test_records, index=valid_test_idx).reindex(
        index=test.index, columns=feature_cols_xt)
    for col in feature_cols_xt:
        if col in test_xt_df.columns:
            test_feat_df_xt[col] = test_xt_df[col]

    test_feat_df_by_target = {
        tt: (test_feat_df_xt if use_cross_target[tt] else test_feat_df_base)
        for tt in target_types
    }

    for tt in sorted(D3D_TARGETS):
        if not d3d_accepted[tt]:
            continue
        own_test_index = test.index[(test['target_type'] == tt) & test_valid_mask]
        d3d_test_df = pd.DataFrame(
            [conformer_3d_feats(s) for s in test.loc[own_test_index, 'smiles']],
            index=own_test_index)
        test_feat_df_by_target[tt] = pd.concat(
            [test_feat_df_by_target[tt], d3d_test_df.reindex(test.index)], axis=1)
    print(f"  3D-conformer test-side features built for shipped targets "
          f"[{time.time()-t0:.0f}s]")

    oof_meta = {}
    cv_scores = {}
    test_predictions = np.full(len(test), np.nan)

    large_targets_ordered = [tt for tt in target_types if tt in LARGE_TARGETS]
    small_targets_ordered = [tt for tt in target_types if tt not in LARGE_TARGETS]

    print("\n" + "=" * 100)
    print("OOF stacking + test prediction -- LARGE_TARGETS first")
    print("=" * 100)

    large_target_fits = {}
    for tt in large_targets_ordered:
        large_target_fits[tt] = process_target(
            tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
            test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
            oof_meta, cv_scores, test_predictions, t0,
            track_fold_models=True, xprop=(xprop_train, xprop_test))

    print("\n" + "=" * 100)
    print("Dense pred_egc/pred_tg cross-target features (model predictions, not lookups)")
    print("=" * 100)

    small_target_set = set(small_targets_ordered)
    dense_train_cols, dense_test_cols = {}, {}
    for tt_source in large_targets_ordered:
        X_tr_source, y_tr_source, fold_models_by_name = large_target_fits[tt_source]
        train_series, test_series = build_dense_predictions(
            tt_source, X_tr_source, y_tr_source, fold_models_by_name, oof_meta,
            X_by_target, test_feat_df_by_target, feature_cols_by_target,
            accepted_tuned_configs, train_valid, test, small_target_set, t0)
        dense_train_cols[train_series.name] = train_series
        dense_test_cols[test_series.name] = test_series

    pred_feat_verdicts = {}
    for tt in small_targets_ordered:
        own_train_index = train_valid.index[train_valid['target_type'] == tt]
        own_test_index = test.index[test['target_type'] == tt]

        new_cols_train = pd.DataFrame(
            {name: series.loc[own_train_index] for name, series in dense_train_cols.items()})
        candidate_raw = pd.concat([X_by_target[tt], new_cols_train], axis=1)
        candidate_cols = fit_feature_pruner(candidate_raw)
        candidate_X = candidate_raw[candidate_cols]

        accepted, _, _, _, _ = paired_delta_verdict(
            tt, X_by_target[tt], candidate_X, y_all, train_valid, t0, label='pred_egc/tg')
        pred_feat_verdicts[tt] = accepted

        if accepted:
            X_by_target[tt] = candidate_X
            feature_cols_by_target[tt] = candidate_cols
            new_cols_test = pd.DataFrame(
                {name: series.loc[own_test_index] for name, series in dense_test_cols.items()})
            test_feat_df_by_target[tt] = pd.concat(
                [test_feat_df_by_target[tt], new_cols_test.reindex(test.index)], axis=1)

    shipped_pred = [tt for tt in small_targets_ordered if pred_feat_verdicts[tt]]
    print(f"\n  Dense pred_egc/pred_tg features SHIP for: "
          f"{shipped_pred if shipped_pred else '(none)'}")
    for tt in small_targets_ordered:
        print(f"    {tt:5s}: {len(feature_cols_by_target[tt])} features "
              f"({'WITH' if pred_feat_verdicts[tt] else 'WITHOUT'} dense pred_egc/pred_tg)")

    print("\n" + "=" * 100)
    print("Dense pred_egb/pred_eea cross-target features for nc/ei/eps")
    print("=" * 100)

    dense_source_names = [t for t in ('egb', 'eea') if t in small_target_set]
    egb_eea_fits = {}
    for tt in dense_source_names:
        egb_eea_fits[tt] = process_target(
            tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
            test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
            oof_meta, cv_scores, test_predictions, t0,
            track_fold_models=True, xprop=(xprop_train, xprop_test))

    egb_eea_consumers = [t for t in ('nc', 'ei', 'eps') if t in small_target_set]
    egb_eea_verdicts, egb_eea_train_cols, egb_eea_test_cols = apply_dense_sources(
        egb_eea_fits, egb_eea_consumers, X_by_target, feature_cols_by_target,
        test_feat_df_by_target, oof_meta, y_all, train_valid, test,
        accepted_tuned_configs, t0, label='pred_egb/eea')
    shipped_ee = [tt for tt in egb_eea_consumers if egb_eea_verdicts[tt]]
    print(f"\n  Dense pred_egb/pred_eea features SHIP for: {shipped_ee if shipped_ee else '(none)'}")

    dense_preds = {}
    for src in ('egb', 'eea'):
        key = f'pred_{src}'
        if key in egb_eea_train_cols:
            dense_preds[src] = (egb_eea_train_cols[key], egb_eea_test_cols[key])

    print("\n" + "=" * 100)
    print("OOF stacking + test prediction -- nc first (sources pred_nc for eps)")
    print("=" * 100)

    nc_X_tr, nc_y_tr, nc_fold_models_by_name = process_target(
        'nc', X_by_target, feature_cols_by_target, y_all, train_valid, test,
        test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
        oof_meta, cv_scores, test_predictions, t0,
        track_fold_models=True, xprop=(xprop_train, xprop_test))

    print("\n" + "=" * 100)
    print("Dense pred_nc cross-target feature for eps (model predictions, not lookups)")
    print("=" * 100)

    pred_nc_train, pred_nc_test = build_dense_predictions(
        'nc', nc_X_tr, nc_y_tr, nc_fold_models_by_name, oof_meta,
        X_by_target, test_feat_df_by_target, feature_cols_by_target,
        accepted_tuned_configs, train_valid, test, {'eps'}, t0)
    dense_preds['nc'] = (pred_nc_train, pred_nc_test)

    candidate_raw = pd.concat([X_by_target['eps'], pred_nc_train.to_frame()], axis=1)
    candidate_cols = fit_feature_pruner(candidate_raw)
    candidate_X = candidate_raw[candidate_cols]

    pred_nc_accepted, _, _, _, _ = paired_delta_verdict(
        'eps', X_by_target['eps'], candidate_X, y_all, train_valid, t0, label='pred_nc')
    print(f"  Dense pred_nc feature for eps: {'SHIP' if pred_nc_accepted else 'reject'}")
    if pred_nc_accepted:
        X_by_target['eps'] = candidate_X
        feature_cols_by_target['eps'] = candidate_cols
        test_feat_df_by_target['eps'] = pd.concat(
            [test_feat_df_by_target['eps'], pred_nc_test.reindex(test.index).to_frame()], axis=1)
    print(f"  eps  : {len(feature_cols_by_target['eps'])} features "
          f"({'WITH' if pred_nc_accepted else 'WITHOUT'} dense pred_nc)")

    print("\n" + "=" * 100)
    print("OOF stacking + test prediction -- remaining small targets")
    print("=" * 100)

    already_processed = {'nc', *dense_source_names}
    for tt in [t for t in small_targets_ordered if t not in already_processed]:
        process_target(
            tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
            test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
            oof_meta, cv_scores, test_predictions, t0,
            track_fold_models=False, xprop=(xprop_train, xprop_test),
            dense_preds=dense_preds)

    print(f"\n{'target':6s}{'stacked CV R2':>18s}")
    for tt in target_types:
        mean, std = cv_scores[tt]
        print(f"{tt:6s}{mean:14.4f} (+/-{std:.3f})")
    mean_r2 = float(np.mean([m for m, _ in cv_scores.values()]))
    print(f"\n>>> Mean CV R2 across all {len(target_types)} targets: {mean_r2:.4f} <<<")
    print("(this is the number that estimates the competition metric -- mean R2 across targets)")

    print("\n" + "=" * 100)
    print("Clipping + writing submission.csv")
    print("=" * 100)

    stacked_out = test[['id', 'target_type']].copy()
    stacked_out['target'] = test_predictions

    clipped_target = stacked_out['target'].copy()
    for tt in target_types:
        clip_lo, clip_hi = clip_bounds_for(tt, y_all, train_valid)
        mask = stacked_out['target_type'] == tt
        vals = stacked_out.loc[mask, 'target']
        n_clipped = ((vals < clip_lo) | (vals > clip_hi)).sum()
        clipped_target.loc[mask] = vals.clip(lower=clip_lo, upper=clip_hi)
        print(f"  {tt:5s}: clip=[{clip_lo:.4g}, {clip_hi:.4g}]  "
              f"{n_clipped}/{mask.sum()} rows clipped")

    submission = stacked_out[['id']].copy()
    submission['target'] = clipped_target

    if SAMPLE_SUB_PATH.exists():
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        assert list(submission.columns) == list(sample_sub.columns), \
            f"column mismatch: {submission.columns.tolist()} vs {sample_sub.columns.tolist()}"
    assert len(submission) == len(test), \
        f"row count mismatch: {len(submission)} vs test.csv's {len(test)}"
    assert submission['target'].notna().all(), "unfilled predictions remain"

    submission.to_csv(OUT_PATH, index=False)
    print(f"\nFormat check OK -- saved {OUT_PATH} with shape {submission.shape}")
    print(f">>> Mean CV R2 across all {len(target_types)} targets: {mean_r2:.4f} <<<")
    _elapsed = int(time.time() - t0)
    print(f"Total elapsed: {_elapsed // 3600:d}:{_elapsed % 3600 // 60:02d}:{_elapsed % 60:02d} "
          f"(hh:mm:ss)")


if __name__ == "__main__":
    main()
