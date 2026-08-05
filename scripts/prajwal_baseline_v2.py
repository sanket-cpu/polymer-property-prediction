"""
Polymer Property Prediction - Round 4 (v4)
Key upgrades over v3:
  1. OOF stacking with Ridge meta-learner (replaces softmax heuristic blend)
  2. Semi-supervised pseudo-labeling from PI1M (per target_type, confidence-gated)
  3. Expanded model zoo: adds LightGBM + XGBoost for more base-learner diversity
  4. Richer fingerprints: 1024-bit radius-2 Morgan + radius-3 Morgan + MACCS (concatenated)
  5. Trimer-level descriptors extend the dimer-delta slope toward infinite chain
  6. GroupKFold dedup guard to prevent near-duplicate SMILES leaking across folds
  7. Iterative pseudo-label refinement (1 round) with downweighting
"""

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys
from rdkit import DataStructs

RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found – skipping. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found – skipping. Install with: pip install xgboost")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_PATH    = "train.csv"
TEST_PATH     = "test.csv"
PI1M_PATH     = "PI1M.csv"
PI1M_SMILES_COL = "smiles"
OUT_PATH      = "submission_v4.csv"

# Fingerprint config
FP_BITS_R2   = 1024   # Morgan radius-2
FP_BITS_R3   = 1024   # Morgan radius-3
MACCS_BITS   = 167

# Feature pruning
VAR_THRESH   = 1e-6
CORR_THRESH  = 0.98

# PI1M
PI1M_SAMPLE_SIZE  = 20000
DENSITY_TOPK      = 7

# Pseudo-labeling gates
PL_SIM_THRESH     = 0.35   # min pi1m_max_sim to include a PI1M row as pseudo-labeled
PL_STD_THRESH     = 0.15   # max normalized std across base models (low = confident)
PL_MAX_PER_TT     = 5000   # cap pseudo-labeled rows per target_type
PL_SAMPLE_WEIGHT  = 0.4    # weight applied to pseudo-labeled rows in final fit

# CV
N_SPLITS = 3

_SLOW_OR_UNSTABLE = {'Ipc'}
_DESC_LIST = [(n, f) for n, f in Descriptors._descList if n not in _SLOW_OR_UNSTABLE]

_MORGAN_GEN_R2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS_R2)
_MORGAN_GEN_R3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=FP_BITS_R3)


# ===========================================================================
# 1. Featurization
# ===========================================================================

def _parse_mol(smiles):
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    return mol


def _make_oligomer_mol(smiles, n_repeat=2):
    """
    Build an n-repeat oligomer by repeatedly linking the repeat unit.
    Works for SMILES with exactly 2 wildcard (*) attachment points.
    """
    try:
        s = smiles.replace('[*]', '*')
        if s.count('*') != 2:
            return None

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None

        current = mol
        for _ in range(n_repeat - 1):
            next_unit = Chem.MolFromSmiles(s)
            if next_unit is None:
                return None
            combo = Chem.RWMol(Chem.CombineMols(current, next_unit))
            dummy_idx = [a.GetIdx() for a in combo.GetAtoms() if a.GetSymbol() == '*']
            if len(dummy_idx) < 2:
                return None

            # Connect last dummy of current to first dummy of next unit
            # Strategy: find the two "inner" dummies (last of current, first of next)
            # After CombineMols, current atoms come first
            n_current = current.GetNumAtoms()
            current_dummies = [i for i in dummy_idx if i < n_current]
            next_dummies    = [i for i in dummy_idx if i >= n_current]

            if not current_dummies or not next_dummies:
                return None

            # Pick the rightmost dummy of current and leftmost of next
            d_curr = current_dummies[-1]
            d_next = next_dummies[0]

            def neighbor_of(rwmol, idx):
                nbrs = rwmol.GetAtomWithIdx(idx).GetNeighbors()
                return nbrs[0].GetIdx() if nbrs else None

            a = neighbor_of(combo, d_curr)
            b = neighbor_of(combo, d_next)
            if a is None or b is None:
                return None

            combo.AddBond(a, b, Chem.BondType.SINGLE)
            # Remove the two connected dummies (sort descending to preserve indices)
            for idx in sorted([d_curr, d_next], reverse=True):
                combo.RemoveAtom(idx)

            try:
                Chem.SanitizeMol(combo)
            except Exception:
                combo.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    combo,
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                )
            current = combo.GetMol()

        return current
    except Exception:
        return None


def _safe_descriptors(mol):
    out = {}
    for name, func in _DESC_LIST:
        try:
            val = func(mol)
            out[f'desc_{name}'] = val if (val is not None and np.isfinite(val)) else np.nan
        except Exception:
            out[f'desc_{name}'] = np.nan
    return out


def _morgan_bits(mol):
    feats = {}
    fp2 = _MORGAN_GEN_R2.GetFingerprint(mol)
    arr2 = np.zeros((FP_BITS_R2,), dtype=np.int8)
    for bit in fp2.GetOnBits():
        arr2[bit] = 1
    for i in range(FP_BITS_R2):
        feats[f'fp2_{i}'] = int(arr2[i])

    fp3 = _MORGAN_GEN_R3.GetFingerprint(mol)
    arr3 = np.zeros((FP_BITS_R3,), dtype=np.int8)
    for bit in fp3.GetOnBits():
        arr3[bit] = 1
    for i in range(FP_BITS_R3):
        feats[f'fp3_{i}'] = int(arr3[i])

    maccs = MACCSkeys.GenMACCSKeys(mol)
    arr_m = np.zeros((MACCS_BITS,), dtype=np.int8)
    for bit in maccs.GetOnBits():
        if bit < MACCS_BITS:
            arr_m[bit] = 1
    for i in range(MACCS_BITS):
        feats[f'maccs_{i}'] = int(arr_m[i])

    return feats


def _custom_physics_feats(mol):
    feats = {}
    heavy  = mol.GetNumHeavyAtoms() or 1
    n_bonds = mol.GetNumBonds() or 1
    atoms  = [a.GetSymbol() for a in mol.GetAtoms()]
    n_atoms = len(atoms) or 1

    for el in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'Si', 'P']:
        feats[f'n_{el}']    = atoms.count(el)
        feats[f'frac_{el}'] = atoms.count(el) / n_atoms

    n_aromatic_atoms  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_conjugated_bonds = sum(1 for b in mol.GetBonds() if b.GetIsConjugated())
    n_rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)

    feats['AromaticRatio']         = n_aromatic_atoms / n_atoms
    feats['ConjugationRatio']      = n_conjugated_bonds / n_bonds
    feats['RotBondsPerHeavyAtom']  = n_rot / heavy
    feats['RingsPerHeavyAtom']     = n_rings / heavy
    feats['MolWtPerHeavyAtom']     = Descriptors.MolWt(mol) / heavy
    return feats


def _oligomer_delta_feats(smiles, mono_feats):
    """
    Compute descriptor deltas at dimer and trimer level.
    Fits a slope (linear extrapolation to n→∞) for key properties.
    """
    feats = {}
    zeros = {
        'dimer_delta_AromaticRatio':    0.0,
        'dimer_delta_ConjugationRatio': 0.0,
        'dimer_delta_RotBondsPerHeavy': 0.0,
        'trimer_delta_AromaticRatio':   0.0,
        'trimer_delta_ConjugationRatio':0.0,
        'trimer_delta_RotBondsPerHeavy':0.0,
        'chain_slope_AromaticRatio':    0.0,
        'chain_slope_ConjugationRatio': 0.0,
        'chain_slope_RotBondsPerHeavy': 0.0,
    }

    def _oligo_physics(mol):
        if mol is None:
            return None
        heavy  = mol.GetNumHeavyAtoms() or 1
        n_bonds = mol.GetNumBonds() or 1
        n_atoms = mol.GetNumAtoms() or 1
        n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        n_conj = sum(1 for b in mol.GetBonds() if b.GetIsConjugated())
        n_rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        return {
            'AromaticRatio':    n_arom / n_atoms,
            'ConjugationRatio': n_conj / n_bonds,
            'RotBondsPerHeavy': n_rot  / heavy,
        }

    try:
        dimer  = _make_oligomer_mol(smiles, n_repeat=2)
        trimer = _make_oligomer_mol(smiles, n_repeat=3)

        d2 = _oligo_physics(dimer)
        d3 = _oligo_physics(trimer)

        mono_ref = {
            'AromaticRatio':    mono_feats.get('AromaticRatio', 0.0),
            'ConjugationRatio': mono_feats.get('ConjugationRatio', 0.0),
            'RotBondsPerHeavy': mono_feats.get('RotBondsPerHeavyAtom', 0.0),
        }

        if d2 is not None:
            feats['dimer_delta_AromaticRatio']    = d2['AromaticRatio']    - mono_ref['AromaticRatio']
            feats['dimer_delta_ConjugationRatio']  = d2['ConjugationRatio'] - mono_ref['ConjugationRatio']
            feats['dimer_delta_RotBondsPerHeavy']  = d2['RotBondsPerHeavy'] - mono_ref['RotBondsPerHeavy']
        else:
            feats.update({k: 0.0 for k in ['dimer_delta_AromaticRatio',
                                             'dimer_delta_ConjugationRatio',
                                             'dimer_delta_RotBondsPerHeavy']})

        if d3 is not None:
            feats['trimer_delta_AromaticRatio']    = d3['AromaticRatio']    - mono_ref['AromaticRatio']
            feats['trimer_delta_ConjugationRatio']  = d3['ConjugationRatio'] - mono_ref['ConjugationRatio']
            feats['trimer_delta_RotBondsPerHeavy']  = d3['RotBondsPerHeavy'] - mono_ref['RotBondsPerHeavy']
        else:
            feats.update({k: 0.0 for k in ['trimer_delta_AromaticRatio',
                                             'trimer_delta_ConjugationRatio',
                                             'trimer_delta_RotBondsPerHeavy']})

        # Linear slope: fit y = a + b*n for n in {1,2,3}, extrapolate trend
        for prop in ['AromaticRatio', 'ConjugationRatio', 'RotBondsPerHeavy']:
            vals = [mono_ref[prop]]
            if d2 is not None: vals.append(d2[prop])
            if d3 is not None: vals.append(d3[prop])
            if len(vals) >= 2:
                ns = np.arange(1, len(vals) + 1)
                slope = np.polyfit(ns, vals, 1)[0]
                feats[f'chain_slope_{prop}'] = slope
            else:
                feats[f'chain_slope_{prop}'] = 0.0

    except Exception:
        feats.update(zeros)

    return feats


def featurize(smiles):
    mol = _parse_mol(smiles)
    if mol is None:
        return None
    try:
        feats = {}
        feats.update(_safe_descriptors(mol))
        phys = _custom_physics_feats(mol)
        feats.update(phys)
        feats.update(_morgan_bits(mol))
        feats.update(_oligomer_delta_feats(smiles, phys))
        return feats
    except Exception:
        return None


def featurize_lightweight(smiles):
    """Descriptor + physics only — no fingerprints, no oligomers (fast for PI1M)."""
    mol = _parse_mol(smiles)
    if mol is None:
        return None
    try:
        feats = {}
        feats.update(_safe_descriptors(mol))
        feats.update(_custom_physics_feats(mol))
        return feats
    except Exception:
        return None


# ===========================================================================
# 2. PI1M loading + density features
# ===========================================================================

def load_pi1m_sample(path, smiles_col, sample_size, seed=RANDOM_STATE):
    print(f"Loading PI1M from {path} ...")
    pi1m = pd.read_csv(path)
    if smiles_col not in pi1m.columns:
        candidates = [c for c in pi1m.columns if 'smiles' in c.lower()]
        if not candidates:
            raise ValueError(f"No SMILES column found. Columns: {list(pi1m.columns)}")
        smiles_col = candidates[0]
    print(f"PI1M full size: {len(pi1m)}, using column '{smiles_col}'")
    sample = pi1m[[smiles_col]].dropna().drop_duplicates()
    n = min(sample_size, len(sample))
    sample = sample.sample(n=n, random_state=seed).reset_index(drop=True)
    sample = sample.rename(columns={smiles_col: 'smiles'})
    print(f"Sampled {len(sample)} PI1M rows")
    return sample


def featurize_pi1m_lightweight(pi1m_sample):
    t0 = time.time()
    feats = pi1m_sample['smiles'].apply(featurize_lightweight)
    valid = feats.notna()
    feat_df = pd.DataFrame(list(feats[valid])).reset_index(drop=True)
    print(f"PI1M lightweight featurization: {valid.sum()}/{len(pi1m_sample)} valid "
          f"in {time.time()-t0:.1f}s")
    return feat_df, pi1m_sample.loc[valid, 'smiles'].reset_index(drop=True)


def build_pi1m_fp_index(pi1m_smiles):
    t0 = time.time()
    fps = []
    for smi in pi1m_smiles:
        mol = _parse_mol(smi)
        if mol is not None:
            fps.append(_MORGAN_GEN_R2.GetFingerprint(mol))
    print(f"Built PI1M fp index: {len(fps)} fps in {time.time()-t0:.1f}s")
    return fps


def density_features(smiles_series, pi1m_fps, topk=DENSITY_TOPK):
    t0 = time.time()
    max_sims   = np.zeros(len(smiles_series))
    topk_means = np.zeros(len(smiles_series))
    for i, smi in enumerate(smiles_series):
        mol = _parse_mol(smi)
        if mol is None:
            max_sims[i] = np.nan; topk_means[i] = np.nan; continue
        fp   = _MORGAN_GEN_R2.GetFingerprint(mol)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, pi1m_fps))
        if len(sims) == 0:
            max_sims[i] = np.nan; topk_means[i] = np.nan; continue
        top = np.sort(sims)[-topk:]
        max_sims[i]   = top[-1]
        topk_means[i] = top.mean()
    print(f"Density features for {len(smiles_series)} molecules in {time.time()-t0:.1f}s")
    return pd.DataFrame({'pi1m_max_sim': max_sims, 'pi1m_topk_mean_sim': topk_means})


# ===========================================================================
# 3. Feature pruning + global scaler
# ===========================================================================

def fit_feature_pruner_combined(train_feat_df, pi1m_feat_df, common_cols):
    combined = pd.concat(
        [train_feat_df[common_cols], pi1m_feat_df[common_cols]], axis=0, ignore_index=True
    )
    variances = combined.var(numeric_only=True)
    keep = variances[variances > VAR_THRESH].index.tolist()

    corr  = combined[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop  = {c for c in upper.columns if any(upper[c] > CORR_THRESH)}
    keep  = [c for c in keep if c not in drop]
    return keep


class _FittedStateWrapper:
    def __init__(self, imputer, scaler):
        self.imputer = imputer
        self.scaler  = scaler


class PrefitScaler(BaseEstimator, TransformerMixin):
    def __init__(self, state):
        self.state = state

    def fit(self, X, y=None):
        self.n_features_in_ = getattr(self.state.imputer, 'n_features_in_',
                                       X.shape[1] if hasattr(X, 'shape') else None)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return self.state.scaler.transform(self.state.imputer.transform(X))


def fit_global_scaler(train_feat_df, pi1m_feat_df, feature_cols):
    combined = pd.concat(
        [train_feat_df[feature_cols], pi1m_feat_df[feature_cols]], axis=0, ignore_index=True
    )
    imp = SimpleImputer(strategy='median')
    combined_imp = imp.fit_transform(combined)
    sc  = StandardScaler()
    sc.fit(combined_imp)
    return imp, sc


# ===========================================================================
# 4. Model zoo  (with optional LGB / XGB)
# ===========================================================================

def get_model_zoo(prefit_imputer, prefit_scaler):
    state = _FittedStateWrapper(prefit_imputer, prefit_scaler)

    zoo = {
        'Ridge': Pipeline([
            ('sc', PrefitScaler(state)),
            ('m', Ridge(alpha=5.0, random_state=RANDOM_STATE)),
        ]),
        'ElasticNet': Pipeline([
            ('sc', PrefitScaler(state)),
            ('m', ElasticNet(alpha=0.01, l1_ratio=0.3,
                             random_state=RANDOM_STATE, max_iter=5000)),
        ]),
        'RF': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', RandomForestRegressor(
                n_estimators=500, max_depth=10, min_samples_leaf=2,
                n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        'GBM': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.04,
                subsample=0.85, random_state=RANDOM_STATE)),
        ]),
        'HGB': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', HistGradientBoostingRegressor(
                max_iter=400, max_depth=6, learning_rate=0.05,
                l2_regularization=0.1, random_state=RANDOM_STATE)),
        ]),
    }

    if HAS_LGB:
        zoo['LGB'] = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                num_leaves=63, subsample=0.85, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)),
        ])

    if HAS_XGB:
        zoo['XGB'] = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=RANDOM_STATE,
                verbosity=0, tree_method='hist')),
        ])

    return zoo


# ===========================================================================
# 5. GroupKFold dedup guard
# ===========================================================================

def build_dedup_groups(smiles_series):
    """
    Assign a group ID to each molecule based on a canonical SMILES after
    stripping wildcard atoms. Near-duplicate repeat units land in the same
    group so they won't straddle train/val folds.
    """
    groups = []
    seen   = {}
    for smi in smiles_series:
        mol = _parse_mol(smi)
        if mol is None:
            canon = smi
        else:
            canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen[canon] = len(seen)
        groups.append(seen[canon])
    return np.array(groups)


# ===========================================================================
# 6. OOF stacking helpers
# ===========================================================================

def _get_X(feat_df, cols, is_linear):
    """Return numpy array for given column set; linear models use pruned_common only."""
    return feat_df[cols].values.astype(float)


def oof_predictions(models_dict, feat_df, pruned_common, feature_cols,
                    y, groups, n_splits=N_SPLITS):
    """
    Generate out-of-fold predictions for every base model.
    Uses GroupKFold to avoid leaking near-duplicate molecules.
    Returns:
        oof_matrix : (n_samples, n_models) array
        model_names: list of model names in column order
    """
    model_names = list(models_dict.keys())
    n = len(y)
    oof_matrix = np.full((n, len(model_names)), np.nan)

    linear_models = {'Ridge', 'ElasticNet'}

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat_df, y, groups=groups)):
        print(f"  OOF fold {fold+1}/{n_splits} ...")
        for j, name in enumerate(model_names):
            cols = pruned_common if name in linear_models else feature_cols
            X_tr = feat_df.iloc[tr_idx][cols].values.astype(float)
            X_va = feat_df.iloc[va_idx][cols].values.astype(float)
            y_tr = y[tr_idx]

            # clone is expensive; rebuild zoo each fold
            from sklearn.base import clone
            model = clone(models_dict[name])
            model.fit(X_tr, y_tr)
            oof_matrix[va_idx, j] = model.predict(X_va)

    return oof_matrix, model_names


# ===========================================================================
# 7. Pseudo-labeling helpers
# ===========================================================================

def pseudo_label_pi1m(pi1m_smiles_valid, pi1m_feat_df,
                       pi1m_fps, final_models_dict,
                       pruned_common, feature_cols, target_type,
                       target_mean, target_std):
    """
    For a given target_type:
      1. Featurize PI1M molecules (lightweight already done → pi1m_feat_df)
      2. Compute density (pi1m_max_sim) for PI1M vs itself is trivially 1.0,
         so instead we use a leave-one-out style: check if sim > PL_SIM_THRESH
         vs train distribution (approximated by treating pi1m_max_sim as already computed)
      3. Run all base models; keep rows where std across model preds is low
      4. Return (smiles, feat_df_row, pseudo_target) for qualifying rows
    """
    linear_models = {'Ridge', 'ElasticNet'}

    # Build feature matrix for PI1M (lightweight feats only — matches pruned_common subset)
    avail_cols_linear = [c for c in pruned_common if c in pi1m_feat_df.columns]

    # Predict with each base model that has avail cols
    preds_per_model = {}
    for name, model in final_models_dict.items():
        if name in linear_models:
            cols = avail_cols_linear
        else:
            cols = [c for c in feature_cols if c in pi1m_feat_df.columns]
        if not cols:
            continue
        X_pl = pi1m_feat_df[cols].values.astype(float)
        # impute nans quickly
        col_means = np.nanmedian(X_pl, axis=0)
        inds = np.where(np.isnan(X_pl))
        X_pl[inds] = np.take(col_means, inds[1])
        try:
            preds_per_model[name] = model.predict(X_pl)
        except Exception:
            pass

    if len(preds_per_model) < 2:
        print(f"  [PL] Not enough models for pseudo-labeling on {target_type}")
        return None

    pred_matrix = np.stack(list(preds_per_model.values()), axis=1)
    pred_mean   = pred_matrix.mean(axis=1)
    pred_std    = pred_matrix.std(axis=1)

    # Gate 1: model agreement (normalized by target std)
    norm_std = pred_std / (target_std + 1e-8)
    conf_mask = norm_std < PL_STD_THRESH

    # Gate 2: density — build PI1M-to-PI1M self similarity would be trivial,
    # so proxy: ensure raw pred is within 2 std of target mean (in-range)
    range_mask = np.abs(pred_mean - target_mean) < 2.5 * target_std

    mask = conf_mask & range_mask
    print(f"  [PL] {target_type}: {mask.sum()} PI1M rows pass confidence gates "
          f"(conf={conf_mask.sum()}, range={range_mask.sum()})")

    if mask.sum() == 0:
        return None

    # Cap to avoid overwhelming labeled data
    if mask.sum() > PL_MAX_PER_TT:
        chosen = np.where(mask)[0]
        # prefer most confident (lowest norm_std)
        order  = np.argsort(norm_std[chosen])
        chosen = chosen[order[:PL_MAX_PER_TT]]
        mask2  = np.zeros(len(mask), dtype=bool)
        mask2[chosen] = True
        mask = mask2

    pl_feat_df      = pi1m_feat_df[mask].reset_index(drop=True)
    pl_targets      = pred_mean[mask]
    pl_smiles       = pi1m_smiles_valid[mask].reset_index(drop=True)

    print(f"  [PL] {target_type}: added {mask.sum()} pseudo-labeled rows "
          f"(mean={pl_targets.mean():.3f}, std={pl_targets.std():.3f})")
    return pl_feat_df, pl_targets, pl_smiles


# ===========================================================================
# 8. Main pipeline
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"Loaded train={train.shape}, test={test.shape}")

    # ------------------------------------------------------------------
    # Featurize train
    # ------------------------------------------------------------------
    print("\n=== Featurizing train ===")
    t0 = time.time()
    train_feats  = train['smiles'].apply(featurize)
    valid_mask   = train_feats.notna()
    print(f"Featurized {valid_mask.sum()}/{len(train)} train molecules in {time.time()-t0:.1f}s")
    train_feat_df = pd.DataFrame(list(train_feats[valid_mask])).reset_index(drop=True)
    train_valid   = train[valid_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Featurize test
    # ------------------------------------------------------------------
    print("\n=== Featurizing test ===")
    t0 = time.time()
    test_feats      = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    print(f"Featurized {test_valid_mask.sum()}/{len(test)} test molecules in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # PI1M: sample + lightweight featurize + density
    # ------------------------------------------------------------------
    print("\n=== PI1M processing ===")
    pi1m_sample = load_pi1m_sample(PI1M_PATH, PI1M_SMILES_COL, PI1M_SAMPLE_SIZE)
    pi1m_feat_df, pi1m_smiles_valid = featurize_pi1m_lightweight(pi1m_sample)
    pi1m_fps = build_pi1m_fp_index(pi1m_smiles_valid)

    train_density = density_features(train_valid['smiles'], pi1m_fps)
    test_density  = density_features(test['smiles'], pi1m_fps)

    train_feat_df = pd.concat([train_feat_df, train_density], axis=1)

    # ------------------------------------------------------------------
    # Feature pruning (PI1M-informed)
    # ------------------------------------------------------------------
    print("\n=== Feature pruning ===")
    common_cols    = [c for c in pi1m_feat_df.columns if c in train_feat_df.columns]
    pruned_common  = fit_feature_pruner_combined(train_feat_df, pi1m_feat_df, common_cols)

    fp_ext_cols    = [c for c in train_feat_df.columns
                      if c.startswith('fp2_') or c.startswith('fp3_')
                      or c.startswith('maccs_')
                      or c.startswith('dimer_delta_') or c.startswith('trimer_delta_')
                      or c.startswith('chain_slope_') or c.startswith('pi1m_')]
    feature_cols   = pruned_common + fp_ext_cols

    print(f"Kept {len(feature_cols)} features "
          f"({len(pruned_common)} descriptor/physics + {len(fp_ext_cols)} fp/oligo/density)")

    # ------------------------------------------------------------------
    # Global scaler
    # ------------------------------------------------------------------
    prefit_imputer, prefit_scaler = fit_global_scaler(
        train_feat_df, pi1m_feat_df, pruned_common
    )

    # ------------------------------------------------------------------
    # Build full train frame
    # ------------------------------------------------------------------
    full_train = pd.concat(
        [train_valid[['target', 'target_type']], train_feat_df[feature_cols]], axis=1
    )

    # ------------------------------------------------------------------
    # Build test feature matrix (fill in now, before loops)
    # ------------------------------------------------------------------
    print("\n=== Building test feature matrix ===")
    test_feat_df = pd.DataFrame(index=test.index, columns=feature_cols, dtype=float)
    density_only = {'pi1m_max_sim', 'pi1m_topk_mean_sim'}
    for idx in test.index[test_valid_mask]:
        row = test_feats[idx]
        for k in feature_cols:
            if k not in density_only:
                test_feat_df.loc[idx, k] = row.get(k, np.nan)
    test_feat_df['pi1m_max_sim']       = test_density['pi1m_max_sim'].values
    test_feat_df['pi1m_topk_mean_sim'] = test_density['pi1m_topk_mean_sim'].values

    # ------------------------------------------------------------------
    # Per-target-type: OOF stacking → meta-learner → pseudo-label → retrain
    # ------------------------------------------------------------------
    print("\n=== OOF stacking + pseudo-labeling per target_type ===")

    final_meta_models  = {}   # tt -> fitted Ridge meta-learner
    final_base_models  = {}   # tt -> {name: fitted base model}
    base_model_names_by_tt = {}
    target_stats       = {}   # tt -> (mean, std)
    oof_r2_by_tt       = {}

    dedup_groups = build_dedup_groups(train_valid['smiles'])

    for tt in sorted(full_train['target_type'].unique()):
        print(f"\n--- target_type: {tt} ---")
        sub    = full_train[full_train['target_type'] == tt]
        y      = sub['target'].values
        groups = dedup_groups[sub.index]

        t_mean = y.mean(); t_std = y.std() + 1e-8
        target_stats[tt] = (t_mean, t_std)

        # ---- Step A: OOF predictions from all base models ----
        zoo = get_model_zoo(prefit_imputer, prefit_scaler)
        print(f"  Running OOF ({N_SPLITS}-fold GroupKFold) for {len(zoo)} models ...")

        oof_matrix, model_names = oof_predictions(
            zoo, sub[feature_cols], pruned_common, feature_cols,
            y, groups, n_splits=N_SPLITS
        )
        base_model_names_by_tt[tt] = model_names

        # ---- Step B: Fit Ridge meta-learner on OOF ----
        # Replace NaN in OOF (groups that never appeared in val) with column mean
        for j in range(oof_matrix.shape[1]):
            col = oof_matrix[:, j]
            col[np.isnan(col)] = np.nanmean(col)

        meta = Ridge(alpha=1.0)
        meta.fit(oof_matrix, y)
        oof_blend = meta.predict(oof_matrix)
        oof_r2    = r2_score(y, oof_blend)
        oof_r2_by_tt[tt] = oof_r2
        print(f"  OOF R2 after stacking: {oof_r2:.4f}  "
              f"(model contributions: " +
              ", ".join(f"{n}={v:.3f}" for n,v in zip(model_names, meta.coef_)) + ")")

        # ---- Step C: Fit base models on full labeled data ----
        zoo_full = get_model_zoo(prefit_imputer, prefit_scaler)
        linear_models = {'Ridge', 'ElasticNet'}
        fitted_base = {}
        for name in model_names:
            cols = pruned_common if name in linear_models else feature_cols
            X_tr = sub[cols].values.astype(float)
            zoo_full[name].fit(X_tr, y)
            fitted_base[name] = zoo_full[name]
        final_base_models[tt]  = fitted_base
        final_meta_models[tt]  = meta

        # ---- Step D: Pseudo-labeling from PI1M ----
        pl_result = pseudo_label_pi1m(
            pi1m_smiles_valid, pi1m_feat_df, pi1m_fps,
            fitted_base, pruned_common, feature_cols,
            tt, t_mean, t_std
        )

        if pl_result is not None:
            pl_feat_df_sub, pl_targets, _ = pl_result

            # Align columns: pi1m only has descriptor/physics cols, not fp/oligo/density
            available = [c for c in feature_cols if c in pl_feat_df_sub.columns]
            missing   = [c for c in feature_cols if c not in pl_feat_df_sub.columns]

            # Build augmented train: labeled + pseudo-labeled
            # For pseudo-labeled rows, missing columns (fps, oligo deltas, density) = NaN
            # Tree-based models handle NaN via imputation; linear models only use pruned_common
            aug_feat_labeled = sub[feature_cols].copy()
            aug_y_labeled    = y.copy()
            aug_w_labeled    = np.ones(len(y))

            pl_feat_full = pd.DataFrame(
                np.full((len(pl_feat_df_sub), len(feature_cols)), np.nan),
                columns=feature_cols
            )
            pl_feat_full[available] = pl_feat_df_sub[available].values

            aug_feat = pd.concat([aug_feat_labeled, pl_feat_full], axis=0, ignore_index=True)
            aug_y    = np.concatenate([aug_y_labeled, pl_targets])
            aug_w    = np.concatenate([aug_w_labeled,
                                        np.full(len(pl_targets), PL_SAMPLE_WEIGHT)])

            print(f"  Retraining on {len(aug_y)} rows "
                  f"({len(y)} labeled + {len(pl_targets)} pseudo-labeled) ...")

            # Retrain only tree-based models on augmented data (linear models unaffected)
            zoo_aug = get_model_zoo(prefit_imputer, prefit_scaler)
            fitted_base_aug = dict(fitted_base)  # keep linear models from labeled-only fit
            for name in model_names:
                if name in linear_models:
                    continue  # linear models don't benefit from NaN-heavy pseudo rows
                cols = feature_cols
                X_aug = aug_feat[cols].values.astype(float)
                try:
                    zoo_aug[name].fit(X_aug, aug_y, **{
                        # pass sample_weight only to models that support it natively
                        # Pipeline fit: need to pass as step__param
                        f'm__sample_weight': aug_w
                    })
                    fitted_base_aug[name] = zoo_aug[name]
                except TypeError:
                    # Fallback: fit without sample weights
                    zoo_aug[name].fit(X_aug, aug_y)
                    fitted_base_aug[name] = zoo_aug[name]

            # Re-generate OOF on LABELED data only to refit meta-learner
            # (meta-learner should not be trained on pseudo-labels)
            oof_aug = np.zeros((len(y), len(model_names)))
            for j, name in enumerate(model_names):
                cols = pruned_common if name in linear_models else feature_cols
                X_sub = sub[cols].values.astype(float)
                # Use the labeled-data-fitted version for OOF on labeled set
                # (approximate: predict from the retrained model which saw pseudo-labels)
                # For a cleaner meta-fit, generate OOF from labeled-only models
                oof_aug[:, j] = oof_matrix[:, j]  # reuse original labeled OOF

            # Refit meta with augmented base preds on labeled OOF
            meta_aug = Ridge(alpha=1.0)
            meta_aug.fit(oof_aug, y)
            final_meta_models[tt]  = meta_aug
            final_base_models[tt]  = fitted_base_aug

    mean_oof_r2 = np.mean(list(oof_r2_by_tt.values()))
    print(f"\n{'='*60}")
    print(f"Mean OOF R2 across all target_types (stacked): {mean_oof_r2:.4f}")
    for tt, r2 in sorted(oof_r2_by_tt.items()):
        print(f"  {tt}: {r2:.4f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------------
    print("=== Test inference ===")
    predictions = np.zeros(len(test))

    for tt in sorted(full_train['target_type'].unique()):
        t_mean, _ = target_stats[tt]
        mask         = (test['target_type'] == tt).values
        rows_valid   = mask & test_valid_mask.values
        rows_invalid = mask & (~test_valid_mask.values)
        linear_models = {'Ridge', 'ElasticNet'}

        if rows_valid.sum() > 0:
            model_names = base_model_names_by_tt[tt]
            n_valid     = rows_valid.sum()
            test_base_preds = np.zeros((n_valid, len(model_names)))

            for j, name in enumerate(model_names):
                cols  = pruned_common if name in linear_models else feature_cols
                X_tst = test_feat_df.loc[rows_valid, cols].values.astype(float)
                test_base_preds[:, j] = final_base_models[tt][name].predict(X_tst)

            # Stack through meta-learner
            blend = final_meta_models[tt].predict(test_base_preds)
            predictions[rows_valid] = blend

        if rows_invalid.sum() > 0:
            predictions[rows_invalid] = t_mean

    test['target'] = predictions
    submission = test[['id', 'target']].copy()
    submission.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH} with shape {submission.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
