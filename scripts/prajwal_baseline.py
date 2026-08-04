"""
Polymer Property Prediction - Round 3 (v3, Tier-1 PI1M-augmented)
Refactored & Fixed for Sklearn Pipeline Compatibility
"""

import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit import DataStructs

RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import KFold, cross_val_score
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- paths ----
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
PI1M_PATH = "PI1M.csv"          # column expected: 'smiles' or 'SMILES'
PI1M_SMILES_COL = "smiles"
OUT_PATH = "submission.csv"

# ---- feature engineering knobs ----
FP_BITS = 256
FP_RADIUS = 2
VAR_THRESH = 1e-6
CORR_THRESH = 0.98
TOP_K_MODELS = 2

# ---- Tier-1 PI1M knobs ----
PI1M_SAMPLE_SIZE = 20000     # rows sampled from PI1M for stats + density
DENSITY_TOPK = 7             # avg similarity to top-K nearest PI1M neighbors

_SLOW_OR_UNSTABLE = {'Ipc'}
_DESC_LIST = [(n, f) for n, f in Descriptors._descList if n not in _SLOW_OR_UNSTABLE]
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


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


def featurize_lightweight(smiles):
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


# ---------------------------------------------------------------------------
# 2. PI1M loading + sampling + lightweight featurization
# ---------------------------------------------------------------------------
def load_pi1m_sample(path, smiles_col, sample_size, seed=RANDOM_STATE):
    print(f"Loading PI1M from {path} ...")
    pi1m = pd.read_csv(path)
    if smiles_col not in pi1m.columns:
        candidates = [c for c in pi1m.columns if 'smiles' in c.lower()]
        if not candidates:
            raise ValueError(f"Could not find a SMILES column in {path}; columns found: {list(pi1m.columns)}")
        smiles_col_local = candidates[0]
    else:
        smiles_col_local = smiles_col
    print(f"PI1M full size: {len(pi1m)} rows, using column '{smiles_col_local}'")

    n = min(sample_size, len(pi1m))
    pi1m_sample = pi1m[[smiles_col_local]].dropna().drop_duplicates()
    pi1m_sample = pi1m_sample.sample(n=min(n, len(pi1m_sample)), random_state=seed).reset_index(drop=True)
    pi1m_sample = pi1m_sample.rename(columns={smiles_col_local: 'smiles'})
    print(f"Sampled {len(pi1m_sample)} PI1M rows for Tier-1 use")
    return pi1m_sample


def featurize_pi1m_lightweight(pi1m_sample):
    t0 = time.time()
    feats = pi1m_sample['smiles'].apply(featurize_lightweight)
    valid = feats.notna()
    feat_df = pd.DataFrame(list(feats[valid])).reset_index(drop=True)
    print(f"PI1M lightweight featurization: {valid.sum()}/{len(pi1m_sample)} valid in {time.time() - t0:.1f}s")
    return feat_df, pi1m_sample.loc[valid, 'smiles'].reset_index(drop=True)


def build_pi1m_fp_index(pi1m_smiles):
    t0 = time.time()
    fps = []
    for smi in pi1m_smiles:
        mol = _parse_mol(smi)
        if mol is None:
            continue
        fps.append(_MORGAN_GEN.GetFingerprint(mol))
    print(f"Built PI1M fingerprint index: {len(fps)} fps in {time.time() - t0:.1f}s")
    return fps


def density_features(smiles_series, pi1m_fps, topk=DENSITY_TOPK):
    t0 = time.time()
    max_sims = np.zeros(len(smiles_series))
    topk_means = np.zeros(len(smiles_series))
    for i, smi in enumerate(smiles_series):
        mol = _parse_mol(smi)
        if mol is None:
            max_sims[i] = np.nan
            topk_means[i] = np.nan
            continue
        fp = _MORGAN_GEN.GetFingerprint(mol)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, pi1m_fps))
        if len(sims) == 0:
            max_sims[i] = np.nan
            topk_means[i] = np.nan
            continue
        top = np.sort(sims)[-topk:]
        max_sims[i] = top[-1]
        topk_means[i] = top.mean()
    print(f"Computed density features for {len(smiles_series)} molecules in {time.time() - t0:.1f}s")
    return pd.DataFrame({
        'pi1m_max_sim': max_sims,
        'pi1m_topk_mean_sim': topk_means,
    })


# ---------------------------------------------------------------------------
# 3. Feature pruning & Prefit Scaler (Fixed)
# ---------------------------------------------------------------------------
def fit_feature_pruner_combined(train_feat_df, pi1m_feat_df, feature_cols_common):
    combined = pd.concat(
        [train_feat_df[feature_cols_common], pi1m_feat_df[feature_cols_common]],
        axis=0, ignore_index=True
    )
    variances = combined.var(numeric_only=True)
    keep = variances[variances > VAR_THRESH].index.tolist()

    corr = combined[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
    keep = [c for c in keep if c not in to_drop]
    return keep


def fit_global_scaler(train_feat_df, pi1m_feat_df, feature_cols):
    combined = pd.concat(
        [train_feat_df[feature_cols], pi1m_feat_df[feature_cols]],
        axis=0, ignore_index=True
    )
    imputer = SimpleImputer(strategy='median')
    combined_imp = imputer.fit_transform(combined)
    scaler = StandardScaler()
    scaler.fit(combined_imp)
    return imputer, scaler


class _FittedStateWrapper:
    """Helper class to hide fitted estimators from sklearn's clone() wipe."""
    def __init__(self, imputer, scaler):
        self.imputer = imputer
        self.scaler = scaler


class PrefitScaler(BaseEstimator, TransformerMixin):
    """Fixed: Transformer wrapper that registers sklearn attributes and bypasses re-fitting."""
    def __init__(self, state):
        self.state = state

    def fit(self, X, y=None):
        # Register sklearn fitting state check attributes
        self.n_features_in_ = getattr(
            self.state.imputer, 
            "n_features_in_", 
            X.shape[1] if hasattr(X, "shape") else None
        )
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_imp = self.state.imputer.transform(X)
        return self.state.scaler.transform(X_imp)


# ---------------------------------------------------------------------------
# 4. Model zoo
# ---------------------------------------------------------------------------
def get_model_zoo(prefit_imputer, prefit_scaler):
    # Wrap the fitted components so clone() deepcopies them instead of resetting them
    state = _FittedStateWrapper(prefit_imputer, prefit_scaler)
    
    return {
        'Ridge': Pipeline([
            ('sc', PrefitScaler(state)),
            ('m', Ridge(alpha=5.0, random_state=RANDOM_STATE)),
        ]),
        'ElasticNet': Pipeline([
            ('sc', PrefitScaler(state)),
            ('m', ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=5000)),
        ]),
        'RF': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        'GBM': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=RANDOM_STATE)),
        ]),
        'HGB': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', HistGradientBoostingRegressor(
                max_iter=300, max_depth=6, learning_rate=0.06,
                l2_regularization=0.1, random_state=RANDOM_STATE)),
        ]),
    }


def softmax_weights(scores):
    arr = np.array(scores, dtype=float)
    arr = arr - arr.max()
    w = np.exp(arr * 5.0)
    return w / w.sum()


# ---------------------------------------------------------------------------
# 5. Main pipeline
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

    # --- featurize test ---
    test_feats = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    print(f"Featurized {test_valid_mask.sum()}/{len(test)} test molecules")

    # --- PI1M: sample, lightweight-featurize, build fp index ---
    pi1m_sample = load_pi1m_sample(PI1M_PATH, PI1M_SMILES_COL, PI1M_SAMPLE_SIZE)
    pi1m_feat_df, pi1m_smiles_valid = featurize_pi1m_lightweight(pi1m_sample)
    pi1m_fps = build_pi1m_fp_index(pi1m_smiles_valid)

    # --- density features for train/test ---
    train_density = density_features(train_valid['smiles'], pi1m_fps)
    test_density = density_features(test['smiles'], pi1m_fps)
    train_feat_df = pd.concat([train_feat_df, train_density], axis=1)

    # --- prune features ---
    common_cols = [c for c in pi1m_feat_df.columns if c in train_feat_df.columns]
    pruned_common = fit_feature_pruner_combined(train_feat_df, pi1m_feat_df, common_cols)

    fp_dimer_density_cols = [c for c in train_feat_df.columns
                              if c.startswith('fp_') or c.startswith('dimer_delta_')
                              or c.startswith('pi1m_')]
    feature_cols = pruned_common + fp_dimer_density_cols
    print(f"Kept {len(feature_cols)} features after PI1M-informed pruning "
          f"({len(pruned_common)} descriptor/physics + {len(fp_dimer_density_cols)} fp/dimer/density)")

    # --- global scaler fit on train + PI1M sample ---
    prefit_imputer, prefit_scaler = fit_global_scaler(train_feat_df, pi1m_feat_df, pruned_common)

    full_train = pd.concat(
        [train_valid[['target', 'target_type']], train_feat_df[feature_cols]], axis=1
    )

    # --- cross-validation & model evaluation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    blend_config = {}
    cv_scores_summary = {}

    for tt in sorted(full_train['target_type'].unique()):
        sub = full_train[full_train['target_type'] == tt]
        y = sub['target'].values

        scores_per_model = {}
        zoo = get_model_zoo(prefit_imputer, prefit_scaler)
        for name, model in zoo.items():
            if name in ('Ridge', 'ElasticNet'):
                X = sub[pruned_common].values
            else:
                X = sub[feature_cols].values
            scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
            scores_per_model[name] = scores.mean()

        ranked = sorted(scores_per_model.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:TOP_K_MODELS]
        names, scores = zip(*top)
        weights = softmax_weights(scores)
        blend_config[tt] = list(zip(names, weights))
        cv_scores_summary[tt] = ranked[0][1]

        print(f"{tt:5s} (n={len(y):4d})  " +
              "  ".join(f"{k}={v:.3f}" for k, v in scores_per_model.items()) +
              "  -> BLEND: " + ", ".join(f"{n}({w:.2f})" for n, w in blend_config[tt]))

    mean_cv_r2 = np.mean(list(cv_scores_summary.values()))
    print(f"\nEstimated mean CV R2 across all targets: {mean_cv_r2:.4f}\n")

    # --- train final ensemble models ---
    final_models = {}
    target_means = {}
    for tt in sorted(full_train['target_type'].unique()):
        sub = full_train[full_train['target_type'] == tt]
        y = sub['target'].values
        target_means[tt] = y.mean()

        zoo = get_model_zoo(prefit_imputer, prefit_scaler)
        fitted = []
        for name, weight in blend_config[tt]:
            model = zoo[name]
            X = sub[pruned_common].values if name in ('Ridge', 'ElasticNet') else sub[feature_cols].values
            model.fit(X, y)
            fitted.append((model, weight, name))
        final_models[tt] = fitted

    # --- test inference ---
    test_feat_df = pd.DataFrame(index=test.index, columns=feature_cols, dtype=float)
    for idx in test.index[test_valid_mask]:
        row = test_feats[idx]
        for k in feature_cols:
            if k in ('pi1m_max_sim', 'pi1m_topk_mean_sim'):
                continue
            test_feat_df.loc[idx, k] = row.get(k, np.nan)
    test_feat_df.loc[test.index, 'pi1m_max_sim'] = test_density['pi1m_max_sim'].values
    test_feat_df.loc[test.index, 'pi1m_topk_mean_sim'] = test_density['pi1m_topk_mean_sim'].values

    predictions = np.zeros(len(test))
    for tt in sorted(full_train['target_type'].unique()):
        mask = (test['target_type'] == tt).values
        if mask.sum() == 0:
            continue
        rows_valid = mask & test_valid_mask.values
        rows_invalid = mask & (~test_valid_mask.values)

        if rows_valid.sum() > 0:
            blend_pred = np.zeros(rows_valid.sum())
            for model, weight, name in final_models[tt]:
                cols = pruned_common if name in ('Ridge', 'ElasticNet') else feature_cols
                X_test = test_feat_df.loc[rows_valid, cols].values.astype(float)
                blend_pred += weight * model.predict(X_test)
            predictions[rows_valid] = blend_pred

        if rows_invalid.sum() > 0:
            predictions[rows_invalid] = target_means[tt]

    test['target'] = predictions
    submission = test[['id', 'target']].copy()
    submission.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with shape {submission.shape}")


if __name__ == "__main__":
    main()
