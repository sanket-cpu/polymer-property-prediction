"""
Polymer Property Prediction - Round 2 (v2, boosted)
Physics-informed + fingerprint descriptor model with weighted ensembling.

Predicts 7 polymer properties (Egc, Egb, Ei, Eea, EPS, Nc, Tg) from polymer SMILES
using RDKit descriptors + Morgan fingerprints + a "dimer" repeat-unit trick to
capture chain-level conjugation effects, followed by per-property model
selection and weighted blending.

No external data, no pretrained weights. Fully self-contained, runs in
seconds-minutes on CPU.

Key changes vs v1 (aimed at CV R2 0.78 -> ~0.84):
  1. Full RDKit descriptor set (~200 descriptors) instead of ~35 hand-picked
     ones, with automatic pruning of constant / near-duplicate columns.
  2. Morgan fingerprint bits (folded, 256-bit) added as coarse substructure
     features -- these carry a lot of the signal GBM/RF exploit for
     electronic properties (Egc, Egb, Ei, Eea).
  3. "Dimer" trick: the repeat unit is joined head-to-tail with a copy of
     itself at the * attachment points before descriptor calculation. This
     lets ring-conjugation / aromaticity / rotatable-bond descriptors see
     across the repeat-unit boundary, which matters for backbone-driven
     properties (band gaps, refractive index) far more than a single
     isolated unit does. Dimer descriptors are added as a *delta* from the
     monomer values, so they encode "what changes when the chain extends."
  4. Median imputation of any NaN feature values (full descriptor list can
     occasionally throw NaN/inf on unusual structures) instead of silently
     dropping columns.
  5. Model zoo expanded to include HistGradientBoostingRegressor (fast,
     usually stronger than the older GradientBoostingRegressor on tabular
     data) and ElasticNet alongside Ridge/RF/GBM.
  6. Instead of taking a single "best" model per target, the top-2 models
     (by CV R2) are blended with softmax-style weights derived from their
     CV scores -- ensembling reduces variance and typically adds a few
     points of R2 over picking one winner.
  7. Feature pruning (variance threshold + correlation threshold) is fit on
     train only and re-used for test, avoiding leakage while keeping the
     matrix small enough that model fitting stays fast.

Usage: adjust TRAIN_PATH / TEST_PATH / OUT_PATH below, then run.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

# The dimer-construction step (see _make_dimer_mol) intentionally attempts
# a bond-surgery + resanitize that fails on some ring topologies; those
# failures are caught and handled (falls back to zero-delta features), so
# RDKit's C++ logger spam ("Can't kekulize...") is expected noise, not a
# sign anything is broken. Silence it so the run log stays readable.
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- paths (edit these to match your environment / Kaggle input folder) ----
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUT_PATH = "submission.csv"

# ---- feature engineering knobs ----
FP_BITS = 256          # folded Morgan fingerprint size (keep small = fast)
FP_RADIUS = 2
VAR_THRESH = 1e-6      # drop near-constant columns
CORR_THRESH = 0.98     # drop one of any pair of columns correlated above this
TOP_K_MODELS = 2        # blend the top-K CV models per target, not just 1

# All RDKit descriptor (name, function) pairs, excluding a couple that are
# slow/unstable (3D descriptors need embedding, which polymer repeat units
# with dummy atoms often fail at).
_SLOW_OR_UNSTABLE = {
    'Ipc',  # can overflow to inf on larger structures
}
_DESC_LIST = [(n, f) for n, f in Descriptors._descList if n not in _SLOW_OR_UNSTABLE]

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


# ---------------------------------------------------------------------------
# 1. Featurization
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
    """
    Join two copies of the repeat unit at their * attachment points to
    approximate a short chain segment. Falls back to None if construction
    fails (rare, e.g. more/less than 2 attachment points).
    """
    try:
        s = smiles.replace('[*]', '*')
        if s.count('*') != 2:
            return None
        # Replace the two '*' in unit A with dummy isotopes, unit B likewise,
        # then bond A's second dummy to B's first dummy via RWMol surgery.
        molA = Chem.MolFromSmiles(s)
        molB = Chem.MolFromSmiles(s)
        if molA is None or molB is None:
            return None

        combo = Chem.RWMol(Chem.CombineMols(molA, molB))
        dummy_idx = [a.GetIdx() for a in combo.GetAtoms() if a.GetSymbol() == '*']
        if len(dummy_idx) != 4:
            return None

        # dummy_idx[0], dummy_idx[1] belong to molA; [2], [3] belong to molB
        def neighbor_of(idx):
            atom = combo.GetAtomWithIdx(idx)
            nbrs = atom.GetNeighbors()
            return nbrs[0].GetIdx() if nbrs else None

        a2 = neighbor_of(dummy_idx[1])
        b1 = neighbor_of(dummy_idx[2])
        if a2 is None or b1 is None:
            return None

        combo.AddBond(a2, b1, Chem.BondType.SINGLE)
        # remove the 4 dummy atoms (remove highest index first to keep indices valid)
        for idx in sorted(dummy_idx, reverse=True):
            combo.RemoveAtom(idx)

        dimer = combo.GetMol()
        try:
            # Preferred path: full sanitize (correct kekulization/aromaticity).
            Chem.SanitizeMol(dimer)
        except Exception:
            # Some ring topologies genuinely can't be re-kekulized after the
            # junction bond is spliced in (rare, structure-dependent -- not
            # a bug to "fix" by forcing it). Fall back to a partial sanitize
            # that skips just the kekulize/aromaticity perception steps, so
            # we still get valid valences and can compute the non-aromaticity
            # -dependent portion of the descriptor set instead of discarding
            # the whole dimer.
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
    """Compute the full RDKit descriptor list, replacing failures with NaN."""
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
    """Small set of hand-crafted, physically-motivated ratios (cheap, robust)."""
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

        # Dimer-delta features: how key backbone-sensitive descriptors shift
        # when the chain is extended by one repeat unit. Captures
        # conjugation/rigidity trends that a single isolated unit misses,
        # which is especially informative for Egc/Egb/Nc.
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


# ---------------------------------------------------------------------------
# 2. Feature matrix cleanup (fit on train, applied to test)
# ---------------------------------------------------------------------------
def fit_feature_pruner(df):
    """
    Returns the list of columns to keep, after dropping near-constant
    columns and one column from each highly-correlated pair. Fit on train
    only to avoid leakage.
    """
    variances = df.var(numeric_only=True)
    keep = variances[variances > VAR_THRESH].index.tolist()

    corr = df[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
    keep = [c for c in keep if c not in to_drop]
    return keep


# ---------------------------------------------------------------------------
# 3. Model zoo
# ---------------------------------------------------------------------------
def get_model_zoo():
    return {
        'Ridge': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('m', Ridge(alpha=5.0, random_state=RANDOM_STATE)),
        ]),
        'ElasticNet': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
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
    """Convert a list of CV R2 scores into positive blend weights."""
    arr = np.array(scores, dtype=float)
    arr = arr - arr.max()  # numerical stability
    w = np.exp(arr * 5.0)  # sharpen so the better model dominates a bit
    return w / w.sum()


# ---------------------------------------------------------------------------
# 4. Main pipeline
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

    # --- prune features on train only ---
    feature_cols = fit_feature_pruner(train_feat_df)
    print(f"Kept {len(feature_cols)}/{train_feat_df.shape[1]} features after pruning")

    full_train = pd.concat(
        [train_valid[['target', 'target_type']], train_feat_df[feature_cols]], axis=1
    )

    # --- per-target-type: CV-evaluate model zoo, blend top-K ---
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    blend_config = {}   # target_type -> list of (model_name, weight)
    cv_scores_summary = {}

    for tt in sorted(full_train['target_type'].unique()):
        sub = full_train[full_train['target_type'] == tt]
        X = sub[feature_cols].values
        y = sub['target'].values

        scores_per_model = {}
        for name, model in get_model_zoo().items():
            scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
            scores_per_model[name] = scores.mean()

        ranked = sorted(scores_per_model.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:TOP_K_MODELS]
        names, scores = zip(*top)
        weights = softmax_weights(scores)
        blend_config[tt] = list(zip(names, weights))
        cv_scores_summary[tt] = ranked[0][1]  # best single-model CV score for reporting

        print(f"{tt:5s} (n={len(y):4d})  " +
              "  ".join(f"{k}={v:.3f}" for k, v in scores_per_model.items()) +
              "  -> BLEND: " + ", ".join(f"{n}({w:.2f})" for n, w in blend_config[tt]))

    mean_cv_r2 = np.mean(list(cv_scores_summary.values()))
    print(f"\nEstimated mean CV R2 across all 7 targets (best single model each): {mean_cv_r2:.4f}")
    print("(actual blended CV R2 is typically equal to or slightly better than this)\n")

    # --- train final blended models on full data ---
    final_models = {}   # target_type -> list of (fitted_model, weight)
    target_means = {}
    for tt in sorted(full_train['target_type'].unique()):
        sub = full_train[full_train['target_type'] == tt]
        X = sub[feature_cols].values
        y = sub['target'].values
        target_means[tt] = y.mean()

        zoo = get_model_zoo()
        fitted = []
        for name, weight in blend_config[tt]:
            model = zoo[name]
            model.fit(X, y)
            fitted.append((model, weight))
        final_models[tt] = fitted

    # --- featurize test ---
    test_feats = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    print(f"Featurized {test_valid_mask.sum()}/{len(test)} test molecules")

    test_feat_df = pd.DataFrame(index=test.index, columns=feature_cols, dtype=float)
    for idx in test.index[test_valid_mask]:
        row = test_feats[idx]
        for k in feature_cols:
            test_feat_df.loc[idx, k] = row.get(k, np.nan)

    predictions = np.zeros(len(test))
    for tt in sorted(full_train['target_type'].unique()):
        mask = (test['target_type'] == tt).values
        if mask.sum() == 0:
            continue
        rows_valid = mask & test_valid_mask.values
        rows_invalid = mask & (~test_valid_mask.values)

        if rows_valid.sum() > 0:
            X_test = test_feat_df.loc[rows_valid, feature_cols].values.astype(float)
            blend_pred = np.zeros(rows_valid.sum())
            for model, weight in final_models[tt]:
                blend_pred += weight * model.predict(X_test)
            predictions[rows_valid] = blend_pred

        if rows_invalid.sum() > 0:
            # fallback for the rare unparsable SMILES: use training mean for that property
            predictions[rows_invalid] = target_means[tt]

    test['target'] = predictions
    submission = test[['id', 'target']].copy()
    submission.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with shape {submission.shape}")


if __name__ == "__main__":
    main()
