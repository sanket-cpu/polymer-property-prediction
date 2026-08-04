"""
Phase 2, step 7 -- PI1M-derived features and diagnostics.

Three pieces, each addressing a different limitation of learning purely
from the ~6-7K labeled train rows:

1. Mol2Vec-style embedding. Ridge/GBM-on-descriptors only "knows" about
   substructures that happen to appear in the labeled train set. Training
   a skip-gram model on PI1M's ~1M unlabeled polymer SMILES lets the
   embedding learn which substructures tend to co-occur / behave similarly
   across a much broader slice of real polymer chemistry, then we just
   look up embeddings for train/test molecules -- this is exactly the kind
   of "no external data, no pretrained weights, train fresh every run"
   move the competition rules allow (PI1M ships with the repo, nothing
   pretrained is uploaded, and this script *is* the full training code).

   Sentence construction follows the published Mol2Vec method (Jaeger et
   al. 2018) directly against RDKit -- no `mol2vec` package dependency:
   for each atom, RDKit's Morgan-fingerprint bitInfo gives an integer
   identifier per (atom, radius) pair, where the identifier encodes that
   atom's local environment out to `radius` bonds. Concatenating each
   atom's radius-0 then radius-1 identifier, atom by atom, turns a
   molecule into a "sentence" of substructure "words" -- skip-gram then
   learns which substructures appear in similar contexts, the same way
   word2vec learns word semantics from text.

2. Refit the descriptor scaler/pruner on a large PI1M sample instead of
   train. Train has ~6-7K rows across 7 targets (as few as 221 for the
   small ones) -- which descriptor columns look "constant" or "redundant"
   from that small, label-biased sample may not hold across the broader
   population of real polymers. Refitting variance/correlation pruning
   and the scaling mean/std on a large *unlabeled* PI1M sample (no target
   leakage risk -- it never sees `target`) gives a more representative
   picture. Scoped to the continuous descriptor+physics block only (not
   fingerprints/dimer-delta -- those are handled by their own pruning in
   step 6, and dimer construction is the slow/fragile part of
   featurization, unnecessary for a population-level scaling reference).

3. PCA-space density diagnostic (train/test vs. PI1M). Fits PCA on the
   PI1M sample's scaled descriptors, projects train/test into that space,
   and flags molecules whose nearest PI1M neighbors are unusually far away
   -- i.e. structurally unlike anything common in the broader polymer
   space PI1M represents. This is purely diagnostic (per your instructions
   -- informs prediction clipping/confidence in Phase 5), not a feature
   used in modeling.

Mol2Vec corpus: 150K PI1M sample (cheap per-molecule cost -- fingerprint
substructure extraction only).
Descriptor-stats sample: 30K PI1M sample (expensive per-molecule cost --
full ~200-descriptor RDKit computation).
Both counts were your call after flagging the runtime/coverage tradeoff.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from prajwal_baseline import (  # noqa: E402
    featurize, fit_feature_pruner, _parse_mol, _safe_descriptors,
    _custom_physics_feats, VAR_THRESH, CORR_THRESH,
)
from cv_harness import load_train_with_groups, get_harness_splits, drop_constant_columns  # noqa: E402

from rdkit import RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PI1M_PATH = PROJECT_ROOT / "data" / "PI1M.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"

MOL2VEC_SAMPLE_N = 150_000
DESC_SAMPLE_N = 30_000
SEED = 42

MOL2VEC_RADIUS = 1
MOL2VEC_DIM = 100
MOL2VEC_WINDOW = 10
MOL2VEC_MIN_COUNT = 3
MOL2VEC_EPOCHS = 10

PROBE_PARAMS = dict(max_iter=150, max_depth=6, learning_rate=0.08,
                     l2_regularization=0.1, random_state=42)

N_PCA_COMPONENTS = 10
KNN_K = 20
SPARSE_PERCENTILE = 90  # flag train/test molecules above this percentile
                         # of PI1M's own internal kNN-distance distribution


# ---------------------------------------------------------------------------
# 1. Mol2Vec
# ---------------------------------------------------------------------------
def mol_to_sentence(mol, radius=MOL2VEC_RADIUS):
    """Mol2Vec 'alternating sentence': for each atom, walk radius 0..R and
    emit that atom's identifier at each radius (skipped if RDKit didn't
    assign one, which happens at the molecule's own edge cases)."""
    radii = list(range(radius + 1))
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    dict_atoms = {a.GetIdx(): {r: None for r in radii} for a in mol.GetAtoms()}
    for identifier, envs in info.items():
        for atom_idx, r in envs:
            if atom_idx in dict_atoms and r in dict_atoms[atom_idx]:
                dict_atoms[atom_idx][r] = identifier

    sentence = []
    for atom_idx in dict_atoms:
        for r in radii:
            ident = dict_atoms[atom_idx][r]
            if ident is not None:
                sentence.append(str(ident))
    return sentence


def load_pi1m_sample(n, seed):
    pi1m = pd.read_csv(PI1M_PATH)
    pi1m.columns = [c.strip() for c in pi1m.columns]
    smiles_col = 'SMILES' if 'SMILES' in pi1m.columns else pi1m.columns[0]
    sample = pi1m[smiles_col].dropna().sample(n=min(n, len(pi1m)), random_state=seed)
    return sample.reset_index(drop=True)


def train_mol2vec(smiles_sample, t0):
    print(f"  parsing {len(smiles_sample)} PI1M molecules for Mol2Vec corpus... "
          f"[{time.time()-t0:.0f}s]")
    mols = smiles_sample.apply(_parse_mol)
    n_bad = mols.isna().sum()
    mols = mols.dropna()
    print(f"  {n_bad} unparsable, {len(mols)} usable [{time.time()-t0:.0f}s]")

    print(f"  building sentences... [{time.time()-t0:.0f}s]")
    sentences = [mol_to_sentence(m) for m in mols]
    sentences = [s for s in sentences if len(s) > 0]
    print(f"  {len(sentences)} sentences, "
          f"avg length {np.mean([len(s) for s in sentences]):.1f} tokens "
          f"[{time.time()-t0:.0f}s]")

    print(f"  training word2vec (dim={MOL2VEC_DIM}, window={MOL2VEC_WINDOW}, "
          f"epochs={MOL2VEC_EPOCHS})... [{time.time()-t0:.0f}s]")
    model = Word2Vec(
        sentences, vector_size=MOL2VEC_DIM, window=MOL2VEC_WINDOW,
        min_count=MOL2VEC_MIN_COUNT, sg=1, workers=8, epochs=MOL2VEC_EPOCHS,
        seed=SEED,
    )
    print(f"  vocab size: {len(model.wv)} [{time.time()-t0:.0f}s]")
    return model


def mol2vec_embed(model, mol):
    tokens = mol_to_sentence(mol)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(MOL2VEC_DIM)
    return np.mean(vecs, axis=0)


def embed_dataframe(model, mols):
    embeds = np.stack([mol2vec_embed(model, m) for m in mols])
    return pd.DataFrame(embeds, columns=[f'mol2vec_{i}' for i in range(MOL2VEC_DIM)])


# ---------------------------------------------------------------------------
# 2. PI1M-refit descriptor scaler / pruner
# ---------------------------------------------------------------------------
def descriptor_only_features(mol):
    feats = {}
    feats.update(_safe_descriptors(mol))
    feats.update(_custom_physics_feats(mol))
    return feats


def build_pi1m_descriptor_sample(n, seed, t0):
    smiles_sample = load_pi1m_sample(n, seed + 1)  # different draw than mol2vec's
    print(f"  parsing + computing descriptors for {len(smiles_sample)} PI1M "
          f"molecules... [{time.time()-t0:.0f}s]")
    mols = smiles_sample.apply(_parse_mol).dropna()
    feats = [descriptor_only_features(m) for m in mols]
    df = pd.DataFrame(feats)
    print(f"  done: {df.shape} [{time.time()-t0:.0f}s]")
    return df


def fit_pi1m_pruner_scaler(pi1m_desc_df):
    keep_cols = fit_feature_pruner(pi1m_desc_df)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(pi1m_desc_df[keep_cols])
    scaler.fit(X_imputed)
    return keep_cols, imputer, scaler


# ---------------------------------------------------------------------------
# 3. PCA density diagnostic
# ---------------------------------------------------------------------------
def pca_density_diagnostic(pi1m_desc_df, keep_cols, imputer, scaler,
                            train_desc_df, test_desc_df):
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)
    pi1m_scaled = scaler.transform(imputer.transform(pi1m_desc_df[keep_cols]))
    pi1m_pca = pca.fit_transform(pi1m_scaled)
    print(f"  PCA explained variance ratio (first {N_PCA_COMPONENTS} PCs): "
          f"{pca.explained_variance_ratio_.sum():.3f} total, "
          f"{[round(v, 3) for v in pca.explained_variance_ratio_[:5]]}... (first 5)")

    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(pi1m_pca)
    # PI1M's own internal kNN distance distribution -- the reference for
    # "how far apart are molecules in this space, normally"
    pi1m_dists, _ = nn.kneighbors(pi1m_pca)
    pi1m_self_dist = pi1m_dists[:, 1:].mean(axis=1)  # exclude self (dist=0)
    threshold = np.percentile(pi1m_self_dist, SPARSE_PERCENTILE)
    print(f"  PI1M internal mean-{KNN_K}NN-distance {SPARSE_PERCENTILE}th "
          f"percentile (sparse-region threshold): {threshold:.3f}")

    def project_and_flag(desc_df, cols, label):
        X = scaler.transform(imputer.transform(desc_df[cols].reindex(columns=cols)))
        pts = pca.transform(X)
        dists, _ = nn.kneighbors(pts, n_neighbors=KNN_K)
        mean_dist = dists.mean(axis=1)
        flagged = mean_dist > threshold
        print(f"  {label}: {flagged.sum()}/{len(flagged)} "
              f"({100*flagged.mean():.1f}%) in sparse PI1M regions")
        return mean_dist, flagged

    return project_and_flag


# ---------------------------------------------------------------------------
# 4. Mol2Vec R2 ablation (isolated from step 6's fingerprint variants)
# ---------------------------------------------------------------------------
def cv_score(X_df, y_series, sub_index, repeats):
    scores = []
    for folds in repeats:
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            Xtr = X_df.loc[tr_idx].values.astype(float)
            Xva = X_df.loc[va_idx].values.astype(float)
            Xtr, Xva = drop_constant_columns(Xtr, Xva)
            ytr = y_series.loc[tr_idx].values
            yva = y_series.loc[va_idx].values
            model = HistGradientBoostingRegressor(**PROBE_PARAMS)
            model.fit(Xtr, ytr)
            scores.append(r2_score(yva, model.predict(Xva)))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    t0 = time.time()

    print("=" * 100)
    print("STEP 7a -- Mol2Vec: train on PI1M sample")
    print("=" * 100)
    mol2vec_smiles = load_pi1m_sample(MOL2VEC_SAMPLE_N, SEED)
    m2v_model = train_mol2vec(mol2vec_smiles, t0)

    print("\n" + "=" * 100)
    print("STEP 7b -- PI1M-refit descriptor scaler/pruner")
    print("=" * 100)
    pi1m_desc_df = build_pi1m_descriptor_sample(DESC_SAMPLE_N, SEED, t0)
    pi1m_keep_cols, pi1m_imputer, pi1m_scaler = fit_pi1m_pruner_scaler(pi1m_desc_df)
    print(f"  PI1M-fit pruner kept {len(pi1m_keep_cols)}/{pi1m_desc_df.shape[1]} "
          f"descriptor+physics columns")

    print("\nLoading train/test, featurizing (baseline + descriptor-only block)...")
    train = load_train_with_groups()
    raw_feats = train['smiles'].apply(featurize)
    valid_mask = raw_feats.notna()
    train_valid = train[valid_mask].reset_index(drop=True)
    raw_baseline_df = pd.DataFrame(list(raw_feats[valid_mask])).reset_index(drop=True)
    y_all = train_valid['target']

    desc_physics_cols = [c for c in raw_baseline_df.columns
                         if not c.startswith('fp_') and not c.startswith('dimer_delta_')]
    train_only_desc_cols = fit_feature_pruner(raw_baseline_df[desc_physics_cols])
    overlap = set(train_only_desc_cols) & set(pi1m_keep_cols)
    print(f"  train-only pruner (on train's own descriptor+physics block) kept "
          f"{len(train_only_desc_cols)} cols; PI1M-fit pruner kept "
          f"{len(pi1m_keep_cols)}; overlap: {len(overlap)}")
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    print("\n" + "=" * 100)
    print("STEP 7c -- PCA density diagnostic (train/test vs. PI1M)")
    print("=" * 100)
    train_mols = train_valid['smiles'].apply(_parse_mol)
    train_desc_df = pd.DataFrame([descriptor_only_features(m) for m in train_mols])

    test = pd.read_csv(TEST_PATH)
    test_mols_parsed = test['smiles'].apply(_parse_mol)
    test_valid_mask = test_mols_parsed.notna()
    test_valid = test[test_valid_mask].reset_index(drop=True)
    test_desc_df = pd.DataFrame(
        [descriptor_only_features(m) for m in test_mols_parsed[test_valid_mask]]
    ).reset_index(drop=True)
    print(f"  train desc matrix: {train_desc_df.shape}, "
          f"test desc matrix: {test_desc_df.shape} [{time.time()-t0:.0f}s]")

    projector = pca_density_diagnostic(
        pi1m_desc_df, pi1m_keep_cols, pi1m_imputer, pi1m_scaler,
        train_desc_df, test_desc_df)
    train_dist, train_flag = projector(train_desc_df, pi1m_keep_cols, "TRAIN")
    test_dist, test_flag = projector(test_desc_df, pi1m_keep_cols, "TEST")

    print("\n  Sparse-region flag rate by target_type:")
    train_valid_flags = pd.Series(train_flag, index=train_valid.index)
    test_valid_flags = pd.Series(test_flag, index=test_valid.index)
    for tt in sorted(train_valid['target_type'].unique()):
        tr_rate = train_valid_flags[train_valid['target_type'] == tt].mean()
        te_rate = test_valid_flags[test_valid['target_type'] == tt].mean() \
            if (test_valid['target_type'] == tt).any() else float('nan')
        print(f"    {tt:5s}  train={100*tr_rate:.1f}%  test={100*te_rate:.1f}%")

    print(f"\n[{time.time()-t0:.0f}s elapsed]")

    print("\n" + "=" * 100)
    print("STEP 7a (cont'd) -- Mol2Vec R2 ablation, isolated from step 6")
    print("=" * 100)
    print("  embedding train molecules...")
    mol2vec_df = embed_dataframe(m2v_model, train_mols)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    no_m2v_cols = fit_feature_pruner(raw_baseline_df)
    X_no_m2v = raw_baseline_df[no_m2v_cols]

    with_m2v_raw = pd.concat([raw_baseline_df, mol2vec_df], axis=1)
    with_m2v_cols = fit_feature_pruner(with_m2v_raw)
    X_with_m2v = with_m2v_raw[with_m2v_cols]

    print(f"  no-mol2vec: {len(no_m2v_cols)} cols | "
          f"with-mol2vec: {len(with_m2v_cols)} cols "
          f"({len(with_m2v_cols) - len(no_m2v_cols)} net new)")

    target_types = sorted(train_valid['target_type'].unique())
    print(f"\n{'target':6s}{'no_mol2vec':>16s}{'with_mol2vec':>16s}{'delta':>10s}")
    for tt in target_types:
        sub_index, repeats = get_harness_splits(train_valid, tt)
        m_no, s_no = cv_score(X_no_m2v, y_all, sub_index, repeats)
        m_yes, s_yes = cv_score(X_with_m2v, y_all, sub_index, repeats)
        delta = m_yes - m_no
        flag = " <-- keep mol2vec" if delta > max(s_no, s_yes) else ""
        print(f"{tt:6s}{m_no:+9.4f}(±{s_no:.3f}){m_yes:+9.4f}(±{s_yes:.3f})"
              f"{delta:+9.4f}{flag}")
        print(f"  [{time.time()-t0:.0f}s elapsed]")

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
    print("Phase 2 step 7 complete. Phase 2 complete -- stopping per workflow.")


if __name__ == "__main__":
    main()
