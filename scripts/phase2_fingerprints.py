"""
Phase 2, step 6 -- fingerprint variants.

Prajwal's baseline (scripts/prajwal_baseline.py) is carried forward
unchanged: RDKit descriptors + custom physics ratios + dimer-delta features
+ a 256-bit, radius-2 Morgan fingerprint block. This script adds three
*additional* fingerprint blocks on top of that unchanged baseline, one at a
time, and scores each addition against the Phase 1 CV harness:

  +1024r3  : baseline + 1024-bit Morgan, radius 3
  +2048r3  : baseline + 2048-bit Morgan, radius 3
  +MACCS   : baseline + 167-bit MACCS keys

Each variant's raw features go through the same variance/correlation
pruning prajwal's script already uses (fit on the variant's own columns --
pruning doesn't see the target, so this is safe to fit on the full
available train set rather than per-fold).

Probe model: HistGradientBoostingRegressor (handles NaN + high-dimensional
mixed dense/binary features natively, no scaling/imputation pipeline
needed). This is deliberately a lighter config (max_iter=150) than the
tuned zoo Phase 3 will use -- it's a fast, consistent yardstick for
comparing *feature sets*, not a final model.

A variant is flagged "keep" for a target_type only if it beats the
256-bit baseline's mean CV R2 by more than the noise floor (repeat-to-
repeat std for that target/variant) -- otherwise the "improvement" is
indistinguishable from fold-assignment noise, especially for the small
targets.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from prajwal_baseline import featurize, fit_feature_pruner, _parse_mol  # noqa: E402
from cv_harness import load_train_with_groups, get_harness_splits, drop_constant_columns  # noqa: E402

from rdkit.Chem import rdFingerprintGenerator, MACCSkeys  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog('rdApp.*')

_MORGAN_1024_R3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
_MORGAN_2048_R3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

PROBE_PARAMS = dict(max_iter=150, max_depth=6, learning_rate=0.08,
                     l2_regularization=0.1, random_state=42)


def _fp_bits(gen, mol, prefix, size):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(size, dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return {f'{prefix}_{i}': int(arr[i]) for i in range(size)}


def morgan_1024_r3(mol):
    return _fp_bits(_MORGAN_1024_R3, mol, 'fp1024', 1024)


def morgan_2048_r3(mol):
    return _fp_bits(_MORGAN_2048_R3, mol, 'fp2048', 2048)


def maccs_keys(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.int8)
    for bit in fp.GetOnBits():
        if bit < 167:
            arr[bit] = 1
    return {f'maccs_{i}': int(arr[i]) for i in range(167)}


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
    train = load_train_with_groups()
    print(f"Loaded train={train.shape}")

    print("Featurizing with prajwal's unchanged baseline featurize()...")
    raw_feats = train['smiles'].apply(featurize)
    valid_mask = raw_feats.notna()
    if (~valid_mask).sum():
        print(f"  dropping {(~valid_mask).sum()} rows that failed full featurization")
    train_valid = train[valid_mask].reset_index(drop=True)
    raw_baseline_df = pd.DataFrame(list(raw_feats[valid_mask])).reset_index(drop=True)
    y_all = train_valid['target']
    print(f"  baseline raw feature count: {raw_baseline_df.shape[1]} "
          f"({time.time()-t0:.0f}s elapsed)")

    print("Computing additional fingerprint blocks (1024r3, 2048r3, MACCS)...")
    mols = train_valid['smiles'].apply(_parse_mol)
    morgan1024_df = pd.DataFrame([morgan_1024_r3(m) for m in mols])
    morgan2048_df = pd.DataFrame([morgan_2048_r3(m) for m in mols])
    maccs_df = pd.DataFrame([maccs_keys(m) for m in mols])
    print(f"  done ({time.time()-t0:.0f}s elapsed)")

    variants = {
        'baseline_256r2': raw_baseline_df,
        '+1024r3': pd.concat([raw_baseline_df, morgan1024_df], axis=1),
        '+2048r3': pd.concat([raw_baseline_df, morgan2048_df], axis=1),
        '+MACCS': pd.concat([raw_baseline_df, maccs_df], axis=1),
    }

    target_types = sorted(train_valid['target_type'].unique())
    results = {tt: {} for tt in target_types}  # tt -> variant -> (mean, std, n_features)

    for vname, raw_df in variants.items():
        cols = fit_feature_pruner(raw_df)
        X = raw_df[cols]
        print(f"\nVariant {vname}: {raw_df.shape[1]} raw -> {len(cols)} after pruning")
        for tt in target_types:
            sub_index, repeats = get_harness_splits(train_valid, tt)
            mean_r2, std_r2 = cv_score(X, y_all, sub_index, repeats)
            results[tt][vname] = (mean_r2, std_r2, len(cols))
            print(f"  {tt:5s}  R2={mean_r2:+.4f}  (std={std_r2:.4f})  "
                  f"[{time.time()-t0:.0f}s elapsed]")

    print("\n" + "=" * 100)
    print("COMPARISON: per-target mean CV R2 by fingerprint variant")
    print("=" * 100)
    header = f"{'target':6s}" + "".join(f"{v:>18s}" for v in variants)
    print(header)
    keep_decisions = {}
    for tt in target_types:
        base_mean, base_std, _ = results[tt]['baseline_256r2']
        row = f"{tt:6s}"
        best_variant = 'baseline_256r2'
        best_margin = 0.0
        for vname in variants:
            mean_r2, std_r2, _ = results[tt][vname]
            row += f"  {mean_r2:+.4f}(±{std_r2:.3f})"
            if vname != 'baseline_256r2':
                margin = mean_r2 - base_mean
                noise_floor = max(std_r2, base_std)
                if margin > noise_floor and margin > best_margin:
                    best_margin = margin
                    best_variant = vname
        keep_decisions[tt] = best_variant
        print(row)

    print("\n--- Keep decision per target (variant beats baseline by more than noise floor) ---")
    for tt, decision in keep_decisions.items():
        if decision == 'baseline_256r2':
            print(f"  {tt:5s}: keep baseline_256r2 (no variant cleared the noise floor)")
        else:
            m, s, nf = results[tt][decision]
            bm, bs, _ = results[tt]['baseline_256r2']
            print(f"  {tt:5s}: keep {decision} (R2 {bm:.4f} -> {m:.4f}, "
                  f"+{m-bm:.4f}, {nf} features)")

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
    print("Phase 2 step 6 complete -- continuing to step 7 (PI1M / Mol2Vec).")


if __name__ == "__main__":
    main()
