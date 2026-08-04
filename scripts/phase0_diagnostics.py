"""
Phase 0 -- Diagnostics only, no modeling.

1. Within each target_type's slice of train.csv, canonicalize SMILES and
   report exact-duplicate and near-duplicate molecule groups, and whether
   their target values agree or conflict.
2. Per-target (all 7) distribution stats (min/max/mean/std/skew) and an
   IQR-based outlier flag.

"Near-duplicate" here means: two rows whose canonical SMILES differ, but
which become identical once stereochemistry is stripped (isomericSmiles=
False). In other words, the same constitutional structure (same atoms,
same bonds, same connectivity) but different stereodescriptors (e.g. a
different R/S center or E/Z double bond). This is a cheap, deterministic
definition -- no similarity threshold to tune -- as opposed to a fingerprint-
Tanimoto-similarity approach, which would catch structurally-similar-but-
different molecules too, at the cost of picking an arbitrary threshold and
O(n^2) pairwise comparisons per target_type slice.

Run: .venv/bin/python scripts/phase0_diagnostics.py
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"

pd.set_option('display.width', 140)


def parse_mol(smiles):
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    return mol


def canonicalize(smiles):
    mol = parse_mol(smiles)
    if mol is None:
        return None, None
    canon = Chem.MolToSmiles(mol, canonical=True)
    canon_no_stereo = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return canon, canon_no_stereo


# ---------------------------------------------------------------------------
# 1. Duplicate / near-duplicate report
# ---------------------------------------------------------------------------
def duplicate_report(train):
    print("=" * 100)
    print("DUPLICATE / NEAR-DUPLICATE REPORT (per target_type slice)")
    print("=" * 100)

    canon_pairs = train['smiles'].apply(canonicalize)
    train = train.copy()
    train['canon'] = [c[0] for c in canon_pairs]
    train['canon_no_stereo'] = [c[1] for c in canon_pairs]

    n_unparsable = train['canon'].isna().sum()
    if n_unparsable:
        print(f"\n[!] {n_unparsable} train SMILES failed to parse even with the "
              f"'*'->'C' fallback -- excluded from the duplicate analysis below.")
    train = train[train['canon'].notna()].reset_index(drop=True)

    overall = {}
    for tt in sorted(train['target_type'].unique()):
        sub = train[train['target_type'] == tt]

        # exact duplicates: same canonical (isomeric) SMILES
        exact_groups = sub.groupby('canon').filter(lambda g: len(g) > 1).groupby('canon')
        n_exact_groups = exact_groups.ngroups
        n_exact_rows = sum(len(g) for _, g in exact_groups)

        exact_consistent, exact_conflicting = 0, 0
        conflict_examples = []
        for canon, g in exact_groups:
            vals = g['target'].values
            if np.ptp(vals) / (abs(vals.mean()) + 1e-9) < 0.01 or np.ptp(vals) < 1e-6:
                exact_consistent += 1
            else:
                exact_conflicting += 1
                if len(conflict_examples) < 3:
                    conflict_examples.append((canon, vals.tolist()))

        # near-duplicates: same canon_no_stereo, but NOT already an exact-dup
        # group (i.e. this collapsing only happens once stereo is stripped)
        near = sub.groupby('canon_no_stereo').filter(lambda g: len(g) > 1)
        near_groups = near.groupby('canon_no_stereo')
        n_near_groups_raw = 0
        n_near_rows = 0
        near_consistent, near_conflicting = 0, 0
        near_conflict_examples = []
        for canon_ns, g in near_groups:
            # only count as "near" (not "exact") if this group spans >1 distinct
            # isomeric canonical SMILES -- otherwise it's just the same exact-dup
            # group being seen again
            if g['canon'].nunique() <= 1:
                continue
            n_near_groups_raw += 1
            n_near_rows += len(g)
            vals = g['target'].values
            if np.ptp(vals) / (abs(vals.mean()) + 1e-9) < 0.01 or np.ptp(vals) < 1e-6:
                near_consistent += 1
            else:
                near_conflicting += 1
                if len(near_conflict_examples) < 3:
                    near_conflict_examples.append((canon_ns, g['canon'].unique().tolist(), vals.tolist()))

        overall[tt] = dict(
            n=len(sub),
            exact_groups=n_exact_groups, exact_rows=n_exact_rows,
            exact_consistent=exact_consistent, exact_conflicting=exact_conflicting,
            near_groups=n_near_groups_raw, near_rows=n_near_rows,
            near_consistent=near_consistent, near_conflicting=near_conflicting,
        )

        print(f"\n--- {tt} (n={len(sub)}) ---")
        print(f"  Exact duplicates:  {n_exact_groups} groups / {n_exact_rows} rows "
              f"({exact_consistent} consistent, {exact_conflicting} conflicting)")
        for canon, vals in conflict_examples:
            print(f"      CONFLICT example: {canon[:60]}...  targets={vals}")
        print(f"  Near duplicates:   {n_near_groups_raw} groups / {n_near_rows} rows "
              f"(stereo-only diff; {near_consistent} consistent, {near_conflicting} conflicting)")
        for canon_ns, variants, vals in near_conflict_examples:
            print(f"      CONFLICT example: {canon_ns[:60]}...  "
                  f"{len(variants)} stereo variants  targets={vals}")

    print("\n--- Summary table ---")
    summary_df = pd.DataFrame(overall).T
    print(summary_df.to_string())
    return train, summary_df


# ---------------------------------------------------------------------------
# 2. Distribution / outlier report
# ---------------------------------------------------------------------------
def distribution_report(train):
    print("\n" + "=" * 100)
    print("PER-TARGET DISTRIBUTION + OUTLIER REPORT")
    print("=" * 100)

    rows = []
    for tt in sorted(train['target_type'].unique()):
        y = train.loc[train['target_type'] == tt, 'target'].astype(float)
        q1, q3 = y.quantile(0.25), y.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = y[(y < lo) | (y > hi)]

        rows.append(dict(
            target_type=tt, n=len(y), min=y.min(), max=y.max(),
            mean=y.mean(), std=y.std(), skew=y.skew(),
            iqr_lo=lo, iqr_hi=hi, n_outliers=len(outliers),
            pct_outliers=100 * len(outliers) / len(y),
        ))

        print(f"\n--- {tt} (n={len(y)}) ---")
        print(f"  min={y.min():.4g}  max={y.max():.4g}  mean={y.mean():.4g}  "
              f"std={y.std():.4g}  skew={y.skew():.3f}")
        print(f"  IQR outlier fence: [{lo:.4g}, {hi:.4g}]  ->  "
              f"{len(outliers)}/{len(y)} rows ({100*len(outliers)/len(y):.1f}%) flagged")
        if len(outliers):
            extreme = outliers.sort_values()
            shown = pd.concat([extreme.head(3), extreme.tail(3)]).drop_duplicates()
            print(f"  Extreme values (sorted, up to 3 low + 3 high): "
                  f"{[round(v, 4) for v in shown.tolist()]}")

    print("\n--- Summary table ---")
    dist_df = pd.DataFrame(rows).set_index('target_type')
    print(dist_df.to_string())
    return dist_df


def main():
    train = pd.read_csv(TRAIN_PATH)
    print(f"Loaded train={train.shape}\n")

    _, dup_summary = duplicate_report(train)
    dist_summary = distribution_report(train)

    print("\n" + "=" * 100)
    print("Phase 0 complete. Reports above -- stopping per workflow, no modeling yet.")
    print("=" * 100)


if __name__ == "__main__":
    main()
