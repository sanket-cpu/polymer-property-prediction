"""
Phase 1 -- CV harness. Every model-comparison / tuning decision from Phase 3
onward is scored through this module, not an ad-hoc split.

Splitter choice (per Phase 0 findings):
  Phase 0 found real exact-duplicate molecules (same canonical SMILES,
  different rows) inside the `tg` slice -- 4 groups / 8 rows, all with
  *conflicting* target values (repeat measurements). If a plain KFold
  happened to split such a pair across train/val, the model could get
  most of the way to predicting the held-out value just by having seen
  the (near-identical) target for the same molecule during training --
  that's leakage, and it inflates CV R2 without reflecting real
  generalization. So every split here is a GroupKFold grouped on exact
  canonical SMILES, applied uniformly across all 7 target_types (not
  just `tg`) for consistency -- for the target_types with zero duplicates,
  every group is just size 1, so GroupKFold behaves identically to a
  shuffled KFold there; it only changes behavior where duplicates exist.

Two harness variants, both GroupKFold-based:
  - `get_cv_splits`: single 5-fold split. Used for tg/egc (thousands of
    rows each -- one 5-fold split is already stable/low-variance).
  - `get_repeated_cv_splits`: 5-fold x 3 seeds (15 folds total, averaged).
    Used for the 5 small targets (egb/ei/eea/eps/nc, 221-337 rows each) --
    with so few rows, which molecules happen to land in the validation
    fold materially swings the R2 of a single split, so any model-
    selection or tuning call on these 5 must average over multiple
    random fold assignments to not be chasing split noise.

`n_repeats=1` for the large targets is not a special case in the code --
it's just `get_repeated_cv_splits` with one repeat, so both call sites
share one implementation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"

SMALL_TARGETS = {'egb', 'ei', 'eea', 'eps', 'nc'}
LARGE_TARGETS = {'tg', 'egc'}

N_SPLITS = 5
N_REPEATS_SMALL = 3
BASE_SEED = 42


def canonical_smiles(smiles):
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_train_with_groups():
    """train.csv with an added `canon` column (exact canonical SMILES,
    used as the GroupKFold group key). Rows that fail to parse even with
    the '*'->'C' fallback are dropped -- Phase 0 found none of these in
    train, but this guards later phases against silent NaN groups."""
    train = pd.read_csv(TRAIN_PATH)
    train['canon'] = train['smiles'].apply(canonical_smiles)
    n_bad = train['canon'].isna().sum()
    if n_bad:
        print(f"[cv_harness] dropping {n_bad} unparsable train rows")
        train = train[train['canon'].notna()].reset_index(drop=True)
    return train


def get_repeated_cv_splits(df, target_type, n_repeats, n_splits=N_SPLITS, base_seed=BASE_SEED):
    """
    Returns (sub_index, repeats) where:
      sub_index : the row indices (into `df`) belonging to this target_type,
                  in the same order used to build each split's train/val
                  index arrays below.
      repeats   : list of length n_repeats, each element a list of
                  n_splits (train_pos, val_pos) tuples -- positions are
                  into sub_index (i.e. 0..len(sub_index)-1), not into df
                  directly. Use `sub_index[train_pos]` to recover df indices.
    """
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values

    repeats = []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + r)
        repeats.append(list(gkf.split(np.zeros(len(sub_index)), groups=groups)))
    return sub_index, repeats


def get_cv_splits(df, target_type, n_splits=N_SPLITS, random_state=BASE_SEED):
    """Single GroupKFold split -- thin wrapper for the large-target case
    (tg/egc), so call sites don't need to special-case n_repeats=1."""
    sub_index, repeats = get_repeated_cv_splits(
        df, target_type, n_repeats=1, n_splits=n_splits, base_seed=random_state)
    return sub_index, repeats[0]


def drop_constant_columns(Xtr, Xva):
    """
    Some GBM implementations (HistGradientBoostingRegressor in particular)
    crash outright on a column that's constant within a training fold, even
    if it varies across the full dataset -- its histogram binning step needs
    >=2 distinct values per column. This is common on the small-target
    slices here: a 40-something-row fold can easily land on all-zero for a
    rare one-hot-ish descriptor/fingerprint column that's non-constant over
    the full ~220-4000 row slice. Drop those columns for this fold only
    (decided from the training split, same mask applied to validation).
    """
    with np.errstate(invalid='ignore'):
        stds = np.nanstd(Xtr, axis=0)
    keep = (stds > 0) & ~np.isnan(stds)
    return Xtr[:, keep], Xva[:, keep]


def n_repeats_for(target_type):
    return N_REPEATS_SMALL if target_type in SMALL_TARGETS else 1


def get_harness_splits(df, target_type):
    """The single entry point later phases should call: picks single vs.
    repeated CV automatically based on target_type, per the Phase 1 rule."""
    return get_repeated_cv_splits(df, target_type, n_repeats=n_repeats_for(target_type))


def get_search_splits(df, target_type, n_splits=3, random_state=BASE_SEED):
    """
    Cheap 3-fold GroupKFold splitter for Optuna's *inner* search loop only
    (Phase 3). Deliberately smaller/cheaper than get_harness_splits (which
    is 5-fold, or 5-fold x3 seeds for the small targets) so a 15-trial
    search stays fast. Still grouped on canonical SMILES for the same
    leakage reason as the main harness.

    Never use this to decide whether a tuned config beats the default --
    that comparison must run on get_harness_splits, or a config that
    happened to overfit the 3-fold search split would look like a genuine
    improvement.
    """
    sub_index, repeats = get_repeated_cv_splits(
        df, target_type, n_repeats=1, n_splits=n_splits, base_seed=random_state)
    return sub_index, repeats[0]


def score_model(model_factory, X_df, y_series, sub_index, repeats):
    """
    Runs `model_factory()` (a zero-arg callable returning a fresh,
    unfitted estimator/pipeline -- fresh per fold, so no state leaks
    across folds) over every fold of every repeat, scores with R2, and
    returns (mean, std) across all folds.

    Shared by every phase from here on (Phase 2's two feature scripts each
    had their own copy of this loop; centralizing it here so the
    per-fold constant-column fix only has to exist -- and be remembered --
    in one place).
    """
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
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


# eps (dielectric constant) and ei (ionization energy) are the two most
# right-skewed targets (Phase 0 skew: eps=1.22, ei=0.79, vs tg's 0.09) and
# the two weakest CV scores after the MACCS/Optuna round -- neither has
# ever had a target transform tried. Both are comfortably positive in
# train (eps min=2.61, ei min=4.03), so a plain log/exp is safe -- no need
# for log1p/expm1's near-zero handling.
TARGET_TRANSFORMS = {
    'eps': (np.log, np.exp),
    'ei': (np.log, np.exp),
}


def wrap_for_target(model_factory, target_type):
    """
    Wraps a model factory so the target gets transformed before fit and
    inverse-transformed after predict, for targets in TARGET_TRANSFORMS --
    a no-op passthrough for every other target. Uses sklearn's built-in
    TransformedTargetRegressor rather than hand-rolling it, so this is
    invisible to score_model/generate_oof/fit_predict_full: they still
    just call .fit(X, y_raw) / .predict(X) and get back real-scale
    predictions, with the transform happening entirely inside the wrapped
    model.
    """
    if target_type not in TARGET_TRANSFORMS:
        return model_factory
    func, inverse_func = TARGET_TRANSFORMS[target_type]

    def wrapped():
        return TransformedTargetRegressor(
            regressor=model_factory(), func=func, inverse_func=inverse_func)
    return wrapped


# ---------------------------------------------------------------------------
# Reporting / self-check
# ---------------------------------------------------------------------------
def _report():
    train = load_train_with_groups()
    print(f"Loaded train={train.shape} (post-parse-check)\n")

    print("=" * 100)
    print("PHASE 1 CV HARNESS -- fold structure per target_type")
    print("=" * 100)

    for tt in sorted(train['target_type'].unique()):
        n_rep = n_repeats_for(tt)
        sub_index, repeats = get_repeated_cv_splits(train, tt, n_repeats=n_rep)
        n = len(sub_index)
        groups = train.loc[sub_index, 'canon'].values
        n_dup_groups = pd.Series(groups).value_counts()
        n_dup_groups = (n_dup_groups[n_dup_groups > 1]).shape[0]

        kind = "repeated 5-fold x3 seeds" if n_rep > 1 else "single 5-fold"
        print(f"\n--- {tt} (n={n}, {'SMALL' if tt in SMALL_TARGETS else 'LARGE'} target) "
              f"-> {kind}, {n_dup_groups} duplicate SMILES group(s) in this slice ---")

        for r_idx, folds in enumerate(repeats):
            sizes = [(len(tr), len(va)) for tr, va in folds]
            print(f"  repeat {r_idx} (seed={BASE_SEED + r_idx}): "
                  f"val fold sizes = {[va for _, va in sizes]}, "
                  f"train fold sizes = {[tr for tr, _ in sizes]}")

        # leakage self-check: every duplicate-SMILES group must stay
        # entirely within train or entirely within val, for every fold
        # of every repeat
        violations = 0
        for folds in repeats:
            for tr_pos, va_pos in folds:
                tr_groups = set(groups[tr_pos])
                va_groups = set(groups[va_pos])
                violations += len(tr_groups & va_groups)
        status = "OK -- no group split across train/val" if violations == 0 else f"[!] {violations} LEAK(S)"
        print(f"  duplicate-group leakage check: {status}")

    print("\n" + "=" * 100)
    print("Phase 1 complete. Harness defined in this module (get_harness_splits) --")
    print("stopping per workflow before any model scoring.")
    print("=" * 100)


if __name__ == "__main__":
    _report()
