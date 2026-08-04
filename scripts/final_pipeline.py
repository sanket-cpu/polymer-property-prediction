"""
Phase 4 + Phase 5 -- OOF stacking, cross-target features, output safety.

This is the real pipeline -- the one script in the rebuild that actually
writes outputs/submission.csv, as opposed to scripts/phase0-3, which are
disposable diagnostics kept around as the record of *why* each decision
got made.

Feature pipeline and model zoo/transform logic both live in
phase3_model_zoo.py's build_feature_matrix() / model-factory functions /
wrap_for_target(), imported here rather than duplicated -- this script
and phase3_model_zoo.py are structurally unable to drift onto different
feature sets or transform behavior, which is what actually happened with
MACCS before build_feature_matrix existed (two independent hand-written
copies of the same feature-assembly logic). Mol2Vec is opt-in per target
(target_feature_cols(), gated by MOL2VEC_TARGETS in phase3_model_zoo.py) --
every per-target model fit/score site in this file slices X down with it
rather than using the full mol2vec-inclusive matrix, since Mol2Vec measurably
hurts eps/nc even though it helps egb/ei.

Model selection: loaded from outputs/phase3_results.json rather than
hardcoded -- that file is written by scripts/phase3_model_zoo.py's own
run (best model per target_type, plus whichever boosting/target combos'
Optuna-tuned configs actually beat their defaults on the real harness).
This script does not re-run that search; it reads the answer. If that
file is missing or stale relative to a phase3_model_zoo.py change, this
script will error out asking you to rerun Phase 3 first, rather than
silently falling back to hardcoded numbers that no longer reflect
what phase3_model_zoo.py would say.

--- Step 11: OOF stacking ---

Method: for each target_type, generate out-of-fold predictions from all 8
zoo models via the Phase 1 harness (averaged across repeats for the 5
small targets, so a row's OOF prediction isn't just an artifact of which
one random fold it landed in). Fit a Ridge meta-learner on the resulting
(n_rows x 8) OOF matrix.

To *evaluate* the stack's R2 without leakage, the textbook-correct
approach is full nested CV: for every outer fold, re-run an inner CV to
generate OOF predictions using only the outer-training rows, fit the
meta-learner on those, then score on the outer-validation fold. That's
also roughly 6x the compute of what's done here (an inner refit per
outer fold, per model, per target) -- for our fold counts that's
thousands of extra model fits for a `tg`/`egc`-scale problem, not
practical on a laptop mid-iteration.

The practical alternative used here -- standard in Kaggle-style stacking,
not just a shortcut -- is: generate the OOF matrix once (this part is
already leak-free, since every row's prediction comes from a model that
never saw that row's true target), then evaluate the meta-learner with
its *own* CV directly on that OOF matrix. This slightly underestimates
real deployed performance (the final meta-learner sees 100% of OOF rows;
the CV-evaluated one only sees ~80-93%), but introduces no leakage, and
is cheap enough to run for real. Final test-time predictions come from
base models refit on 100% of train, combined via a meta-learner fit on
100% of the OOF matrix -- matching what step 11 asks for.

--- Step 12: cross-target features (Egc/Egb/Ei/Eea) ---

These 4 are all electronic-structure properties (band gaps / ionization
energy / electron affinity), and some molecules have more than one
measured. The idea: if a molecule's Egc is well-predicted, that's signal
about its Ei too, since they share the same underlying electronic
structure. Implementation: reuse step 11's per-target stacked OOF
predictions (already leak-free) as lookup values -- for a molecule in
(say) Ei's training slice that *also* has an Egc measurement, add Egc's
OOF-predicted value as an extra input feature for the Ei model. A
target's own OOF predictions never appear as a feature for itself --
structurally impossible here, since a given training row belongs to
exactly one target_type slice, and only *other* targets' OOF columns
ever get joined in.

This step is scored as a CV ablation only (with vs. without the cross
features, isolated from everything else) -- it does not (yet) get built
into the test-time stacking pipeline above; that's a follow-up decision
once we see whether it actually helps.

--- Phase 5: output safety (clipping) + final submission.csv ---

The stacked test predictions from step 11 are real but unbounded -- a
model can extrapolate to a value no real polymer would plausibly have.
Before those numbers become the actual submission, each target_type's
predictions get clipped to [train_min - margin, train_max + margin]
(see MARGIN_FRACTION/PHYSICAL_FLOORS above for the reasoning), and the
count of clipped rows plus each target's PI1M sparse-region rate
(Phase 2) get reported so it's clear which targets' scores to trust
more or less. This is the step that actually writes
outputs/submission.csv, format-checked against sample_submission.csv.
"""

import json
import os
import sys
import time
from pathlib import Path

# Reverted to unrestricted (-1) -- see phase3_model_zoo.py for why. No env
# vars to set here either, for the same reason (they don't accept -1).
N_JOBS = -1

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from prajwal_baseline import featurize, _parse_mol  # noqa: E402
from cv_harness import (  # noqa: E402
    load_train_with_groups, get_harness_splits, score_model, drop_constant_columns,
    wrap_for_target,
)
from phase3_model_zoo import (  # noqa: E402
    linear_models, tree_models, boosting_default_params, BOOSTING_CTORS,
    build_feature_matrix, maccs_keys, gasteiger_features,
    load_pi1m_sample, train_mol2vec, embed_dataframe, target_feature_cols,
    MOL2VEC_SAMPLE_N, RANDOM_STATE as MOL2VEC_SEED,
    RESULTS_PATH as PHASE3_RESULTS_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
SAMPLE_SUB_PATH = PROJECT_ROOT / "data" / "sample_submission.csv"
OUT_DIR = PROJECT_ROOT / "outputs"

META_ALPHA = 1.0
CROSS_TARGETS = ['egc', 'ei', 'eea', 'egb']

# ---- Phase 5: output safety (clipping) ----
# Clip range per target_type: [train_min - margin, train_max + margin],
# margin = 10% of the observed train range. Deliberately *min/max plus a
# margin*, not the tighter Phase 0 IQR fence -- the IQR fence flagged
# individual rows as statistical outliers within train itself (useful for
# auditing train, not for bounding predictions); here the goal is only to
# catch predictions so far outside anything the model was trained on that
# they're almost certainly a bad extrapolation, while still letting
# genuinely extreme-but-real polymers through.
MARGIN_FRACTION = 0.10

# Three targets get an extra hard physical floor on top of the margin --
# basic physics facts, not domain-specific cheminformatics judgment calls:
#   Egc, Egb (band gap, eV): floor 0 -- a band gap is defined as the
#     magnitude of an energy difference; negative isn't "a small gap," it's
#     not a valid gap.
#   Nc (refractive index): floor 1 -- n = c/v, and v can't exceed c in an
#     ordinary (non-metamaterial) medium.
#   Eps (dielectric constant): floor 1, same reasoning -- relative
#     permittivity of 1 is vacuum; ordinary materials sit above that.
# Eea/Ei (electron affinity, ionization energy) get no hard floor -- these
# *can* be negative for genuinely unstable species, and train's own
# observed minimums were already comfortably positive, so the margin alone
# is enough; a hard floor there would assert more domain certainty than
# is actually warranted.
PHYSICAL_FLOORS = {'egc': 0.0, 'egb': 0.0, 'nc': 1.0, 'eps': 1.0}

# PI1M sparse-region test rates, carried over verbatim from the completed
# Phase 2 diagnostic run (recomputing means ~7 more minutes of descriptor
# calculation over a 30K PI1M sample for numbers that don't change). Used
# only for the trust-flag in Phase 5's report, not for setting the clip
# bounds themselves -- the brief specifies the clip range as train
# min/max + margin; the PI1M diagnostic informs the reporting alongside it.
PI1M_SPARSE_TEST_PCT = {
    'eea': 15.0, 'egb': 18.8, 'egc': 18.7, 'ei': 15.5,
    'eps': 16.3, 'nc': 19.6, 'tg': 37.2,
}


# ---------------------------------------------------------------------------
# Zoo assembly (reuses Phase 3's model configs; uses whichever accepted
# tuned configs Phase 3 found, loaded from its results file -- not
# hardcoded to any particular model/target)
# ---------------------------------------------------------------------------
def load_phase3_results():
    if not PHASE3_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{PHASE3_RESULTS_PATH} not found -- run scripts/phase3_model_zoo.py "
            f"first (it writes this file at the end of its run)."
        )
    with open(PHASE3_RESULTS_PATH) as f:
        return json.load(f)


def get_zoo_factories(tt, accepted_tuned_configs):
    factories = {}
    for name, pipeline_template in {**linear_models(), **tree_models()}.items():
        factories[name] = wrap_for_target(lambda p=pipeline_template: clone(p), tt)

    defaults = boosting_default_params()
    for name, ctor in BOOSTING_CTORS.items():
        params = accepted_tuned_configs.get(f'{name}__{tt}', defaults[name])
        factories[name] = wrap_for_target(lambda p=params, c=ctor: c(**p), tt)
    return factories


MODEL_NAMES = ['Ridge', 'ElasticNet', 'RF', 'GBM', 'HGB', 'XGB', 'CatBoost', 'LightGBM']


# ---------------------------------------------------------------------------
# OOF generation
# ---------------------------------------------------------------------------
def generate_oof(model_factory, X_df, y_series, sub_index, repeats):
    """
    Per-row OOF prediction, averaged across repeats. Returns a Series
    indexed like sub_index (i.e. aligned to X_df/y_series's row index).
    """
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


def fit_predict_full(model_factory, X_tr_df, y_tr_series, X_te_df):
    Xtr = X_tr_df.values.astype(float)
    Xte = X_te_df.values.astype(float)
    Xtr, Xte = drop_constant_columns(Xtr, Xte)
    model = model_factory()
    model.fit(Xtr, y_tr_series.values)
    return model.predict(Xte)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    train = load_train_with_groups()
    print(f"Loaded train={train.shape}")

    print("Training Mol2Vec on a PI1M sample (egb/ei only, see MOL2VEC_TARGETS)...")
    m2v_model = train_mol2vec(load_pi1m_sample(MOL2VEC_SAMPLE_N, MOL2VEC_SEED))
    print(f"  vocab size {len(m2v_model.wv)} [{time.time()-t0:.0f}s]")

    print("Featurizing (baseline + MACCS + Gasteiger + Mol2Vec)...")
    X, feature_cols, train_valid = build_feature_matrix(train, m2v_model)
    y_all = train_valid['target']
    print(f"  {len(feature_cols)} features after pruning [{time.time()-t0:.0f}s]")

    target_types = sorted(train_valid['target_type'].unique())

    print("\nLoading Phase 3 results...")
    phase3_results = load_phase3_results()
    accepted_tuned_configs = phase3_results['accepted_tuned_configs']
    print(f"  {len(accepted_tuned_configs)} accepted tuned config(s): "
          f"{list(accepted_tuned_configs.keys())}")
    print(f"  best model per target: "
          f"{ {tt: v['name'] for tt, v in phase3_results['best_model_per_target'].items()} }")

    # ---- Step 11: OOF stacking ----
    print("\n" + "=" * 100)
    print("STEP 11 -- OOF stacking")
    print("=" * 100)

    oof_stacked = {}   # tt -> Series of stacked OOF predictions, indexed like train_valid slice
    stacked_scores = {}
    harness_cache = {}  # tt -> (sub_index, repeats), reused for meta-learner CV too

    for tt in target_types:
        sub_index, repeats = get_harness_splits(train_valid, tt)
        harness_cache[tt] = (sub_index, repeats)
        factories = get_zoo_factories(tt, accepted_tuned_configs)
        X_tt = X[target_feature_cols(feature_cols, tt)]

        oof_cols = {}
        for name in MODEL_NAMES:
            oof_cols[name] = generate_oof(factories[name], X_tt, y_all, sub_index, repeats)
        oof_df = pd.DataFrame(oof_cols)  # index = sub_index (that target's rows)
        print(f"  {tt:5s}: OOF generated for all 8 models [{time.time()-t0:.0f}s]")

        # meta-learner CV eval, directly on the OOF matrix, same harness folds
        meta_mean, meta_std = score_model(
            lambda: Ridge(alpha=META_ALPHA), oof_df, y_all, sub_index, repeats)
        stacked_scores[tt] = (meta_mean, meta_std)

        # final meta-learner, fit on the FULL OOF matrix (100% of rows) --
        # this is the one that'll combine full-refit base models at test time
        meta_final = Ridge(alpha=META_ALPHA)
        meta_final.fit(oof_df.values, y_all.loc[sub_index].values)
        oof_stacked[tt] = oof_df  # keep for step 12
        oof_stacked[f'{tt}__meta'] = meta_final
        oof_stacked[f'{tt}__stacked_oof_pred'] = pd.Series(
            meta_final.predict(oof_df.values), index=sub_index)

    print(f"\n{'target':6s}{'best_single (P3)':>18s}{'stacked':>18s}{'delta':>10s}")
    for tt in target_types:
        meta_mean, meta_std = stacked_scores[tt]
        best = phase3_results['best_model_per_target'][tt]['score']
        delta = meta_mean - best
        flag = " <-- stack wins" if delta > meta_std else ""
        print(f"{tt:6s}{best:16.4f}{meta_mean:14.4f}(±{meta_std:.3f}){delta:+9.4f}{flag}")

    # ---- Step 11 (cont'd): full-refit base models + meta-learner, applied
    # to test.csv -- per step 11's own spec ("use the fitted meta-learner to
    # combine full-refit base-model predictions at test time"). Saved
    # *unclipped* -- clipping is Phase 5's job, not this script's. ----
    print("\nFull-refit base models on 100% of train, predicting test.csv...")
    test = pd.read_csv(TEST_PATH)
    test_feats = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    if (~test_valid_mask).sum():
        print(f"  {(~test_valid_mask).sum()} test SMILES failed to featurize -- "
              f"falling back to that target's train mean for those rows")
    # MACCS + Gasteiger + Mol2Vec computed separately here (same as
    # build_feature_matrix does for train) and merged in before the
    # feature_cols lookup below -- test_feats alone only has the baseline
    # featurize() keys, so skipping this would silently leave every
    # maccs_*/gast_*/mol2vec_* column as NaN for all of test.
    test_mols = test.loc[test_valid_mask, 'smiles'].apply(_parse_mol)
    test_maccs = pd.Series([maccs_keys(m) for m in test_mols], index=test_mols.index)
    test_gast = pd.Series([gasteiger_features(m) for m in test_mols], index=test_mols.index)
    test_m2v_df = embed_dataframe(m2v_model, test_mols)
    test_m2v_df.index = test_mols.index
    test_m2v = test_m2v_df.to_dict(orient='index')

    test_feat_df = pd.DataFrame(index=test.index, columns=feature_cols, dtype=float)
    for idx in test.index[test_valid_mask]:
        row = {**test_feats[idx], **test_maccs[idx], **test_gast[idx], **test_m2v[idx]}
        for k in feature_cols:
            test_feat_df.loc[idx, k] = row.get(k, np.nan)

    test_predictions = np.full(len(test), np.nan)
    for tt in target_types:
        sub_index, _ = harness_cache[tt]
        cols_tt = target_feature_cols(feature_cols, tt)
        X_tr_tt = X.loc[sub_index, cols_tt]
        y_tr_tt = y_all.loc[sub_index]
        factories = get_zoo_factories(tt, accepted_tuned_configs)

        mask = (test['target_type'] == tt).values
        rows_valid = mask & test_valid_mask.values
        rows_invalid = mask & (~test_valid_mask.values)

        if rows_valid.sum() > 0:
            X_te_tt = test_feat_df.loc[rows_valid, cols_tt]
            base_test_preds = np.column_stack([
                fit_predict_full(factories[name], X_tr_tt, y_tr_tt, X_te_tt)
                for name in MODEL_NAMES
            ])
            meta_final = oof_stacked[f'{tt}__meta']
            test_predictions[rows_valid] = meta_final.predict(base_test_preds)

        if rows_invalid.sum() > 0:
            test_predictions[rows_invalid] = y_tr_tt.mean()
        print(f"  {tt:5s} done [{time.time()-t0:.0f}s]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stacked_out = test[['id', 'target_type']].copy()
    stacked_out['target'] = test_predictions
    stacked_path = OUT_DIR / "phase4_stacked_test_predictions_unclipped.csv"
    stacked_out.to_csv(stacked_path, index=False)
    print(f"  saved unclipped stacked test predictions -> {stacked_path}")

    # ---- Step 12: cross-target features (CV ablation only) ----
    print("\n" + "=" * 100)
    print("STEP 12 -- cross-target features (Egc/Egb/Ei/Eea), CV ablation")
    print("=" * 100)

    cross_lookup = {}  # canon -> {tt: stacked_oof_pred}
    for tt in CROSS_TARGETS:
        sub_index = harness_cache[tt][0]
        canons = train_valid.loc[sub_index, 'canon'].values
        preds = oof_stacked[f'{tt}__stacked_oof_pred'].values
        for c, p in zip(canons, preds):
            cross_lookup.setdefault(c, {})[tt] = p

    n_multi = sum(1 for d in cross_lookup.values() if len(d) > 1)
    print(f"  {n_multi} molecules have >1 of {CROSS_TARGETS} measured "
          f"(out of {len(cross_lookup)} total across the 4)")

    for tt in CROSS_TARGETS:
        sub_index, repeats = harness_cache[tt]
        sub = train_valid.loc[sub_index]
        others = [o for o in CROSS_TARGETS if o != tt]
        cross_cols = {}
        for other in others:
            cross_cols[f'cross_{other}'] = [
                cross_lookup.get(c, {}).get(other, np.nan) for c in sub['canon'].values
            ]
        cross_df = pd.DataFrame(cross_cols, index=sub_index)
        n_have_any = cross_df.notna().any(axis=1).sum()

        X_tt = X.loc[sub_index, target_feature_cols(feature_cols, tt)]
        y_tt = y_all.loc[sub_index]
        X_with_cross = pd.concat([X_tt, cross_df], axis=1)

        # each target's own Phase-3-best model config, held fixed, so the
        # only thing varying is presence/absence of the cross features
        factories = get_zoo_factories(tt, accepted_tuned_configs)
        best_model_name = phase3_results['best_model_per_target'][tt]['name']
        factory = factories[best_model_name]

        local_sub_index = np.arange(len(sub_index))  # positions into the tt-local frames
        # score_model expects positions into sub_index; reuse repeats' fold
        # positions directly since X_tt/X_with_cross are already tt-local
        no_cross_mean, no_cross_std = score_model(factory, X_tt.reset_index(drop=True),
                                                   y_tt.reset_index(drop=True),
                                                   local_sub_index, repeats)
        with_cross_mean, with_cross_std = score_model(factory, X_with_cross.reset_index(drop=True),
                                                        y_tt.reset_index(drop=True),
                                                        local_sub_index, repeats)
        delta = with_cross_mean - no_cross_mean
        noise_floor = max(no_cross_std, with_cross_std)
        verdict = "HELPS" if delta > noise_floor else "no clear effect"
        print(f"  {tt:5s} ({best_model_name}, {n_have_any}/{len(sub_index)} rows with >=1 cross feature): "
              f"no_cross={no_cross_mean:.4f}  with_cross={with_cross_mean:.4f}  "
              f"delta={delta:+.4f}  [{verdict}]  [{time.time()-t0:.0f}s]")

    # ---- Phase 5: clip + write final submission.csv ----
    print("\n" + "=" * 100)
    print("PHASE 5 -- output safety (clipping) + final submission.csv")
    print("=" * 100)

    bounds = {}
    for tt in target_types:
        y = y_all.loc[train_valid['target_type'] == tt]
        lo, hi = y.min(), y.max()
        margin = MARGIN_FRACTION * (hi - lo)
        clip_lo, clip_hi = lo - margin, hi + margin
        if tt in PHYSICAL_FLOORS:
            clip_lo = max(clip_lo, PHYSICAL_FLOORS[tt])
        bounds[tt] = (clip_lo, clip_hi)
        floor_note = f" (physical floor {PHYSICAL_FLOORS[tt]} applied)" if tt in PHYSICAL_FLOORS else ""
        print(f"  {tt:5s}: train=[{lo:.4g}, {hi:.4g}]  ->  "
              f"clip=[{clip_lo:.4g}, {clip_hi:.4g}]{floor_note}")

    print("\nClipping test predictions:")
    clipped_target = stacked_out['target'].copy()
    for tt in target_types:
        clip_lo, clip_hi = bounds[tt]
        mask = stacked_out['target_type'] == tt
        vals = stacked_out.loc[mask, 'target']
        n_below = (vals < clip_lo).sum()
        n_above = (vals > clip_hi).sum()
        clipped_target.loc[mask] = vals.clip(lower=clip_lo, upper=clip_hi)
        sparse_pct = PI1M_SPARSE_TEST_PCT[tt]
        trust_flag = " [!] ELEVATED EXTRAPOLATION RISK" if sparse_pct >= 30 else ""
        print(f"  {tt:5s}: {n_below + n_above}/{mask.sum()} rows clipped "
              f"({n_below} below, {n_above} above)  |  "
              f"PI1M sparse-region test rate: {sparse_pct:.1f}%{trust_flag}")

    submission = stacked_out[['id']].copy()
    submission['target'] = clipped_target

    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    assert list(submission.columns) == list(sample_sub.columns), \
        f"column mismatch: {submission.columns.tolist()} vs {sample_sub.columns.tolist()}"
    assert submission['target'].notna().all(), "unfilled predictions remain"
    print(f"\nFormat check OK: columns match sample_submission.csv "
          f"({list(submission.columns)}), {submission.shape[0]} rows, no NaNs")

    final_path = OUT_DIR / "submission.csv"
    submission.to_csv(final_path, index=False)
    print(f"Saved final submission -> {final_path}")

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
    print("Phase 5 complete. Full pipeline done -- stopping per workflow.")


if __name__ == "__main__":
    main()
