# CLAUDE.md

## Project

AISEHack 2.0 (ANRF) — Polymer Property Prediction track, Phase 1 Kaggle qualifier.
Predicting `Tg` (glass transition temperature) and `Egc` (chain bandgap) from polymer
SMILES. Phase 1 is a qualifier, not the final — top teams advance to a Phase 2 online
round, then an offline grand finale. Deadline as told to Claude: July 24. The public
event page lists Phase 1 closing July 16 with shortlist announcements the third week
of July — verify the real date on the Kaggle Rules tab if it matters, that's the
authoritative source, not this file.

Public leaderboard as of Jul 4: our best entry sits at 0.898 (~rank 23 of 24+ visible).
Top 2 entries sit at 0.994 / 1.000, isolated by a large gap from an otherwise smooth
field below them — plausibly not legitimate under the rules below. Realistic target
through legitimate modeling is closing in on rank 3 (~0.932), not the top 2.

## Constraints — do not violate

- No external datasets, no pre-trained weights, no private artifacts. Violating this
  is disqualifying regardless of leaderboard rank — the organizers explicitly state
  leaderboard placement does not guarantee eligibility.

Kaggle-notebook reproducibility, the 12h session-time limit, and swapping data paths
to Kaggle's mount were flagged earlier and have been deliberately deprioritized —
not being worked on right now. Revisit only if/when it actually matters for how a
submission gets made.

## Decision: baseline.py only — GNN work discontinued

We built a standalone GNN (`scripts/gnn.py`) and later a GNN+LightGBM fusion script.
Auditing the fusion script found it reintroduced a severe CV-fold leak — it grouped
by raw `smiles` instead of `canon_smiles`, and baseline.py had already established
~70% of train SMILES are non-canonical — plus a rebuilt, weaker tabular pipeline
(fewer features, far less tuning, no XGBoost, single seed). That combination is why
local CV read 0.92 while the actual leaderboard score came back at 0.891, *below*
baseline.py's own ~0.898. "Does the GNN help" was never actually tested cleanly as a
result, and given the deadline, we're not spending more time finding out.

**Going forward: only `scripts/baseline.py` is being developed.** `scripts/gnn.py`
and the fusion script are kept for reference, not under active iteration.

## scripts/baseline.py — current state

LightGBM + XGBoost blend. ~210 RDKit descriptors + ECFP4 (r=2) + ECFP6 (r=3) +
MACCS + topology + electronic features, computed for both the monomer and a
stitched 3-repeat-unit chain. 5-fold `StratifiedGroupKFold` grouped by
`canon_smiles`, stratified by `target_type` — this is correct, don't change it.
Duplicate `(canon_smiles, target_type)` rows are mean-merged before anything else.
Optuna-tuned per target (LGB: 30 trials, XGB: 10), per-target blend-weight grid
search, multi-seed averaging (42/0/123). Validated: ~0.90 CV / 0.898 LB, CV and LB
agree closely — this is the trustworthy foundation, keep its correctness intact
while extending it below.

## Changes to make, in priority order

**1. Add CatBoost + a real stacker**
- Add `CatBoostRegressor` as a third per-target model, tuned via Optuna the same
  way LGB/XGB already are.
- Replace the `np.arange(0.1, 1.0, 0.1)` blend-weight grid search with an OOF
  stacking meta-learner (RidgeCV on `[lgb_oof, xgb_oof, catboost_oof]` per target).
  Score it via a nested holdout — fit on OOF from folds 1-4, evaluate on fold 5's
  OOF, rotate — rather than fitting and scoring on the same OOF rows, which would
  reintroduce the same class of optimism found in the fusion script, just smaller.

**2. Cheap hygiene fixes**
- Permutation-importance pruning uses `n_repeats=1` on a single probe fold — noisy.
  Bump to 3-5; the probe model (300 trees) is cheap, so this barely affects runtime.
- Test `N_CHAIN_UNITS=2` and `=5` against the current `3`. Quick to run, and a real
  physical hyperparameter — both Tg and Egc have chain-length-dependent convergence
  behavior in actual polymer physics, not just a knob to guess at.

**3. Diagnostic (not a code change, do before chasing more features)**
- The script already logs duplicate `(canon_smiles, target_type)` label-conflict
  spread at the top. Once OOF predictions exist, check whether the worst residuals
  cluster on those same conflicted molecules — if so, that's the data's noise
  ceiling, not a modeling gap, and tells you where not to keep pushing.

## Conventions to keep

- Tg and Egc are always modeled separately — never pool target types into one model.
- Fold groups are always `canon_smiles`, never raw `smiles`.
- Dedup: mean-merge rows sharing `(canon_smiles, target_type)` before anything else
  touches the data.