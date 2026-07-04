# Project Context: AISEHack 2.0 — Polymer Property Prediction

## Who's working on this
I (Sanket) am a backend AI engineer (LLM pipelines, FastAPI, LangGraph — not a
chemist or cheminformatics person). This is my first time working with SMILES,
RDKit, or molecular property prediction. Please explain *why* behind any
chemistry-specific or domain-specific step, not just the *what* — I'm learning
this domain as we go. Don't over-explain general ML concepts I'd already know
(regression, train/val splits, overfitting) — focus explanations on the
chemistry/cheminformatics-specific parts.

## The competition
- Kaggle: ANRF AISEHack 2.0 — Polymer Property Prediction
- Goal: predict two polymer properties directly from chemical structure
  - **Tg** — glass transition temperature, in °C (continuous, regression)
  - **Egc** — chain band gap, in eV (continuous, regression)
- Input representation: **SMILES** strings (text notation for chemical
  structure; for polymers here, one repeating unit, with `*` marking the
  attachment points)
- Metric: **mean R² across Tg and Egc** (not RMSE, not wMAE — R²)
- Timeline: Start 24 June 2026, final submission deadline 24 July 2026

## Data format (important — long format, not wide)
- `train.csv`: columns `smiles`, `target`, `target_type`. **6,171 rows total,
  combining BOTH properties** — i.e. each row is one (molecule, property)
  measurement, not one row per molecule with both Tg and Egc filled in. A
  given SMILES may appear once (only one property measured), twice (both), or
  not at all for the other property.
- `test.csv`: columns `id`, `smiles`, `target_type`. Public test = 1,543 rows,
  private test = 2,572 rows.
- `sample_submission.csv`: format to match exactly — `id,target`.
- `base_line_model.ipynb`: organizer-provided baseline — RDKit descriptors +
  Ridge regression. This was the floor to beat; already substantially beaten
  (see score history below).

## Confirmed competition rule (Rules tab, section 6.2.3 "Public Code Usage")
Public code/architecture scripts (e.g. a GNN or Transformer implementation
copied from a public GitHub repo) ARE allowed to use, provided:
- **No external data** is used (can't supplement train.csv with other
  polymer datasets, even similar public ones)
- **No pretrained weights, checkpoints, embeddings, or processed artifacts**
  are uploaded — rules out ChemBERTa/PolyBERT/any pretrained chemistry
  language model fine-tuning, and rules out precomputing features elsewhere
  and uploading them as a file
- The **entire pipeline must execute reproducibly inside the Kaggle
  notebook environment** — feature extraction, training, and inference all
  run fresh each execution, nothing cached/loaded from outside

Practical effect: RDKit descriptor computation (live, in-notebook) is fine.
A from-scratch GNN, even using a public reference implementation's *code*
and training its weights fresh during the run, is fine. Pretrained
chemistry-LM fine-tuning is fully off the table — don't suggest it.

## Current pipeline state (as of 2026-07-03)
scripts/baseline.py currently implements:

- **Features (8,975 pre-pruning, 1,628 post-pruning)**: RDKit descriptors
  (~210), ECFP4 + ECFP6 Morgan fingerprints (2048 bits each), MACCS keys
  (167), 2 topology features (backbone span between `*` atoms), 5 electronic
  features, 1 Tg-specific feature (backbone_rotatable_bonds — monomer only),
  1 conjugation feature (max_conjugation_path — monomer + chain) — ALL
  computed on BOTH the monomer SMILES and a trimer chain (3 repeat units
  stitched together via `*` attachment points). No topology features on chain
  (no `*` atoms after capping). ~62/6165 train SMILES fail chain building
  and fall back to monomer features.
- **Pruning**: Permutation importance (replaces gain-importance). 300-tree
  probe on fold 0, keeps features with positive permutation importance on val
  set. TG kept 1,228 / 8,975; EGC kept 672 / 8,975; union = 1,628. Unbiased
  vs fingerprint bits unlike gain importance.
- **Models**: LightGBM + XGBoost blend, each tuned separately per target
  (Tg, Egc) via Optuna (30 LGB trials / 10 XGB trials). Blend weights are
  fitted per target via OOF grid search (w∈{0.1…0.9}).
- **Validation**: 5-fold StratifiedGroupKFold, grouped by *canonicalized*
  SMILES (~70% of raw train.csv SMILES are non-canonical — must canonicalize
  before grouping or CV leaks), stratified by target_type.
- **Pre-processing**: 6 duplicate (canon SMILES, target_type) groups
  mean-merged before feature extraction (all Tg, spreads 4.9–24°C).
- **Tg log-transform**: REVERTED. Tested and confirmed to hurt Tg R²
  (-0.007 vs benchmark). No transform in current code.
- **Robustness**: None-guards on all RDKit parse calls, float32 cast to
  catch RDKit Ipc descriptor overflow before XGBoost crashes.

## Known score history (use to judge whether a change is a real improvement)
- **0.8987** — first well-tuned number. Single LGB, 20 trials, no blend,
  no chain. R²(Tg)=0.8925, R²(Egc)=0.9049.
- **0.8978** — LGB+XGB blend, 30/10 trials, WITH log-transform, no chain.
  Log-transform hurt Tg (0.8918 vs 0.8925).
- **0.9052 CV / 0.894 Kaggle** — LGB+XGB blend, 30/10 trials, NO
  log-transform, WITH chain extension (trimer), dedup merge, fitted blend
  weights, gain-importance pruning (2,952 features). R²(Tg)=0.9050,
  R²(Egc)=0.9055. Blend weights: TG=50/50, EGC=0.4 LGB/0.6 XGB.
- **0.9073 CV / ~0.896 Kaggle (est.)** — current best. Same as above but
  WITH backbone_rotatable_bonds + max_conjugation_path features AND
  permutation importance pruning (1,628 features). R²(Tg)=0.9062 (+0.0012),
  R²(Egc)=0.9084 (+0.0029). Blend weights: TG=0.2 LGB/0.8 XGB, EGC=0.3
  LGB/0.7 XGB. Egc fold std=0.0174. Runtime: 4:50. CV-to-Kaggle gap: ~0.011
  (consistent across runs — likely distribution shift).
- **Observed fold-to-fold noise floor**: Tg std ~0.007, Egc std ~0.015.
  Differences < 0.01–0.02 in mean R² may not be reliably distinguishable
  from CV noise on a single seed.

## Hard-learned operational rules — follow these strictly
1. **Before running anything with a long expected runtime, print/confirm the
   actual values of N_TRIALS and N_XGB_TRIALS (or equivalent config) and
   give a runtime estimate. Wait for go-ahead before executing.** We've hit
   this exact mistake twice: a "final" run turned out to still be using
   smoke-test trial counts (N_TRIALS=2), producing misleading results that
   looked like real regressions.
2. **Only implement the specific changes requested in a given prompt. Do not
   add features, swap models, change trial counts, or start new model
   families unless explicitly asked — even if it seems like a good idea.**
   This happened once already (an XGB blend + trial count increase got
   silently added when 3 specific bug fixes were requested instead). If
   something seems worth doing but wasn't asked for, propose it and wait
   rather than doing it.
3. **When a metric changes after a code change, show the actual code lines
   responsible (e.g. the exact r2_score() call and its arguments) rather
   than just asserting the change is correct or is "expected noise."** We've
   caught real ambiguity this way before (verifying original-units vs
   log-space scoring) — a plausible-sounding explanation isn't a substitute
   for showing the lines.
4. **If a decision is close/ambiguous (e.g. keep vs revert a change), say so
   explicitly rather than silently picking a side.**

## GNN: deferred, not abandoned
Explored self-supervised pretraining + multi-task GNN as an "innovative"
direction (contrastive pretraining on train+test SMILES structure, no labels
needed, fully compliant since nothing external is loaded — then fine-tune
with two heads for Tg/Egc). Decided to shelve this for now and push the
GBM approach further first. Explicitly decided AGAINST adding CatBoost or
any third ensemble member as a "quick win" — judged as low-margin relative
to effort. If GNN is revisited later: keep it as a separate script during
development (different paradigm — PyTorch training loop vs sklearn-style
.fit()), share the same CV fold assignments (same seed, same
canonical-SMILES groups) so predictions can be honestly blended with the
GBM's, and only merge into one Kaggle-submittable notebook at the end (rule
6.2.3 requires the final submission to run as a single reproducible
notebook — the script split is a dev-time convenience only). Library
leaning: plain PyTorch with hand-written message passing over PyTorch
Geometric, given the earlier RDKit DLL block on Windows — torch-geometric's
companion packages (torch-scatter, torch-sparse) carry similar install risk.

## Realistic score ceiling — grounded in external research, not a guess
Researched the closely related NeurIPS Open Polymer Prediction 2025
competition (same Tg property, similar SMILES-to-property task) to
calibrate expectations rather than assume 0.93 is reachable by default:
- Top solutions on that competition got their real gains from external data
  (not allowed for us), polymer-specific structural feature engineering,
  and handling a documented distribution-shift issue in Tg labels
  specifically — not from more exotic model architectures. Multiple model
  types converged on similar scores there; ensembling mattered more than
  model complexity.
- **Realistic target for the current GBM-only architecture: 0.90–0.92**,
  not 0.93. Going from ~0.90 to 0.93 R² means cutting unexplained variance
  by ~30% — a big ask from tuning/blending alone, against our measured
  fold-noise floor (std 0.006–0.014). 0.93 isn't ruled out, but shouldn't
  be the assumed baseline expectation for planning purposes. Some of the
  remaining gap may reflect real label-measurement noise, not a fixable
  modeling gap — can't confirm this for our specific dataset, but it's a
  documented issue in this exact problem domain.
- **Chain extension (DONE)**: stitching 3 repeat units together before feature
  extraction. Drove Tg R² from 0.8925 → 0.9050 (+0.0125). Implemented via
  `_build_chain()` in baseline.py — connects units via `*` attachment points,
  removes junction `*` atoms, caps terminals with implicit H.

## Prioritized idea list for further score improvement (current plan)
- **DONE**: Tier 1 (dedup merge, fitted blend weights, log-transform reverted).
- **DONE**: Chain extension (trimer). Drove Tg from 0.8925 → 0.9050.
- **DONE**: backbone_rotatable_bonds + max_conjugation_path + permutation
  importance pruning. Drove CV from 0.9052 → 0.9073 (+0.0021). Egc gained
  more (+0.0029) than Tg (+0.0012) — max_conjugation_path working as
  expected. Feature count dropped 2,952 → 1,628 and model improved.
- **Next**: multi-seed CV (2-3 seeds averaged) — required to confirm whether
  any future change is real vs. fold-noise, given Egc std=0.0174.
- **Deferred**: GNN (self-supervised + multi-task), see above. CatBoost /
  third ensemble member explicitly rejected as low-margin.

## Working style preferences
- Prefer plain `.py` scripts over Jupyter notebooks (no Jupyter installed;
  also easier to iterate/rerun cleanly).
- Before any nontrivial architecture decision or library choice, briefly
  explain the tradeoff and check in, rather than silently picking one.
- I want to actually understand each step, not just get working code — treat
  explanations as part of the deliverable, not an afterthought.
- When something surprising happens (score drop, identical params across
  runs, etc.), explain the likely mechanism using specifics from the actual
  output/code — not a generic explanation that could apply to any run.