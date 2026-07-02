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

## Steps 1–4 status: DONE. Current pipeline state (as of latest run)
The original workflow (data exploration → baseline → validation harness →
feature engineering) is complete. scripts/baseline.py currently implements:

- **Features (~4487 pre-pruning, ~1857 post-pruning)**: RDKit descriptors
  (~210), ECFP4 + ECFP6 Morgan fingerprints (2048 bits each), MACCS keys
  (167), 2 topology features (backbone span between `*` atoms), 5 electronic
  features (aromatic ring count, aromatic atom fraction, rotatable bonds,
  sp2 atom fraction, non-aromatic double bond count — chosen for Egc/band-gap
  relevance via conjugation).
- **Models**: LightGBM + XGBoost blend (currently naive 50/50 average —
  under revision, see Tier 1 below), each tuned separately per target
  (Tg, Egc) via Optuna.
- **Validation**: 5-fold StratifiedGroupKFold, grouped by *canonicalized*
  SMILES (important: ~70% of raw train.csv SMILES are non-canonical —
  canonicalize before grouping or CV leaks), stratified by target_type.
- **Robustness**: None-guards on all RDKit parse calls (0 unparseable SMILES
  confirmed in current train/test), float32 cast to catch RDKit Ipc
  descriptor overflow before it silently becomes inf and crashes XGBoost.
- **Known data quality issue**: 6 duplicate (canonical SMILES, target_type)
  groups exist, all in Tg. 5 have small spread (a few °C, likely measurement
  noise); 1 has a 24°C spread (smiles starting
  `*C(=O)Nc1ccc(Oc2ccc(-c3ccc(Oc4ccc(NC(=O)c5ccc6c(c5` — worth resolving,
  e.g. mean-merging duplicates, see Tier 1).
- **Tg log-transform**: implemented (log1p with offset=119 to handle
  negative Tg values down to -118°C) but NOT YET CONFIRMED to actually help
  — see "Known score history" below. May be kept or reverted based on
  results.

## Known score history (use to judge whether a change is a real improvement)
- **0.8987** — first well-tuned, trustworthy number. Single model (LGB
  only), 20 real Optuna trials, RDKit descriptors + Morgan(ECFP4) + MACCS +
  topology only (no electronic features, no log-transform, no blend).
  R²(Tg)=0.8925, R²(Egc)=0.9049. **This is the benchmark to beat.**
- All LGB+XGB blend runs so far were confounded by either (a) N_TRIALS
  accidentally left at a smoke-test value (2), or (b) not yet having fitted
  blend weights / resolved duplicates / settled the log-transform question.
  Treat any blend-era CV number as provisional until a run completes with
  N_TRIALS=30, N_XGB_TRIALS=10 AND Tier 1 fixes applied.
- **Observed fold-to-fold noise floor**: Tg fold std has ranged 0.0063–0.0144
  across different runs. Differences smaller than ~0.01–0.02 in overall mean
  R² between two runs may not be reliably distinguishable from CV noise on a
  single seed. Don't over-interpret small deltas without multi-seed CV
  (averaging across 2-3 different fold-split random seeds) to confirm.

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
- **New idea, borrowed from a top solution on the related competition**:
  "chain extension" — instead of computing features on a single repeat
  unit, bond 2-3 copies of the repeat unit together (via the `*` attachment
  points) before running RDKit/fingerprint feature extraction. Real
  polymers are long chains; a single repeat unit may not fully capture
  backbone flexibility (Tg-relevant) or conjugation length (Egc-relevant)
  that only emerges over a longer stretch. Not yet implemented — high
  priority.

## IMPORTANT correction: Tier 1 is NOT actually complete
Earlier notes assumed Tier 1 (fit blend weights, merge duplicate labels,
settle log-transform) was done. Checking the actual current script:
duplicates are only detected/printed, NOT merged; the LGB/XGB blend is
still a naive 50/50 average, NOT fitted weights. Do not treat any CV score
from the current script as reflecting a completed Tier 1 — finish these
first before judging headroom or trying further ideas.

## Prioritized idea list for further score improvement (current plan)
- **0 (do first)**: actually complete Tier 1 — fit blend weights per target
  (e.g. small Ridge meta-model on out-of-fold LGB+XGB predictions, or grid
  search over blend weight, per target since Tg/Egc will likely differ);
  mean-merge the 6 duplicate-label groups; confirm log-transform keep/revert
  decision against the 0.8987 single-model benchmark with a real (non-smoke-
  test) trial count.
- **1**: chain-extension features (see above) — likely highest-value single
  addition given it's untried and grounded in a real winning solution.
- **2**: Tg-specific features — H-bond donor/acceptor count (raises Tg via
  chain-chain interaction strength), backbone-path-only rotatable bond count
  (more precise rigidity signal than current whole-molecule topology
  features); true conjugation-path-length graph feature for Egc (longest
  unbroken conjugated path, not just aromatic/sp2 counts — more faithful to
  Hückel-theory band gap reasoning than current proxies).
- **3**: switch feature pruning from LGB gain-importance (biased against
  sparse fingerprint bits) to permutation importance.
- **4**: multi-seed CV (2-3 different fold-split seeds, averaged) — required
  before trusting whether any of the above are real improvements vs. noise,
  given the observed 0.006-0.014 fold std.
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