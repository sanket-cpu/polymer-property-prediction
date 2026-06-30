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
  Ridge regression. Treat this as the floor to beat, not a starting point to
  necessarily build directly on top of.

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

## Agreed workflow (work through these in order, don't skip ahead)
1. **Explore the raw data** — no modeling yet.
   - Count rows by `target_type` (Tg vs Egc) — check balance.
   - Check for SMILES overlap between the two target types.
   - Distribution (min/max/mean) of Tg and Egc values — sanity check for
     anything implausible.
   - Validate every SMILES (train + test) actually parses in RDKit; flag any
     that fail — this matters, broken SMILES break feature extraction later.
2. **Run the organizer's baseline notebook as-is.** Confirm it produces a
   submission.csv matching `sample_submission.csv` format exactly. This step
   exists purely to prove the end-to-end submission pipeline works before
   building anything fancier on top of it.
3. **Build a proper validation harness** — this is NOT the Kaggle test set,
   it's our own held-out split from train.csv, used to estimate model quality
   before submitting.
   - **Group by unique SMILES** when splitting (a SMILES appearing for both
     Tg and Egc should never have its two rows land in different folds — that's
     leakage).
   - **Stratify by target_type** so each fold has a representative mix of Tg
     and Egc rows.
   - Use **5-fold cross-validation** (not a single 80/20 split) — given only
     ~6,171 rows split across two targets, a single split is too noisy to
     trust.
   - Score using **R² per target, then mean of the two** — matching the
     actual competition metric exactly, so our internal numbers mean
     something.
4. **Feature engineering** — turn SMILES into numbers a model can use.
   - Start with RDKit's full descriptor set (200+, broader than whatever the
     baseline used) plus Morgan fingerprints, concatenated.
   - Likely candidate model on top: gradient-boosted trees (LightGBM/XGBoost)
     rather than Ridge, given typical performance on descriptor-style tabular
     features.
   - A GNN (graph neural network) on the actual molecular graph is the
     stronger longer-term direction (more faithful to real chemistry than a
     flattened SMILES string), but is a stretch goal given time constraints —
     don't start here, build up to it.

## Working style preferences
- Prefer plain `.py` scripts over Jupyter notebooks (no Jupyter installed;
  also easier to iterate/rerun cleanly).
- Don't jump ahead to later steps (e.g. don't start GNN work while step 1
  exploration is still incomplete).
- Before any nontrivial architecture decision or library choice, briefly
  explain the tradeoff and check in, rather than silently picking one.
- I want to actually understand each step, not just get working code — treat
  explanations as part of the deliverable, not an afterthought.