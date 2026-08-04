# Polymer Property Prediction

Solution for the ANRF AISEHack 2.0 — Polymer Property Prediction Kaggle competition, Round 2.

## Task

Predict **7 polymer properties** directly from chemical structure (SMILES strings of the repeat unit, `*` marking attachment points):

| Property | Description | Unit |
|----------|-------------|------|
| **Tg** | Glass transition temperature | °C |
| **Egc** | Chain band gap | eV |
| **Egb** | Bulk/backbone band gap | eV |
| **Eps** | Dielectric constant | — |
| **Nc** | Refractive index | — |
| **Ei** | Ionization energy | eV |
| **Eea** | Electron affinity | eV |

**Metric:** mean R² across all 7 properties.

## Data

Long format — each row is one (molecule, property) measurement, not one row per molecule with all 7 properties filled in. A given SMILES may appear for one, several, or all target types.

| File | Rows | Description |
|------|------|-------------|
| `data/train.csv` | 7,409 | `smiles`, `target`, `target_type` |
| `data/test.csv` | 4,940 | `id`, `smiles`, `target_type` |
| `data/sample_submission.csv` | 10 | Format example only (`id`, `target`) — **not** a full-length template; real row count comes from `test.csv` |
| `data/PI1M.csv` | ~1M | Unlabeled public polymer SMILES, used only for an optional self-supervised embedding (see Mol2Vec below) — no property labels |
| `data/base_line_model.ipynb` | — | Organizer-provided baseline (RDKit descriptors + Ridge) |

Target row counts are heavily imbalanced — `tg` (4,143) and `egc` (2,028) dominate; `egb`, `eps`, `nc`, `ei`, `eea` each have only 221–337 rows, which drives most of the validation-methodology decisions below.

## Approach

### Features
- **Baseline** (`prajwal_baseline.py`): full RDKit descriptor set (~200), hand-crafted physics ratios (aromaticity/conjugation/rotatable-bond fractions), a "dimer" trick (join two repeat units head-to-tail, take the *delta* in backbone-sensitive descriptors vs. the monomer — captures chain-extension effects a single unit misses), and 256-bit Morgan fingerprints. Pruned by variance + correlation thresholding, fit on train only.
- **MACCS keys** (167-bit): folded into the global feature set — didn't clear the per-target noise floor in isolation, but moved all 7 targets in the same direction, which read as signal rather than noise.
- **Gasteiger partial charges** (5 summary stats: max/min/mean/sum/std): targets `eps`/`ei` specifically, since both are polarity/charge-driven properties the mostly topological descriptor set doesn't directly capture.
- **log/exp target transform** for `eps`/`ei` (the two most right-skewed, weakest-CV targets), via `sklearn`'s `TransformedTargetRegressor`.
- **Mol2Vec** (skip-gram embedding trained fresh on a 150K PI1M SMILES sample, no property labels touched): opt-in per target, not global — an isolated ablation + a later stacked-pipeline test found it helps `egb` (+0.01–0.02 R²) but is flat-to-negative for `eps`/`nc`/`ei`. Implemented in the local dev pipeline (`phase3_model_zoo.py`, `final_pipeline.py`) but **not currently included in the Kaggle submission script** — the measured gain was small and concentrated in one target, not worth the added dependency/runtime for this round.

### Model
- **8-model zoo** per target: Ridge, ElasticNet, RandomForest, GradientBoosting, HistGradientBoosting, XGBoost, CatBoost, LightGBM.
- **Optuna** (TPE sampler, 50 trials, wide search space) tunes the 3 boosting models per target; a tuned config only replaces its default if it beats it by more than the CV noise floor on the real harness (not the cheap search split) — out of 21 model/target combos tried, only 2 cleared that bar (CatBoost on `egc` and `tg`).
- **Out-of-fold stacking**: a Ridge meta-learner combines all 8 models' leak-free OOF predictions per target — this was the single most consistent win across the pipeline, improving 6 of 7 targets.
- **Output-safety clipping**: predictions clipped to `[train_min − margin, train_max + margin]` (10% margin), plus hard physical floors (band gap ≥ 0, refractive index/dielectric constant ≥ 1).

### Validation
- **GroupKFold on canonical SMILES** — a handful of exact-duplicate molecules in `tg` have conflicting repeat measurements; grouping prevents the same molecule's two rows from splitting across train/val (leakage).
- **Repeated CV** (5-fold × 3 seeds, averaged) for the 5 small targets (221–337 rows), where which molecules land in the validation fold materially swings a single split's R². Large targets (`tg`, `egc`) use a single 5-fold split.
- Scored as **R² per target, then mean of all 7** — matching the competition metric.

### Results so far
| Milestone | Public LB |
|---|---|
| Organizer baseline (RDKit descriptors + Ridge) | — |
| Phase 0–5 rebuild (own CV harness, feature engineering, model zoo, stacking, clipping) | 0.837 |
| + MACCS keys, wider Optuna search (50 trials), Gasteiger charges, eps/ei log-transform | **0.849** |

## Project Structure

```
polymer-property-prediction/
├── data/
│   ├── train.csv, test.csv, sample_submission.csv
│   ├── PI1M.csv                     # unlabeled, used only by the optional Mol2Vec step
│   └── base_line_model.ipynb        # organizer-provided baseline
├── scripts/
│   ├── prajwal_baseline.py          # baseline featurize()/fit_feature_pruner() -- imported, not rerun
│   ├── cv_harness.py                # GroupKFold splitters, score_model, wrap_for_target (log-transform)
│   ├── phase3_model_zoo.py          # feature assembly (build_feature_matrix), model zoo, Optuna tuning
│   │                                 #   -> writes outputs/phase3_results.json
│   ├── final_pipeline.py            # OOF stacking + clipping, reads phase3_results.json
│   │                                 #   -> writes outputs/submission.csv
│   ├── phase0_diagnostics.py        # exploratory data checks (Phase 0, diagnostic only)
│   ├── phase2_fingerprints.py       # MACCS/fingerprint ablation (diagnostic only)
│   ├── phase2_pi1m.py               # Mol2Vec + PCA-density ablation vs. PI1M (diagnostic only)
│   └── sanket_baseline.py           # earlier local exploration script
├── kaggle_submission.py             # SELF-CONTAINED single-file version for the actual Kaggle
│                                     #   notebook submission -- no imports from scripts/, everything
│                                     #   (including the Optuna search) runs fresh in one execution
├── run_pipeline.sh                  # local dev convenience: runs phase3_model_zoo.py only if
│                                     #   outputs/phase3_results.json is missing (or --retune passed),
│                                     #   then always runs final_pipeline.py
├── outputs/
│   ├── phase3_results.json          # best model + accepted tuned configs per target
│   └── submission.csv               # generated predictions
├── pyproject.toml / uv.lock / .python-version   # uv-managed environment, Python 3.13
└── CLAUDE.md                        # project context & workflow notes
```

### Why the local dev pipeline and the Kaggle script are separate

`phase3_model_zoo.py` (expensive: full Optuna search) and `final_pipeline.py` (cheap: stacking + clipping) are split so iterating on stacking/clipping logic doesn't force a ~90-minute retune every run. `kaggle_submission.py` is a **different artifact**, not just a copy — Kaggle's execution environment doesn't have `scripts/` available, and per the competition rules, every stage (including the Optuna search itself) must execute fresh inside the single graded run, so nothing from `outputs/phase3_results.json` can be loaded or hardcoded in — it's recomputed live. It also intentionally omits Mol2Vec (unconfirmed gain, extra dependency + PI1M input requirement) to keep the submission on the smaller, fully-validated feature set.

## Setup

```bash
uv sync
```

## Running

```bash
# Local dev iteration (fast path: skips Optuna retuning if outputs/phase3_results.json exists)
./run_pipeline.sh

# Force a fresh ~2-4 hour Optuna retune first
./run_pipeline.sh --retune

# Individual scripts
uv run python scripts/phase3_model_zoo.py     # research: model zoo + tuning -> phase3_results.json
uv run python scripts/final_pipeline.py       # production: stacking + clipping -> submission.csv

# The actual Kaggle submission artifact (self-contained, no local imports) -- run locally to verify,
# or upload as a Kaggle Script/Notebook kernel
uv run python kaggle_submission.py
```

Outputs `outputs/submission.csv` (or `./submission.csv` for `kaggle_submission.py`) in the format expected by Kaggle: `id,target`.
