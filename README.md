# Polymer Property Prediction

Solution for the [ANRF AISEHack 2.0 — Polymer Property Prediction](https://www.kaggle.com/) Kaggle competition.

## Task

Predict two properties of polymers directly from their chemical structure (SMILES strings):

| Property | Description | Unit |
|----------|-------------|------|
| **Tg** | Glass transition temperature — the temperature at which a polymer shifts from rigid/glassy to soft/rubbery | °C |
| **Egc** | Chain band gap — the minimum energy needed to excite an electron, relevant to a polymer's electrical/optical behaviour | eV |

**Metric:** mean R² across both properties (higher is better).

## Data

The dataset uses a **long format** — each row is one (molecule, property) measurement, not one row per molecule with both properties. A given polymer may have only Tg measured, only Egc, or both.

| File | Rows | Description |
|------|------|-------------|
| `data/train.csv` | 6,171 | `smiles`, `target`, `target_type` |
| `data/test.csv` | 4,115 | `id`, `smiles`, `target_type` |
| `data/sample_submission.csv` | — | Expected output format: `id`, `target` |

## Approach

### Pre-processing
- **SMILES canonicalization:** ~70% of raw train SMILES are non-canonical. All SMILES are canonicalized before grouping to prevent the same physical molecule appearing in both train and validation folds.
- **Duplicate merging:** 6 (canonical SMILES, target_type) groups have multiple measurements (all Tg, spreads 5–24 °C — measurement noise). These are mean-merged before feature extraction, reducing train rows from 6,171 → 6,165.

### Features (~8,972 pre-pruning, ~2,958 post-pruning)

Features are computed on two representations of each repeat unit: the **monomer** (single repeat unit as given) and the **chain extension** (3 copies stitched together via the `*` attachment points).

**Why chain extension?** Real polymers are long chains. A single repeat unit doesn't fully capture backbone flexibility (Tg-relevant) or conjugation length (Egc-relevant) — these properties only emerge over a longer stretch. Extending to a trimer before feature extraction lets RDKit descriptors and fingerprints see substructures that span repeat-unit junctions.

| Feature group | Monomer | Chain (trimer) |
|---|---|---|
| RDKit descriptors (~210) | ✓ | ✓ |
| ECFP4 Morgan fingerprints (2,048 bits, radius=2) | ✓ | ✓ |
| ECFP6 Morgan fingerprints (2,048 bits, radius=3) | ✓ | ✓ |
| MACCS keys (167 bits) | ✓ | ✓ |
| Topology (2): backbone span between `*` atoms | ✓ | — |
| Electronic (5): aromatic rings, sp2 fraction, rotatable bonds, aromatic atom fraction, non-aromatic double bonds | ✓ | ✓ |

A 300-tree LightGBM probe prunes zero-importance features before tuning, reducing ~8,972 → ~2,958 features. This ensures Optuna finds hyperparameters that are valid for the actual feature set the final models will see.

### Models

**LightGBM + XGBoost blend** with per-target hyperparameters tuned via Optuna (TPE sampler):
- LGB: 30 trials per target
- XGB: 10 trials per target

Blend weights are fitted per target by grid-searching `w(LGB) ∈ {0.1, 0.2, …, 0.9}` on out-of-fold predictions, rather than using a fixed 50/50 average. Both LGB and XGB use different tree-growth strategies (leaf-wise vs level-wise), so they make different errors on different molecules — blending reduces variance even when one model is stronger overall.

### Validation

**5-fold StratifiedGroupKFold** grouped by canonical SMILES, stratified by `target_type`. Ensures the same molecule never appears in both train and validation folds, and each fold has a balanced mix of Tg and Egc rows.

Final score is mean R²(Tg, Egc) across all folds with the fitted blend weights applied — exactly matching the Kaggle metric.

### Prediction

Test predictions are the **fold-ensembled average** of all 5 LGB models and all 5 XGB models per target, blended with the fitted per-target weights.

## Project Structure

```
polymer-property-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── base_line_model.ipynb       # organizer-provided baseline
├── scripts/
│   ├── exploration.py              # data exploration & sanity checks
│   └── baseline.py                 # full pipeline: features -> prune -> tune -> CV -> submission
├── outputs/
│   └── submission.csv              # generated predictions
└── CLAUDE.md                       # project context & workflow notes
```

## Setup

```bash
# Create venv (requires uv)
uv venv --python 3.11

# Install dependencies
uv pip install pandas numpy rdkit scikit-learn lightgbm xgboost joblib optuna
```

**Note (Windows):** Recent Windows security updates (June 2026) block unsigned RDKit `.pyd` DLLs. Run via WSL:
```bash
wsl
cd /mnt/d/AI\ Projects/polymer-property-prediction
python3 scripts/baseline.py
```

## Running

```bash
# Full pipeline: tune + CV + generate submission (~3 hours)
python3 scripts/baseline.py

# Quick smoke-test (~15 min): set N_TRIALS=2, N_XGB_TRIALS=2 at the top of baseline.py first
python3 scripts/baseline.py
```

Outputs `outputs/submission.csv` in the format expected by Kaggle.
