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

### Features (~2,260 per molecule)
- **RDKit descriptors** (~210): physicochemical properties computed from the molecular graph — molecular weight, ring counts, charge descriptors, etc.
- **Morgan fingerprints** (2,048 bits, radius 2): circular fingerprints encoding local substructure patterns around each atom — the standard way to represent molecular structure as a fixed-length bit vector.
- **Polymer topology features** (2): backbone span (`star_distance` — shortest bond-path between the two `*` attachment points of the repeat unit) and its fraction of the molecule's topological diameter. Captures how flexible or rigid the polymer backbone is, which is a known driver of Tg.

### Model
- **LightGBM** (gradient-boosted trees) — handles correlated features and NaN values natively, no feature scaling required.
- Hyperparameters tuned separately for Tg and Egc using **Optuna** (TPE sampler, 25 trials per target).

### Validation
- **5-fold cross-validation** grouped by canonical SMILES and stratified by `target_type` — ensures the same molecule never appears in both train and validation folds (which would inflate scores), and each fold has a balanced mix of Tg and Egc rows.
- Scored as **mean R²(Tg, Egc)** — exactly matching the Kaggle metric.

### Prediction
- Test predictions are the **average of all 5 fold models** (ensemble/bagging), rather than a single model retrained on all data. This reduces variance for free.

## Project Structure

```
polymer-property-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── base_line_model.ipynb       # organizer-provided baseline
├── scripts/
│   ├── exploration.py              # Step 1: data exploration & sanity checks
│   └── baseline.py                 # Full pipeline: features -> tune -> CV -> submission
├── outputs/
│   └── submission.csv              # Generated predictions
└── CLAUDE.md                       # Project context & workflow notes
```

## Setup

```bash
# Create venv (requires uv)
uv venv polymer-property-prediction --python 3.14

# Install dependencies
uv pip install --python polymer-property-prediction/Scripts/python.exe \
    pandas numpy rdkit scikit-learn lightgbm joblib optuna
```

## Running

```bash
# Data exploration (run first)
polymer-property-prediction/Scripts/python.exe scripts/exploration.py

# Full pipeline: tune + CV + generate submission
polymer-property-prediction/Scripts/python.exe scripts/baseline.py
```

Outputs `outputs/submission.csv` in the format expected by Kaggle.
