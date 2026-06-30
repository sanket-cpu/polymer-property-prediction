"""
Step 1: Data exploration for AISEHack polymer property prediction.
Run from the project root: python scripts/exploration.py
"""

import sys
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # suppress RDKit stderr noise
    RDKIT_AVAILABLE = True
except ImportError:
    print("WARNING: RDKit not installed — SMILES validation check will be skipped.")
    RDKIT_AVAILABLE = False

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


# ── Formatting helpers ─────────────────────────────────────────────────────────

def header(title):
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def sub(title):
    print(f"\n  -- {title} --")


# ── 1. Load & basic shape ──────────────────────────────────────────────────────

header("1. BASIC SHAPE INFO")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"\n  train.csv  ->  {len(train):,} rows,  {train.shape[1]} columns: {list(train.columns)}")
print(f"  test.csv   ->  {len(test):,} rows,  {test.shape[1]} columns: {list(test.columns)}")


# ── 2. Row counts by target_type ───────────────────────────────────────────────

header("2. ROW COUNTS BY TARGET TYPE  (train only — test has no targets)")

counts = train["target_type"].value_counts().sort_index()
total  = len(train)
for ttype, n in counts.items():
    print(f"  {ttype.upper():5s}  {n:,} rows  ({n / total * 100:.1f}% of train)")
print(f"\n  Total     {total:,} rows")


# ── 3. SMILES overlap between Tg and Egc ──────────────────────────────────────

header("3. SMILES OVERLAP BETWEEN Tg AND Egc")

tg_smiles  = set(train.loc[train["target_type"] == "tg",  "smiles"])
egc_smiles = set(train.loc[train["target_type"] == "egc", "smiles"])
overlap    = tg_smiles & egc_smiles
all_unique = tg_smiles | egc_smiles

print(f"\n  Unique SMILES with a Tg measurement   : {len(tg_smiles):,}")
print(f"  Unique SMILES with an Egc measurement : {len(egc_smiles):,}")
print(f"  Total unique SMILES in train          : {len(all_unique):,}")
print(f"\n  SMILES appearing in BOTH Tg AND Egc   : {len(overlap):,}  "
      f"({len(overlap) / len(all_unique) * 100:.1f}% of all unique SMILES)")

if len(overlap) > 0:
    print(f"\n  [!] These {len(overlap):,} molecules have measurements for both properties.")
    print("  When splitting into train/validation folds, BOTH rows for a given SMILES")
    print("  must land on the same side — otherwise the model has already seen the")
    print("  molecule, and validation scores will be artificially inflated (leakage).")
else:
    print("\n  No overlap — each SMILES was measured for only one property.")


# ── 4. Target value distributions ─────────────────────────────────────────────

header("4. TARGET VALUE DISTRIBUTIONS")

for ttype, unit in [("tg", "°C"), ("egc", "eV")]:
    vals = train.loc[train["target_type"] == ttype, "target"].dropna()
    print(f"\n  {ttype.upper()}  (unit: {unit},  n = {len(vals):,})")
    print(f"    Min      {vals.min():>10.3f}")
    print(f"    Max      {vals.max():>10.3f}")
    print(f"    Mean     {vals.mean():>10.3f}")
    print(f"    Median   {vals.median():>10.3f}")
    print(f"    Std      {vals.std():>10.3f}")


# ── 5. SMILES validity ─────────────────────────────────────────────────────────

header("5. SMILES VALIDITY  (RDKit parse check)")

if not RDKIT_AVAILABLE:
    print("\n  Skipped — RDKit not installed.")
else:
    def validate_smiles(df, label, smiles_col="smiles"):
        all_smiles = df[smiles_col].tolist()
        failed = []
        for i, s in enumerate(all_smiles):
            if not isinstance(s, str) or s.strip() == "":
                failed.append((i, s, "empty or non-string"))
            elif Chem.MolFromSmiles(s) is None:
                failed.append((i, s, "RDKit parse failure"))

        n_total = len(all_smiles)
        n_fail  = len(failed)
        print(f"\n  {label}")
        print(f"    Total SMILES  : {n_total:,}")
        print(f"    Valid         : {n_total - n_fail:,}")
        print(f"    Invalid       : {n_fail:,}  ({n_fail / n_total * 100:.2f}%)")
        if n_fail > 0:
            print(f"    First up to 5 failing entries:")
            for idx, s, reason in failed[:5]:
                display = s if isinstance(s, str) and len(s) <= 80 else (str(s)[:77] + "...")
                print(f"      row {idx:>5d}  [{reason}]  {display}")
        return failed

    train_failures = validate_smiles(train, "train.csv")
    test_failures  = validate_smiles(test,  "test.csv")


# ── 6. Missing values ──────────────────────────────────────────────────────────

header("6. MISSING VALUES")

for label, df in [("train.csv", train), ("test.csv", test)]:
    missing = df.isnull().sum()
    any_missing = missing[missing > 0]
    print(f"\n  {label}  ({len(df):,} rows)")
    if len(any_missing) == 0:
        print("    No missing values in any column.")
    else:
        for col, n in any_missing.items():
            print(f"    {col:<15s}  {n:,} missing  ({n / len(df) * 100:.2f}%)")


# ── Done ───────────────────────────────────────────────────────────────────────

header("DONE")
print("  All Step 1 checks complete.\n")
