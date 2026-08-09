"""
ANRF AISEHack 2.0 -- Polymer Property Prediction. Single self-contained
script for a Kaggle Script/Notebook kernel: builds on the pipeline that
scored 0.849 on the public leaderboard, in one file with no imports from
this repo's scripts/ directory (Kaggle's kernel environment doesn't have
those files -- everything importable only from this repo has been inlined
below).

What's IN, and why:
  - RDKit descriptors + physics ratios + dimer-delta + 256-bit Morgan
    fingerprints (prajwal's original featurize()), + 167-bit MACCS keys,
    + Gasteiger partial-charge summary stats, + ~85 Fragments SMARTS-based
    functional-group counts, + 6 attachment-point features describing the
    backbone between the two `*` atoms specifically (path length, backbone
    vs. side-chain atom fraction, same-ring / aromaticity / sp3 at the
    junction -- confirmed on the real leaderboard: 0.849->0.858 after
    adding these), + 3 junction-bond features (conjugated / aromatic /
    in-ring status of the specific bond formed when two repeat units chain
    together in _make_dimer_mol -- a more direct read on whether
    conjugation actually carries from one repeat unit into the next than
    the whole-dimer-average dimer_delta_* features give), + 3 conjugation-
    extent features (size of the largest contiguous conjugated system in
    the molecule, via union-find over conjugated bonds -- distinguishes one
    big linked conjugated system from several small disconnected ones,
    which whole-molecule averages like AromaticRatio/ConjugationRatio
    can't; motivated by scripts/diagnose_weak_targets.py finding OOF error
    on eps/nc specifically -- both polarizability-driven properties --
    correlates with ring/conjugation complexity, unlike the other 3 small
    targets), + full monomer->dimer descriptor deltas (~216 columns, CLAUDE.md
    task list Task 4 -- the dimer is already built for every molecule for
    the 3 hand-picked dimer_delta_* ratios above; running the complete
    _safe_descriptors() set on it too and taking deltas across all of them
    captures how every RDKit-computed property shifts when the chain
    extends, not just aromaticity/conjugation/rotatable-bonds specifically).
    All computed fresh from train.csv/test.csv, no external data.
  - Cross-target features (v2, re-opening CLAUDE.md's Step 12 -- previously
    diagnostic-only, see "What's OUT" note below in earlier versions). A
    canonical-SMILES audit found ~98% of the eea/ei/eps/nc/egb test
    molecules already exist elsewhere in train.csv under a *different*
    target_type, with a median of 2-3 other properties already known for
    that same molecule -- signal the earlier diagnostic-only version never
    actually fed into predictions. build_cross_target_lookup() builds a
    wide canon_smiles -> {eea, egb, egc, ei, eps, nc, tg} table from
    train_valid (mean if a canon SMILES has duplicate rows under the same
    target_type), and cross_target_feats() joins it into both
    build_feature_matrix() (train) and the test-featurization block in
    main() as xtarget_<prop> numeric columns + known_<prop> boolean flags,
    computed BEFORE fit_feature_pruner so they go through the same
    variance/correlation pruning as every other feature. A row's own
    target_type column is always blanked (NaN/known=0) before the join --
    that value is the label being predicted, and letting it in would be
    direct leakage, not signal. Whether this block actually ships is
    decided live, every run, by evaluate_cross_target_features(): a single
    representative untuned model (LightGBM, same convention
    scripts/ablation_study.py already established for feature-block-level
    decisions, as opposed to model/hyperparameter-level ones) scored with
    vs. without the block on the real harness (get_harness_splits, not the
    cheap search split) -- only kept if it clears the noise floor (larger
    of the two configs' fold-to-fold std), identical accept/reject pattern
    to every other tuned decision in this file.
  - 3D-conformer descriptors (D3D_TARGETS only -- eps/ei/nc, see
    conformer_3d_feats): dipole moment magnitude (MMFF partial charges
    weighted by 3D position) + RDKit's Descriptors3D shape descriptors
    (radius of gyration, asphericity, eccentricity, spherocity, PMI
    ratios), off an ETKDGv3-embedded + MMFF-optimized conformer of the
    *dimer* (reuses _make_dimer_mol exactly as built for the dimer-delta
    features above, rather than re-deriving a separate `*`-capping
    scheme -- a raw wildcard atom has no sane geometry to embed, but
    _make_dimer_mol already resolves both attachment points). Real
    per-molecule wall-clock cost (embedding + force-field optimization),
    so computed only for D3D_TARGETS' own rows -- not shared across all 7
    targets the way every feature block above is -- and gated to eps/ei/nc
    specifically: exactly the polarizability-/charge-distribution-driven
    properties (dielectric constant, ionization energy, refractive index)
    3D shape and dipole moment are physically expected to matter for.
    Embedding failures (unusual topologies, same edge cases
    _make_dimer_mol already tolerates for the 2D dimer-delta features)
    fall back to NaN rather than crashing the run. Tested via the same
    paired-delta accept/reject pattern as the cross-target block above,
    per target, before it's allowed to feed the tuning phase below.
  - log/exp target transform for eps, ei (the two most right-skewed,
    weakest-CV targets -- see TARGET_TRANSFORMS below).
  - 8-model zoo (Ridge, KNN, RF, ExtraTrees, GBM, XGBoost, CatBoost,
    LightGBM) per target_type, combined via out-of-fold stacking with a
    Ridge meta-learner. KNN and ExtraTrees for genuine instance-based/
    local-structure signal and stacking diversity. A general SVR/
    KernelRidge was considered and dropped: their O(n^2)-O(n^3) training
    cost is a real risk on tg (4,143 rows) inside a once-and-done timed
    run. A 9th model, KernelRidge with a Tanimoto/Jaccard kernel on raw
    Morgan fingerprint bits, was tried narrowly for eps/ei/nc and reverted
    -- net-negative on the final stack (0.8763 without it vs. 0.8755 with
    it, even after live-tuning its alpha), despite the a priori case for
    instance-based fingerprint similarity being a genuinely different
    signal from the other 8 models' descriptor-space view. ElasticNet
    and HGB (originally in a 10-model zoo) were dropped after
    scripts/ablation_study.py (Task 2) found ElasticNet correlates with
    Ridge at 0.98-1.00 and HGB correlates with LightGBM at exactly 1.00 on
    every single target -- true near-duplicates, not just similar models --
    with consistently negligible-to-negative individual contribution to
    the stack. Fewer near-duplicate columns feeding the meta-learner
    directly reduces multicollinearity-driven coefficient instability, so
    this targets leaderboard *variance*, not just mean score.
  - Final refit-predict stage is bagged over BAG_SEEDS (3->5->10 seeds --
    this benefit is invisible in local CV since CV only ever sees OOF
    predictions, never the bagged refit, but the 0.849->0.858 leaderboard
    jump landed well above what the CV-visible feature gains alone
    predicted, and bagging's variance reduction is the most plausible
    explanation for the rest) -- each tree/boosting model is refit with a
    different random_state and the raw predictions averaged before the
    meta-learner is applied, standard variance reduction. Only this (cheap)
    final stage is bagged, not OOF generation or meta-learner fitting,
    which stay at one canonical seed.
  - Phase 5 output-safety clipping (train min/max + 10% margin, plus hard
    physical floors on band gap / refractive index / dielectric constant).
  - Pseudo-labeling on test.csv's own rows, per target_type. test.csv is
    competition-provided, not external data (Section 6.2.1 only bans
    outside datasets) -- this is standard semi-supervised learning on the
    competition's own files, entirely within the single run. A test row's
    prediction is trusted as a pseudo-label only if the 8 base models
    (already bagged) agree with each other by less than half of that
    target's CV-estimated typical residual size; confident rows get
    safety-clipped (clip_bounds_for) and added to the base models' training
    data at reduced sample weight for a second, lighter-bagged refit. The
    meta-learner (oof_meta[tt]) is NOT refit on anything here -- it stays
    exactly as fit on the real labeled OOF, applied unchanged to the
    retrained base models' new outputs; see main() for why that's a
    deliberate choice, not an oversight. Targets the small targets'
    real bottleneck directly: too few rows to pin down a stable estimate,
    which no amount of feature engineering or tuning fixes.

Tried and reverted: a target-aware SelectKBest cap (tighter feature limit for
the 5 small targets, applied to all 10 models instead of just the 3 that
already had a fixed k=100) was implemented and smoke-tested, but made every
single target worse -- including tg/egc, whose cap should have been a no-op.
Root cause: 7 of the 10 models (RF/ExtraTrees/GBM/HGB/XGB/CatBoost/LightGBM)
previously had no feature cap at all, and evidently used signal spread across
more columns than a univariate-correlation filter (f_regression) preserves --
the filter's inherent blind spot to nonlinear/interaction-only signal ended up
mattering more than the overfitting risk it was meant to fix. Reverted
rather than iterated on, since the regression was unambiguous across all 7
targets on the first test.

Rules compliance note on hyperparameter tuning: an earlier version of this
script hardcoded the 2 CatBoost configs (egc, tg) that a local Optuna search
found to beat their defaults. That's a violation of "all stages -- including
model definition/initialization and training -- must execute entirely
within the notebook during a single run, manual intervention at any stage
not permitted": those exact hyperparameter values are the *output* of a
data-dependent search run outside the graded execution, so injecting them
as constants is manual intervention at the training stage, even though the
models themselves still fit fresh on train.csv. Fixed by moving the Optuna
search itself into this script (see TUNE_BOOSTING below) -- the winning
configs are now discovered live, every run, inside the single execution.
Ordinary fixed hyperparameters elsewhere in this file (n_estimators=300,
SELECT_K=100, VAR_THRESH, the clipping margin, etc.) are NOT the same kind
of issue -- they're engineering defaults chosen by judgment, never fit or
searched against this dataset, so there's nothing external being replayed.

What's OUT versus the local dev pipeline, and why:
  1. No Mol2Vec / PI1M-derived features -- and this one is NOT a judgment
     call, it's a hard rules requirement. Competition Rules Section 6.2.1
     ("No External Data"): "Use of any external, private, or previously
     prepared datasets (public or private)... [is] strictly prohibited...
     Any violation will result in immediate disqualification, regardless
     of leaderboard position." PI1M.csv is exactly that -- a previously
     prepared public dataset -- and the rule carves out no exception for
     "unlabeled" or "structure-only" use. This also rules out any future
     PI1M-based pseudo-labeling idea for this script, not just Mol2Vec.
     (Separately: a local ablation found Mol2Vec helps egb/ei but hurts
     eps/nc, and even setting the rule aside, that result was never
     validated stacked on top of the current feature set -- so it wasn't
     a strong candidate anyway.)
  (Step 12 cross-target features are back IN as of v2 -- see above. The
  earlier diagnostic-only pass never fed its findings into predictions,
  which didn't match a fresh canonical-SMILES audit's ~98%-overlap result,
  so it was rebuilt as a real, live-tested input feature.)

Estimated runtime: aiming to stay under ~2.5 hours on CPU, still dominated
by the live Optuna search. N_OPTUNA_TRIALS is 35 -- wide enough coverage
that the original 15-trial search found only 1/21 combos worth accepting.
N_OPTUNA_TRIALS and BAG_SEEDS are the two knobs to shrink further if a run
is running long. (N_OPTUNA_TRIALS was briefly parked at 2 for smoke-testing
control flow during development -- reverted to 35 for every real run.)

Input path handling: looks for train.csv anywhere under /kaggle/input/ if
that directory exists (Kaggle mounts competition data there, under a
folder name that matches the competition slug, which this script doesn't
hardcode); otherwise falls back to ./data/ for local testing. Output is
written to ./submission.csv, which resolves to /kaggle/working/submission.csv
under Kaggle's default working directory -- exactly where a Code
Competition looks for it.
"""

import inspect
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem, Descriptors, Descriptors3D, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys,
    rdPartialCharges, Fragments,
)

# The dimer-construction step intentionally attempts a bond-surgery +
# resanitize that fails on some ring topologies; those failures are caught
# and handled (falls back to zero-delta features), so RDKit's C++ logger
# spam ("Can't kekulize...") is expected noise. Silenced so the run log
# stays readable.
RDLogger.DisableLog('rdApp.*')

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_JOBS = -1
# Trimmed from 50 -> 35 to create time budget for the Tier-1 additions below
# (Fragments features, 2 more zoo models, seed-bagged final refit) while
# keeping the 2.5hr ceiling -- still far wider coverage than the original
# 15-trial search that found only 1/21 combos worth accepting.
N_OPTUNA_TRIALS = 35
# Seeds for bagging the *final* refit-predict stage only (not OOF/meta-learner
# fitting, which always use the single canonical seed every score in this
# script is measured against). Averages away RF/ExtraTrees/GBM/XGB/CatBoost/
# LightGBM's own bootstrap/split-order randomness -- standard variance
# reduction, and cheap since it only multiplies the fast final-refit stage,
# not the expensive tuning stage. This benefit is invisible in local CV (CV
# is computed from OOF folds, never the bagged refit) -- the 0.849->0.858
# leaderboard jump after adding the attachment-point features was bigger
# than the local CV delta predicted, most plausibly because bagging's
# variance reduction was doing real work on the actual test set the whole
# time without ever showing up in any number we could see locally. Bumped
# 3->5->10 seeds on that basis: this is the one lever directly targeting
# leaderboard *variance* rather than mean score, the cost is cheap (only
# multiplies the fast final-refit stage), and we have runtime headroom.
BAG_SEEDS = list(range(10))

# Pseudo-labeling config. PL_CONF_FRACTION: a test row's pseudo-label is
# only trusted if the 8 base models (already bagged across BAG_SEEDS)
# disagree with each other by less than this fraction of the target's own
# CV-estimated typical residual size -- tied to a number this script
# already computes and trusts (cv_scores[tt]), not an arbitrary constant.
# PL_SAMPLE_WEIGHT: pseudo-labeled rows count for less than real labels
# when retraining. PL_MIN_ROWS: skip the (expensive) augmented retrain
# entirely if too few rows passed the gate to be worth it.
# PL_BAG_SEEDS: the augmented-retrain pass reuses only the first half of
# BAG_SEEDS -- it still gets real bagging benefit, just cheaper, since this
# whole round already roughly doubles the final-refit stage's cost.
PL_CONF_FRACTION = 0.5
PL_SAMPLE_WEIGHT = 0.5
PL_MIN_ROWS = 5
PL_BAG_SEEDS = BAG_SEEDS[:len(BAG_SEEDS) // 2]
# Looser-than-default confidence thresholds tried live, per target, by
# evaluate_pl_conf_fraction -- only shipped for a target if that target's
# own paired delta clears its own noise floor. Restricted to
# PL_CONF_SEARCH_TARGETS because the simulation replays a full
# pseudo-labeling round inside every fold (see that function's docstring),
# which is only affordable on the ~230-row targets; eps and nc are also
# the two the loosening is most plausible for, being the weakest-CV
# targets with the least real training signal to begin with.
PL_CONF_CANDIDATES = [0.65, 0.8]
# ei added (was {'eps', 'nc'}): it has the 2nd-largest CV deficit of any
# target and already gets the 3D-conformer block (D3D_TARGETS), but was the
# only top-3-deficit target still on the untuned default PL_CONF_FRACTION.
# Same proven mechanism (evaluate_pl_conf_fraction), one more target.
PL_CONF_SEARCH_TARGETS = {'eps', 'ei', 'nc'}

# GNN track CLOSED. The 3-seed multi-task export (gnn_mt_export.log) settled
# it: pooled OOF R2 lost to the tuned 8-model stack on every one of the 5
# candidate targets, by real margins, not noise --
#   eps 0.7529 vs 0.8014 (-0.0485), ei 0.7968 vs 0.8329 (-0.0361),
#   nc  0.8364 vs 0.8558 (-0.0194), egc 0.9037 vs 0.9201 (-0.0164),
#   tg  0.8997 vs 0.9111 (-0.0114).
# An empty set short-circuits the whole GNN path in process_target (the
# `if tt in GNN_STACK_TARGETS` guard is never true), so a run loads no
# gnn_oof_*/gnn_test_* files and never invokes evaluate_gnn_stack_column.
# load_gnn_predictions/evaluate_gnn_stack_column are left defined but unused;
# re-enable by listing targets here only if that verdict is ever overturned.
GNN_STACK_TARGETS = set()


# ---------------------------------------------------------------------------
# Paths -- Kaggle input dir auto-detected, local ./data/ as fallback
# ---------------------------------------------------------------------------
def _find_input_dir():
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        hits = list(kaggle_root.rglob("train.csv"))
        if hits:
            return hits[0].parent
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    return here / "data"


INPUT_DIR = _find_input_dir()
TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"
SAMPLE_SUB_PATH = INPUT_DIR / "sample_submission.csv"
OUT_PATH = Path("submission.csv")

# ---------------------------------------------------------------------------
# Featurization knobs
# ---------------------------------------------------------------------------
FP_BITS = 256
FP_RADIUS = 2
VAR_THRESH = 1e-6
CORR_THRESH = 0.98
SELECT_K = 100

_SLOW_OR_UNSTABLE = {'Ipc'}  # can overflow to inf on larger structures
_DESC_LIST = [(n, f) for n, f in Descriptors._descList if n not in _SLOW_OR_UNSTABLE]
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
# RDKit's ~85 SMARTS-based functional-group counters (amines, esters,
# aromatic rings, halides, etc.) -- genuinely different signal from the
# topology/counting descriptors and fingerprints above, and virtually free
# to compute. Collected once at import time, same pattern as _DESC_LIST.
_FRAGMENT_FUNCS = [(n, getattr(Fragments, n)) for n in dir(Fragments) if n.startswith('fr_')]

SMALL_TARGETS = {'egb', 'ei', 'eea', 'eps', 'nc'}
LARGE_TARGETS = {'tg', 'egc'}
# 3D-conformer descriptor block (see conformer_3d_feats) targets -- a
# subset of SMALL_TARGETS. Restricted to eps/ei/nc (not all 5 small
# targets, and never the ~4,000-row LARGE_TARGETS): ETKDG embedding +
# MMFF optimization is real per-molecule wall-clock cost, and eps/ei/nc
# are exactly the polarizability-/charge-distribution-driven properties
# (dielectric constant, ionization energy, refractive index) a molecule's
# 3D shape and dipole moment are physically expected to matter for, unlike
# eea/egb.
D3D_TARGETS = {'eps', 'ei', 'nc'}
N_SPLITS = 5
N_REPEATS_SMALL = 3

# eps (dielectric constant) and ei (ionization energy) are the two most
# right-skewed targets and the two weakest CV scores; both are comfortably
# positive in train, so a plain log/exp is safe.
TARGET_TRANSFORMS = {
    'eps': (np.log, np.exp),
    'ei': (np.log, np.exp),
}

MARGIN_FRACTION = 0.10
# ei (ionization energy) can't be negative, same physical reasoning as
# egc/egb's existing floors -- was missing.
PHYSICAL_FLOORS = {'egc': 0.0, 'egb': 0.0, 'nc': 1.0, 'eps': 1.0, 'ei': 0.0}
MODEL_NAMES = ['Ridge', 'KNN', 'RF', 'ExtraTrees', 'GBM', 'XGB', 'CatBoost', 'LightGBM']
META_ALPHA = 1.0


# ---------------------------------------------------------------------------
# 1. Featurization (RDKit descriptors + physics ratios + full dimer-delta
#    descriptor set + junction-bond conjugation/aromaticity/ring status +
#    Morgan fingerprints + MACCS keys + Gasteiger charges + Fragments
#    functional-group counts + attachment-point backbone features +
#    conjugation-extent features)
# ---------------------------------------------------------------------------
def _parse_mol(smiles):
    """Parse a polymer repeat-unit SMILES (with * attachment points)."""
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        # fallback: cap dummy attachment atoms with carbon to keep valence valid
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    return mol


def _make_dimer_mol(smiles):
    """Join two copies of the repeat unit at their * attachment points to
    approximate a short chain segment -- lets ring-conjugation/aromaticity/
    rotatable-bond descriptors see across the repeat-unit boundary, which
    matters for backbone-driven properties (band gaps, refractive index)
    more than a single isolated unit does. None if construction fails."""
    try:
        s = smiles.replace('[*]', '*')
        if s.count('*') != 2:
            return None
        molA = Chem.MolFromSmiles(s)
        molB = Chem.MolFromSmiles(s)
        if molA is None or molB is None:
            return None

        combo = Chem.RWMol(Chem.CombineMols(molA, molB))
        dummy_idx = [a.GetIdx() for a in combo.GetAtoms() if a.GetSymbol() == '*']
        if len(dummy_idx) != 4:
            return None

        def neighbor_of(idx):
            atom = combo.GetAtomWithIdx(idx)
            nbrs = atom.GetNeighbors()
            return nbrs[0].GetIdx() if nbrs else None

        a2 = neighbor_of(dummy_idx[1])
        b1 = neighbor_of(dummy_idx[2])
        if a2 is None or b1 is None:
            return None

        # Atom map numbers survive RemoveAtom's index renumbering below --
        # standard RDKit way to keep tracking specific atoms across an edit,
        # used here so junction_bond_feats() can relocate the exact junction
        # bond in the final `dimer` mol after both atom removal and
        # sanitization have happened.
        combo.GetAtomWithIdx(a2).SetAtomMapNum(1)
        combo.GetAtomWithIdx(b1).SetAtomMapNum(2)

        combo.AddBond(a2, b1, Chem.BondType.SINGLE)
        for idx in sorted(dummy_idx, reverse=True):
            combo.RemoveAtom(idx)

        dimer = combo.GetMol()
        try:
            Chem.SanitizeMol(dimer)
        except Exception:
            # Some ring topologies genuinely can't be re-kekulized after the
            # junction bond is spliced in -- fall back to a partial sanitize
            # that skips kekulize/aromaticity perception so we still get
            # valid valences instead of discarding the whole dimer.
            dimer.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                dimer,
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
            )
        return dimer
    except Exception:
        return None


def junction_bond_feats(dimer):
    """Character of the specific bond formed when two repeat units chain
    together (the one `_make_dimer_mol` adds between `a2`/`b1`, tagged with
    atom map numbers 1/2 so it survives the atom-removal renumbering).
    dimer_delta_* above measures whole-molecule averages before vs. after
    joining; this looks at just the junction itself, which is the more
    direct question for Egc: does conjugation actually carry across from
    one repeat unit into the next, or does this specific link break it?
    An sp3-to-aromatic junction, for instance, averages to something
    ambiguous across dimer_delta_AromaticRatio but is unambiguously a
    conjugation break right at the bond in question. None (defaults) if
    dimer construction failed or the map numbers can't be found."""
    defaults = {'junction_bond_conjugated': 0.0, 'junction_bond_aromatic': 0.0,
                'junction_bond_in_ring': 0.0}
    if dimer is None:
        return defaults
    try:
        a2_atom = next((a for a in dimer.GetAtoms() if a.GetAtomMapNum() == 1), None)
        b1_atom = next((a for a in dimer.GetAtoms() if a.GetAtomMapNum() == 2), None)
        if a2_atom is None or b1_atom is None:
            return defaults
        bond = dimer.GetBondBetweenAtoms(a2_atom.GetIdx(), b1_atom.GetIdx())
        if bond is None:
            return defaults
        return {
            'junction_bond_conjugated': float(bond.GetIsConjugated()),
            'junction_bond_aromatic': float(bond.GetIsAromatic()),
            'junction_bond_in_ring': float(bond.IsInRing()),
        }
    except Exception:
        return defaults


def _safe_descriptors(mol):
    out = {}
    for name, func in _DESC_LIST:
        try:
            val = func(mol)
            if val is None or not np.isfinite(val):
                val = np.nan
        except Exception:
            val = np.nan
        out[f'desc_{name}'] = val
    return out


def _morgan_bits(mol):
    fp = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros((FP_BITS,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return {f'fp_{i}': int(arr[i]) for i in range(FP_BITS)}


def _custom_physics_feats(mol):
    feats = {}
    heavy = mol.GetNumHeavyAtoms() or 1
    n_bonds = mol.GetNumBonds() or 1
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    n_atoms = len(atoms) or 1

    for el in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'Si', 'P']:
        feats[f'n_{el}'] = atoms.count(el)
        feats[f'frac_{el}'] = atoms.count(el) / n_atoms

    n_aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_conjugated_bonds = sum(1 for b in mol.GetBonds() if b.GetIsConjugated())
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)

    feats['AromaticRatio'] = n_aromatic_atoms / n_atoms
    feats['ConjugationRatio'] = n_conjugated_bonds / n_bonds
    feats['RotBondsPerHeavyAtom'] = n_rot / heavy
    feats['RingsPerHeavyAtom'] = n_rings / heavy
    feats['MolWtPerHeavyAtom'] = Descriptors.MolWt(mol) / heavy
    return feats


def featurize(smiles):
    mol = _parse_mol(smiles)
    if mol is None:
        return None

    feats = {}
    try:
        mono_desc = _safe_descriptors(mol)
        feats.update(mono_desc)
        feats.update(_custom_physics_feats(mol))
        feats.update(_morgan_bits(mol))

        dimer = _make_dimer_mol(smiles)
        if dimer is not None:
            dimer_heavy = dimer.GetNumHeavyAtoms() or 1
            dimer_bonds = dimer.GetNumBonds() or 1
            d_arom = sum(1 for a in dimer.GetAtoms() if a.GetIsAromatic()) / dimer_heavy
            d_conj = sum(1 for b in dimer.GetBonds() if b.GetIsConjugated()) / dimer_bonds
            d_rot = rdMolDescriptors.CalcNumRotatableBonds(dimer) / dimer_heavy
            feats['dimer_delta_AromaticRatio'] = d_arom - feats['AromaticRatio']
            feats['dimer_delta_ConjugationRatio'] = d_conj - feats['ConjugationRatio']
            feats['dimer_delta_RotBondsPerHeavy'] = d_rot - feats['RotBondsPerHeavyAtom']

            # Task 4 (CLAUDE.md task list): the 3 ratios above were always a
            # hand-picked subset of what's available off the dimer object
            # that's already being built for every molecule. Running the
            # full ~200-descriptor _safe_descriptors() set on it too and
            # taking monomer->dimer deltas across all of them is close to
            # free (the dimer's already constructed) and is a genuinely
            # different kind of signal from the 3 hand-picked ratios: how
            # every RDKit-computed property shifts when the chain extends,
            # not just aromaticity/conjugation/rotatable-bonds specifically.
            dimer_desc = _safe_descriptors(dimer)
            for key, mono_val in mono_desc.items():
                dimer_val = dimer_desc.get(key, np.nan)
                if np.isfinite(mono_val) and np.isfinite(dimer_val):
                    feats[f'dimer_delta_{key}'] = dimer_val - mono_val
                else:
                    feats[f'dimer_delta_{key}'] = np.nan
        else:
            feats['dimer_delta_AromaticRatio'] = 0.0
            feats['dimer_delta_ConjugationRatio'] = 0.0
            feats['dimer_delta_RotBondsPerHeavy'] = 0.0
            for key in mono_desc:
                feats[f'dimer_delta_{key}'] = 0.0
        feats.update(junction_bond_feats(dimer))
    except Exception:
        return None
    return feats


def maccs_keys(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.int8)
    for bit in fp.GetOnBits():
        if bit < 167:
            arr[bit] = 1
    return {f'maccs_{i}': int(arr[i]) for i in range(167)}


def gasteiger_features(mol):
    """Gasteiger partial charges, per atom, summarized. Targets eps
    (dielectric constant) and ei (ionization energy) specifically -- both
    are driven by molecular polarity/charge distribution, which the mostly
    topology-/counting-based descriptor set above doesn't directly
    capture."""
    mol = Chem.Mol(mol)  # ComputeGasteigerCharges mutates in place
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    charges = charges[np.isfinite(charges)]
    if len(charges) == 0:
        return {'gast_max': 0.0, 'gast_min': 0.0, 'gast_mean': 0.0,
                'gast_sum': 0.0, 'gast_std': 0.0}
    return {
        'gast_max': float(charges.max()), 'gast_min': float(charges.min()),
        'gast_mean': float(charges.mean()), 'gast_sum': float(charges.sum()),
        'gast_std': float(charges.std()),
    }


def fragment_counts(mol):
    """~85 RDKit SMARTS-based functional-group counts (see _FRAGMENT_FUNCS
    above). Counts should never raise/NaN in practice, but each call is
    wrapped defensively -- same pattern as _safe_descriptors -- since a
    single bad match on an unusual structure shouldn't kill the whole row."""
    out = {}
    for name, func in _FRAGMENT_FUNCS:
        try:
            out[f'frag_{name}'] = func(mol)
        except Exception:
            out[f'frag_{name}'] = 0
    return out


def attachment_point_feats(mol):
    """Features about the backbone between the two `*` attachment points
    where this repeat unit links to the next one in the chain. Every other
    feature block above treats the molecule as an unordered bag of
    atoms/rings/fingerprint bits -- none of them encode backbone length or
    rigidity *between* the two attachment points specifically, which is
    exactly what drives chain mobility (Tg: short/rigid spacer -> high Tg,
    long/flexible spacer -> low Tg) and conjugation length (Egc) in a real
    polymer chain. Molecules where `*` didn't survive parsing (rare fallback
    in _parse_mol) get the same all-zero-default treatment as a failed
    dimer construction below."""
    defaults = {
        'attach_path_len': 0.0, 'attach_path_len_frac': 0.0,
        'attach_backbone_frac': 0.0, 'attach_same_ring': 0.0,
        'attach_neighbor_aromatic_frac': 0.0, 'attach_neighbor_sp3_frac': 0.0,
    }
    dummy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    if len(dummy_idx) != 2:
        return defaults
    try:
        path = Chem.GetShortestPath(mol, dummy_idx[0], dummy_idx[1])
        if not path:
            return defaults
        heavy = mol.GetNumHeavyAtoms() or 1
        path_len = len(path) - 1  # bonds from one * to the other
        backbone_atoms = len(path) - 2  # path atoms minus the two * endpoints

        nbrs = []
        for idx in dummy_idx:
            atom_nbrs = mol.GetAtomWithIdx(idx).GetNeighbors()
            if atom_nbrs:
                nbrs.append(atom_nbrs[0])

        same_ring = 0.0
        if len(nbrs) == 2:
            ring_info = mol.GetRingInfo()
            same_ring = float(any(
                nbrs[0].GetIdx() in ring and nbrs[1].GetIdx() in ring
                for ring in ring_info.AtomRings()
            ))

        arom_frac = float(np.mean([n.GetIsAromatic() for n in nbrs])) if nbrs else 0.0
        sp3_frac = float(np.mean(
            [n.GetHybridization() == Chem.HybridizationType.SP3 for n in nbrs]
        )) if nbrs else 0.0

        return {
            'attach_path_len': float(path_len),
            'attach_path_len_frac': path_len / heavy,
            'attach_backbone_frac': backbone_atoms / heavy,
            'attach_same_ring': same_ring,
            'attach_neighbor_aromatic_frac': arom_frac,
            'attach_neighbor_sp3_frac': sp3_frac,
        }
    except Exception:
        return defaults


def conjugation_extent_feats(mol):
    """Size of the largest contiguous conjugated system in the molecule --
    a cheap, hand-built proxy for how far pi-conjugation extends, testing
    the same hypothesis a GNN would otherwise learn implicitly.

    Motivated directly by scripts/diagnose_weak_targets.py: eps (dielectric
    constant) and nc (refractive index) -- both fundamentally about
    polarizability -- showed OOF error correlating with ring/conjugation
    complexity (+0.18 to +0.31 vs. residual, consistent across two
    structurally different models), while the other 3 small targets showed
    no such pattern. AromaticRatio/ConjugationRatio above already count
    conjugated bonds molecule-wide, but can't distinguish one big linked
    conjugated system (e.g. 4 fused thiophenes) from several small
    disconnected ones (e.g. 2 separate isolated phenyl rings) -- same total
    count, very different polarizability. Finding the largest connected
    component of the "conjugated bond" subgraph (union-find over atoms
    joined by a conjugated bond) makes that distinction directly."""
    heavy = mol.GetNumHeavyAtoms() or 1
    conj_bonds = [b for b in mol.GetBonds() if b.GetIsConjugated()]
    if not conj_bonds:
        return {'max_conj_component_size': 0.0, 'max_conj_component_frac': 0.0,
                'n_conj_components': 0.0}

    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for b in conj_bonds:
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        parent.setdefault(a1, a1)
        parent.setdefault(a2, a2)
        r1, r2 = find(a1), find(a2)
        if r1 != r2:
            parent[r1] = r2

    component_sizes = {}
    for atom_idx in parent:
        root = find(atom_idx)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    max_size = max(component_sizes.values())
    return {
        'max_conj_component_size': float(max_size),
        'max_conj_component_frac': max_size / heavy,
        'n_conj_components': float(len(component_sizes)),
    }


_D3D_DEFAULTS = {
    'd3d_dipole': np.nan, 'd3d_radius_of_gyration': np.nan, 'd3d_asphericity': np.nan,
    'd3d_eccentricity': np.nan, 'd3d_spherocity': np.nan, 'd3d_npr1': np.nan, 'd3d_npr2': np.nan,
}


def conformer_3d_feats(smiles):
    """3D-conformer descriptors: dipole moment magnitude + RDKit's
    Descriptors3D shape descriptors, off an ETKDGv3-embedded + MMFF-
    optimized conformer -- D3D_TARGETS only (eps/ei/nc), called directly
    from main() on that subset of rows, not from compute_raw_features (see
    D3D_TARGETS' comment for why: this is real per-molecule wall-clock
    cost that isn't worth paying for every row of every target_type).

    Embeds the *dimer* (_make_dimer_mol(smiles)), not the raw monomer --
    reuses the exact same dimer construction the dimer_delta_* features
    above already build for every molecule, rather than re-deriving a
    separate capping scheme for the two `*` attachment points. That
    reuse matters here specifically: a raw wildcard atom has no sane
    valence/geometry to embed, but _make_dimer_mol already resolves both
    attachment points for us -- one becomes an explicit junction bond
    between the two repeat units, the other two vanish when the dummy
    atoms are removed and RDKit fills the resulting open valences with
    implicit hydrogens on resanitize. Embedding that already-capped
    structure is what gives ETKDG sane geometry to work with, instead of
    the '*' atom itself breaking valence.

    Every failure point (dimer construction, embedding, MMFF setup) falls
    back to NaN for every column rather than raising -- across ~1,000
    real molecules, a handful of embed failures on unusual topologies
    (same ring-topology edge cases _make_dimer_mol already tolerates for
    the 2D dimer-delta features) are expected, and must never crash the
    whole run over one bad molecule."""
    dimer = _make_dimer_mol(smiles)
    if dimer is None:
        return dict(_D3D_DEFAULTS)
    try:
        mol_h = Chem.AddHs(dimer)
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_STATE
        if AllChem.EmbedMolecule(mol_h, params) == -1:
            return dict(_D3D_DEFAULTS)
        if AllChem.MMFFOptimizeMolecule(mol_h) == -1:
            return dict(_D3D_DEFAULTS)

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h)
        if mmff_props is None:
            return dict(_D3D_DEFAULTS)
        conf = mol_h.GetConformer()
        dipole_vec = np.zeros(3)
        for atom in mol_h.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            dipole_vec += mmff_props.GetMMFFPartialCharge(idx) * np.array([pos.x, pos.y, pos.z])

        return {
            'd3d_dipole': float(np.linalg.norm(dipole_vec)),
            'd3d_radius_of_gyration': Descriptors3D.RadiusOfGyration(mol_h),
            'd3d_asphericity': Descriptors3D.Asphericity(mol_h),
            'd3d_eccentricity': Descriptors3D.Eccentricity(mol_h),
            'd3d_spherocity': Descriptors3D.SpherocityIndex(mol_h),
            'd3d_npr1': Descriptors3D.NPR1(mol_h),
            'd3d_npr2': Descriptors3D.NPR2(mol_h),
        }
    except Exception:
        return dict(_D3D_DEFAULTS)


def fit_feature_pruner(df):
    """Columns to keep after dropping near-constant columns and one column
    from each highly-correlated pair. Fit on train only to avoid leakage."""
    variances = df.var(numeric_only=True)
    keep = variances[variances > VAR_THRESH].index.tolist()

    corr = df[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
    keep = [c for c in keep if c not in to_drop]
    return keep


def compute_raw_features(df):
    """RDKit featurization only (baseline + MACCS + Gasteiger + Fragments +
    attachment-point + conjugation-extent), unpruned. Factored out of
    build_feature_matrix() (v2) so the cross-target accept/reject test
    below can score the pruned feature set with vs. without the
    cross-target block (see evaluate_cross_target_features) without paying
    RDKit's featurization cost twice. Takes a df with a 'smiles' column,
    handles the featurize()-can-return-None filtering itself, and returns
    (raw_df, valid_df)."""
    raw_feats = df['smiles'].apply(featurize)
    valid_mask = raw_feats.notna()
    if (~valid_mask).sum():
        print(f"  dropping {(~valid_mask).sum()} rows that failed to featurize")
    valid_df = df[valid_mask].reset_index(drop=True)
    raw_baseline_df = pd.DataFrame(list(raw_feats[valid_mask])).reset_index(drop=True)

    mols = valid_df['smiles'].apply(_parse_mol)
    maccs_df = pd.DataFrame([maccs_keys(m) for m in mols])
    gast_df = pd.DataFrame([gasteiger_features(m) for m in mols])
    frag_df = pd.DataFrame([fragment_counts(m) for m in mols])
    attach_df = pd.DataFrame([attachment_point_feats(m) for m in mols])
    conj_df = pd.DataFrame([conjugation_extent_feats(m) for m in mols])

    raw_df = pd.concat([raw_baseline_df, maccs_df, gast_df, frag_df, attach_df, conj_df], axis=1)
    return raw_df, valid_df


def build_cross_target_lookup(train_valid):
    """Wide canon_smiles -> {target_type: mean known target value} table
    (v2, re-opening CLAUDE.md's Step 12 as a real feature -- see module
    docstring). Built once from train_valid, mean-aggregated across any
    duplicate (canon, target_type) rows. The returned DataFrame's columns
    (sorted target_type names) are the contract cross_target_feats() below
    reads back -- no separate target-type list needs to be threaded
    through."""
    all_types = sorted(train_valid['target_type'].unique())
    wide = (train_valid.groupby(['canon', 'target_type'])['target']
            .mean().unstack('target_type').reindex(columns=all_types))
    return wide


def cross_target_feats(df, lookup):
    """Joins `lookup` onto df by canon SMILES: xtarget_<prop> (known value,
    NaN if this molecule has no train row under that target_type) +
    known_<prop> (0/1 presence flag) for every target_type in
    lookup.columns. df's own target_type column is always blanked to
    NaN/0 before returning, regardless of what the lookup table says --
    that column is the label this row is being scored on, and letting it
    through would be direct leakage, not signal, even though the
    aggregation in build_cross_target_lookup mixes in other rows' values
    for that same (canon, target_type) pair. Works unchanged for both
    train (real target_type per row) and test (target_type present, no
    target column needed -- lookup was built from train only)."""
    all_types = lookup.columns
    vals = lookup.reindex(df['canon'].values)
    vals.index = df.index
    for tt in all_types:
        own_mask = (df['target_type'] == tt).values
        if own_mask.any():
            vals.loc[own_mask, tt] = np.nan
    known = vals.notna().astype(int).add_prefix('known_')
    vals = vals.add_prefix('xtarget_')
    return pd.concat([vals, known], axis=1)


def get_xtarget_eval_splits(df, target_type, n_repeats=N_REPEATS_SMALL, base_seed=RANDOM_STATE):
    """3-seed repeat sampler for evaluate_cross_target_features specifically
    -- independent of the n_repeats_for/SMALL_TARGETS gating that limits
    repeats to the 5 small targets elsewhere in the harness (get_harness_splits).
    A single-seed (or small-target-only 3-seed) comparison here rejected the
    cross-target block on every target, but the delta pattern (biggest gains
    on eps/nc, near-zero on egc/tg) tracked the underlying chemistry too
    closely to be noise, and eps/nc's own noise floors were barely above
    their deltas -- 3 independent seeds per target (including the large
    ones), averaged, tightens that estimate. Returns seed_folds as a list of
    3 single-repeat fold lists (not pooled into one GroupKFold's worth of
    folds) so the caller can score+average per seed rather than pooling raw
    fold scores across seeds."""
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values
    seed_folds = []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=base_seed + r)
        seed_folds.append(list(gkf.split(np.zeros(len(sub_index)), groups=groups)))
    return sub_index, seed_folds


def paired_delta_verdict(tt, X_a, X_b, y_all, train_valid, t0, label='xtarget'):
    """One target's paired accept/reject of candidate block X_b against
    baseline X_a, on get_xtarget_eval_splits' 3-seed GroupKFold splits,
    scored with a single representative untuned LightGBM proxy rather than
    the full 10-model stack + live Optuna tuning: matches the convention
    scripts/ablation_study.py already established in this codebase for
    feature-block-level decisions specifically (as opposed to the
    per-model hyperparameter tuning elsewhere in this file, which
    necessarily retrains the exact model being tuned) -- re-running the
    entire stack+tuning pipeline twice just to test one feature block would
    cost as much as the rest of this script combined, for a decision
    that's binary either way.

    Paired, not independent, comparison: X_a and X_b are scored on the
    identical sub_index/seed_folds splits, so fold i's baseline score and
    fold i's candidate score share the exact same train/val split and the
    exact same fold-to-fold difficulty variance. Comparing b_mean - a_mean
    against max(a_std, b_std) (independent-samples logic) throws that
    pairing away and folds ordinary fold-to-fold difficulty swings into
    the noise floor alongside the thing actually being measured -- the
    candidate block's treatment effect. Instead, delta_i = b_score_i -
    a_score_i is computed per fold (via score_model_folds, which preserves
    fold order so the two raw score arrays line up positionally), and the
    effective delta/noise floor are mean(delta_i)/std(delta_i) across all
    folds (all 3 seeds x 5 splits) -- a paired t-test-style comparison,
    which is only ever equal to or *tighter* than the unpaired version,
    never looser: it cancels out shared fold-to-fold variance instead of
    letting it inflate the noise floor on both sides independently.

    Shared by evaluate_cross_target_features (baseline feature set vs.
    +cross-target-lookup block) and the dense pred_egc/pred_tg accept/
    reject in main() (a target's existing feature set vs. +pred_egc/
    pred_tg), which run the exact same statistical test against different
    candidate blocks.

    Returns (accepted, a_mean, b_mean, delta, noise_floor)."""
    sub_index, seed_folds = get_xtarget_eval_splits(train_valid, tt)
    factory = get_zoo_factories(tt, {})['LightGBM']

    a_fold_scores, b_fold_scores = [], []
    for folds in seed_folds:
        a_fold_scores.append(score_model_folds(factory, X_a, y_all, sub_index, [folds]))
        b_fold_scores.append(score_model_folds(factory, X_b, y_all, sub_index, [folds]))
    a_fold_scores = np.concatenate(a_fold_scores)
    b_fold_scores = np.concatenate(b_fold_scores)

    a_mean = float(a_fold_scores.mean())
    b_mean = float(b_fold_scores.mean())
    fold_deltas = b_fold_scores - a_fold_scores
    delta = float(fold_deltas.mean())
    noise_floor = float(fold_deltas.std())
    accepted = delta > noise_floor
    print(f"  {tt:5s}: baseline={a_mean:.4f}  +{label}={b_mean:.4f}  "
          f"paired delta={delta:+.4f} (+/-{noise_floor:.3f})  -> "
          f"{'ACCEPT' if accepted else 'reject'}  [{time.time()-t0:.0f}s]")
    return accepted, a_mean, b_mean, delta, noise_floor


def evaluate_cross_target_features(X_base, X_xt, y_all, train_valid, target_types, t0):
    """Step 12 re-open (v2): accept/reject the cross-target lookup block
    (see paired_delta_verdict for the statistical test itself) against
    each target's own noise floor, on the real harness -- not the cheap
    search split, exactly as asked.

    Returns a per-target verdict dict ({target_type: bool}), not a single
    pooled bool -- pooling the accept/reject decision to one mean-across-
    targets call let a strong target (e.g. egc/tg, where the xtarget block
    is genuinely near-zero signal) mask a real per-target effect elsewhere
    (eps/nc), or the reverse: a couple of targets with a real effect
    dragging the pooled mean over the line for targets where the block
    does nothing. Each target ships the xtarget block, or doesn't, purely
    on its own paired delta vs. its own paired noise floor -- main() then
    builds a separate feature matrix per target from this dict instead of
    one shared matrix for all 7."""
    print("\n" + "=" * 100)
    print("Step 12 re-open -- cross-target feature block, accept/reject vs. noise floor "
          "(LightGBM proxy, real harness, 3-seed paired delta)")
    print("=" * 100)

    base_means, xt_means = [], []
    verdicts = {}
    for tt in target_types:
        accepted, base_mean, xt_mean, _, _ = paired_delta_verdict(
            tt, X_base, X_xt, y_all, train_valid, t0, label='xtarget')
        verdicts[tt] = accepted
        base_means.append(base_mean)
        xt_means.append(xt_mean)

    shipped = [tt for tt in target_types if verdicts[tt]]
    print(f"\n  MEAN across {len(target_types)} targets (informational only -- ship/no-ship is "
          f"now decided per-target above, not pooled): baseline={float(np.mean(base_means)):.4f}  "
          f"+xtarget={float(np.mean(xt_means)):.4f}")
    print(f"  Cross-target features SHIP for: {shipped if shipped else '(none)'}")
    return verdicts


def build_feature_matrix(df, cross_target_lookup=None):
    """Single source of truth for the feature set -- baseline + MACCS +
    Gasteiger + Fragments (+ cross-target features, if cross_target_lookup
    is given), pruned. Takes a df with a 'smiles' column, handles the
    featurize()-can-return-None filtering itself, and returns (X,
    feature_cols, valid_df). main() calls compute_raw_features/
    fit_feature_pruner directly instead of this wrapper, so it can build
    the with- and without-cross-target matrices for
    evaluate_cross_target_features() from one shared RDKit pass; this
    wrapper stays as the simple single-call entry point other callers
    (e.g. scripts/ablation_study.py's convention) expect."""
    raw_df, valid_df = compute_raw_features(df)
    if cross_target_lookup is not None:
        raw_df = pd.concat([raw_df, cross_target_feats(valid_df, cross_target_lookup)], axis=1)
    feature_cols = fit_feature_pruner(raw_df)
    X = raw_df[feature_cols]
    return X, feature_cols, valid_df


# ---------------------------------------------------------------------------
# 2. CV harness -- GroupKFold on canonical SMILES (duplicate molecules with
#    conflicting repeat measurements must never be split across train/val,
#    or the model gets leakage from having seen the ~same target during
#    training). Repeated (5-fold x3 seeds) for the 5 small targets
#    (221-337 rows), since which molecules land in the validation fold
#    materially swings R2 at that scale.
# ---------------------------------------------------------------------------
def canonical_smiles(smiles):
    s = smiles.replace('[*]', '*')
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s.replace('*', 'C'))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_train_with_groups():
    train = pd.read_csv(TRAIN_PATH)
    # Stable row identity, captured before any filtering: this is the row's
    # position in the raw train.csv and it survives both the canon drop
    # below and compute_raw_features' separate featurize drop. The saved
    # GNN prediction columns (see load_gnn_predictions) are keyed on it --
    # gnn_prototype.py and this script filter rows at different points, so
    # positional alignment between the two frames is not safe.
    train['orig_row'] = np.arange(len(train))
    train['canon'] = train['smiles'].apply(canonical_smiles)
    n_bad = train['canon'].isna().sum()
    if n_bad:
        print(f"  dropping {n_bad} unparsable train rows")
        train = train[train['canon'].notna()].reset_index(drop=True)
    return train


def n_repeats_for(target_type):
    return N_REPEATS_SMALL if target_type in SMALL_TARGETS else 1


def get_harness_splits(df, target_type, base_seed=RANDOM_STATE):
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values

    repeats = []
    for r in range(n_repeats_for(target_type)):
        gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=base_seed + r)
        repeats.append(list(gkf.split(np.zeros(len(sub_index)), groups=groups)))
    return sub_index, repeats


def get_search_splits(df, target_type, n_splits=3, base_seed=RANDOM_STATE):
    """Cheap 3-fold GroupKFold splitter for Optuna's *inner* search loop
    only -- deliberately smaller/cheaper than get_harness_splits so the
    search stays fast. Never used to decide whether a tuned config beats
    the default; that comparison always runs on get_harness_splits."""
    sub = df[df['target_type'] == target_type]
    sub_index = sub.index.values
    groups = sub['canon'].values
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    return sub_index, list(gkf.split(np.zeros(len(sub_index)), groups=groups))


def constant_column_mask(Xtr):
    """Boolean column mask marking Xtr's non-constant, non-NaN-std columns
    -- factored out of drop_constant_columns so a caller that needs to
    apply the *same* fold-fitted mask to a different matrix later
    (generate_oof_with_fold_models, for the dense cross-target prediction
    features below) can reuse it instead of recomputing stats on the wrong
    array."""
    with np.errstate(invalid='ignore'):
        stds = np.nanstd(Xtr, axis=0)
    return (stds > 0) & ~np.isnan(stds)


def drop_constant_columns(Xtr, Xva):
    """A column that's constant within a training fold (even if it varies
    globally -- common on the small-target slices here) breaks SelectKBest's
    f_regression scorer (Ridge/KNN's feature-selection step): zero variance
    means a divide-by-zero in the correlation computation."""
    keep = constant_column_mask(Xtr)
    return Xtr[:, keep], Xva[:, keep]


def wrap_for_target(model_factory, target_type):
    """Wraps a model factory so the target gets log-transformed before fit
    and exp'd back after predict, for eps/ei -- no-op for the other 5."""
    if target_type not in TARGET_TRANSFORMS:
        return model_factory
    func, inverse_func = TARGET_TRANSFORMS[target_type]

    def wrapped():
        return TransformedTargetRegressor(
            regressor=model_factory(), func=func, inverse_func=inverse_func)
    return wrapped


# ---------------------------------------------------------------------------
# 3. Model zoo
# ---------------------------------------------------------------------------
def linear_models(seed_offset=0):
    """seed_offset shifts random_state for the final-refit bagging step
    (BAG_SEEDS) -- a no-op for Ridge's actual output here (our solver
    config has no real random_state-driven stochasticity), and KNN has no
    random_state at all, but keeping the parameter uniform across
    linear_models/tree_models keeps get_zoo_factories simple.

    ElasticNet removed from the zoo (scripts/ablation_study.py, Task 2):
    correlates with Ridge at 0.98-1.00 on every single target -- not
    similar, statistically indistinguishable -- and its own solo
    contribution to the stack was consistently near-zero-to-negative
    across all 7 targets, not just one. It also proved outright unstable
    as a meta-learner candidate (see select_meta_learner): L1 shrinkage on
    a highly multicollinear OOF matrix collapsed its score catastrophically
    on several targets. Dropping it removes a near-duplicate, low-value
    column feeding the meta-learner -- directly reduces the multicollinearity
    that drives coefficient instability, i.e. targets score *variance*, not
    just mean."""
    rs = RANDOM_STATE + seed_offset
    return {
        'Ridge': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('kbest', SelectKBest(f_regression, k=SELECT_K)),
            ('m', Ridge(alpha=5.0, random_state=rs)),
        ]),
        # KNN: genuinely different paradigm (instance-based/local structure)
        # from every other model in the zoo -- kept to k=SELECT_K dims via
        # the same SelectKBest step Ridge uses, so it stays fast and isn't
        # fighting the curse of dimensionality on ~650 raw columns.
        'KNN': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('kbest', SelectKBest(f_regression, k=SELECT_K)),
            ('m', KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=N_JOBS)),
        ]),
    }


def tree_models(seed_offset=0):
    """HGB removed from the zoo (scripts/ablation_study.py, Task 2):
    correlates with LightGBM at exactly 1.00 on every single target --
    the two produce statistically indistinguishable predictions, and
    LightGBM already gets tuned live (Task 3) while HGB never did. Same
    variance-reduction rationale as dropping ElasticNet above: one fewer
    near-duplicate column feeding the meta-learner."""
    rs = RANDOM_STATE + seed_offset
    return {
        'RF': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                n_jobs=N_JOBS, random_state=rs)),
        ]),
        # ExtraTrees: same family as RF but splits on randomly-chosen
        # thresholds instead of searching for the best one -- more
        # randomization, different bias/variance tradeoff, cheap to add
        # since it's the same cost order as RF.
        'ExtraTrees': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', ExtraTreesRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                n_jobs=N_JOBS, random_state=rs)),
        ]),
        'GBM': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=rs)),
        ]),
    }


def model_names_for(tt):
    """Per-target model roster. Currently uniform (the 8 always-on zoo
    models for every target_type) -- kept as a function rather than a
    bare MODEL_NAMES reference at each call site because
    build_dense_predictions/process_target/bagged_refit_predict use
    tt_source's own roster to build the base-model matrix a target's
    meta-learner expects, and this used to vary per target (a 9th,
    TANIMOTO_TARGETS-only model, removed after it proved net-negative on
    the final stack)."""
    return MODEL_NAMES


def boosting_default_params():
    return {
        'XGB': dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0),
        'CatBoost': dict(iterations=300, depth=6, learning_rate=0.05,
                          l2_leaf_reg=3.0, random_state=RANDOM_STATE, verbose=False,
                          thread_count=N_JOBS),
        'LightGBM': dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.8,
                          random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1),
    }


BOOSTING_CTORS = {'XGB': XGBRegressor, 'CatBoost': CatBoostRegressor, 'LightGBM': LGBMRegressor}

# Task 3 (CLAUDE.md task list) tuning extension: RF/ExtraTrees/GBM +
# Ridge/KNN get the same live-search-and-accept/reject treatment already
# applied to the 3 boosting models above. HGB and ElasticNet are excluded
# entirely (see MODEL_NAMES comment above -- both dropped from the zoo as
# near-duplicates of already-tuned siblings), so there's nothing left to
# tune for either.
TUNABLE_TREE_CTORS = {'RF': RandomForestRegressor, 'ExtraTrees': ExtraTreesRegressor,
                       'GBM': GradientBoostingRegressor}
TUNABLE_LINEAR_CTORS = {'Ridge': Ridge, 'KNN': KNeighborsRegressor}
EXTENDED_TUNABLE = list(TUNABLE_TREE_CTORS) + list(TUNABLE_LINEAR_CTORS)


def extended_default_params():
    """Current hardcoded defaults from tree_models()/linear_models(), as a
    plain dict per model -- the baseline extended_search_space's tuned
    dims get merged on top of, and what a target falls back to if nothing
    tuned clears the noise floor."""
    return {
        'RF': dict(n_estimators=400, max_depth=8, min_samples_leaf=2, n_jobs=N_JOBS),
        'ExtraTrees': dict(n_estimators=400, max_depth=8, min_samples_leaf=2, n_jobs=N_JOBS),
        'GBM': dict(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9,
                    min_samples_leaf=1),
        'Ridge': dict(alpha=5.0, select_k=SELECT_K),
        'KNN': dict(n_neighbors=10, select_k=SELECT_K, weights='distance'),
    }


def extended_search_space(trial, name):
    """RF/ExtraTrees/GBM: tree hyperparameters only, no SelectKBest --
    extending feature selection to these previously-uncapped tree models
    is exactly what an earlier attempt in this file's history (a blanket
    per-target SelectKBest cap applied to all 10 models) found actively
    harmful across every target, including ones it shouldn't have touched
    at all. Not repeating that here.

    Ridge/KNN: their own hyperparameters plus select_k, tuned per-model
    rather than one shared k -- each may have a genuinely different
    optimal feature count (KNN's distance metric degrades with
    dimensionality differently than Ridge's regularization does)."""
    if name in ('RF', 'ExtraTrees'):
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600),
            max_depth=trial.suggest_int('max_depth', 4, 15),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            n_jobs=N_JOBS,
        )
    if name == 'GBM':
        return dict(
            # n_estimators capped at 300 (not 500-800 like the other
            # boosting models) -- sklearn's GradientBoostingRegressor has
            # no histogram-based optimization like XGB/LightGBM/CatBoost,
            # and a high-n_estimators GBM search landed a single (GBM, tg)
            # combo at 37.6 minutes in a real run, ~20x every other combo's
            # cost. Capping here directly bounds that worst case instead of
            # hoping the sampler avoids it.
            n_estimators=trial.suggest_int('n_estimators', 100, 300),
            max_depth=trial.suggest_int('max_depth', 2, 6),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        )
    if name == 'Ridge':
        return dict(
            alpha=trial.suggest_float('alpha', 0.1, 50.0, log=True),
            select_k=trial.suggest_int('select_k', 30, 200),
        )
    if name == 'KNN':
        return dict(
            n_neighbors=trial.suggest_int('n_neighbors', 3, 30),
            select_k=trial.suggest_int('select_k', 30, 200),
            weights='distance',
        )
    raise ValueError(name)


def build_extended_pipeline(name, params, seed_offset=0):
    """Builds the Pipeline for one of the 6 newly-tunable models from a
    params dict (from extended_search_space, merged with defaults, or
    read back from accepted_tuned_configs) -- mirrors tree_models()'s and
    linear_models()'s existing Pipeline shapes exactly, just with tuned
    values swapped in for the hardcoded defaults."""
    rs = RANDOM_STATE + seed_offset
    if name in TUNABLE_TREE_CTORS:
        ctor = TUNABLE_TREE_CTORS[name]
        return Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('m', ctor(**params, random_state=rs)),
        ])
    ctor = TUNABLE_LINEAR_CTORS[name]
    params = dict(params)
    k = params.pop('select_k')
    if name != 'KNN':
        params['random_state'] = rs
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('kbest', SelectKBest(f_regression, k=k)),
        ('m', ctor(**params)),
    ])


def tune_extended_model(name, target_type, X_df, y_series, search_sub_index, search_folds,
                         n_trials=N_OPTUNA_TRIALS):
    """Same pattern as tune_boosting_model, applied to Task 3's 6 newly-
    tunable models."""
    def objective(trial):
        params = extended_search_space(trial, name)
        factory = wrap_for_target(lambda: build_extended_pipeline(name, params), target_type)
        mean_r2, _ = score_model(factory, X_df, y_series, search_sub_index, [search_folds])
        return mean_r2

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_all_extended_models(X_by_target, y_all, train_valid, target_types, t0):
    """Same accept/reject-against-the-real-harness pattern as
    tune_all_boosting_models, extended to RF/ExtraTrees/GBM/Ridge/KNN
    (HGB and ElasticNet excluded -- see EXTENDED_TUNABLE comment above).

    Search is gated to LARGE_TARGETS only: a full run against all 5 newly-
    tunable models x 35 trials x all 5 SMALL_TARGETS rejected against the noise floor on
    every single (model, target) combo -- pure wasted runtime on targets
    too small (221-337 rows) for a tuned config to ever clear the noise
    floor. Small targets get extended_default_params() directly, no search.

    X_by_target is a {target_type: DataFrame} map (evaluate_cross_target_features'
    verdict is per-target now, so each target may be scored on a different
    feature matrix -- X_base or X_xt), not one shared X for every target."""
    print("\n" + "=" * 100)
    print(f"Live Optuna tuning (Task 3 extension) -- {N_OPTUNA_TRIALS} trials, 3-fold search, "
          f"per newly-tunable model per target (LARGE_TARGETS only -- see docstring)")
    print("=" * 100)

    search_targets = [tt for tt in target_types if tt in LARGE_TARGETS]
    search_cache = {tt: get_search_splits(train_valid, tt) for tt in search_targets}
    harness_cache = {tt: get_harness_splits(train_valid, tt) for tt in search_targets}

    defaults = extended_default_params()
    accepted_tuned_configs = {}
    for name in EXTENDED_TUNABLE:
        for tt in search_targets:
            X = X_by_target[tt]
            search_sub_index, search_folds = search_cache[tt]
            best_params = tune_extended_model(name, tt, X, y_all, search_sub_index, search_folds)
            full_params = {**defaults[name], **best_params}

            harness_sub_index, harness_repeats = harness_cache[tt]
            default_factory = wrap_for_target(
                lambda p=defaults[name]: build_extended_pipeline(name, p), tt)
            default_mean, default_std = score_model(
                default_factory, X, y_all, harness_sub_index, harness_repeats)
            tuned_factory = wrap_for_target(
                lambda p=full_params: build_extended_pipeline(name, p), tt)
            tuned_mean, tuned_std = score_model(
                tuned_factory, X, y_all, harness_sub_index, harness_repeats)

            noise_floor = max(tuned_std, default_std)
            if tuned_mean - default_mean > noise_floor:
                accepted_tuned_configs[(name, tt)] = full_params
                verdict = f"ACCEPT tuned ({default_mean:.4f} -> {tuned_mean:.4f})"
            else:
                verdict = f"keep default (tuned {tuned_mean:.4f} vs default {default_mean:.4f}, " \
                          f"didn't clear noise floor {noise_floor:.4f})"
            print(f"  {name:10s} {tt:5s}: {verdict}  [{time.time()-t0:.0f}s]")

    return accepted_tuned_configs


def get_zoo_factories(tt, accepted_tuned_configs, seed_offset=0):
    """All model factories for one target_type -- the 8 always-on models.
    Boosting models (XGB/CatBoost/LightGBM) and Task 3's 5 newly-tunable
    models (RF/ExtraTrees/GBM/Ridge/KNN) use their accepted tuned config
    where one exists (accepted_tuned_configs, built live by
    tune_all_boosting_models()/tune_all_extended_models() below), default
    hyperparameters otherwise. Every factory wrapped with the eps/ei
    log-transform where applicable.

    seed_offset is used only by the final-refit bagging stage (BAG_SEEDS):
    OOF generation and meta-learner fitting always call this with the
    default seed_offset=0, so every score reported by this script is
    measured at one canonical seed."""
    factories = {}
    for name, pipeline_template in {**linear_models(seed_offset), **tree_models(seed_offset)}.items():
        tuned = accepted_tuned_configs.get((name, tt)) if name in EXTENDED_TUNABLE else None
        if tuned is not None:
            factories[name] = wrap_for_target(
                lambda p=tuned, n=name, so=seed_offset: build_extended_pipeline(n, p, so), tt)
        else:
            factories[name] = wrap_for_target(lambda p=pipeline_template: clone(p), tt)

    defaults = boosting_default_params()
    for name, ctor in BOOSTING_CTORS.items():
        base_params = accepted_tuned_configs.get((name, tt), defaults[name])
        params = {**base_params, 'random_state': RANDOM_STATE + seed_offset}
        factories[name] = wrap_for_target(lambda p=params, c=ctor: c(**p), tt)

    return factories


def boosting_search_space(trial, name):
    if name == 'XGB':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.01, 20.0, log=True),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
        )
    if name == 'CatBoost':
        return dict(
            # iterations capped at 400 (not 800) -- a (CatBoost, tg) combo
            # landed at 40.5 minutes in a real run, ~20x every other
            # combo's cost, almost entirely from trials near the top of
            # this range on tg's 4,143 rows. Capping directly bounds the
            # worst case rather than hoping the sampler avoids it.
            iterations=trial.suggest_int('iterations', 100, 400),
            depth=trial.suggest_int('depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 20.0, log=True),
            random_state=RANDOM_STATE, verbose=False, thread_count=N_JOBS,
        )
    if name == 'LightGBM':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            num_leaves=trial.suggest_int('num_leaves', 15, 255),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1,
        )
    raise ValueError(name)


def tune_boosting_model(name, target_type, X_df, y_series, search_sub_index, search_folds,
                         n_trials=N_OPTUNA_TRIALS):
    """Optuna TPE search over boosting_search_space, scored on a cheap
    3-fold GroupKFold search split (get_search_splits) -- never the real
    accept/reject harness, or a config that just got lucky on the search
    split would look like a genuine improvement."""
    ctor = BOOSTING_CTORS[name]

    def objective(trial):
        params = boosting_search_space(trial, name)
        factory = wrap_for_target(lambda: ctor(**params), target_type)
        mean_r2, _ = score_model(factory, X_df, y_series, search_sub_index, [search_folds])
        return mean_r2

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_all_boosting_models(X_by_target, y_all, train_valid, target_types, t0):
    """Runs the live Optuna search for every (boosting model, target_type)
    combo -- this IS the model-selection/tuning stage, executing here
    inside the single run rather than being replayed from an offline
    result (see module docstring). A tuned config only replaces the
    default if it beats the default's score on the real harness
    (get_harness_splits) by more than the noise floor (larger of the two
    configs' fold-to-fold std), so a config that got lucky on the cheap
    3-fold search doesn't get promoted.

    Search is gated to LARGE_TARGETS only: a full run against all 3
    boosting models x 35 trials x all 5 SMALL_TARGETS rejected against the
    noise floor on every single (model, target) combo -- pure wasted
    runtime on targets too small (221-337 rows) for a tuned config to ever
    clear the noise floor. Small targets get boosting_default_params()
    directly, no search.

    X_by_target is a {target_type: DataFrame} map (evaluate_cross_target_features'
    verdict is per-target now, so each target may be scored on a different
    feature matrix -- X_base or X_xt), not one shared X for every target."""
    print("\n" + "=" * 100)
    print(f"Live Optuna tuning -- {N_OPTUNA_TRIALS} trials, 3-fold search, "
          f"per boosting model per target (LARGE_TARGETS only -- see docstring)")
    print("=" * 100)

    # Per-target splits, computed once and reused across all 3 boosting
    # models below (they don't depend on model name), instead of redoing
    # the same GroupKFold split 3x per target.
    search_targets = [tt for tt in target_types if tt in LARGE_TARGETS]
    search_cache = {tt: get_search_splits(train_valid, tt) for tt in search_targets}
    harness_cache = {tt: get_harness_splits(train_valid, tt) for tt in search_targets}

    defaults = boosting_default_params()
    accepted_tuned_configs = {}
    for name in BOOSTING_CTORS:
        ctor = BOOSTING_CTORS[name]
        for tt in search_targets:
            X = X_by_target[tt]
            search_sub_index, search_folds = search_cache[tt]
            best_params = tune_boosting_model(name, tt, X, y_all, search_sub_index, search_folds)
            full_params = {**defaults[name], **best_params}

            harness_sub_index, harness_repeats = harness_cache[tt]
            default_factory = wrap_for_target(lambda p=defaults[name]: ctor(**p), tt)
            default_mean, default_std = score_model(
                default_factory, X, y_all, harness_sub_index, harness_repeats)
            tuned_factory = wrap_for_target(lambda p=full_params: ctor(**p), tt)
            tuned_mean, tuned_std = score_model(
                tuned_factory, X, y_all, harness_sub_index, harness_repeats)

            noise_floor = max(tuned_std, default_std)
            if tuned_mean - default_mean > noise_floor:
                accepted_tuned_configs[(name, tt)] = full_params
                verdict = f"ACCEPT tuned ({default_mean:.4f} -> {tuned_mean:.4f})"
            else:
                verdict = f"keep default (tuned {tuned_mean:.4f} vs default {default_mean:.4f}, " \
                          f"didn't clear noise floor {noise_floor:.4f})"
            print(f"  {name:9s} {tt:5s}: {verdict}  [{time.time()-t0:.0f}s]")

    return accepted_tuned_configs


def score_model_folds(model_factory, X_df, y_series, sub_index, repeats):
    """Same fold iteration as score_model, but returns the raw per-fold R2
    array instead of collapsing straight to (mean, std). Needed wherever a
    caller must pair per-fold scores from two different feature sets/models
    scored on the identical splits (e.g. evaluate_cross_target_features's
    paired baseline-vs-xtarget delta) -- comparing independent aggregate
    means/stds throws away the fact that fold i's baseline score and fold
    i's xtarget score came from the exact same train/val split."""
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
    return np.array(scores)


def score_model(model_factory, X_df, y_series, sub_index, repeats):
    scores = score_model_folds(model_factory, X_df, y_series, sub_index, repeats)
    return float(scores.mean()), float(scores.std())


def generate_oof(model_factory, X_df, y_series, sub_index, repeats):
    """Per-row OOF prediction, averaged across repeats -- leak-free input
    to the stacking meta-learner."""
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


def generate_oof_with_fold_models(model_factory, X_df, y_series, sub_index, repeats):
    """Same as generate_oof, but also returns every per-fold fitted model
    (paired with the exact column mask drop_constant_columns computed for
    that fold), instead of discarding them once the OOF array is filled.

    Needed for the dense cross-target prediction features (pred_egc/
    pred_tg/pred_nc, see build_dense_predictions below): a source
    target's own OOF array only covers its own training rows -- other
    targets' rows were never part of its own CV folds at all, so there's
    no OOF value to look up for them directly. Averaging predictions from
    every one of the source target's per-fold models (each fit on ~80% of
    its own data, never the full 100%) is the closest available
    approximation to genuine OOF-quality generalization for rows outside
    its own training set: unlike the single 100%-refit model used for the
    source target's own official test predictions, no individual fold
    model here has seen every one of its training molecules, which caps
    how much a molecule that also happens to appear (or nearly so, in
    descriptor space) under a different target_type can bias its own
    prediction."""
    n = len(sub_index)
    accum = np.zeros(n)
    fold_models = []
    for folds in repeats:
        fold_pred = np.full(n, np.nan)
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            Xtr_full = X_df.loc[tr_idx].values.astype(float)
            Xva_full = X_df.loc[va_idx].values.astype(float)
            col_mask = constant_column_mask(Xtr_full)
            Xtr, Xva = Xtr_full[:, col_mask], Xva_full[:, col_mask]
            ytr = y_series.loc[tr_idx].values
            model = model_factory()
            model.fit(Xtr, ytr)
            fold_pred[va_pos] = model.predict(Xva)
            fold_models.append((model, col_mask))
        accum += fold_pred
    oof = accum / len(repeats)
    return pd.Series(oof, index=sub_index), fold_models


def apply_fold_ensemble(fold_models, X_ext_df):
    """Averages predictions from every (fitted_model, column_mask) pair
    produced by generate_oof_with_fold_models, applied to X_ext_df -- rows
    outside the target_type those fold models were trained/validated on
    (a different target_type's own rows, scored off the row-level
    descriptor features every row already has regardless of its own
    target_type, since X_by_target[tt] spans the full train_valid index,
    not just tt's own subset)."""
    Xext = X_ext_df.values.astype(float)
    preds = np.zeros(len(Xext))
    for model, col_mask in fold_models:
        preds += model.predict(Xext[:, col_mask])
    return preds / len(fold_models)


def fit_predict_multi(model_factory, X_tr_df, y_tr_series, X_te_dfs, sample_weight=None):
    """One fit, several predict targets -- the shared implementation behind
    fit_predict_full below.

    Exists because evaluate_pl_conf_fraction needs each fold's base models
    to score the validation fold *and* rate the test rows' confidence, and
    calling fit_predict_full twice would train all 8 models twice for
    nothing. The training-fold constant-column mask is computed once and
    applied to every prediction matrix, exactly as drop_constant_columns
    does for the single-target case.

    sample_weight (used by pseudo-labeling below) needs a different fit()
    kwarg name depending on whether the model is a bare estimator
    ('sample_weight') or a Pipeline ('m__sample_weight', routing to the
    final step) -- and TransformedTargetRegressor (eps/ei) passes fit_params
    straight through to whichever of those its wrapped .regressor is, no
    extra prefix. Checked via the real fit() signature rather than
    hardcoding which models support it (KNeighborsRegressor doesn't --
    weighting isn't meaningful for a pure distance-based fit -- so it's
    silently fit unweighted rather than raising)."""
    Xtr = X_tr_df.values.astype(float)
    keep = constant_column_mask(Xtr)
    model = model_factory()

    fit_kwargs = {}
    if sample_weight is not None:
        inner = getattr(model, 'regressor', model)
        final_est = inner.named_steps['m'] if hasattr(inner, 'named_steps') else inner
        if 'sample_weight' in inspect.signature(final_est.fit).parameters:
            kwarg = 'm__sample_weight' if hasattr(inner, 'named_steps') else 'sample_weight'
            fit_kwargs[kwarg] = sample_weight

    model.fit(Xtr[:, keep], y_tr_series.values, **fit_kwargs)
    return [model.predict(X.values.astype(float)[:, keep]) for X in X_te_dfs]


def fit_predict_full(model_factory, X_tr_df, y_tr_series, X_te_df, sample_weight=None):
    return fit_predict_multi(
        model_factory, X_tr_df, y_tr_series, [X_te_df], sample_weight)[0]


def _find_gnn_file(name):
    """Locate a saved GNN prediction file next to the data or next to this
    script. Returns None if absent -- every GNN code path in this file is
    opt-in on these files existing, so the pipeline stays fully runnable
    (and submittable) without them."""
    candidates = [INPUT_DIR]
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        candidates.append(Path.cwd())
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return None


def load_gnn_predictions(tt, train_valid, test):
    """Load gnn_prototype.py's saved seed-bagged predictions for tt.

    IMPORTANT -- these files are an OFFLINE EVALUATION MECHANISM, not a
    submission path. Competition rule 6.2.3 requires the entire pipeline to
    execute inside the notebook with nothing loaded from outside, so
    uploading these CSVs alongside the kernel would be a rule violation.
    They exist so the GNN's contribution can be measured here without
    torch and LightGBM sharing a process (which segfaults on macOS -- see
    gnn_prototype.py). If the column is ever accepted, shipping it means
    training chemprop inside the notebook, not shipping these files.

    Returns (oof_series over train_valid's index, test_series over test's
    index) or None if the files are missing or don't cover tt's rows."""
    oof_path = _find_gnn_file(f"gnn_oof_{tt}.csv")
    test_path = _find_gnn_file(f"gnn_test_{tt}.csv")
    if oof_path is None or test_path is None:
        return None

    oof_raw = pd.read_csv(oof_path)
    test_raw = pd.read_csv(test_path)

    row_to_index = pd.Series(train_valid.index.values,
                             index=train_valid['orig_row'].to_numpy())
    mapped = oof_raw['orig_row'].map(row_to_index)
    keep = mapped.notna()
    oof = pd.Series(np.nan, index=train_valid.index, dtype=float)
    oof.loc[mapped[keep].astype(int).to_numpy()] = oof_raw.loc[keep, 'gnn_pred'].to_numpy()

    test_series = pd.Series(np.nan, index=test.index, dtype=float)
    in_range = test_raw['test_row'].isin(test.index)
    test_series.loc[test_raw.loc[in_range, 'test_row'].to_numpy()] = \
        test_raw.loc[in_range, 'gnn_pred'].to_numpy()

    own_rows = train_valid.index[train_valid['target_type'] == tt]
    if oof.loc[own_rows].isna().any():
        n_missing = int(oof.loc[own_rows].isna().sum())
        print(f"  {tt:5s}: GNN OOF file covers only "
              f"{len(own_rows) - n_missing}/{len(own_rows)} rows -- skipping GNN column")
        return None
    return oof, test_series


def evaluate_gnn_stack_column(tt, oof_df, gnn_oof, y_all, train_valid, t0):
    """Accept/reject the saved GNN prediction as a 9th column of tt's OOF
    matrix, against the real tuned 8-model stack -- not the deliberately
    weak untuned LightGBM proxy used for feature-block decisions elsewhere,
    which would overstate the gain by comparing against a much lower bar.

    Uses the SAME split structure and paired statistic as
    paired_delta_verdict: get_xtarget_eval_splits' 3 independent seeds x
    5 folds (15 folds), which forces 3 seeds even for the large targets --
    not get_harness_splits, whose large-target repeat gating collapses
    egc/tg to a single 5-fold seed. That distinction is the whole point of
    this function's history: an earlier version passed get_harness_splits'
    `repeats` straight through, so egc/tg's noise floor was the std of only
    5 paired deltas -- both a thinner estimate and biased low (the
    sample-std bias factor is ~0.94 at n=5 vs ~0.98 at n=15), which put the
    egc/tg floors at roughly half of every other paired accept/reject's
    floor on the same targets and same folds this session, i.e. a
    systematically too-easy bar. Scoring per seed and concatenating, as
    below, makes the floor directly comparable to the cross-target/dense/3D
    verdicts.

    The 8-model oof_df and the GNN column are both fixed per-row held-out
    predictions; re-splitting them across 3 seeds to CV the meta-learner
    treats the GNN column identically to the 8 base columns and introduces
    no leakage -- exactly what get_xtarget_eval_splits is for.

    Returns (accepted, candidate_oof_df)."""
    sub_index, seed_folds = get_xtarget_eval_splits(train_valid, tt)
    cand = oof_df.copy()
    cand['GNN'] = gnn_oof.loc[sub_index].to_numpy()
    meta = META_LEARNER_CANDIDATES['Ridge']

    a_scores, b_scores = [], []
    for folds in seed_folds:
        a_scores.append(score_model_folds(meta, oof_df, y_all, sub_index, [folds]))
        b_scores.append(score_model_folds(meta, cand, y_all, sub_index, [folds]))
    a = np.concatenate(a_scores)
    b = np.concatenate(b_scores)
    deltas = b - a
    delta, noise_floor = float(deltas.mean()), float(deltas.std())
    accepted = delta > noise_floor

    print(f"  {tt:5s}: GNN 9th-column test ({len(a)} folds, 3-seed paired) -- "
          f"8-model stack={a.mean():.4f}, +GNN={b.mean():.4f}, "
          f"delta={delta:+.4f} vs noise floor {noise_floor:.4f} "
          f"({'ACCEPT' if accepted else 'reject'}) [{time.time()-t0:.0f}s]")
    return accepted, cand


def evaluate_pl_conf_fraction(tt, X_tt, y_all, train_valid, X_te_tt,
                               accepted_tuned_configs, cv_r2, t0):
    """Live accept/reject search over PL_CONF_FRACTION for one target.

    WHY THIS NEEDS ITS OWN SIMULATION, unlike every other tuned decision in
    this file: pseudo-labeling reads *test.csv* rows and only ever changes
    the test-side prediction. It is not on the CV path at all -- no fold
    score anywhere in this script moves when PL_CONF_FRACTION changes, so
    score_model/paired_delta_verdict physically cannot rate it. To get a
    real harness-scored verdict, the whole pseudo-labeling round has to be
    replayed *inside* each fold: train the base models on the training
    fold, use them to rate and pseudo-label the real test rows exactly as
    production does, retrain on the augmented set, and score the held-out
    fold. That is what this does, on get_harness_splits' folds -- the same
    splits as everything else.

    Two deliberate simplifications, both applied identically to the default
    and to every candidate, so the *paired delta* (the thing the verdict
    turns on) stays meaningful even though the absolute R2 here won't match
    the production stacked number:
      - base models are combined by plain mean rather than by the fitted
        Ridge meta-learner. Using oof_meta[tt] would be more faithful to
        production, but it was fit on OOF predictions covering these very
        validation rows, so it would leak their labels into the simulation.
      - the per-fold base models are fit at a single seed, not bagged over
        BAG_SEEDS. pred_std here is the disagreement *between the 8 model
        families*, which is what the gate actually keys on and is
        dominated by family differences rather than seed noise; bagging
        would multiply this search's cost by 10 to shave a little off it.

    Returns the fraction to ship for this target."""
    sub_index, repeats = get_harness_splits(train_valid, tt)
    names = model_names_for(tt)
    factories = get_zoo_factories(tt, accepted_tuned_configs)
    clip_lo, clip_hi = clip_bounds_for(tt, y_all, train_valid)
    fracs = [PL_CONF_FRACTION] + PL_CONF_CANDIDATES
    per_frac = {f: [] for f in fracs}

    for folds in repeats:
        for tr_pos, va_pos in folds:
            tr_idx, va_idx = sub_index[tr_pos], sub_index[va_pos]
            X_tr, X_va = X_tt.loc[tr_idx], X_tt.loc[va_idx]
            y_tr, y_va = y_all.loc[tr_idx], y_all.loc[va_idx].values

            # One unaugmented pass per fold: same fitted models give both
            # the val-fold reference predictions and the test-row spread
            # the confidence gate thresholds on.
            va_cols, te_cols = [], []
            for n in names:
                p_va, p_te = fit_predict_multi(factories[n], X_tr, y_tr, [X_va, X_te_tt])
                va_cols.append(p_va)
                te_cols.append(p_te)
            base_te = np.column_stack(te_cols)
            pred_std = base_te.std(axis=1)
            first_pass = base_te.mean(axis=1)
            unaug_score = r2_score(y_va, np.column_stack(va_cols).mean(axis=1))
            expected_resid = y_tr.std() * np.sqrt(max(1 - cv_r2, 0.01))

            # Distinct candidates often select the identical test rows on a
            # set this small; score each distinct mask once.
            seen = {}
            for f in fracs:
                conf_mask = pred_std < f * expected_resid
                key = conf_mask.tobytes()
                if key in seen:
                    per_frac[f].append(seen[key])
                    continue
                if conf_mask.sum() < PL_MIN_ROWS:
                    score = unaug_score
                else:
                    pseudo_y = np.clip(first_pass[conf_mask], clip_lo, clip_hi)
                    X_aug = pd.concat([X_tr, X_te_tt.loc[conf_mask]], ignore_index=True)
                    y_aug = pd.concat([y_tr, pd.Series(pseudo_y)], ignore_index=True)
                    w_aug = np.concatenate([
                        np.ones(len(y_tr)),
                        np.full(int(conf_mask.sum()), PL_SAMPLE_WEIGHT)])
                    aug_va = np.column_stack([
                        fit_predict_multi(factories[n], X_aug, y_aug, [X_va], w_aug)[0]
                        for n in names])
                    score = r2_score(y_va, aug_va.mean(axis=1))
                seen[key] = score
                per_frac[f].append(score)

    default_scores = np.array(per_frac[PL_CONF_FRACTION])
    best_frac, best_delta = PL_CONF_FRACTION, 0.0
    print(f"  {tt:5s}: PL_CONF_FRACTION search over {fracs} "
          f"({len(default_scores)} folds, simulated PL round per fold) "
          f"[{time.time()-t0:.0f}s]")
    print(f"    default {PL_CONF_FRACTION:.2f}: sim mean R2={default_scores.mean():.4f}")
    for f in PL_CONF_CANDIDATES:
        cand = np.array(per_frac[f])
        # Same paired test as paired_delta_verdict: per-fold differences on
        # identical splits, compared against the spread of those
        # differences, not against the two configs' independent stds.
        deltas = cand - default_scores
        delta, noise_floor = float(deltas.mean()), float(deltas.std())
        cleared = delta > noise_floor
        print(f"    cand    {f:.2f}: sim mean R2={cand.mean():.4f}  "
              f"delta={delta:+.4f} vs noise floor {noise_floor:.4f} "
              f"({'ACCEPT' if cleared else 'reject'})")
        if cleared and delta > best_delta:
            best_frac, best_delta = f, delta

    print(f"  {tt:5s}: PL_CONF_FRACTION -> {best_frac:.2f}"
          f"{' (default kept)' if best_frac == PL_CONF_FRACTION else ' (ACCEPTED)'}")
    return best_frac


def bagged_refit_predict(tt, X_tr, y_tr, X_te, accepted_tuned_configs, seeds, sample_weight=None):
    """Bags the final refit across seeds -- averages away each tree/
    boosting model's own bootstrap/split-order randomness. Only this stage
    is bagged, not OOF generation or meta-learner fitting, which stay at
    the single canonical seed every score in this script is measured
    against.

    Factored out of main()'s per-target test-prediction loop (used to be a
    closure over that loop's own X_te_tt/tt) so it can also be called with
    a *different* target_type's rows as X_te -- needed by
    build_dense_predictions below to compute the test-side half of the
    dense pred_egc/pred_tg features using egc/tg's own already-trained
    bagged ensemble, not just tt's own test set."""
    names = model_names_for(tt)
    preds = np.zeros((len(X_te), len(names)))
    for seed_offset in seeds:
        factories = get_zoo_factories(tt, accepted_tuned_configs, seed_offset)
        preds += np.column_stack([
            fit_predict_full(factories[name], X_tr, y_tr, X_te, sample_weight)
            for name in names
        ])
    return preds / len(seeds)


# Task 3 (CLAUDE.md task list) meta-learner alternative. Deliberately no
# tree-based candidate (XGB/CatBoost/LightGBM): the OOF matrix for the
# small targets is ~220-340 rows x 10 columns, and a tree-based stacker has
# more than enough capacity to fit noise at that size -- a hard constraint,
# not a style preference. ElasticNet was also tested here and removed: on a
# highly multicollinear OOF matrix (base models correlate at 0.9+ with each
# other), its L1 shrinkage collapsed catastrophically on several targets
# (e.g. nc: mean=-0.0221 vs Ridge's 0.8234) -- not just "didn't win," an
# actively unstable candidate not worth continuing to test.
META_LEARNER_CANDIDATES = {
    'Ridge': lambda: Ridge(alpha=META_ALPHA),
    'Ridge_positive': lambda: Ridge(alpha=META_ALPHA, positive=True),
}


def select_meta_learner(oof_df, y_all, sub_index, repeats):
    """Tests Ridge(positive=True) against the default Ridge, on the OOF
    matrix via the real harness -- same accept/reject-against-noise-floor
    pattern used for every tuned hyperparameter in this script, applied to
    the meta-learner choice instead. The alternative only replaces Ridge
    if it beats Ridge's mean by more than the noise floor (larger of the
    two configs' fold-to-fold std)."""
    default_mean, default_std = score_model(
        META_LEARNER_CANDIDATES['Ridge'], oof_df, y_all, sub_index, repeats)
    best_name, best_mean, best_std = 'Ridge', default_mean, default_std

    for name in ('Ridge_positive',):
        mean, std = score_model(META_LEARNER_CANDIDATES[name], oof_df, y_all, sub_index, repeats)
        noise_floor = max(std, default_std)
        cleared = mean - default_mean > noise_floor
        print(f"    meta-learner {name:14s}: mean={mean:.4f} vs Ridge {default_mean:.4f} "
              f"({'ACCEPT' if cleared else 'reject'}, noise floor {noise_floor:.4f})")
        if cleared and mean > best_mean:
            best_name, best_mean, best_std = name, mean, std

    return best_name, META_LEARNER_CANDIDATES[best_name], best_mean, best_std


def clip_bounds_for(tt, y_all, train_valid):
    """Train-range + margin clip bounds for one target_type, with hard
    physical floors where they apply. Factored out of the final
    output-safety clip so pseudo-labeling below can reuse the exact same
    bounds to safety-clip pseudo-labels before they're used as training
    signal -- a pseudo-label that's wildly out of physical range shouldn't
    become training data just because it happened to pass the confidence
    gate."""
    y = y_all.loc[train_valid['target_type'] == tt]
    lo, hi = y.min(), y.max()
    margin = MARGIN_FRACTION * (hi - lo)
    clip_lo, clip_hi = lo - margin, hi + margin
    if tt in PHYSICAL_FLOORS:
        clip_lo = max(clip_lo, PHYSICAL_FLOORS[tt])
    return clip_lo, clip_hi


def process_target(tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
                    test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
                    oof_meta, cv_scores, test_predictions, t0,
                    track_fold_models=False):
    """OOF stacking (generate OOF for every one of tt's own base models --
    see model_names_for -- select + fit the meta-learner) immediately
    followed by that target's own
    full-refit bagged test prediction (with the same pseudo-labeling
    second pass this script has always run) -- one combined per-target
    step. main() used to run these as two separate loops, each over all 7
    targets; now it's one call per target, so a caller can run it for
    LARGE_TARGETS first and the remaining 5 targets second (needed for the
    dense pred_egc/pred_tg features, which depend on egc/tg's own
    finished training before the other 5 targets' feature matrices can be
    finalized), and again for nc specifically ahead of eps within the
    small-targets phase (needed for the dense pred_nc feature, same
    pattern one level deeper -- see build_dense_predictions and main()).

    Mutates oof_meta/cv_scores/test_predictions in place -- the same
    dicts/array main() threads through the rest of the pipeline
    regardless of which phase a target is processed in.

    When track_fold_models is True, also returns (X_tr_tt, y_tr_tt,
    fold_models_by_name) so the caller can build dense cross-target
    features from tt's own trained models (see build_dense_predictions);
    returns (X_tr_tt, y_tr_tt, None) otherwise. Only ever True for
    LARGE_TARGETS and nc -- the only targets that currently source a
    pred_ feature for anyone else."""
    sub_index, repeats = get_harness_splits(train_valid, tt)
    names = model_names_for(tt)
    factories = get_zoo_factories(tt, accepted_tuned_configs)
    X_tt = X_by_target[tt]

    fold_models_by_name = {} if track_fold_models else None
    oof_cols = {}
    for name in names:
        if track_fold_models:
            oof_series, fold_models = generate_oof_with_fold_models(
                factories[name], X_tt, y_all, sub_index, repeats)
            fold_models_by_name[name] = fold_models
        else:
            oof_series = generate_oof(factories[name], X_tt, y_all, sub_index, repeats)
        oof_cols[name] = oof_series
    oof_df = pd.DataFrame(oof_cols)[names]
    print(f"  {tt:5s}: OOF generated for all {len(names)} models [{time.time()-t0:.0f}s]")

    # ---- GNN as a 9th stack column (egc/tg only, and only if
    # gnn_prototype.py --export has been run). Decided live by the same
    # paired accept/reject as every other optional block here, but against
    # the real tuned stack rather than a proxy model. ----
    gnn_test_col = None
    if tt in GNN_STACK_TARGETS:
        loaded = load_gnn_predictions(tt, train_valid, test)
        if loaded is None:
            print(f"  {tt:5s}: no saved GNN predictions found -- 8-model stack unchanged")
        else:
            gnn_oof, gnn_test = loaded
            accepted, cand_oof = evaluate_gnn_stack_column(
                tt, oof_df, gnn_oof, y_all, train_valid, t0)
            if accepted:
                if track_fold_models:
                    # The dense pred_<tt> feature evaluates this target's
                    # meta-learner on *other* targets' rows, which the saved
                    # GNN column doesn't cover (it spans tt's own rows only).
                    # Rather than materializing GNN predictions for every
                    # other target's molecules, the dense feature keeps using
                    # an 8-model meta-learner -- a deliberately small
                    # difference, since that feature is an input to other
                    # targets, not tt's own output.
                    dense_meta = META_LEARNER_CANDIDATES['Ridge']()
                    dense_meta.fit(oof_df.values, y_all.loc[sub_index].values)
                    oof_meta[(tt, 'dense')] = dense_meta
                oof_df = cand_oof
                gnn_test_col = gnn_test

    # Meta-learner selection (Task 3): tests Ridge(positive=True) against
    # the default Ridge on the OOF matrix via the real harness, same
    # accept/reject-against-noise-floor pattern as every tuned
    # hyperparameter above. cv_scores[tt] is the number that estimates
    # what the competition metric will actually show, not just a single
    # model's.
    meta_name, meta_ctor, meta_mean, meta_std = select_meta_learner(
        oof_df, y_all, sub_index, repeats)
    cv_scores[tt] = (meta_mean, meta_std)
    print(f"  {tt:5s}: meta-learner selected: {meta_name} (R2={meta_mean:.4f}) "
          f"[{time.time()-t0:.0f}s]")

    meta_final = meta_ctor()
    meta_final.fit(oof_df.values, y_all.loc[sub_index].values)
    oof_meta[tt] = meta_final

    feature_cols_tt = feature_cols_by_target[tt]
    test_feat_df = test_feat_df_by_target[tt]
    X_tr_tt = X_tt.loc[sub_index]
    y_tr_tt = y_all.loc[sub_index]

    mask = (test['target_type'] == tt).values
    rows_valid = mask & test_valid_mask.values
    rows_invalid = mask & (~test_valid_mask.values)

    if rows_valid.sum() > 0:
        X_te_tt = test_feat_df.loc[rows_valid, feature_cols_tt]

        def _with_gnn(base_matrix):
            """Append the saved GNN test column when it was accepted into
            this target's stack, so the matrix width matches the 9-column
            matrix oof_meta[tt] was fitted on. Unlike the 8 base models the
            GNN column is not retrained during the pseudo-labeling round
            below -- it is a fixed saved array, so it contributes the same
            values to both passes."""
            if gnn_test_col is None:
                return base_matrix
            col = gnn_test_col.reindex(X_te_tt.index).to_numpy()
            return np.column_stack([base_matrix, col])

        base_only_preds = bagged_refit_predict(
            tt, X_tr_tt, y_tr_tt, X_te_tt, accepted_tuned_configs, BAG_SEEDS)
        base_test_preds = _with_gnn(base_only_preds)
        first_pass_preds = oof_meta[tt].predict(base_test_preds)

        # ---- pseudo-labeling: test.csv is competition-provided, not
        # external data (compliant with Section 6.2.1) -- use the rows
        # this model is confident about as extra training signal for the
        # base models specifically. Confidence = how much the 8
        # independently-trained base models (already bagged) agree with
        # each other on that row; the threshold is relative to this
        # target's own CV-estimated typical residual size
        # (cv_scores[tt]), so small/noisy targets get a naturally
        # stricter bar than well-behaved ones, not a fixed constant.
        # Deliberately does NOT refit oof_meta[tt] on anything -- it
        # stays exactly as fit on the real labeled OOF above, applied
        # unchanged to the retrained base models' new outputs. Refitting
        # it here would need a second real OOF pass to stay leak-free;
        # keeping it fixed is simpler and avoids silently pretending to
        # do that (see prajwal_baseline_v2.py's code review this
        # session for exactly that bug: a fake meta-learner "refit" that
        # silently reused the old OOF instead of actually incorporating
        # pseudo-label information).
        # Disagreement is measured over the 8 base models only, never the
        # appended GNN column. PL_CONF_FRACTION's 0.5 was calibrated
        # against 8-model spread; letting a 9th, deliberately dissimilar
        # model widen that spread would silently re-tune the gate (fewer
        # rows passing) as a side effect of accepting the GNN, rather than
        # as a decision anyone made.
        pred_std = base_only_preds.std(axis=1)
        expected_resid = y_tr_tt.std() * np.sqrt(max(1 - cv_scores[tt][0], 0.01))
        pl_frac = PL_CONF_FRACTION
        if tt in PL_CONF_SEARCH_TARGETS:
            pl_frac = evaluate_pl_conf_fraction(
                tt, X_tt, y_all, train_valid, X_te_tt,
                accepted_tuned_configs, cv_scores[tt][0], t0)
        conf_mask = pred_std < pl_frac * expected_resid

        if conf_mask.sum() >= PL_MIN_ROWS:
            clip_lo, clip_hi = clip_bounds_for(tt, y_all, train_valid)
            pseudo_y = np.clip(first_pass_preds[conf_mask], clip_lo, clip_hi)

            X_pl = X_te_tt.loc[conf_mask]
            X_aug = pd.concat([X_tr_tt, X_pl], ignore_index=True)
            y_aug = pd.concat([y_tr_tt, pd.Series(pseudo_y)], ignore_index=True)
            w_aug = np.concatenate([
                np.ones(len(y_tr_tt)), np.full(int(conf_mask.sum()), PL_SAMPLE_WEIGHT),
            ])

            base_test_preds_pl = _with_gnn(bagged_refit_predict(
                tt, X_aug, y_aug, X_te_tt, accepted_tuned_configs, PL_BAG_SEEDS, w_aug))
            print(f"  {tt:5s}: {int(conf_mask.sum())}/{len(conf_mask)} test rows "
                  f"pseudo-labeled, base models retrained [{time.time()-t0:.0f}s]")
            test_predictions[rows_valid] = oof_meta[tt].predict(base_test_preds_pl)
        else:
            test_predictions[rows_valid] = first_pass_preds

    if rows_invalid.sum() > 0:
        test_predictions[rows_invalid] = y_tr_tt.mean()
    print(f"  {tt:5s} done [{time.time()-t0:.0f}s]")

    return X_tr_tt, y_tr_tt, fold_models_by_name


def build_dense_predictions(tt_source, X_tr_source, y_tr_source, fold_models_by_name,
                             oof_meta, X_by_target, test_feat_df_by_target,
                             feature_cols_by_target, accepted_tuned_configs,
                             train_valid, test, other_target_types, t0):
    """Builds pred_<tt_source> -- dense: every row of every target_type in
    other_target_types gets a value, no exact-canonical-SMILES-match
    requirement, unlike the sparse xtarget_ lookup columns -- from
    tt_source's (egc/tg, or nc for the nc->eps chain) own already-finished
    training:

    * train-side, via the K-fold ensemble of tt_source's own per-fold
      models (fold_models_by_name, from generate_oof_with_fold_models --
      no single model here saw 100% of tt_source's own training data, so a
      molecule that also happens to appear under a different target_type
      can't be memorized the way a single 100%-refit model could
      memorize it, which is what would otherwise let the other targets'
      own CV indirectly leak tt_source's training labels for their
      overlapping molecules);
    * test-side, via tt_source's existing 100%-refit bagged ensemble
      (bagged_refit_predict, the same mechanism used for tt_source's own
      official test predictions) -- no leakage concern applies there,
      since test rows of any target_type were never part of tt_source's
      training set regardless of which model produces the feature.

    Both halves are then passed through tt_source's already-fitted
    meta-learner (oof_meta[tt_source]) so the feature represents the full
    stacked ensemble's estimate, not one proxy base model -- which means
    the base-model matrix built here must use tt_source's own model
    roster (model_names_for(tt_source), 9 columns when tt_source is nc,
    since nc's own meta-learner was fit on a 9-column OOF matrix), not the
    fixed 8-model MODEL_NAMES.

    Feature-vector lookup needs no SMILES matching either: X_by_target[tt]
    and test_feat_df_by_target[tt] already span the *entire* train_valid /
    test row range (every target_type's rows, not just tt's own subset),
    since descriptor features are computed per-molecule independent of
    which property that row happens to measure -- so tt_source's own
    feature columns (including, for nc, its own fp_ selector's column
    positions) are already sitting there for every other target's rows
    too, just never used for tt_source's own training/validation.

    Returns (train_series, test_series), each named f'pred_{tt_source}'
    and indexed over other_target_types' own rows only."""
    # (tt_source, 'dense') is present only when tt_source accepted a GNN
    # column into its own stack; it is that target's meta-learner refit on
    # the 8 base models alone, because the GNN column doesn't span the
    # other targets' rows this feature is built for. Falls back to the
    # target's own meta-learner in every other case, which is what this
    # always used before.
    meta_final = oof_meta.get((tt_source, 'dense'), oof_meta[tt_source])
    X_source_full = X_by_target[tt_source]
    names = model_names_for(tt_source)

    other_train_index = train_valid.index[train_valid['target_type'].isin(other_target_types)]
    X_ext_train = X_source_full.loc[other_train_index]
    base_ext_train = pd.DataFrame(
        {name: apply_fold_ensemble(fold_models_by_name[name], X_ext_train) for name in names},
        index=other_train_index,
    )[names]
    train_series = pd.Series(
        meta_final.predict(base_ext_train.values), index=other_train_index,
        name=f'pred_{tt_source}')

    other_test_index = test.index[test['target_type'].isin(other_target_types)]
    feature_cols_source = feature_cols_by_target[tt_source]
    X_te_ext = test_feat_df_by_target[tt_source].loc[other_test_index, feature_cols_source]
    base_ext_test = bagged_refit_predict(
        tt_source, X_tr_source, y_tr_source, X_te_ext, accepted_tuned_configs, BAG_SEEDS)
    test_series = pd.Series(
        meta_final.predict(base_ext_test), index=other_test_index,
        name=f'pred_{tt_source}')

    print(f"  pred_{tt_source}: dense feature built for {len(other_train_index)} train rows, "
          f"{len(other_test_index)} test rows [{time.time()-t0:.0f}s]")
    return train_series, test_series


def apply_dense_sources(source_fits, consumers, X_by_target, feature_cols_by_target,
                        test_feat_df_by_target, oof_meta, y_all, train_valid, test,
                        accepted_tuned_configs, t0, label):
    """Build a dense pred_<source> column from each already-trained source
    target and offer them, together, as candidate features to each consumer
    target -- accepting per consumer via the same paired_delta_verdict gate
    the pred_egc/pred_tg and pred_nc blocks use. Mutates X_by_target /
    feature_cols_by_target / test_feat_df_by_target in place for the
    consumers that clear their own noise floor. Returns {consumer: bool}.

    Factored out so a second group of sources (egb/eea -> nc/ei/eps) can
    reuse the identical build+prune+gate+augment logic as the original
    egc/tg block, rather than duplicating it a third time."""
    dense_train_cols, dense_test_cols = {}, {}
    for tt_source, (X_tr_s, y_tr_s, fold_models) in source_fits.items():
        tr_series, te_series = build_dense_predictions(
            tt_source, X_tr_s, y_tr_s, fold_models, oof_meta,
            X_by_target, test_feat_df_by_target, feature_cols_by_target,
            accepted_tuned_configs, train_valid, test, set(consumers), t0)
        dense_train_cols[tr_series.name] = tr_series
        dense_test_cols[te_series.name] = te_series

    verdicts = {}
    for tt in consumers:
        own_tr = train_valid.index[train_valid['target_type'] == tt]
        own_te = test.index[test['target_type'] == tt]
        new_tr = pd.DataFrame({n: s.loc[own_tr] for n, s in dense_train_cols.items()})
        cand_raw = pd.concat([X_by_target[tt], new_tr], axis=1)
        cand_cols = fit_feature_pruner(cand_raw)
        cand_X = cand_raw[cand_cols]
        accepted, _, _, _, _ = paired_delta_verdict(
            tt, X_by_target[tt], cand_X, y_all, train_valid, t0, label=label)
        verdicts[tt] = accepted
        if accepted:
            X_by_target[tt] = cand_X
            feature_cols_by_target[tt] = cand_cols
            new_te = pd.DataFrame({n: s.loc[own_te] for n, s in dense_test_cols.items()})
            test_feat_df_by_target[tt] = pd.concat(
                [test_feat_df_by_target[tt], new_te.reindex(test.index)], axis=1)
    return verdicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    print(f"Reading data from: {INPUT_DIR}")
    train = load_train_with_groups()
    print(f"Loaded train={train.shape}")

    print("Featurizing train (baseline + MACCS + Gasteiger + Fragments + "
          "attachment-point + conjugation-extent)...")
    raw_df, train_valid = compute_raw_features(train)
    y_all = train_valid['target']
    target_types = sorted(train_valid['target_type'].unique())
    print(f"  {raw_df.shape[1]} raw features before pruning [{time.time()-t0:.0f}s]")

    # ---- Step 12 re-open (v2): build the cross-target lookup + feature
    # block, then accept/reject it against the noise floor on the real
    # harness before it's allowed to feed the (expensive) tuning/stacking
    # pipeline below -- see evaluate_cross_target_features docstring for
    # why this uses a cheap LightGBM proxy instead of the full stack. ----
    cross_target_lookup = build_cross_target_lookup(train_valid)

    feature_cols_base = fit_feature_pruner(raw_df)
    X_base = raw_df[feature_cols_base]

    xt_block = cross_target_feats(train_valid, cross_target_lookup)
    raw_df_xt = pd.concat([raw_df, xt_block], axis=1)
    feature_cols_xt = fit_feature_pruner(raw_df_xt)
    X_xt = raw_df_xt[feature_cols_xt]

    use_cross_target = evaluate_cross_target_features(
        X_base, X_xt, y_all, train_valid, target_types, t0)

    # Per-target feature matrix: each target_type ships X_xt/feature_cols_xt
    # if its own paired verdict was ACCEPT, X_base/feature_cols_base
    # otherwise -- evaluate_cross_target_features' verdict is per-target
    # now (see its docstring), not one shared feature set pooled across
    # all 7 targets.
    X_by_target = {tt: (X_xt if use_cross_target[tt] else X_base) for tt in target_types}
    feature_cols_by_target = {
        tt: (feature_cols_xt if use_cross_target[tt] else feature_cols_base)
        for tt in target_types
    }
    shipped = [tt for tt in target_types if use_cross_target[tt]]
    print(f"\n  Cross-target features shipped for: {shipped if shipped else '(none)'} "
          f"[{time.time()-t0:.0f}s]")
    for tt in target_types:
        print(f"    {tt:5s}: {len(feature_cols_by_target[tt])} features "
              f"({'WITH' if use_cross_target[tt] else 'WITHOUT'} cross-target block)")

    # ---- 3D-conformer descriptor block (D3D_TARGETS only -- eps/ei/nc,
    # see conformer_3d_feats/D3D_TARGETS): ETKDG-embedded + MMFF-optimized
    # dipole moment + RDKit Descriptors3D shape descriptors. Real
    # per-molecule wall-clock cost (embedding + force-field optimization),
    # so computed only for these 3 targets' own rows -- unlike every
    # feature block above, which is computed once for the whole
    # train_valid frame and shared across all 7 targets. A pure
    # structural property of the molecule itself, with no dependence on
    # any trained model (unlike the dense pred_egc/pred_tg/pred_nc
    # features below), so it's decided here, before tuning -- same
    # placement as the cross-target block above -- so an accepted block
    # actually gets to feed the tuning phase, not just the final stack. ----
    print("\n" + "=" * 100)
    print(f"3D-conformer descriptor block -- D3D_TARGETS only ({sorted(D3D_TARGETS)})")
    print("=" * 100)

    d3d_row_index = train_valid.index[train_valid['target_type'].isin(D3D_TARGETS)]
    d3d_train_df = pd.DataFrame(
        [conformer_3d_feats(s) for s in train_valid.loc[d3d_row_index, 'smiles']],
        index=d3d_row_index)
    n_embedded = int(d3d_train_df.notna().all(axis=1).sum())
    print(f"  D3D features computed for {len(d3d_row_index)} train rows, "
          f"{n_embedded} fully embedded (rest fell back to NaN) [{time.time()-t0:.0f}s]")

    d3d_accepted = {}
    for tt in sorted(D3D_TARGETS):
        candidate_raw = pd.concat([X_by_target[tt], d3d_train_df], axis=1)
        candidate_cols = fit_feature_pruner(candidate_raw)
        candidate_X = candidate_raw[candidate_cols]
        accepted, _, _, _, _ = paired_delta_verdict(
            tt, X_by_target[tt], candidate_X, y_all, train_valid, t0, label='d3d')
        d3d_accepted[tt] = accepted
        if accepted:
            X_by_target[tt] = candidate_X
            feature_cols_by_target[tt] = candidate_cols

    shipped_d3d = [tt for tt in sorted(D3D_TARGETS) if d3d_accepted[tt]]
    print(f"\n  3D-conformer descriptor block SHIPS for: {shipped_d3d if shipped_d3d else '(none)'}")
    for tt in sorted(D3D_TARGETS):
        print(f"    {tt:5s}: {len(feature_cols_by_target[tt])} features "
              f"({'WITH' if d3d_accepted[tt] else 'WITHOUT'} 3D-conformer block)")

    # ---- live hyperparameter tuning (must run in this execution -- see
    # module docstring's rules-compliance note) ----
    accepted_tuned_configs = tune_all_boosting_models(
        X_by_target, y_all, train_valid, target_types, t0)
    accepted_tuned_configs.update(
        tune_all_extended_models(X_by_target, y_all, train_valid, target_types, t0))
    print(f"\n  {len(accepted_tuned_configs)} accepted tuned config(s): "
          f"{list(accepted_tuned_configs.keys())}")

    # ---- featurize test (moved ahead of OOF stacking: doesn't depend on
    # any model fitting, and the LARGE_TARGETS-first processing below needs
    # test features ready before either phase's model fitting starts) ----
    print("\nFeaturizing test...")
    test = pd.read_csv(TEST_PATH)
    test_feats = test['smiles'].apply(featurize)
    test_valid_mask = test_feats.notna()
    if (~test_valid_mask).sum():
        print(f"  {(~test_valid_mask).sum()} test SMILES failed to featurize -- "
              f"falling back to that target's train mean for those rows")
    test_mols = test.loc[test_valid_mask, 'smiles'].apply(_parse_mol)
    test_maccs = pd.Series([maccs_keys(m) for m in test_mols], index=test_mols.index)
    test_gast = pd.Series([gasteiger_features(m) for m in test_mols], index=test_mols.index)
    test_frag = pd.Series([fragment_counts(m) for m in test_mols], index=test_mols.index)
    test_attach = pd.Series([attachment_point_feats(m) for m in test_mols], index=test_mols.index)
    test_conj = pd.Series([conjugation_extent_feats(m) for m in test_mols], index=test_mols.index)

    valid_test_idx = test.index[test_valid_mask]
    test_records = [
        {**test_feats[idx], **test_maccs[idx], **test_gast[idx], **test_frag[idx],
         **test_attach[idx], **test_conj[idx]}
        for idx in valid_test_idx
    ]
    # Two reindexed views mirroring X_base/X_xt, built unconditionally the
    # same way X_base/X_xt are (single vectorized construction instead of a
    # scalar .loc[idx, k] write per (row, feature) pair -- reindex fills
    # any feature_cols missing from a row's merged dict with NaN, and
    # fills invalid rows (excluded above) with NaN across every column) --
    # each target_type picks whichever it shipped per
    # evaluate_cross_target_features' per-target verdict, not one shared
    # test feature matrix for every target.
    test_feat_df_base = pd.DataFrame(test_records, index=valid_test_idx).reindex(
        index=test.index, columns=feature_cols_base)

    # Same cross_target_feats() join used for train, keyed off test's own
    # canon SMILES -- reuses cross_target_lookup (built from train only,
    # test has no 'target' column to leak from). Overwrites the NaN
    # xtarget_/known_ columns the reindex above created for whichever of
    # those columns survived pruning.
    test['canon'] = test['smiles'].apply(canonical_smiles)
    test_xt_df = cross_target_feats(test, cross_target_lookup)
    test_feat_df_xt = pd.DataFrame(test_records, index=valid_test_idx).reindex(
        index=test.index, columns=feature_cols_xt)
    for col in feature_cols_xt:
        if col in test_xt_df.columns:
            test_feat_df_xt[col] = test_xt_df[col]

    test_feat_df_by_target = {
        tt: (test_feat_df_xt if use_cross_target[tt] else test_feat_df_base)
        for tt in target_types
    }

    # 3D-conformer test-side features -- only for D3D_TARGETS that actually
    # shipped the train-side block above (d3d_accepted, decided before
    # tuning); test_feat_df_by_target[tt] must carry every column
    # feature_cols_by_target[tt] expects, same as the dense pred_ features'
    # test-side merge below.
    for tt in sorted(D3D_TARGETS):
        if not d3d_accepted[tt]:
            continue
        own_test_index = test.index[(test['target_type'] == tt) & test_valid_mask]
        d3d_test_df = pd.DataFrame(
            [conformer_3d_feats(s) for s in test.loc[own_test_index, 'smiles']],
            index=own_test_index)
        test_feat_df_by_target[tt] = pd.concat(
            [test_feat_df_by_target[tt], d3d_test_df.reindex(test.index)], axis=1)
    print(f"  3D-conformer test-side features built for shipped targets "
          f"[{time.time()-t0:.0f}s]")

    oof_meta = {}
    cv_scores = {}  # tt -> (mean, std) of the stacked meta-learner, via its own CV
    test_predictions = np.full(len(test), np.nan)

    large_targets_ordered = [tt for tt in target_types if tt in LARGE_TARGETS]
    small_targets_ordered = [tt for tt in target_types if tt not in LARGE_TARGETS]

    # ---- Phase 1: OOF stacking + test prediction, LARGE_TARGETS first --
    # egc/tg need to be fully trained (including their own final bagged-
    # refit models) before the dense pred_egc/pred_tg features below can
    # be built. ----
    print("\n" + "=" * 100)
    print("OOF stacking + test prediction -- LARGE_TARGETS first")
    print("=" * 100)

    large_target_fits = {}  # tt -> (X_tr_tt, y_tr_tt, fold_models_by_name)
    for tt in large_targets_ordered:
        large_target_fits[tt] = process_target(
            tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
            test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
            oof_meta, cv_scores, test_predictions, t0,
            track_fold_models=True)

    # ---- Round 2 extension: dense pred_egc/pred_tg cross-target features
    # for the other 5 targets -- egc/tg's *model predictions*, not the
    # sparse exact-canonical-SMILES-match xtarget_egc/xtarget_tg lookup
    # columns built earlier, so every molecule gets a value. Injected as
    # candidate columns on top of each small target's already-decided
    # feature set, re-pruned, and accepted/rejected with the same paired-
    # delta test as the xtarget block above -- per target, real harness,
    # ships only where it clears that target's own noise floor. ----
    print("\n" + "=" * 100)
    print("Dense pred_egc/pred_tg cross-target features (model predictions, not lookups)")
    print("=" * 100)

    small_target_set = set(small_targets_ordered)
    dense_train_cols, dense_test_cols = {}, {}
    for tt_source in large_targets_ordered:
        X_tr_source, y_tr_source, fold_models_by_name = large_target_fits[tt_source]
        train_series, test_series = build_dense_predictions(
            tt_source, X_tr_source, y_tr_source, fold_models_by_name, oof_meta,
            X_by_target, test_feat_df_by_target, feature_cols_by_target,
            accepted_tuned_configs, train_valid, test, small_target_set, t0)
        dense_train_cols[train_series.name] = train_series
        dense_test_cols[test_series.name] = test_series

    pred_feat_verdicts = {}
    for tt in small_targets_ordered:
        own_train_index = train_valid.index[train_valid['target_type'] == tt]
        own_test_index = test.index[test['target_type'] == tt]

        new_cols_train = pd.DataFrame(
            {name: series.loc[own_train_index] for name, series in dense_train_cols.items()})
        candidate_raw = pd.concat([X_by_target[tt], new_cols_train], axis=1)
        candidate_cols = fit_feature_pruner(candidate_raw)
        candidate_X = candidate_raw[candidate_cols]

        accepted, _, _, _, _ = paired_delta_verdict(
            tt, X_by_target[tt], candidate_X, y_all, train_valid, t0, label='pred_egc/tg')
        pred_feat_verdicts[tt] = accepted

        if accepted:
            X_by_target[tt] = candidate_X
            feature_cols_by_target[tt] = candidate_cols
            new_cols_test = pd.DataFrame(
                {name: series.loc[own_test_index] for name, series in dense_test_cols.items()})
            test_feat_df_by_target[tt] = pd.concat(
                [test_feat_df_by_target[tt], new_cols_test.reindex(test.index)], axis=1)

    shipped_pred = [tt for tt in small_targets_ordered if pred_feat_verdicts[tt]]
    print(f"\n  Dense pred_egc/pred_tg features SHIP for: {shipped_pred if shipped_pred else '(none)'}")
    for tt in small_targets_ordered:
        print(f"    {tt:5s}: {len(feature_cols_by_target[tt])} features "
              f"({'WITH' if pred_feat_verdicts[tt] else 'WITHOUT'} dense pred_egc/pred_tg)")

    # ---- Round 2b: egb/eea as dense sources for the 3 weakest targets
    # (nc/ei/eps). egb (~0.94 CV) and eea (~0.89) are both well-predicted,
    # and the physics ties the weak targets to them directly: ionization
    # energy ei ~= band gap egb - electron affinity eea, and eps/nc are
    # coupled to the same electronic polarizability. So egb/eea's *model
    # predictions* are candidate signal for ei/eps/nc that the sparse
    # xtarget_ lookups (exact-SMILES-match only) can't supply densely.
    # egb/eea are trained here FIRST (track_fold_models=True) -- their own
    # feature sets were already finalized by the pred_egc/tg round above,
    # and they consume nothing new -- so their fold-ensembles are available
    # to source dense columns for the 3 consumers before those are trained.
    # Same build + paired-delta gate as every other dense block; ships per
    # consumer only where it clears that target's own noise floor. ----
    print("\n" + "=" * 100)
    print("Dense pred_egb/pred_eea cross-target features for nc/ei/eps")
    print("=" * 100)

    dense_source_names = [t for t in ('egb', 'eea') if t in small_target_set]
    egb_eea_fits = {}
    for tt in dense_source_names:
        egb_eea_fits[tt] = process_target(
            tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
            test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
            oof_meta, cv_scores, test_predictions, t0,
            track_fold_models=True)

    egb_eea_consumers = [t for t in ('nc', 'ei', 'eps') if t in small_target_set]
    egb_eea_verdicts = apply_dense_sources(
        egb_eea_fits, egb_eea_consumers, X_by_target, feature_cols_by_target,
        test_feat_df_by_target, oof_meta, y_all, train_valid, test,
        accepted_tuned_configs, t0, label='pred_egb/eea')
    shipped_ee = [tt for tt in egb_eea_consumers if egb_eea_verdicts[tt]]
    print(f"\n  Dense pred_egb/pred_eea features SHIP for: {shipped_ee if shipped_ee else '(none)'}")

    # ---- Phase 2: OOF stacking + test prediction, small targets. nc
    # first (track_fold_models=True, same as LARGE_TARGETS above) so its
    # own fold-ensemble + full-refit models are available to build a
    # dense pred_nc feature for eps specifically -- nc->eps mirrors the
    # egc/tg->small-targets chaining above, one level deeper, since nc is
    # itself one of the targets that chain already runs for. eea/egb/ei
    # don't depend on anything new here and are processed afterward,
    # order among themselves doesn't matter. ----
    print("\n" + "=" * 100)
    print("OOF stacking + test prediction -- nc first (sources pred_nc for eps)")
    print("=" * 100)

    nc_X_tr, nc_y_tr, nc_fold_models_by_name = process_target(
        'nc', X_by_target, feature_cols_by_target, y_all, train_valid, test,
        test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
        oof_meta, cv_scores, test_predictions, t0,
        track_fold_models=True)

    print("\n" + "=" * 100)
    print("Dense pred_nc cross-target feature for eps (model predictions, not lookups)")
    print("=" * 100)

    pred_nc_train, pred_nc_test = build_dense_predictions(
        'nc', nc_X_tr, nc_y_tr, nc_fold_models_by_name, oof_meta,
        X_by_target, test_feat_df_by_target, feature_cols_by_target,
        accepted_tuned_configs, train_valid, test, {'eps'}, t0)

    candidate_raw = pd.concat([X_by_target['eps'], pred_nc_train.to_frame()], axis=1)
    candidate_cols = fit_feature_pruner(candidate_raw)
    candidate_X = candidate_raw[candidate_cols]

    pred_nc_accepted, _, _, _, _ = paired_delta_verdict(
        'eps', X_by_target['eps'], candidate_X, y_all, train_valid, t0, label='pred_nc')
    print(f"  Dense pred_nc feature for eps: {'SHIP' if pred_nc_accepted else 'reject'}")
    if pred_nc_accepted:
        X_by_target['eps'] = candidate_X
        feature_cols_by_target['eps'] = candidate_cols
        test_feat_df_by_target['eps'] = pd.concat(
            [test_feat_df_by_target['eps'], pred_nc_test.reindex(test.index).to_frame()], axis=1)
    print(f"  eps  : {len(feature_cols_by_target['eps'])} features "
          f"({'WITH' if pred_nc_accepted else 'WITHOUT'} dense pred_nc)")

    print("\n" + "=" * 100)
    print("OOF stacking + test prediction -- remaining small targets")
    print("=" * 100)

    # nc processed above (Phase 2), egb/eea processed in Round 2b as dense
    # sources -- so only ei/eps remain, now carrying whichever dense
    # pred_egb/pred_eea (+ pred_nc for eps) columns cleared their gates.
    already_processed = {'nc', *dense_source_names}
    for tt in [t for t in small_targets_ordered if t not in already_processed]:
        process_target(
            tt, X_by_target, feature_cols_by_target, y_all, train_valid, test,
            test_valid_mask, test_feat_df_by_target, accepted_tuned_configs,
            oof_meta, cv_scores, test_predictions, t0,
            track_fold_models=False)

    print(f"\n{'target':6s}{'stacked CV R2':>18s}")
    for tt in target_types:
        mean, std = cv_scores[tt]
        print(f"{tt:6s}{mean:14.4f} (+/-{std:.3f})")
    mean_r2 = float(np.mean([m for m, _ in cv_scores.values()]))
    print(f"\n>>> Mean CV R2 across all {len(target_types)} targets: {mean_r2:.4f} <<<")
    print("(this is the number that estimates the competition metric -- mean R2 across targets)")

    # ---- output safety: clip to train range + margin, hard physical floors ----
    print("\n" + "=" * 100)
    print("Clipping + writing submission.csv")
    print("=" * 100)

    stacked_out = test[['id', 'target_type']].copy()
    stacked_out['target'] = test_predictions

    clipped_target = stacked_out['target'].copy()
    for tt in target_types:
        clip_lo, clip_hi = clip_bounds_for(tt, y_all, train_valid)
        mask = stacked_out['target_type'] == tt
        vals = stacked_out.loc[mask, 'target']
        n_clipped = ((vals < clip_lo) | (vals > clip_hi)).sum()
        clipped_target.loc[mask] = vals.clip(lower=clip_lo, upper=clip_hi)
        print(f"  {tt:5s}: clip=[{clip_lo:.4g}, {clip_hi:.4g}]  "
              f"{n_clipped}/{mask.sum()} rows clipped")

    submission = stacked_out[['id']].copy()
    submission['target'] = clipped_target

    if SAMPLE_SUB_PATH.exists():
        # sample_submission.csv is a short format example (e.g. 10 rows), not
        # a full-length template -- only its columns are a real spec to check
        # against. Row count is checked against test.csv itself instead.
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        assert list(submission.columns) == list(sample_sub.columns), \
            f"column mismatch: {submission.columns.tolist()} vs {sample_sub.columns.tolist()}"
    assert len(submission) == len(test), \
        f"row count mismatch: {len(submission)} vs test.csv's {len(test)}"
    assert submission['target'].notna().all(), "unfilled predictions remain"

    submission.to_csv(OUT_PATH, index=False)
    print(f"\nFormat check OK -- saved {OUT_PATH} with shape {submission.shape}")
    print(f">>> Mean CV R2 across all {len(target_types)} targets: {mean_r2:.4f} <<<")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
