#!/usr/bin/env bash
# Single entry point: runs Phase 3 (model search/tuning) only if its results
# file is missing or --retune is passed, then always runs the production
# pipeline (Phase 4 stacking + Phase 5 clipping -> outputs/submission.csv).
#
# Phase 3 is the expensive step (~60-90 min: Optuna search over 3 boosting
# models x 7 targets). Deliberately NOT merged into one script -- most
# iteration (feature tweaks, stacking/clipping changes) only needs
# final_pipeline.py rerun, which reads Phase 3's saved picks from
# outputs/phase3_results.json instead of re-searching. Forcing a full
# retune on every run would make each iteration ~90 min instead of ~15.
#
# Usage:
#   ./run_pipeline.sh            # skip Phase 3 if results already exist
#   ./run_pipeline.sh --retune   # force a fresh Phase 3 search first
set -e
cd "$(dirname "$0")"

RESULTS_FILE="outputs/phase3_results.json"

if [[ "$1" == "--retune" || ! -f "$RESULTS_FILE" ]]; then
    echo "=== Running Phase 3 (model search/tuning) ==="
    .venv/bin/python scripts/phase3_model_zoo.py
else
    echo "=== Skipping Phase 3 -- $RESULTS_FILE already exists (pass --retune to force) ==="
fi

echo "=== Running final pipeline (stacking + clipping -> outputs/submission.csv) ==="
.venv/bin/python scripts/final_pipeline.py
