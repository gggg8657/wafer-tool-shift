#!/usr/bin/env bash
# Re-run the wafer-budget active-learning grid after the k-center memory fix.
# The first attempt died silently two strategies in: the naive k-center distance
# materialized |pool| x |chosen| x dim floats and was killed with no traceback,
# so the stage logged "done" and wrote no JSON.
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "determinism repeats done" logs/determinism_repeats.log 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] determinism repeats finished, re-running wafer-budget AL" >> logs/chain_s1.log
GPU=0 bash scripts/al_wafer_budget.sh
echo "[$(date +%H:%M:%S)] === chain_s7 done ===" >> logs/chain_s1.log
