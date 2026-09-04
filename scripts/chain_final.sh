#!/usr/bin/env bash
# Runs once both sweeps are done: active learning, then the report, then push.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/chain_final.log; }
say "waiting for both sweeps"
while pgrep -f "scripts/sweep.sh" >/dev/null || pgrep -f "scripts/sweep_c.sh" >/dev/null; do
  sleep 60
done
say "active learning"
CUDA_VISIBLE_DEVICES=0 $PY scripts/active_learning.py > logs/active_learning.log 2>&1 \
  && say "  done" || say "  FAILED (see logs/active_learning.log)"
say "report"
$PY scripts/report.py
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "Benchmark results across every protocol, representation and objective

RESULTS.md regenerated from the run JSONs by scripts/report.py.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  timeout 180 git push -q origin main && say "pushed" || say "push failed (commit is local)"
fi
say "=== complete: $(ls runs/*.json | wc -l) cells ==="
