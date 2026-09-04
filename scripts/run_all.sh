#!/usr/bin/env bash
# Everything, in dependency order. Safe to run after a disconnect: each stage
# skips cells whose result file already exists is not implemented, but the
# stages themselves are idempotent enough to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/run_all.log; }
mkdir -p logs runs

say "waiting for any in-flight sweep"
while pgrep -f "scripts/sweep.sh" >/dev/null; do sleep 60; done
while pgrep -f "rpca_prepare.py" >/dev/null; do sleep 30; done
while pgrep -f "pretrain_ssl.py" >/dev/null; do sleep 60; done

say "stage C: the newly added methods"
bash scripts/sweep_c.sh 2>&1 | tail -5

say "active learning: which lots to pay to measure"
CUDA_VISIBLE_DEVICES=2 $PY scripts/active_learning.py \
  > logs/active_learning.log 2>&1 && say "  active learning done" \
  || say "  active learning FAILED (see logs/active_learning.log)"

say "report"
$PY scripts/report.py
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "Benchmark results: all protocols, representations, objectives

Regenerated RESULTS.md from every run JSON. Committed by scripts/run_all.sh.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  timeout 180 git push -q origin main && say "pushed" || say "push failed (commit is local)"
fi
say "=== all stages complete: $(ls runs/*.json | wc -l) cells ==="
