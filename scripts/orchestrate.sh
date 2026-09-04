#!/usr/bin/env bash
# Everything, in dependency order, across whatever GPUs are given.
#
#   GPUS="0 1 2 3" tmux new -d -s wts "bash scripts/orchestrate.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
GPUS=${GPUS:-"0 2 3"}
read -r -a G <<<"$GPUS"
EPOCHS=${EPOCHS:-12}
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/orchestrate.log; }

say "=== phase 1: caches, in parallel ==="
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/pretrain_ssl.py --epochs 8 --batch 512 \
  > logs/ssl.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=${G[1]:-${G[0]}} $PY scripts/rpca_prepare.py > logs/rpca.log 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=${G[2]:-${G[0]}} $PY scripts/rpca_lambda.py > logs/rpca_lambda.log 2>&1 &
p3=$!
wait $p3 || say "  lambda sweep failed"
wait $p2 || say "  RPCA failed"
wait $p1 || say "  SSL pretraining failed"
say "  caches: rpca=$([ -f data/rpca.pt ] && echo yes || echo no) ssl=$([ -f runs/ssl_pretrain.pt ] && echo yes || echo no)"
tail -2 logs/ssl.log | tee -a logs/orchestrate.log

say "=== phase 2: stage A+B (representations, then objectives) ==="
GPUS="$GPUS" EPOCHS=$EPOCHS bash scripts/sweep.sh 2>&1 | grep -E "cells done|stage|done:" | tee -a logs/orchestrate.log

say "=== phase 3: stage C (the newly added methods) ==="
GPUS="$GPUS" EPOCHS=$EPOCHS bash scripts/sweep_c.sh 2>&1 | grep -E "cells done|stage|done:" | tee -a logs/orchestrate.log

say "=== phase 4: active learning ==="
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/active_learning.py > logs/active_learning.log 2>&1 \
  && say "  done" || say "  FAILED (see logs/active_learning.log)"

say "=== phase 5: report ==="
$PY scripts/report.py
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "Benchmark results across every protocol, representation and objective

RESULTS.md regenerated from the run JSONs by scripts/report.py. Includes the
two corrections found while wiring the last batch of methods: the first
lot-adversarial head was vacuous (41,608 lots hashed to 64 buckets is an
unpredictable label, so gradient reversal had nothing to reverse) and the RPCA
threshold of 4 wafers made the corpus average look as if no shared structure
existed. Both are documented in the code and the report rather than quietly
fixed.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  timeout 180 git push -q origin main && say "pushed" || say "push failed (commit is local)"
fi
say "=== complete: $(ls runs/*.json | wc -l) cells ==="
