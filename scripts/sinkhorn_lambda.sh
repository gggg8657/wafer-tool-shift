#!/usr/bin/env bash
# Why did the entropic-OT objective collapse, and is any weight of it usable?
#
#   bash scripts/sinkhorn_lambda.sh
#
# At the default --ot-lambda 1.0 both sinkhorn cells sit at macro-F1 0.10 (the
# majority class) with train loss flat at 0.662 from epoch 4 -- the model is
# predicting the class prior, which is what an embedding collapsed to a point
# produces. The OT penalty is minimized exactly by that collapse, so the
# question is whether there is a weight small enough that the classifier still
# fits and large enough that the penalty does anything. If macro-F1 only
# recovers where the penalty is effectively off, the objective is broken here
# and the row comes out of the table.
#
# lambda 0 is the ERM control run through the *same* code path, so the
# comparison is not confounded by the objective wrapper.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/sinkhorn_lambda.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for lam in 0.0 0.003 0.01 0.03 0.1 0.3 1.0; do
  jobs+=("--encoder cnn_bn --objective sinkhorn --protocol lot --ot-lambda $lam --tag ot$lam")
done

say "=== sinkhorn lambda sweep: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/s_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== sinkhorn lambda sweep done ==="
