#!/usr/bin/env bash
# Is the long tail a pooling problem?
#
#   bash scripts/pooling_sweep.sh
#
# What is left after the other explanations were measured and discarded: class
# imbalance does nothing (five focal gammas against a bit-exact gamma=0 control,
# class-balanced weights, positive weighting -- nothing, nothing, worse), and
# input resolution is not a lever at all, because 97.7% of wafers are UPSAMPLED
# to reach 64x64 so the resize adds pixels rather than removing detail.
#
# The remaining cheap hypothesis is architectural. A `Scratch` is a thin
# connected line of failed dies. Global average pooling reduces the final
# feature map to a mean over the whole wafer, which is close to
# indistinguishable from a slightly elevated background failure rate; the
# spatial fact that the failures form a line is exactly what the mean throws
# away. Max pooling keeps the extreme a thin structure produces.
#
#   mean      the original
#   meanmax   the treatment
#   meanmean  the control -- identical parameter count, no extra information
#
# The control is the point. The RPCA fourth channel looked like a +0.014 win
# until it was run against a channel of zeros and turned out to be worth the
# same as nothing. `meanmean` asks that question before the claim is made
# instead of after, and tests/test_smoke.py asserts it matches `meanmax` in
# parameter count and carries a duplicated mean.
#
# Read Scratch and Loc F1, not macro-F1: the hypothesis is about thin
# structures, and a macro average over nine classes is mostly about the other
# seven.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for pool in mean meanmax meanmean; do
  for seed in 0 1 2; do
    jobs+=("--encoder cnn_gn --objective erm --protocol lot --seed $seed --pool $pool --tag pool$pool")
  done
done

say "=== pooling sweep: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/p_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== pooling sweep done ==="
