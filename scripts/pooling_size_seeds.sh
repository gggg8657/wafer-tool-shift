#!/usr/bin/env bash
# Is max pooling actually harmful on the geometry-holdout protocol?
#
#   bash scripts/pooling_size_seeds.sh
#
# The protocol sweep found the pooling win exists on `lot` alone. On `size` --
# which holds out whole geometries -- the mean effect is -0.0590 macro-F1 and
# -0.0777 on Scratch, both large and neither separated: the margin is 0.0068
# against a `size` run-to-run floor of 0.0133, because `size` seed ranges are
# enormous. So the largest apparent effect in this sweep is also the least
# established, which is the shape of half the errors in this repository's
# history.
#
# Three seeds cannot settle it, and not for want of care: with three per arm an
# exact permutation test has 20 arrangements and its smallest attainable
# two-sided p is 0.1. Eight per arm gives 12,870 and can reach 0.00016. That is
# the same instrument that resolved GroupNorm against BatchNorm when the range
# test could not, and it is used here for the same reason -- the range test
# grows stricter with sample size, so a sweep run to answer a question is
# penalised for having more data.
#
# H47, before the run: the -0.059 is real and eight seeds give a permutation
# p below 0.05. If it does not, then max pooling is simply not established
# either way on `size`, the sweep's only claim is the `lot` result, and the
# honest summary of the weekend's one positive finding is that it holds on one
# protocol out of four and is unmeasured on the rest.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling_size_seeds.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 3 4 5 6 7; do
  for pool in mean meanmax; do
    jobs+=("--encoder cnn_gn --objective erm --protocol size --seed $seed --pool $pool --tag pool$pool")
  done
done

say "=== size pooling, seeds 3-7: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/ps_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
$PY scripts/gn_vs_bn.py --tag poolsize --out runs/pooling_size_perm.json >> "$LOG" 2>&1 || true
say "=== size pooling seeds done ==="
