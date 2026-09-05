#!/usr/bin/env bash
# Bring the capacity control to the treatment's sample size.
#
#   bash scripts/pooling_control_seeds.sh
#
# The `lot` pooling result is at eight seeds per arm: +0.0150 macro-F1
# (permutation p = 0.012) and +0.0473 on Scratch (p = 0.00047). The claim that
# it is max pooling rather than a wider head rests on `meanmean` -- the mean
# concatenated with itself, identical parameter count, no extra information --
# which is still at **three** seeds.
#
# So the load-bearing control for the weekend's only positive result is at the
# sample size that has been wrong four times, while the result it supports is at
# eight. That is the same asymmetry caught twice already: once when a hand-off
# under-reported a positive because I had learned to distrust positives, and
# once when I demanded eight seeds of an effect I expected to be noise and
# accepted three for one I expected to be real. Here it runs the third way --
# the convenient direction, since a control that "buys nothing" at n=3 is
# exactly what the claim needs.
#
# Five more seeds, then the same permutation test on both metrics.
#
# H56, before the run: the control stays null. Its three-seed deltas were
# -0.0020 macro-F1 and -0.0082 Scratch, both slightly negative, and there is no
# mechanism by which duplicating a vector adds information. If it instead comes
# out positive and significant, the pooling result is a capacity result and the
# mechanism story is wrong.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling_control_seeds.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 3 4 5 6 7; do
  jobs+=("--encoder cnn_gn --objective erm --protocol lot --seed $seed --pool meanmean --tag poolmeanmean")
done

say "=== capacity control, seeds 3-7: ${#jobs[@]} cells ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/pc_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

$PY scripts/verify_stage.py \
  --glob 'runs/lot__cnn_gn__erm__poolmeanmean__s*.json' \
  --expect 8 --label "capacity control, eight seeds" | tee -a "$LOG" || exit 1
for m in macro_f1 class:Scratch; do
  # printf, not echo: echo appends a newline, which tr turns
  # into a trailing underscore, and the file the generators
  # look for then does not exist
  tagm=$(printf "%s" "$m" | tr -cs 'A-Za-z0-9' '_')
  $PY scripts/gn_vs_bn.py --protocol lot \
    --arm-a cnn_gn:poolmeanmean --arm-b cnn_gn:poolmean \
    --label-a meanmean --label-b mean --metric "$m" \
    --out "runs/pooling_lot_control_perm_${tagm}.json" >> "$LOG" 2>&1 \
    || say "  !! permutation test failed for $m"
  $PY scripts/verify_stage.py --glob "runs/pooling_lot_control_perm_${tagm}.json" \
    --expect 1 --label "control $m permutation summary" | tee -a "$LOG" || exit 1
done
say "=== capacity control seeds done ==="
