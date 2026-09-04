#!/usr/bin/env bash
# Does the pooling result survive the protocols it would be sold on?
#
#   bash scripts/pooling_protocols.sh
#
# `meanmax` is the only positive result this weekend produced: on `lot` it beats
# global average pooling by +0.0173 and moves `Scratch` by +0.0566, both clearing
# the run-to-run floor, while its capacity control `meanmean` -- same parameter
# count, mean concatenated with itself -- buys nothing.
#
# That is one protocol and one encoder. A benchmark whose entire point is that
# results move when the protocol changes has no business reporting a single-
# protocol win, and this one would be quoted as "the long tail is an
# architecture problem" if left as it stands. So: `iid`, `size` and `lot_time`
# as well, three seeds, all three pooling variants including the control, run
# together.
#
# The interesting case is `lot_time`. Max pooling keeps the extreme a thin
# structure produces; under forward-only shift the extremes are exactly what a
# new tool changes, so a mean might be the more robust statistic and the sign
# could flip. If it does, "use max pooling" becomes "use max pooling when your
# test set looks like your training set", which is a materially weaker claim and
# the one the data would support.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling_protocols.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for proto in iid size lot_time; do
  for pool in mean meanmax meanmean; do
    for seed in 0 1 2; do
      jobs+=("--encoder cnn_gn --objective erm --protocol $proto --seed $seed --pool $pool --tag pool$pool")
    done
  done
done

say "=== pooling across protocols: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/pp_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== pooling across protocols done ==="
