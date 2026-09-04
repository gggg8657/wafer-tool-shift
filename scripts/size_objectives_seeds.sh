#!/usr/bin/env bash
# The `size` objective grid at three seeds, in one session.
#
#   bash scripts/size_objectives_seeds.sh
#
# This is the last claim of consequence in the repository with no error bar.
# Every objective cell on the `size` protocol is single-seed, and two of them
# are the largest effects anywhere in these tables -- GroupDRO at -0.18 to -0.27
# and logit adjustment at -0.08 to -0.10. They have been quoted all weekend as
# the part of the domain-generalization negative result that "stands", on the
# grounds that the `size` hash was never degenerate.
#
# They may well stand. But `size` is also the protocol with by far the largest
# measured seed spread -- a half-range of 0.031 to 0.036, because each seed
# holds out different geometries and geometries are heterogeneous -- and at
# n=1 a -0.27 and a -0.03 are equally unfalsifiable. A benchmark that has spent
# the weekend withdrawing its own claims for lack of error bars should not keep
# its biggest one on that basis.
#
# ERM is re-run here too, tagged the same way, so every comparison is internal
# to one session. Mixing a Friday baseline with weekend cells folds a session
# offset into the delta; the focal sweep demonstrated that a bit-exact no-op
# came out +0.0083 that way.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/size_objectives.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for enc in cnn_bn feat; do
  for obj in erm group_dro logit_adjust irm coral dann mixup_domain; do
    for seed in 0 1 2; do
      jobs+=("--encoder $enc --objective $obj --protocol size --seed $seed --tag sizeseed")
    done
  done
done

say "=== size objective grid: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/z_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== size objective grid done ==="
