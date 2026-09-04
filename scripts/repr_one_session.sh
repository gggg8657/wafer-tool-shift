#!/usr/bin/env bash
# The representation grid, every cell measured in ONE session on ONE pair of GPUs.
#
#   bash scripts/repr_one_session.sh
#
# Why this is a different experiment from "add seeds to the existing cells":
# six identical invocations of one cell span 0.0054, and the stored seed-0 cells
# -- measured on GPUs 2/3 in a different session -- sit ~0.005 below every one of
# those repeats. So a table built from a Friday seed 0 plus two seeds run this
# weekend mixes a session offset into what is presented as seed variance. The
# focal sweep showed exactly this: `--focal-gamma 0` is bit-exact cross-entropy
# and its cells still came out +0.0083 above the ERM cells, purely because they
# were run in a different batch.
#
# So every cell here is run now, back to back, on GPUs 0 and 1, tagged `sess2`.
# The result is internally comparable and is reported *beside* the original
# table rather than replacing it -- the two are not comparable to each other and
# neither is discarded.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/repr_one_session.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); seen=""
for f in runs/*__erm__s*.json; do
  b=$(basename "$f" .json)
  [ "$(echo "$b" | awk -F'__' '{print NF}')" -eq 4 ] || continue
  proto=$(echo "$b" | awk -F'__' '{print $1}')
  enc=$(echo "$b"   | awk -F'__' '{print $2}')
  case " $seen " in *" $proto:$enc "*) continue;; esac
  seen="$seen $proto:$enc"
  for seed in 0 1 2; do
    jobs+=("--encoder $enc --objective erm --protocol $proto --seed $seed --tag sess2")
  done
done

say "=== representation grid, one session: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/v_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== representation grid done ==="
