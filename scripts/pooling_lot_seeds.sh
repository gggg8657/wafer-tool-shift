#!/usr/bin/env bash
# The weekend's one positive result, held to the standard that just killed a
# negative one.
#
#   bash scripts/pooling_lot_seeds.sh
#
# `meanmax` beats global average pooling on `lot` by +0.0173 macro-F1 and
# +0.0566 on Scratch, both clearing the run-to-run floor, with a capacity
# control that buys nothing. That is at three seeds per arm.
#
# Three seeds is the sample size that has been wrong all weekend. An exact
# permutation test with three per arm has 20 arrangements and a smallest
# attainable two-sided p of 0.1, whatever the data says. The same eight-seed
# test applied to the `size` protocol -- where the three-seed mean was a
# striking -0.0590 -- returned -0.0281 and p = 0.162, so the three-seed estimate
# there was an unlucky draw twice the size of the effect.
#
# There is no principled reason to demand eight seeds of a result I expected to
# be negative and accept three for one I expected to be positive. Scepticism
# applied asymmetrically is not scepticism. So: both arms to eight seeds on
# `lot`, read with the same permutation test.
#
# H50, before the run: the `lot` effect survives, with a permutation p below
# 0.05 on macro-F1 and a smaller p on Scratch, because Scratch is where the
# mechanism says the effect should live and its three-seed margin (0.0180) was
# nearly three times the floor rather than barely over it. If it does not
# survive, the weekend produced no positive result at all and WEEKEND.md goes
# back to saying so.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling_lot_seeds.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 3 4 5 6 7; do
  for pool in mean meanmax; do
    jobs+=("--encoder cnn_gn --objective erm --protocol lot --seed $seed --pool $pool --tag pool$pool")
  done
done

say "=== lot pooling, seeds 3-7: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/pl_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

$PY scripts/verify_stage.py \
  --glob 'runs/lot__cnn_gn__erm__poolmean__s*.json' \
  --glob 'runs/lot__cnn_gn__erm__poolmeanmax__s*.json' \
  --expect 8 --label "lot pooling, eight seeds per arm" | tee -a "$LOG" || exit 1

$PY scripts/gn_vs_bn.py --protocol lot \
  --arm-a cnn_gn:poolmeanmax --arm-b cnn_gn:poolmean \
  --label-a meanmax --label-b mean \
  --out runs/pooling_lot_perm.json >> "$LOG" 2>&1 || say "  !! permutation test failed"
$PY scripts/verify_stage.py --glob runs/pooling_lot_perm.json --expect 1 \
  --label "lot pooling permutation summary" | tee -a "$LOG" || exit 1
say "=== lot pooling seeds done ==="
