#!/usr/bin/env bash
# Take the remaining focal gammas to eight seeds.
#
# `null_power_audit.py` now reports power per *comparison* rather than per
# family, which changed the picture: gamma = 2.0 is settled at eight seeds
# (diff -0.00032, p = 0.9442, floor 0.00016) and gammas 0.5, 1.0 and 5.0 are
# still at two seeds with a floor of 0.3333. So "focal loss contributes
# nothing" is established at one gamma out of four.
#
# The paper names focal loss because the target that started this work named it.
# Settling one value and leaving three at a floor of 0.33 would be the same
# defect this audit exists to find, one level down.
#
# H70, before the run: no gamma separates from the bit-exact gamma = 0 control
# at eight seeds. The reason to expect it is not the two-seed evidence, which is
# worth nothing, but that gamma = 2.0 -- the canonical value and the largest
# effect among the four -- is null at a floor of 0.00016, and focal loss
# reweights a loss whose class imbalance the `Scratch` result showed to be an
# architecture problem rather than a weighting one.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/focal_complete.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for g in 0.5 1.0 5.0; do
    if [ -f "runs/lot__cnn_gn__focal__focal${g}__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    jobs+=("--encoder cnn_gn --objective focal --protocol lot --seed $seed --focal-gamma $g --tag focal${g}")
  done
done

say "=== focal completion: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/fc_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for g in 0.0 0.5 1.0 2.0 5.0; do
  $PY scripts/verify_stage.py --glob "runs/lot__cnn_gn__focal__focal${g}__s*.json" \
    --expect 8 --label "focal gamma=$g at eight seeds" | tee -a "$LOG" || exit 1
done
$PY scripts/null_power_audit.py | tee -a "$LOG"
say "=== focal completion done ==="
