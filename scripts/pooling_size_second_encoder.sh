#!/usr/bin/env bash
# Does the geometry-holdout failure replicate on the second encoder?
#
# H73 showed the pooling gain replicates on `cnn_bn` for `lot`: `meanmax` beats
# `mean` on Scratch by +0.0322 (p = 0.01010) while the capacity control stays
# null. So the gain is not a property of GroupNorm.
#
# The mechanism makes a stronger and more falsifiable claim than that.
# `resize_nearest` upsamples by indexing, so a one-die scratch arrives as a band
# roughly 64/w pixels wide -- a width set by the wafer's native geometry.
# Measured with no model at all: geometry explains 0.7689 of the variance in a
# max-pooled line-filter response against 0.1485 of the mean-pooled one, and at
# native resolution that collapses to 0.0629. That is a property of the *input
# pipeline*, which does not know which normalisation layer follows it.
#
# If that is why `meanmax` fails when geometry is held out, the failure has to
# replicate on `cnn_bn` too. On `cnn_gn` the `size` effect is -0.0427 on Scratch
# (p = 0.20233) and -0.0281 on macro-F1 (p = 0.16177): negative point estimates,
# neither established. So the prediction is about a *sign and a non-result*, not
# a significant reversal, and it is stated that way.
#
# H74, before the run: on `size` with `cnn_bn`, `meanmax` does not beat `mean`
# on Scratch F1 -- the point estimate is at or below zero, or if positive it
# does not reach p < 0.05.
#
# What would falsify it, and it would cost a central story: if `meanmax`
# significantly beats `mean` on `size`/`cnn_bn`, then the geometry explanation
# is wrong. The `size` failure would be a GroupNorm quirk rather than a fact
# about the resize, and section 2.0's mechanism would have to be withdrawn
# despite having survived its own model-free control. That is the outcome I
# would least like and the reason this is worth running.
#
# Also tops up `cnn_gn`/`size`/`poolmeanmean` from three seeds to eight: the
# capacity control on this protocol is the one arm still at three, and a
# three-seed control cannot support the comparison the eight-seed treatment is
# making.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling_size_second_encoder.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for pool in mean meanmax meanmean; do
    if [ -f "runs/size__cnn_bn__erm__pool${pool}__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    jobs+=("--encoder cnn_bn --objective erm --protocol size --seed $seed --pool $pool --tag pool${pool}")
  done
  # the cnn_gn capacity control on size, still at three seeds
  if [ ! -f "runs/size__cnn_gn__erm__poolmeanmean__s${seed}.json" ]; then
    jobs+=("--encoder cnn_gn --objective erm --protocol size --seed $seed --pool meanmean --tag poolmeanmean")
  else
    have=$((have+1))
  fi
done

say "=== pooling on size: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/ps2_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for pool in mean meanmax meanmean; do
  $PY scripts/verify_stage.py --glob "runs/size__cnn_bn__erm__pool${pool}__s*.json" \
    --expect 8 --label "size cnn_bn pool$pool at eight seeds" | tee -a "$LOG" || exit 1
done
$PY scripts/verify_stage.py --glob "runs/size__cnn_gn__erm__poolmeanmean__s*.json" \
  --expect 8 --label "size cnn_gn capacity control at eight seeds" | tee -a "$LOG" || exit 1

for metric in macro_f1 class:Scratch; do
  m=$(printf '%s' "$metric" | tr -cs 'A-Za-z0-9' '_')
  $PY scripts/gn_vs_bn.py --protocol size --objective erm \
    --arm-a "cnn_bn:poolmeanmax" --arm-b "cnn_bn:poolmean" \
    --label-a "meanmax" --label-b "mean" --metric "$metric" \
    --out "runs/pooling_size_bn_perm_${m}.json" | tee -a "$LOG"
  $PY scripts/gn_vs_bn.py --protocol size --objective erm \
    --arm-a "cnn_bn:poolmeanmean" --arm-b "cnn_bn:poolmean" \
    --label-a "meanmean" --label-b "mean" --metric "$metric" \
    --out "runs/pooling_size_bn_control_perm_${m}.json" | tee -a "$LOG"
  $PY scripts/gn_vs_bn.py --protocol size --objective erm \
    --arm-a "cnn_gn:poolmeanmean" --arm-b "cnn_gn:poolmean" \
    --label-a "meanmean" --label-b "mean" --metric "$metric" \
    --out "runs/pooling_size_control_perm_${m}.json" | tee -a "$LOG"
done
say "=== pooling on size done ==="
