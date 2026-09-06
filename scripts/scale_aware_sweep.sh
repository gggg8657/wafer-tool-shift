#!/usr/bin/env bash
# Does a scale-aware receptive field rescue the pooling gain on `size`?
#
#   bash scripts/scale_aware_sweep.sh
#
# `meanmax` pooling improves `Scratch` F1 on every protocol where test wafers
# share geometry with training (iid p=0.00093, lot 0.00047, lot_time 0.00031)
# and fails on `size`, which holds geometry out. `scripts/pooling_mechanism.py`
# established, with no model involved, why: `resize_nearest` upsamples by
# indexing, so a one-die scratch arrives as a band ~64/w pixels wide, and native
# geometry explains 0.7689 of the variance in a max-pooled line-filter response
# against 0.1485 of the mean-pooled one. At native resolution that falls to
# 0.0629, so the resize creates the dependence rather than the wafers carrying
# it.
#
# The first repair proposed -- and written into two documents before it was
# tested -- was to pool the response back onto the native die grid before the
# max. It does not work: eta^2 stays at 0.7511, because convolution and
# downsampling do not commute. Dilating the *filter* by round(64/w) gives
# 0.0361, below the native-resolution control itself.
#
# `--scale-aware` implements that in the encoder's first block. It adds no
# parameters, so this is compared against `meanmax` at identical capacity, and
# at w = 64 the path is bit-identical to the unscaled one (both asserted in
# tests). Every difference measured here comes from the 97.7% of wafers that
# are upsampled.
#
# H66, before the run: on `size`, `meanmax --scale-aware` beats `meanmax` on
# Scratch F1, and the sign of the pooling effect against `mean` turns from
# negative to positive. On `lot`, where geometry is shared, it changes little in
# either direction.
#
# Why it may fail, and I give this real weight after H63 went 2/4: the
# mechanism test used fixed line filters on the input plane, whereas a trained
# encoder has three blocks and can learn to compensate for a scale it sees
# consistently in training. If it already compensates, dilating the first block
# buys nothing and H66 is simply wrong. A second, duller possibility is that
# `size` is too noisy to resolve anything -- ERM's own three-seed spread there
# is 0.0735 -- which is why this runs eight seeds per arm from the start.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/scale_aware.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

# The comparison arms already exist: the pooling sweep used `cnn_gn`, and
# `poolmeanmax` and `poolmean` are both at eight seeds on `lot` and `size`.
# Only the scale-aware arm is new, so this is 16 cells rather than 48. Using a
# different encoder here would have made the treatment incomparable to the
# control it is supposed to beat.
ENC=cnn_gn
jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for proto in size lot; do
    f="runs/${proto}__${ENC}__erm__poolmeanmaxSA__s${seed}.json"
    if [ -f "$f" ]; then have=$((have+1)); else
      jobs+=("--encoder $ENC --objective erm --protocol $proto --seed $seed --pool meanmax --scale-aware --tag poolmeanmaxSA")
    fi
  done
done

say "=== scale-aware sweep: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/sa_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for proto in size lot; do
  for tag in poolmeanmaxSA poolmeanmax poolmean; do
    $PY scripts/verify_stage.py \
      --glob "runs/${proto}__${ENC}__erm__${tag}__s*.json" --expect 8 \
      --label "$proto $tag at eight seeds" | tee -a "$LOG" || exit 1
  done
done

for proto in size lot; do
  for metric in macro_f1 class:Scratch; do
    m=$(printf '%s' "$metric" | tr -cs 'A-Za-z0-9' '_')
    # against plain meanmax: does the receptive field change anything?
    $PY scripts/gn_vs_bn.py --protocol "$proto" --objective erm \
      --arm-a "$ENC:poolmeanmaxSA" --arm-b "$ENC:poolmeanmax" \
      --label-a "meanmax + scale-aware" --label-b "meanmax" \
      --metric "$metric" \
      --out "runs/scale_aware_${proto}_${m}.json" | tee -a "$LOG"
    # against plain mean: the claim the size result is actually about, which is
    # whether the pooling gain exists at all once geometry is held out
    $PY scripts/gn_vs_bn.py --protocol "$proto" --objective erm \
      --arm-a "$ENC:poolmeanmaxSA" --arm-b "$ENC:poolmean" \
      --label-a "meanmax + scale-aware" --label-b "mean" \
      --metric "$metric" \
      --out "runs/scale_aware_vs_mean_${proto}_${m}.json" | tee -a "$LOG"
  done
done
say "=== scale-aware sweep done ==="
