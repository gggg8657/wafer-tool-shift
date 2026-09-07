#!/usr/bin/env bash
# The resize-free encoder, under geometry holdout.
#
# H65 established, with no model involved, that `resize_nearest` couples a
# defect's apparent width to its native geometry: geometry explains 0.7689 of
# the variance in a max-pooled line-filter response against 0.1485 of the
# mean-pooled one, collapsing to 0.0629 at native resolution. Every attempt to
# act on that has failed -- dilating the first block is catastrophic (H66, H69)
# and the `size` behaviour did not replicate across encoders (H74).
#
# There is one encoder here that avoids the problem instead of patching it.
# `spectral` is a Fourier neural operator: learned weights multiply a fixed
# number of low-frequency coefficients, so the same weights apply to a 25x27
# and a 53x58 wafer with no resampling at all. If the resize contributes a
# geometry-specific cost to thin-structure detection, an encoder that never
# resizes should pay less of it when geometry is held out.
#
# The comparison is degradation, not level. `spectral` is worse than the CNN on
# `Scratch` in absolute terms everywhere, and that is a fact about capacity and
# architecture, not about resizing. What the mechanism predicts is a smaller
# *drop* from `lot` to `size`.
#
# H78, before the run: `spectral`'s Scratch F1 degrades less from `lot` to
# `size` than the `meanmax` CNN's does. The CNN-meanmax drop is the one the
# mechanism attributes to the resize interacting with a max statistic; the
# spectral encoder has no resize for it to interact with.
#
# Why it might fail, and this is the outcome that would matter: if `spectral`
# degrades as much or more, then geometry holdout damages thin-class detection
# for reasons that have nothing to do with resampling -- and the model-free
# measurement, while still true about apparent width, would have no demonstrated
# consequence for any trained model at all. That is a weaker position than the
# paper currently holds and it is where two failed remedies already point.
#
# Also worth stating: `spectral` differs from the CNN in more than the resize
# -- different operator, different capacity, different inductive bias -- so a
# confirmation here is weak evidence and a falsification is strong. Only one
# resize-free encoder exists in this repository, and one is not a controlled
# comparison.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/spectral_geometry.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for proto in lot size; do
    if [ -f "runs/${proto}__spectral__erm__spec8__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    jobs+=("--encoder spectral --objective erm --protocol $proto --seed $seed --tag spec8")
  done
done

say "=== spectral on lot+size: ${#jobs[@]} cells to run, $have present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/sg_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for proto in lot size; do
  $PY scripts/verify_stage.py --glob "runs/${proto}__spectral__erm__spec8__s*.json" \
    --expect 8 --label "spectral $proto at eight seeds" | tee -a "$LOG" || exit 1
done
say "=== spectral geometry done ==="
