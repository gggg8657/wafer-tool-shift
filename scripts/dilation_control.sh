#!/usr/bin/env bash
# Is the damage from dilating at all, or from adapting the dilation to 64/w?
#
# H66 was falsified hard. Making the first conv block's receptive field track
# each wafer's native geometry cost 0.1758 of Scratch F1 on `lot` (p = 0.00016)
# and 0.1094 on `size` (p = 0.01523) against plain `meanmax` -- worse than the
# baseline it was supposed to rescue, on both protocols.
#
# The implementation is not the explanation. The grouped-dilation path matches a
# per-sample reference to 7.5e-09, reduces bit-exactly to the unscaled path at
# dilation 1, is invariant to batch order, and gradients reach the first block.
#
# Nor is the dilation extreme: it is 2 for 77.9% of wafers, 1 for 15.8%, 3 for
# 6.2%, and never exceeds 5. A 3x3 filter at d=2 spans 5 pixels. That a change
# this small costs 0.18 of a class F1 is the surprising part.
#
# So there are two live explanations and they have opposite consequences:
#
#   (a) dilating the first block hurts, whatever the dilation is -- in which
#       case H65's mechanism is untouched and only this lever is dead;
#   (b) *adapting* the dilation per wafer hurts, e.g. because the encoder can no
#       longer rely on a fixed relationship between pixels and dies -- in which
#       case geometry-adaptive preprocessing is the thing that fails.
#
# `--dilate-fixed 2` holds the dilation constant at the value 77.9% of wafers
# would have received. It is bit-identical to the plain path at d=1 and adds no
# parameters (both asserted in tests).
#
# H69, before the run: fixed d=2 is also worse than plain `meanmax`, and by a
# similar margin -- explanation (a). If instead fixed d=2 lands close to plain
# `meanmax`, then adaptivity is what costs, which is a more interesting result
# and one I do not expect.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
ENC=cnn_gn
LOG=logs/dilation_control.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  if [ -f "runs/lot__${ENC}__erm__poolmeanmaxD2__s${seed}.json" ]; then
    have=$((have+1)); continue
  fi
  jobs+=("--encoder $ENC --objective erm --protocol lot --seed $seed --pool meanmax --dilate-fixed 2 --tag poolmeanmaxD2")
done

say "=== dilation control: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/dc2_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

$PY scripts/verify_stage.py --glob "runs/lot__${ENC}__erm__poolmeanmaxD2__s*.json" \
  --expect 8 --label "fixed dilation 2 at eight seeds" | tee -a "$LOG" || exit 1

for metric in macro_f1 class:Scratch; do
  m=$(printf '%s' "$metric" | tr -cs 'A-Za-z0-9' '_')
  $PY scripts/gn_vs_bn.py --protocol lot --objective erm \
    --arm-a "$ENC:poolmeanmaxD2" --arm-b "$ENC:poolmeanmax" \
    --label-a "meanmax + fixed dilation 2" --label-b "meanmax" \
    --metric "$metric" --out "runs/dilation_control_lot_${m}.json" | tee -a "$LOG"
  $PY scripts/gn_vs_bn.py --protocol lot --objective erm \
    --arm-a "$ENC:poolmeanmaxD2" --arm-b "$ENC:poolmeanmaxSA" \
    --label-a "meanmax + fixed dilation 2" --label-b "meanmax + scale-aware" \
    --metric "$metric" --out "runs/dilation_control_vs_sa_lot_${m}.json" \
    | tee -a "$LOG"
done
say "=== dilation control done ==="
