#!/usr/bin/env bash
# Does the one positive result survive a second encoder?
#
# Section 2.0 of `WEEKEND.md` ends: "It is still one encoder." That caveat has
# been standing since the pooling result was established and has never been
# tested. Every `poolmean` / `poolmeanmax` / `poolmeanmean` cell in this
# repository is `cnn_gn`.
#
# It matters more than a robustness check usually would, because this is the
# only thing in the project that worked. Everything else was withdrawn, and a
# result that holds on one of two nearly identical encoders is a different
# claim from one that holds on the architecture.
#
# `cnn_bn` differs from `cnn_gn` in the normalisation layer alone -- same three
# blocks, same width, same head, same 290,217 parameters under `meanmax`. So
# this isolates the interaction between pooling and normalisation and nothing
# else.
#
# H73, before the run: the `Scratch` gain replicates. `meanmax` beats `mean` on
# `Scratch` F1 at p < 0.05 with eight seeds per arm, and `meanmean` -- the
# capacity control, identical parameter count, no extra information -- does not.
#
# The reason to expect that: the mechanism is about what global average pooling
# *discards*. A `Scratch` is a thin connected line, and averaged over the wafer
# it is close to a slightly elevated background failure rate, which is a fact
# about the class and the operator, not about the normalisation layer.
#
# The reason it might fail, and it is specific: BatchNorm mixes statistics
# across whatever is in the batch, which this project has already found to be a
# domain leak when batches span lots. Max pooling passes through extreme
# activations, and under BN one wafer's extreme shifts the normalisation of the
# others in its batch -- so the statistic `meanmax` adds is exactly the kind BN
# is least able to keep separate. If that interaction is real, `meanmax` should
# help less on `cnn_bn`, or not at all.
#
# Both outcomes are worth having. Replication makes the pooling result
# architectural; failure makes it a GroupNorm result and says so in the title
# of the section.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
ENC=cnn_bn
LOG=logs/pooling_second_encoder.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for pool in mean meanmax meanmean; do
    if [ -f "runs/lot__${ENC}__erm__pool${pool}__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    jobs+=("--encoder $ENC --objective erm --protocol lot --seed $seed --pool $pool --tag pool${pool}")
  done
done

say "=== pooling on $ENC: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/p2e_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for pool in mean meanmax meanmean; do
  $PY scripts/verify_stage.py --glob "runs/lot__${ENC}__erm__pool${pool}__s*.json" \
    --expect 8 --label "$ENC pool$pool at eight seeds" | tee -a "$LOG" || exit 1
done

for metric in macro_f1 class:Scratch; do
  m=$(printf '%s' "$metric" | tr -cs 'A-Za-z0-9' '_')
  # the treatment
  $PY scripts/gn_vs_bn.py --protocol lot --objective erm \
    --arm-a "$ENC:poolmeanmax" --arm-b "$ENC:poolmean" \
    --label-a "meanmax" --label-b "mean" --metric "$metric" \
    --out "runs/pooling_bn_perm_${m}.json" | tee -a "$LOG"
  # the capacity control, which is what makes it a result rather than a number
  $PY scripts/gn_vs_bn.py --protocol lot --objective erm \
    --arm-a "$ENC:poolmeanmean" --arm-b "$ENC:poolmean" \
    --label-a "meanmean" --label-b "mean" --metric "$metric" \
    --out "runs/pooling_bn_control_perm_${m}.json" | tee -a "$LOG"
done
say "=== pooling on $ENC done ==="
