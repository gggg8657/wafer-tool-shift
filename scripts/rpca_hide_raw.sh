#!/usr/bin/env bash
# Does the RPCA residual actually describe the defect worse than the raw mask?
#
#   bash scripts/rpca_hide_raw.sh
#
# The decomposition removes 61.6% of the failed-die mass on the Edge-Ring wafers
# it fires on -- 77.7% of all Edge-Ring wafers -- because Edge-Ring is a
# lot-level process condition and "what the lot shares" IS the defect. And yet
# Edge-Ring F1 is 0.9822 with the residual against 0.9836 without it: untouched.
#
# The reason is a design decision in `stack_channels`, made deliberately: the
# fourth channel is concatenated to the *intact* one-hot, so channel 2 still
# carries the raw failed-die mask and the encoder can ignore channel 3 entirely.
# The safeguard that stops the decomposition hurting is the same thing that
# stops it helping. It has never been possible for the fourth channel to matter.
#
# Zeroing channel 2 removes the fallback and makes "residual or raw mask?" a
# real question for the first time. Two arms, three seeds, one session:
#
#   --sig-channel failmask --hide-raw-fail   the wafer's true failed dies, once
#   --sig-channel residual --hide-raw-fail   the same, minus what its lot shares
#
# H33, before the run: with no fallback the residual falls clearly below the
# raw mask, and the gap concentrates in `Edge-Ring` -- the class the
# decomposition mutilates -- while `Scratch` and `Loc`, which it never fires on
# (enrichment 0.05x and 0.45x), are unaffected. If the gap is instead flat
# across classes, the mechanism is wrong and the difference is something about
# the fourth channel in general.
#
# This is the experiment that should have accompanied the original claim. It
# tests what the decomposition does to the representation, rather than whether
# a model that can route around it still scores the same.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/rpca_hide_raw.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 0 1 2; do
  jobs+=("--encoder rpca_cnn --objective erm --protocol lot --seed $seed --sig-channel failmask --hide-raw-fail --tag hideraw2_failmask")
  jobs+=("--encoder rpca_cnn --objective erm --protocol lot --seed $seed --sig-channel residual --hide-raw-fail --tag hideraw2_residual")
done

say "=== hide-raw-fail: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/h_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== hide-raw-fail done ==="
