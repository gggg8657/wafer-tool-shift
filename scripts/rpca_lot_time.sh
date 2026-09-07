#!/usr/bin/env bash
# The one place a genuine tool signature should help, at eight seeds.
#
# `paper_draft.md` says: "On `lot_time` the residual is ahead of both controls
# at every seed, which is the one place a genuine tool signature should help;
# the separation is below the controls' half-range at n=3, so we record it as a
# hypothesis and not a result." That sentence has been standing all weekend and
# it is the last hypothesis in the paper without a powered test behind it.
#
# At three seeds the residual leads the zeros channel by about 0.013 -- smaller
# than `lot_time`'s own run-to-run floor of 0.0187, so it is inside the spread
# of re-running one configuration unchanged. And an exact permutation test at
# three per arm cannot return below 0.10 whatever the effect.
#
# Fresh arms rather than topping up the existing three. The current cells carry
# tags from an earlier session (`sess2`, `zerochan`) and this project has
# already been bitten once by mixing seeds measured in different sessions on
# different GPUs -- spread across sessions is about twice the within-session
# figure. Eight seeds measured together under the same tags as the `lot`
# ablation is the comparable experiment.
#
# H72, before the run: the residual does not separate from the zeros channel on
# `lot_time` either. The reason is structural and is the same argument that
# predicted the tight `lot` null correctly: `stack_channels` concatenates the
# fourth channel to an *intact* one-hot, so the encoder reads an untouched copy
# of whatever the decomposition removed, and this is a property of the input
# pipeline that does not know which protocol it is under. On `lot` the two tied
# to 0.000046, 118 times below that protocol's floor.
#
# What would make me wrong: the residual leading at 3 of 3 seeds is not nothing
# (a coin gives that a quarter of the time), and `lot_time` is the protocol
# where lots are ordered in production time, so a lot signature is more likely
# to carry information there than anywhere else. If the structural argument is
# right the lead is chance; if it survives eight seeds, the argument is wrong
# and section 4.1 needs rewriting rather than just extending.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/rpca_lot_time.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for ch in residual failmask zeros; do
    if [ -f "runs/lot_time__rpca_cnn__erm__rpca2_${ch}__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    jobs+=("--encoder rpca_cnn --objective erm --protocol lot_time --seed $seed --sig-channel $ch --tag rpca2_${ch}")
  done
done

say "=== rpca lot_time: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/rlt_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for ch in residual failmask zeros; do
  $PY scripts/verify_stage.py \
    --glob "runs/lot_time__rpca_cnn__erm__rpca2_${ch}__s*.json" --expect 8 \
    --label "lot_time rpca2_$ch at eight seeds" | tee -a "$LOG" || exit 1
done

for metric in macro_f1 class:Scratch; do
  m=$(printf '%s' "$metric" | tr -cs 'A-Za-z0-9' '_')
  $PY scripts/gn_vs_bn.py --protocol lot_time --objective erm \
    --arm-a "rpca_cnn:rpca2_residual" --arm-b "rpca_cnn:rpca2_zeros" \
    --label-a "RPCA residual" --label-b "channel of zeros" \
    --metric "$metric" --out "runs/rpca_lot_time_${m}.json" | tee -a "$LOG"
done
say "=== rpca lot_time done ==="
