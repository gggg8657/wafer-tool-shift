#!/usr/bin/env bash
# Give the two underpowered nulls a test that could reject them.
#
# `scripts/null_power_audit.py` reports that two of the three nulls these
# documents rest on come from tests that cannot return p < 0.05 at any effect
# size:
#
#   RPCA fourth channel vs a channel of zeros   n=3 per arm, floor p = 0.1000
#   focal loss vs its gamma = 0 control          n=2 per arm, floor p = 0.3333
#
# That is the same defect that made "nothing separates from ERM on `size`" look
# like a finding for most of this weekend. It resolved the moment the arms went
# to eight seeds, and against the objectives.
#
# Neither claim is expected to reverse, and it matters why not. The RPCA
# withdrawal does not rest on this ablation: the low-rank part is rank 0 for
# 94.83% of decomposed wafers, the residual is bit-identical to the raw failed-
# die mask for 95.27% of all wafers, and `stack_channels` concatenates the
# fourth channel to an *intact* one-hot, so the encoder reads an untouched copy
# of whatever the decomposition removed. The control could not have failed to
# tie. The ablation corroborates a mechanism; it is not the evidence.
#
# Focal is weaker still at n=2 and its own control is exact -- gamma = 0 is
# bit-identical to cross-entropy, asserted in tests -- so what is missing is
# only seeds.
#
# H68, before the run: at eight seeds neither null reverses. No RPCA arm
# separates from the zeros channel at p < 0.05, and no gamma separates from
# gamma = 0. If either does, the corresponding withdrawal in `paper_draft.md`
# is wrong and has to come back out.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/null_power_fix.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for t in rpca2_residual rpca2_failmask rpca2_zeros; do
    if [ -f "runs/lot__rpca_cnn__erm__${t}__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    ch=${t#rpca2_}
    jobs+=("--encoder rpca_cnn --objective erm --protocol lot --seed $seed --sig-channel $ch --tag $t")
  done
  for g in 0.0 2.0; do
    if [ -f "runs/lot__cnn_gn__focal__focal${g}__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    jobs+=("--encoder cnn_gn --objective focal --protocol lot --seed $seed --focal-gamma $g --tag focal${g}")
  done
done

say "=== null power fix: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/npf_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for t in rpca2_residual rpca2_failmask rpca2_zeros; do
  $PY scripts/verify_stage.py --glob "runs/lot__rpca_cnn__erm__${t}__s*.json" \
    --expect 8 --label "$t at eight seeds" | tee -a "$LOG" || exit 1
done
for g in 0.0 2.0; do
  $PY scripts/verify_stage.py --glob "runs/lot__cnn_gn__focal__focal${g}__s*.json" \
    --expect 8 --label "focal gamma=$g at eight seeds" | tee -a "$LOG" || exit 1
done

$PY scripts/null_power_audit.py | tee -a "$LOG"
say "=== null power fix done ==="
