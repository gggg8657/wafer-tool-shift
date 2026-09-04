#!/usr/bin/env bash
# Does lot-adversarial SSL pretraining actually hurt, or is the fine-tuning
# recipe simply tuned for a random initialization?
#
#   bash scripts/ssl_lr_sweep.sh
#
# The fixed sslinit cells came in 0.0504 *below* from-scratch on `lot` (0.8143
# vs 0.8647, three seeds each, seed half-ranges +/-0.004 and +/-0.009), which is
# ten times the seed spread and so is a real effect. But every cell so far uses
# lr 2e-3 on a OneCycle schedule, and that was chosen for a random init. The
# pretrained weights are not at random-init scale -- measured from the
# checkpoint, body.17.weight has mean |w| 0.0852 against 0.0147 for a fresh
# CnnResized, a factor of 5.8 -- so a schedule that peaks at 2e-3 may simply be
# destroying the representation it was handed. (The classifier head is *not* a
# confound: it was never trained during pretraining and its statistics match a
# fresh init to three decimals.)
#
# The control matters as much as the treatment. Sweeping the LR only for the
# pretrained cells and comparing the best of those against from-scratch at
# 2e-3 would be selection on the treatment arm, so from-scratch is run at every
# LR too and the comparison is made LR by LR.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/ssl_lr.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for lr in 2e-4 5e-4 1e-3; do          # 2e-3 is already measured for both arms
  for seed in 0 1; do
    jobs+=("--encoder cnn_gn --objective erm --protocol lot --seed $seed --lr $lr --init-from runs/ssl_pretrain.pt --tag sslinit_lr$lr")
    jobs+=("--encoder cnn_gn --objective erm --protocol lot --seed $seed --lr $lr --tag scratch_lr$lr")
  done
done

say "=== ssl fine-tuning LR sweep: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/l_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== ssl fine-tuning LR sweep done ==="
