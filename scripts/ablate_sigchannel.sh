#!/usr/bin/env bash
# Ablation: what is the rpca_cnn fourth channel actually worth?
#
#   bash scripts/ablate_sigchannel.sh
#
# `rpca_cnn` is the best cell on the lot protocol (0.8813 vs 0.8671 for the
# plain 3-channel cnn_gn), and the stated reason is that the fourth channel
# carries the wafer's defect with its lot signature removed. But the RPCA
# low-rank part is rank 0 for 94.8% of decomposed lots, and the residual is
# bit-identical to the raw failed-die mask for 95.3% of all wafers. So there are
# three competing explanations for the win and this sweep separates them:
#
#   residual  the RPCA decomposition is doing the work        (the claim)
#   failmask  a redundant copy of the fail mask is enough     (no decomposition)
#   zeros     an extra channel of nothing is enough           (capacity/init only)
#
# Three seeds each, because the claimed effect (0.014) is smaller than any
# seed spread anyone has measured on this corpus -- which is itself unmeasured,
# and is the second thing this sweep produces.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/ablate_sigchannel.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# a log name that is always a plain filename: no slashes from path arguments,
# which is the bug that silently dropped every sslinit cell in stage C
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for proto in lot lot_time size; do
  for seed in 1 2; do          # seed 0 already measured, untagged, in runs/
    jobs+=("--encoder rpca_cnn --objective erm --protocol $proto --seed $seed")
    jobs+=("--encoder cnn_gn   --objective erm --protocol $proto --seed $seed")
  done
  for seed in 0 1 2; do
    jobs+=("--encoder rpca_cnn --objective erm --protocol $proto --seed $seed --sig-channel failmask --tag failmask")
    jobs+=("--encoder rpca_cnn --objective erm --protocol $proto --seed $seed --sig-channel zeros    --tag zerochan")
  done
done

say "=== sig-channel ablation: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/a_$(slug "$spec").log" 2>&1 &
  pids+=($!)
  i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"
    pids=(); say "  $i / ${#jobs[@]} cells done"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== sig-channel ablation done ==="
