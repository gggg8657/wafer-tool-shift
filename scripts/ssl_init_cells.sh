#!/usr/bin/env bash
# The SSL-initialized cells that stage C reported as launched and never ran.
#
#   bash scripts/ssl_init_cells.sh
#
# Question: does lot-adversarial masked pretraining on the 638,506 unlabelled
# wafers help a cnn_gn under shift? Read the pretraining log before believing
# anything here: nuisance (lot) CE went 5.09 -> 4.89 -> 5.04 against a chance of
# 5.77 over 8 epochs, so the adversary has traction but never removes the lot
# identity from the embedding. The honest prior is "little or no effect".
#
# The comparison is against the same encoder from scratch at the same seed --
# runs/<proto>__cnn_gn__erm__s{0,1,2}.json -- so three seeds are run here too.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/ssl_init.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

if [ ! -f runs/ssl_pretrain.pt ]; then say "runs/ssl_pretrain.pt missing, nothing to do"; exit 1; fi
jobs=()
for proto in lot size lot_time; do
  for seed in 0 1 2; do
    jobs+=("--encoder cnn_gn --objective erm --protocol $proto --seed $seed --init-from runs/ssl_pretrain.pt --tag sslinit")
  done
done

say "=== ssl-init cells: ${#jobs[@]} on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  lg="logs/i_$(slug "$spec").log"
  say "launch gpu$g -> $lg: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" >> "$lg" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== ssl-init cells done ==="
