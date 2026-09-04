#!/usr/bin/env bash
# Stage C: everything that was proposed but not yet measured.
#
#   bash scripts/sweep_c.sh
#
# Runs after scripts/sweep.sh. Each cell is still judged against the same ERM
# baseline on the same protocol, so "borrowed from another field" never counts
# as evidence on its own.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-2 3})
LOG=logs/sweep_c.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

jobs=()
# new representations, including on the forward-only protocol
for proto in lot size lot_time; do
  jobs+=("--encoder graph    --objective erm --protocol $proto")
  jobs+=("--encoder rpca_cnn --objective erm --protocol $proto")
done
# the descriptors, plus the lot signature RPCA removed, on every protocol
for proto in lot size lot_time; do
  jobs+=("--encoder feat --objective erm --protocol $proto --rpca-features --tag rpcafeat")
done
# purged forward-only protocol for the stage-A representations
for enc in feat cnn_bn cnn_gn spectral; do
  jobs+=("--encoder $enc --objective erm --protocol lot_time")
done
# independence, transport and anchor objectives
for enc in feat cnn_bn; do
  for obj in hsic sinkhorn anchor; do
    jobs+=("--encoder $enc --objective $obj --protocol lot")
  done
  jobs+=("--encoder $enc --objective anchor --protocol lot_time")
done
# Fourier amplitude swap as a training augmentation
jobs+=("--encoder cnn_bn --objective erm --protocol lot  --fda-aug 0.5 --tag fda")
jobs+=("--encoder cnn_bn --objective erm --protocol size --fda-aug 0.5 --tag fda")
# lot-adversarial self-supervised initialization, if pretraining finished
if [ -f runs/ssl_pretrain.pt ]; then
  for proto in lot size lot_time; do
    jobs+=("--encoder cnn_gn --objective erm --protocol $proto --init-from runs/ssl_pretrain.pt --tag sslinit")
  done
else
  say "note: runs/ssl_pretrain.pt missing, skipping the SSL-initialized cells"
fi

say "=== stage C: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  tta=""
  [[ "$spec" == *"cnn_"* ]] && tta="--tta"
  tag=$(echo "$spec" | tr -d ' -' | tr '[:upper:]' '[:lower:]' | cut -c1-70)
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" $tta \
    >> "logs/c_${tag}.log" 2>&1 &
  pids+=($!)
  i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"
    pids=(); say "  $i / ${#jobs[@]} cells done"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== stage C done: $(ls runs/*.json | wc -l) result files ==="
