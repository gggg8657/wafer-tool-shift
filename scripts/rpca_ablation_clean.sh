#!/usr/bin/env bash
# The RPCA fourth-channel ablation again, all four arms in one session.
#
#   bash scripts/rpca_ablation_clean.sh
#
# The original ablation was the weekend's first withdrawal and its conclusion --
# a channel of zeros buys what the decomposition buys -- has been quoted since.
# Checking the file timestamps shows the comparison was not symmetric: the
# `residual` arm and the `cnn_gn` baseline each contain one cell measured in
# Friday's session on GPUs 2/3, while `failmask` and `zerochan` were measured
# entirely in this weekend's. Two arms carry a session offset and two do not.
#
# Restricted to within-session cells the picture is sharper and less flattering
# to the claim -- residual +0.0002 over three channels against +0.0057 for the
# raw mask and +0.0054 for zeros -- but that rests on n=2 for the residual arm,
# and selecting cells by when they ran is itself a subgroup choice of exactly
# the kind this repository has spent the weekend catching.
#
# So: all four arms, three seeds each, twelve cells, run together. No arm gets a
# cell the others do not.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/rpca_clean.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 0 1 2; do
  jobs+=("--encoder cnn_gn   --objective erm --protocol lot --seed $seed --tag rpca2_3ch")
  jobs+=("--encoder rpca_cnn --objective erm --protocol lot --seed $seed --sig-channel residual --tag rpca2_residual")
  jobs+=("--encoder rpca_cnn --objective erm --protocol lot --seed $seed --sig-channel failmask --tag rpca2_failmask")
  jobs+=("--encoder rpca_cnn --objective erm --protocol lot --seed $seed --sig-channel zeros    --tag rpca2_zeros")
done

say "=== clean RPCA ablation: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/q_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== clean RPCA ablation done ==="
