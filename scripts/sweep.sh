#!/usr/bin/env bash
# The benchmark matrix, two GPUs in parallel.
#
#   bash scripts/sweep.sh
#
# Stage A establishes the picture: every representation under plain ERM on all
# three protocols, so the iid-to-lot gap and the unseen-geometry number are
# measured before any robustness method is allowed to claim credit.
# Stage B then asks whether any of the borrowed objectives beat ERM on the two
# shifted protocols, at the same budget and with the same domain-disjoint model
# selection.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-15}
GPUS=(${GPUS:-2 3})
LOG=logs/sweep.log
mkdir -p logs runs

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

jobs_a=()
for proto in iid lot size; do
  for enc in feat cnn_bn cnn_gn spectral; do
    jobs_a+=("--encoder $enc --objective erm --protocol $proto")
  done
done

jobs_b=()
for proto in lot size; do
  for enc in feat cnn_bn; do
    for obj in logit_adjust group_dro dann irm coral mixup_domain; do
      jobs_b+=("--encoder $enc --objective $obj --protocol $proto")
    done
  done
done

run_all(){
  local -n arr=$1
  local i=0 pids=()
  for spec in "${arr[@]}"; do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    tta=""
    [[ "$spec" == *"cnn_"* ]] && tta="--tta"
    tag=$(echo "$spec" | tr -d ' -' | tr '[:upper:]' '[:lower:]')
    say "launch gpu$g: $spec"
    CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" $tta \
      >> "logs/${tag}.log" 2>&1 &
    pids+=($!)
    i=$((i+1))
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
      wait "${pids[@]}" || say "  (a cell failed; continuing)"
      pids=()
      say "  $i / ${#arr[@]} cells done"
    fi
  done
  if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
}

say "=== stage A: representations under ERM (${#jobs_a[@]} cells) ==="
run_all jobs_a
say "=== stage B: borrowed objectives vs ERM (${#jobs_b[@]} cells) ==="
run_all jobs_b
say "=== sweep done: $(ls runs/*.json | wc -l) result files ==="
