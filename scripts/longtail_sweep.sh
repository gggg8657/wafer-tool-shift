#!/usr/bin/env bash
# The long tail: can focal loss or class balancing move Scratch and Loc?
#
#   bash scripts/longtail_sweep.sh
#
# Per-class F1 on the lot-disjoint split is none 0.992 and Edge-Ring 0.984
# against Scratch 0.747 and Loc 0.766, and 85% of the corpus is `none` predicted
# confidently -- so almost all the gradient mass comes from examples that are
# already correct. That is precisely focal loss's stated target, which makes it
# the right first thing to try and not merely the thing the brief named.
#
# Two *different* corrections, swept separately and never combined in one cell,
# because they assume different things:
#   focal        reweights by difficulty      (1 - p_y)^gamma
#   class-weight reweights by prior           effective-number class weights
#
# gamma = 0 is exactly cross-entropy (asserted in tests/test_smoke.py) and is
# the control: if it does not reproduce ERM the rest of the sweep is measured
# against the wrong baseline.
#
# The honest prior is that this does nothing. The sibling repo measured
# pos-weighting, focal, balanced sampling and a 9-way softmax spanning 0.016 in
# macro-F1 against a 0.012 seed spread on its own protocol, and logit adjustment
# -- the other prior correction -- is already the *worst* objective in this
# repo's table at -0.04 to -0.08. Recorded here before the run so the result
# cannot be reinterpreted afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/longtail.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 0 1; do
  for g in 0.0 0.5 1.0 2.0 5.0; do
    jobs+=("--encoder cnn_gn --objective focal --protocol lot --seed $seed --focal-gamma $g --tag focal$g")
  done
  jobs+=("--encoder cnn_gn --objective erm --protocol lot --seed $seed --class-weight --tag cw")
done

say "=== long-tail sweep: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/t_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== long-tail sweep done ==="
