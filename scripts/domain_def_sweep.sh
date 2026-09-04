#!/usr/bin/env bash
# Is "no borrowed DG objective beats ERM on `lot`" a result, or an artefact of
# how `domain` was defined?
#
#   bash scripts/domain_def_sweep.sh
#
# The objectives were never shown lots. `batch_of` handed them `lot % 32`:
# 10,762 lots averaged into 32 buckets of ~336 lots, whose mean pairwise label
# total-variation is 0.0208 against 0.1666 between real lots. Equalizing 32
# near-identical distributions is something every invariance penalty achieves by
# doing nothing, so ERM-equivalence was close to guaranteed by construction.
#
# One change, everything else fixed: the domain vocabulary. `time_decile` is the
# lot's production-order decile -- ten fixed, ordered, lot-level groups with a
# mean pairwise label TV of 0.1822, and the axis this repo has already shown to
# carry the largest real shift.
#
#   hash32       the control, and the definition every published cell used
#   time_decile  the treatment
#
# `erm` is run under both. It ignores the domain entirely, so the two must come
# out identical at a given seed; if they do not, the plumbing changed something
# it should not have and nothing else in this sweep is interpretable.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/domain_def.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

OBJS="erm group_dro irm coral dann mixup_domain hsic"
jobs=()
for o in $OBJS; do
  for seed in 1 2; do   # hash32 control; seed 0 is already in runs/, untagged
    jobs+=("--encoder cnn_bn --objective $o --protocol lot --seed $seed")
  done
  for seed in 0 1 2; do
    jobs+=("--encoder cnn_bn --objective $o --protocol lot --seed $seed --domain-def time_decile --tag dtime")
  done
done

say "=== domain-definition sweep: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/d_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== domain-definition sweep done ==="
