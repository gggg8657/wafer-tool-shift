#!/usr/bin/env bash
# Active learning re-run with the budget counted in WAFERS, not lots.
#
#   bash scripts/al_wafer_budget.sh
#
# The published curve says random acquisition beats entropy, coreset and
# diverse at every lot budget. Reading `wafers_mean` out of the same JSON shows
# why that comparison could not have gone any other way: at a 400-lot budget
# random labelled 6,284 wafers and entropy labelled 1,104. The heuristics were
# losing while training on a fifth of the data.
#
# The mechanism is in `lot_scores`: a lot's score is the *mean* of its wafers'
# scores, and the maximum of noisy means favours small samples, so "take the
# top-scoring lots" is partly "take the smallest lots". Random has no such bias.
# Measured mean lot size at a 400-lot budget: random 15.7 wafers, entropy 2.8.
#
# Interpolating the stored curve onto a wafer axis reverses the ranking, but
# that rests on 3 seeds and on interpolation between six points, so it is a
# reason to run the experiment and not a result. This runs it directly: the
# budget is wafers, every strategy stops at the same supervision volume, and
# the number of lots each needed is recorded because that is the *other* cost
# and the two cost models disagree.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
GPU=${GPU:-0}
mkdir -p logs runs
echo "[$(date +%H:%M:%S)] === active learning, wafer budget ===" | tee -a logs/al_wafer.log
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/active_learning.py \
  --budget-unit wafers \
  --budgets 400,800,1600,3200,6400 \
  --seeds 3 \
  --out runs/active_learning_wafers.json >> logs/al_wafer.log 2>&1
echo "[$(date +%H:%M:%S)] === al wafer budget done ===" | tee -a logs/al_wafer.log
