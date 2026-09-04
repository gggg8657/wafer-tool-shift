#!/usr/bin/env bash
# The remaining stages, in one ordered chain on the two leased GPUs.
#
#   nohup bash scripts/chain_rest.sh &
#
# Replaces five separate one-stage waiter scripts, each of which polled for the
# previous one's marker line. One script makes the order explicit and reviewable
# instead of being an emergent property of five grep loops.
#
# Order is by what each stage decides, not by when it was written:
#   1 backfill   the seen/unseen-geometry decomposition of the test split, on
#                every ERM cell -- `lot_time` turns out to hold only 19
#                geometries against 338 in training with 14.15% of its test
#                wafers of a geometry never trained on, so the repo's largest
#                result needs this before it needs anything else. Also carries
#                the run-to-run determinism floor and the metrics added today.
#   2 mixed      MIXED-SYNTH, both protocols, focal included
#   3 ssl_lr     the LR confound on the -5 to -8 point SSL result
#   4 longtail   focal gamma sweep with gamma=0 as its own control
#   5 al_wafer   active learning re-run with the budget in wafers rather than
#                lots, because the published curve compared heuristics that had
#                bought a fifth of random's data
#
# `domain_def` is not here: it is stage 2 of chain_s1 and is already queued.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a logs/chain_s1.log; }

while ! grep -q "chain_s2 done" logs/chain_s1.log 2>/dev/null; do sleep 30; done
say "--- chain_rest: backfill + geometry decomposition ---"
GPUS="0 1" bash scripts/backfill_metrics.sh || say "  (backfill gate refused; continuing)"
say "--- chain_rest: MIXED-SYNTH ---"
GPUS="0 1" bash scripts/mixed_sweep.sh
say "--- chain_rest: ssl fine-tuning LR ---"
GPUS="0 1" bash scripts/ssl_lr_sweep.sh
say "--- chain_rest: long tail ---"
GPUS="0 1" bash scripts/longtail_sweep.sh
say "--- chain_rest: active learning on a wafer budget ---"
GPU=0 bash scripts/al_wafer_budget.sh
say "=== chain_rest done ==="
