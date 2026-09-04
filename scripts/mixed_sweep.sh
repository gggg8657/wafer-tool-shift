#!/usr/bin/env bash
# Multi-label F1 on the SYNTHETIC mixed-type set, under both protocols.
#
#   bash scripts/mixed_sweep.sh
#
# The target handed down is "multi-label F1 >= 0.95 under scarce labels, with
# focal loss". WM-811K is single-label, so the multi-label number has to be
# measured on something else; MixedWM38 could not be acquired unattended (HF
# copy gated, Zenodo record 403), so this is the overlay construction, labelled
# synthetic in every artefact it writes.
#
# Both protocols are run so the gap is visible rather than hidden:
#   mixed_iid.pt  random split of the sources  -- the optimistic protocol
#   mixed.pt      lot-disjoint sources         -- the honest protocol
# and three losses, so "with focal loss" is a measured comparison and not an
# assumption.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/mixed_sweep.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for data in data/mixed_iid.pt data/mixed.pt; do
  for loss in bce focal posweight; do
    for seed in 0 1; do
      jobs+=("--data $data --loss $loss --seed $seed")
    done
  done
done

say "=== MIXED-SYNTH sweep: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_mixed.py $spec --epochs "$EPOCHS" \
    >> "logs/m_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== MIXED-SYNTH sweep done ==="
