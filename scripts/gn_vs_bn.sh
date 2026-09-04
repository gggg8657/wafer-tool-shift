#!/usr/bin/env bash
# GroupNorm against BatchNorm on `lot`, at eight seeds, run together.
#
#   bash scripts/gn_vs_bn.sh
#
# This is the one comparison in the repository that keeps landing exactly on the
# resolution limit. Mixed-session, three seeds: ranges disjoint by 0.0004,
# below the 0.0054 floor. One session, three seeds: ranges *overlap* by 0.0003.
# Same conclusion twice -- not established -- but from opposite sides, which is
# what an effect the size of the noise looks like.
#
# It is worth resolving rather than leaving ambiguous, because it is a named
# claim ("GroupNorm beats BatchNorm everywhere") with a mechanism behind it:
# BatchNorm mixes statistics across whatever is in the batch, and a batch here
# spans lots, so the normalization choice is a domain-leak question rather than
# a tuning one. A mechanism is a reason to expect an effect, not evidence of one.
#
# ---------------------------------------------------------------------------
# A note on why this sweep is analysed differently from every other one here.
#
# Everywhere else the verdict is "do the observed seed ranges overlap, and is
# the margin above the run-to-run floor". That criterion is deliberately
# conservative, but it has a property that makes it wrong for *establishing* an
# effect: the range of a sample grows with the sample size. Going from three
# seeds to eight makes non-overlap strictly harder to achieve, so a sweep run to
# settle a question would be penalised for having more data.
#
# So this one is read with an exact permutation test on the eight-versus-eight
# labels: assumption-free, and with C(16,8) = 12,870 arrangements it can reach a
# two-sided p below 0.001, which three seeds never can. `scripts/gn_vs_bn.py`
# does the analysis and reports the range test beside it.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/gn_vs_bn.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 0 1 2 3 4 5 6 7; do
  for enc in cnn_gn cnn_bn; do
    jobs+=("--encoder $enc --objective erm --protocol lot --seed $seed --tag gnbn")
  done
done

say "=== GN vs BN, ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/g_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
$PY scripts/gn_vs_bn.py >> "$LOG" 2>&1 || say "  !! permutation test failed"
# the analysis step is the one that failed silently elsewhere: verify it
# actually wrote its summary before declaring the stage complete
$PY scripts/verify_stage.py --glob runs/gn_vs_bn.json --expect 1 \
  --label "GN vs BN permutation summary" | tee -a "$LOG" || exit 1
say "=== GN vs BN done ==="
