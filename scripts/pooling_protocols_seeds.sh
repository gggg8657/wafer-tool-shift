#!/usr/bin/env bash
# The protocol-dependence claim, at the standard the lot claim now meets.
#
#   bash scripts/pooling_protocols_seeds.sh
#
# `meanmax` beats global average pooling on `lot`: +0.0150 macro-F1 at eight
# seeds per arm, exact permutation p = 0.012, and +0.0473 on `Scratch` at
# p = 0.00047. The claim that it does *not* generalize -- unestablished on
# `iid` and `lot_time`, negative and unestablished on `size` -- rests on three
# seeds for `iid` and `lot_time`.
#
# `size` has already been taken to eight (p = 0.162 macro-F1, p = 0.202 on
# Scratch: unestablished either way). `iid` and `lot_time` have not, and a
# "does not generalize" claim built on the sample size that has been wrong four
# times this weekend is not a claim, it is the absence of one. The whole point
# of the finding is the contrast between protocols, so both sides of the
# contrast need the same instrument.
#
# H54, before the run: `iid` comes out positive but weaker than `lot` and does
# not reach p < 0.05, because its three-seed effect was +0.0090 against `lot`'s
# +0.0173 and its baseline is higher. `lot_time` is a genuine null. If `iid`
# does reach significance, the claim becomes "helps where geometry is shared,
# hurts where it is not", which is stronger and simpler than the current one.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling_protocols_seeds.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for proto in iid lot_time; do
  for seed in 3 4 5 6 7; do
    for pool in mean meanmax; do
      jobs+=("--encoder cnn_gn --objective erm --protocol $proto --seed $seed --pool $pool --tag pool$pool")
    done
  done
done

say "=== pooling seeds 3-7 on iid and lot_time: ${#jobs[@]} cells ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/pq_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for proto in iid lot_time; do
  $PY scripts/verify_stage.py \
    --glob "runs/${proto}__cnn_gn__erm__poolmean__s*.json" \
    --glob "runs/${proto}__cnn_gn__erm__poolmeanmax__s*.json" \
    --expect 8 --label "$proto pooling, eight seeds per arm" | tee -a "$LOG" || exit 1
  for m in macro_f1 class:Scratch; do
    tagm=$(echo "$m" | tr -cs 'A-Za-z0-9' '_')
    $PY scripts/gn_vs_bn.py --protocol "$proto" \
      --arm-a cnn_gn:poolmeanmax --arm-b cnn_gn:poolmean \
      --label-a meanmax --label-b mean --metric "$m" \
      --out "runs/pooling_${proto}_perm_${tagm}.json" >> "$LOG" 2>&1 \
      || say "  !! permutation test failed for $proto/$m"
    $PY scripts/verify_stage.py --glob "runs/pooling_${proto}_perm_${tagm}.json" \
      --expect 1 --label "$proto $m permutation summary" | tee -a "$LOG" || exit 1
  done
done
say "=== pooling protocol seeds done ==="
