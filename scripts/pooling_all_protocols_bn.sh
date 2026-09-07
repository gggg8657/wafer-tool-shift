#!/usr/bin/env bash
# Complete the normalisation-by-pooling design on the remaining two protocols.
#
# Entry 99: `meanmax` closes the GroupNorm-BatchNorm gap on macro-F1, on `lot`
# (+0.0162 -> +0.0055) and on `size` (+0.0521 -> -0.0063), shrinking on 8 of 8
# seeds both times, p = 0.00781 each. So the GroupNorm advantage -- one of this
# paper's two surviving positive results -- is established under `mean` and not
# under `meanmax`, and the two headline findings do not compose.
#
# That was noticed post hoc, on `size`, and then replicated on `lot`. Two
# protocols agreeing is what makes it more than an artefact of looking. `iid`
# and `lot_time` have `cnn_gn` pooling arms at eight seeds and no `cnn_bn` ones
# at all, so the design can be completed rather than left at half.
#
# H76, before the run: the interaction replicates on both. On macro-F1 the
# GN-BN gap is smaller under `meanmax` than under `mean` on `iid` and on
# `lot_time`, at seven or eight of eight seeds each.
#
# Why it should: if `meanmax` supplies something BatchNorm otherwise lacks, that
# is a property of the two normalisations and the pooled statistic, not of what
# is held out. The effect was 5.5x larger on `size` than on `lot`, so magnitude
# clearly depends on the protocol -- but the sign should not.
#
# Why it might not, and this is the interesting failure: `lot_time` is the only
# protocol whose test set is a narrow geometry slice, and `iid` is the only one
# with no shift at all. If the interaction needs *some* shift to appear it
# should vanish on `iid`, which would make it a fact about generalisation rather
# than about the normalisations -- and would mean entry 99's explanation, that
# max pooling buys BatchNorm what GroupNorm was providing, is wrong.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/pooling_all_protocols_bn.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=(); have=0
for seed in 0 1 2 3 4 5 6 7; do
  for proto in iid lot_time; do
    for pool in mean meanmax meanmean; do
      if [ -f "runs/${proto}__cnn_bn__erm__pool${pool}__s${seed}.json" ]; then
        have=$((have+1)); continue
      fi
      jobs+=("--encoder cnn_bn --objective erm --protocol $proto --seed $seed --pool $pool --tag pool${pool}")
    done
  done
done

say "=== pooling on iid+lot_time / cnn_bn: ${#jobs[@]} cells, $have present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/pab_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for proto in iid lot_time; do
  for pool in mean meanmax meanmean; do
    $PY scripts/verify_stage.py \
      --glob "runs/${proto}__cnn_bn__erm__pool${pool}__s*.json" --expect 8 \
      --label "$proto cnn_bn pool$pool at eight seeds" | tee -a "$LOG" || exit 1
  done
done

$PY scripts/norm_pooling_interaction.py | tee -a "$LOG"
$PY scripts/combination_table.py | tail -20 | tee -a "$LOG"
say "=== done ==="
