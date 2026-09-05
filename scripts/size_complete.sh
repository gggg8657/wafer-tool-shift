#!/usr/bin/env bash
# Take the `size`-protocol objective arms to eight seeds.
#
#   bash scripts/size_complete.sh
#
# `paper_draft.md` claimed for most of this weekend that "the `size` half of
# the negative result stands as measured" while the `lot` half was withdrawn.
# Scoring it (scripts/size_power_check.py) shows the claim was not merely
# stale, it was backwards. Every `size` arm holds three seeds. A two-sample
# exact permutation test at three per arm admits twenty arrangements, so the
# smallest two-sided p obtainable is 0.10 -- no objective on this protocol
# could have reached 0.05 whatever its effect size. group_dro (-0.1534) and
# logit_adjust (-0.1117) already sit exactly on that floor with fully disjoint
# seed ranges. They are the largest effects measured anywhere in this project
# and the instrument cannot certify any of them.
#
# The two halves fail differently and it is worth keeping them apart: `lot`
# could not have shown an effect (the domain vocabulary was degenerate, TV
# 0.0208), whereas `size` has a real vocabulary (TV 0.2592) and large effects
# but no resolution. This script fixes the second problem only.
#
# H64, before the run: group_dro and logit_adjust separate from ERM at p < 0.05
# and are worse. coral, dann, irm and mixup_domain do not.
#
# The reasoning, and the reason it might fail: the four small effects
# (-0.011 to -0.026) are well inside ERM's own three-seed spread on this
# protocol, which is 0.0735 -- an order of magnitude noisier than the `dtime`
# arms, where +-0.013 was typical. But on `dtime` eight seeds turned two
# effects of about -0.015 into p = 0.0017 and p = 0.0051, so small effects do
# resolve when the variance is small. Whether these resolve depends entirely on
# whether `size`'s variance falls with more seeds or is intrinsic to holding
# geometry out. I do not know which, and that is the actual question here.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/size_complete.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

OBJS=(erm coral dann irm group_dro mixup_domain logit_adjust)

# resumable: dg_complete.sh was killed at 14/20 and re-ran nothing usefully
# until it was taught to skip finished cells. Built in from the start here.
jobs=(); have=0
for seed in 3 4 5 6 7; do
  for obj in "${OBJS[@]}"; do
    if [ -f "runs/size__cnn_bn__${obj}__sizeseed__s${seed}.json" ]; then
      have=$((have+1)); continue
    fi
    jobs+=("--encoder cnn_bn --objective $obj --protocol size --seed $seed --tag sizeseed")
  done
done

say "=== size completion: ${#jobs[@]} cells to run, $have already present ==="
i=0; pids=()
for spec in ${jobs[@]+"${jobs[@]}"}; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/sc_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for obj in "${OBJS[@]}"; do
  $PY scripts/verify_stage.py --glob "runs/size__cnn_bn__${obj}__sizeseed__s*.json" \
    --expect 8 --label "$obj on size at eight seeds" | tee -a "$LOG" || exit 1
done

$PY scripts/size_power_check.py | tee -a "$LOG"
$PY scripts/verify_stage.py --glob runs/size_power_check.json --expect 1 \
  --label "size power summary" | tee -a "$LOG" || exit 1
say "=== size completion done ==="
