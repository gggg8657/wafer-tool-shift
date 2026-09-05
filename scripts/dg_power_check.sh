#!/usr/bin/env bash
# Is the domain-generalization negative result under-powered?
#
#   bash scripts/dg_power_check.sh
#
# The headline negative -- "not one of seven borrowed objectives separates from
# ERM anywhere" -- rests entirely on three-seed cells read with the range
# screen. That screen has just been shown to produce **false negatives**: on the
# `iid` protocol it called a pooling effect of +0.0113 "below the floor", and at
# eight seeds the same comparison is p = 0.00031.
#
# Two objectives under the real domain vocabulary have effects larger in
# magnitude than the one the screen missed:
#
#   group_dro     -0.0257   (3 seeds, its own range 0.0139)
#   mixup_domain  -0.0143
#
# So the negative result is stated at a confidence the evidence does not
# support. That matters more than it would for a positive claim, because a
# null asserted from an under-powered test is exactly the failure this log
# spent section 40 on: an experiment with no power returning silence, and
# silence being read as evidence.
#
# Both objectives and ERM go to eight seeds under `dtime`, read with the
# permutation test on macro-F1.
#
# H60, before the run: `group_dro` separates from ERM at p < 0.05 and is
# genuinely worse under real domains -- it is the largest and most consistent
# effect in that table. `mixup_domain` does not. If group_dro separates, the
# claim changes from "nothing is distinguishable from ERM" to "one objective is
# established as worse and the rest are unestablished", which is a different
# and more useful sentence.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/dg_power.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 3 4 5 6 7; do
  for obj in erm group_dro mixup_domain; do
    jobs+=("--encoder cnn_bn --objective $obj --protocol lot --seed $seed --domain-def time_decile --tag dtime")
  done
done

say "=== DG power check: ${#jobs[@]} cells ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/dg_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for obj in erm group_dro mixup_domain; do
  $PY scripts/verify_stage.py --glob "runs/lot__cnn_bn__${obj}__dtime__s*.json" \
    --expect 8 --label "$obj at eight seeds" | tee -a "$LOG" || exit 1
done
for obj in group_dro mixup_domain; do
  $PY scripts/gn_vs_bn.py --protocol lot --objective erm \
    --arm-a "cnn_bn:dtime" --arm-b "cnn_bn:dtime" \
    --out /dev/null >/dev/null 2>&1 || true
done
# the two arms differ by objective, not encoder or tag, so call the tool with
# the objective baked into the arm spec via a symlinked view is unnecessary --
# compare directly with a tiny inline driver instead
$PY - <<'PYEOF' | tee -a "$LOG"
import json, glob, sys
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("g", "scripts/gn_vs_bn.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def arm(obj, metric):
    out = {}
    for f in glob.glob(f"runs/lot__cnn_bn__{obj}__dtime__s*.json"):
        r = json.load(open(f))
        out[r["seed"]] = (r["test"]["per_class_f1"][metric.split(":")[1]]
                          if metric.startswith("class:") else r["test"][metric])
    return out

res = {}
for metric in ("macro_f1",):
    base = arm("erm", metric)
    for obj in ("group_dro", "mixup_domain"):
        x = arm(obj, metric)
        shared = sorted(set(base) & set(x))
        a = [x[s] for s in shared]; b = [base[s] for s in shared]
        if len(a) < 2:
            print(f"ERROR: {obj} has {len(a)} shared seeds"); sys.exit(2)
        p, n = m.perm_p(a, b)
        res[f"{obj}__{metric}"] = {
            "n_per_arm": len(a), "objective": obj, "metric": metric,
            "mean_objective": sum(a) / len(a), "mean_erm": sum(b) / len(b),
            "difference": sum(a) / len(a) - sum(b) / len(b),
            "p_two_sided": p, "arrangements": n,
            "ranges_overlap": not (min(a) > max(b) or min(b) > max(a)),
        }
        print(f"{obj:14s} {metric:10s} n={len(a)} diff "
              f"{res[f'{obj}__{metric}']['difference']:+.4f}  p={p:.5f}")
json.dump(res, open("runs/dg_power_check.json", "w"), indent=2)
PYEOF
$PY scripts/verify_stage.py --glob runs/dg_power_check.json --expect 1 \
  --label "DG power check summary" | tee -a "$LOG" || exit 1
say "=== DG power check done ==="
