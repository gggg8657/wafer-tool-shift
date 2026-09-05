#!/usr/bin/env bash
# Finish the domain-generalization table at eight seeds.
#
#   bash scripts/dg_complete.sh
#
# Two of the six objectives under the real domain vocabulary turned out to be
# significantly worse than ERM once taken to eight seeds -- group_dro at
# p = 0.0017 and mixup_domain at p = 0.0051 -- after the three-seed screen had
# called both unestablished. The other four are still at three seeds and still
# described as "genuinely unestablished", which is the same sentence that was
# wrong about the first two.
#
# Their three-seed effects are smaller (coral -0.0071, dann -0.0060, irm
# -0.0049, hsic -0.0026) than the +0.0113 the screen missed on iid, so there is
# no strong prior either way. But "unestablished" asserted from an instrument
# now known to produce false negatives is not a finding, and the table should
# either say these are worse than ERM or say so with power behind it.
#
# H63, before the run: coral and dann separate from ERM at p < 0.05 and are
# worse; irm and hsic do not. The reasoning is only that the first two are
# roughly twice the size of the last two and every DG effect measured on this
# corpus has been negative, so the ones with more signal should resolve first.
# This is a weak prior and I expect to get at least one of the four wrong.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/dg_complete.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

jobs=()
for seed in 3 4 5 6 7; do
  for obj in coral dann irm hsic; do
    jobs+=("--encoder cnn_bn --objective $obj --protocol lot --seed $seed --domain-def time_decile --tag dtime")
  done
done

say "=== DG completion: ${#jobs[@]} cells ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/dc_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

for obj in coral dann irm hsic; do
  $PY scripts/verify_stage.py --glob "runs/lot__cnn_bn__${obj}__dtime__s*.json" \
    --expect 8 --label "$obj at eight seeds" | tee -a "$LOG" || exit 1
done

$PY - <<'PYEOF' | tee -a "$LOG"
import json, glob, sys, importlib.util
spec = importlib.util.spec_from_file_location("g", "scripts/gn_vs_bn.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def arm(obj):
    return {json.load(open(f))["seed"]: json.load(open(f))["test"]["macro_f1"]
            for f in glob.glob(f"runs/lot__cnn_bn__{obj}__dtime__s*.json")}

base = arm("erm")
res = json.load(open("runs/dg_power_check.json")) \
    if glob.glob("runs/dg_power_check.json") else {}
for obj in ("coral", "dann", "irm", "hsic"):
    x = arm(obj)
    shared = sorted(set(base) & set(x))
    a = [x[s] for s in shared]; b = [base[s] for s in shared]
    if len(a) < 2:
        print(f"ERROR: {obj} has {len(a)} shared seeds"); sys.exit(2)
    p, n = m.perm_p(a, b)
    res[f"{obj}__macro_f1"] = {
        "n_per_arm": len(a), "objective": obj, "metric": "macro_f1",
        "mean_objective": sum(a) / len(a), "mean_erm": sum(b) / len(b),
        "difference": sum(a) / len(a) - sum(b) / len(b),
        "p_two_sided": p, "arrangements": n,
        "ranges_overlap": not (min(a) > max(b) or min(b) > max(a)),
    }
    print(f"{obj:14s} n={len(a)} diff {res[f'{obj}__macro_f1']['difference']:+.4f}"
          f"  p={p:.5f}")
json.dump(res, open("runs/dg_power_check.json", "w"), indent=2)
PYEOF
$PY scripts/verify_stage.py --glob runs/dg_power_check.json --expect 1 \
  --label "DG table summary" | tee -a "$LOG" || exit 1
say "=== DG completion done ==="
