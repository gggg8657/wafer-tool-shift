#!/usr/bin/env bash
# Re-run measured cells so they carry the metrics added after they ran, behind a
# gate whose tolerance is *measured* rather than guessed.
#
#   bash scripts/backfill_metrics.sh
#
# Since these cells were measured, summarize() gained mean_domain_macro_f1,
# frac_domains_below_half, per-class conformal coverage and the empty-set rate,
# and run_bench gained the seen/unseen-geometry decomposition of the test split.
# Old cells report blanks for all of them.
#
# Re-running overwrites a measured result under its own filename, which is only
# legitimate if the re-run reproduces it. The previous version of this gate
# demanded agreement to 1e-9, which assumes the pipeline is bit-reproducible on
# a GPU -- an assumption nobody here had tested, and one that non-deterministic
# convolution backward kernels routinely break. So the gate now measures the
# floor first: the same cell is run twice under the current code, the difference
# between those two runs is the run-to-run noise floor, and the stored value has
# to agree with the re-run to within it (plus a small margin). A floor that is
# itself large is a finding and is written to runs/determinism.json for the
# reports to cite, because it bounds every same-seed comparison in this repo.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/backfill.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

# --tta is included because the stored seed-0 cell was measured with it. TTA
# runs after res["test"] is computed and so cannot move test macro-F1, but the
# gate is a like-for-like check and should not rely on that reasoning being
# right; it reruns the recorded invocation.
REF="--encoder cnn_gn --objective erm --protocol lot --seed 0 --tta"
say "measuring the run-to-run floor: $REF, twice, identical arguments"
rm -rf /tmp/det_a /tmp/det_b
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/run_bench.py $REF --epochs "$EPOCHS" \
  --out /tmp/det_a >> "$LOG" 2>&1
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/run_bench.py $REF --epochs "$EPOCHS" \
  --out /tmp/det_b >> "$LOG" 2>&1

$PY - runs/lot__cnn_gn__erm__s0.json /tmp/det_a/lot__cnn_gn__erm__s0.json \
     /tmp/det_b/lot__cnn_gn__erm__s0.json runs/determinism.json <<'PYEOF'
import json, sys
stored, a, b, out = sys.argv[1:5]
g = lambda p: json.load(open(p))["test"]["macro_f1"]
sa, fa, fb = g(stored), g(a), g(b)
floor = abs(fa - fb)
drift = abs(fa - sa)
rec = {
    "what": "two identical invocations of one cell under the current code",
    "cell": "lot / cnn_gn / erm / seed 0", "epochs": 12,
    "run_a_macro_f1": fa, "run_b_macro_f1": fb,
    "run_to_run_abs_diff": floor,
    "bit_reproducible": floor == 0.0,
    "stored_macro_f1": sa,
    "stored_vs_rerun_abs_diff": drift,
    "note": ("The run-to-run floor bounds every same-seed comparison in this "
             "repository: two cells whose test macro-F1 differ by less than it "
             "have not been shown to differ at all, whatever their arguments."),
}
json.dump(rec, open(out, "w"), indent=2)
print(json.dumps(rec, indent=2))
tol = max(3 * floor, 2e-4)
print(f"\ngate: |stored - rerun| = {drift:.3e} against tolerance {tol:.3e}")
sys.exit(0 if drift <= tol else 1)
PYEOF
gate=$?
if [ $gate -ne 0 ]; then
  say "!! the re-run does not reproduce the stored number within the measured"
  say "!! run-to-run floor. NOT backfilling: overwriting these files would"
  say "!! replace a measured result with a materially different one under the"
  say "!! same name. Something in the code changed the numbers -- find it first."
  exit 1
fi
say "stored values reproduce within the measured floor; proceeding"

# every untagged ERM cell, at every seed already present. Tagged variants are
# excluded: their flags (--rpca-features, --fda-aug, --tta) are not recoverable
# from the JSON without guessing, and a guessed argument makes a cell a
# different experiment wearing the same filename.
jobs=()
for f in runs/*__erm__s*.json; do
  b=$(basename "$f" .json)
  [ "$(echo "$b" | awk -F'__' '{print NF}')" -eq 4 ] || continue
  proto=$(echo "$b" | awk -F'__' '{print $1}')
  enc=$(echo "$b"   | awk -F'__' '{print $2}')
  seed=$(echo "$b"  | awk -F'__' '{print $4}' | tr -d 's')
  tta=""; [[ "$enc" == cnn_* ]] && tta="--tta"
  jobs+=("--encoder $enc --objective erm --protocol $proto --seed $seed $tta")
done

say "=== backfill: ${#jobs[@]} cells on GPUs ${GPUS[*]} ==="
i=0; pids=()
for spec in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  say "launch gpu$g: $spec"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $spec --epochs "$EPOCHS" \
    >> "logs/b_$(slug "$spec").log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then
    wait "${pids[@]}" || say "  (a cell failed; continuing)"; pids=(); say "  $i / ${#jobs[@]}"
  fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi
say "=== backfill done ==="
