#!/usr/bin/env bash
# Re-run the plain-ERM grid so its cells carry the metrics added after they ran.
#
#   bash scripts/backfill_metrics.sh
#
# `wts.metrics.summarize` gained mean_domain_macro_f1, frac_domains_below_half,
# per-class conformal coverage and the empty-set rate *after* the 59 original
# cells were measured, so those cells report blanks for columns the analysis now
# depends on. Re-running is only legitimate if it reproduces the numbers already
# published, so that is checked first and the sweep aborts if it does not:
# re-running a cell that does not reproduce would silently replace a measured
# result with a different one under the same filename.
#
# Only untagged ERM cells are backfilled. The tagged variants encode flags
# (--rpca-features, --fda-aug, --tta) that cannot be reconstructed from the JSON
# without guessing, and guessing an argument is how a cell quietly becomes a
# different experiment with the same name.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
EPOCHS=${EPOCHS:-12}
GPUS=(${GPUS:-0 1})
LOG=logs/backfill.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-90; }

# ---- determinism gate ------------------------------------------------------
REF_JSON=runs/lot__cnn_gn__erm__s0.json
say "determinism check on $REF_JSON"
rm -rf /tmp/backfill_check
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/run_bench.py \
  --encoder cnn_gn --objective erm --protocol lot --seed 0 \
  --epochs "$EPOCHS" --out /tmp/backfill_check >> "$LOG" 2>&1
if ! $PY - "$REF_JSON" /tmp/backfill_check/lot__cnn_gn__erm__s0.json <<'PYEOF'
import json, sys
old = json.load(open(sys.argv[1]))["test"]["macro_f1"]
new = json.load(open(sys.argv[2]))["test"]["macro_f1"]
print(f"  stored {old:.6f}  rerun {new:.6f}  delta {new-old:+.2e}")
sys.exit(0 if abs(new - old) < 1e-9 else 1)
PYEOF
then
  say "!! re-run does not reproduce the stored number. NOT backfilling."
  say "!! the pipeline is not deterministic across the code changes made since"
  say "!! these cells were measured, so overwriting them would replace a"
  say "!! measured result with a different one under the same name."
  exit 1
fi
say "reproduces exactly; proceeding"

jobs=()
for f in runs/*__erm__s0.json; do
  b=$(basename "$f" .json)
  IFS='_' read -r -a parts <<< "$(echo "$b" | sed 's/__/ /g' | tr ' ' '_')"
  proto=$(echo "$b" | awk -F'__' '{print $1}')
  enc=$(echo "$b"   | awk -F'__' '{print $2}')
  nf=$(echo "$b" | awk -F'__' '{print NF}')
  [ "$nf" -eq 4 ] || continue          # untagged only: proto__enc__erm__s0
  tta=""; [[ "$enc" == cnn_* ]] && tta="--tta"
  jobs+=("--encoder $enc --objective erm --protocol $proto --seed 0 $tta")
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
