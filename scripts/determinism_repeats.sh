#!/usr/bin/env bash
# How far apart are two identical invocations of the same cell, really?
#
#   bash scripts/determinism_repeats.sh [N]
#
# The backfill gate refused, reporting that a re-run of lot/cnn_gn/erm/seed 0
# differed from the stored value by 0.0082 against a "floor" of 0.0012. But that
# floor was the difference between exactly two runs -- a one-sample estimate of
# a spread, which can be many times too small by luck. A gate whose tolerance is
# estimated that badly will refuse correct backfills and, worse, could admit
# wrong ones.
#
# So: run the same cell N times with identical arguments and report the observed
# range. If that range covers the stored value, the gate was measuring noise and
# the tolerance needs to come from the range rather than from one pair. If the
# repeats cluster away from the stored value, the code changed the numbers and
# the backfill must stay blocked until the change is found.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
N=${1:-6}
PROTO=${PROTO:-lot}
ENC=${ENC:-cnn_gn}
GPUS=(${GPUS:-0 1})
LOG=logs/determinism_repeats.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

REF="--encoder $ENC --objective erm --protocol $PROTO --seed 0 --tta"
OUT="runs/determinism__${PROTO}__${ENC}.json"
STORED="runs/${PROTO}__${ENC}__erm__s0.json"
say "=== $N identical invocations of: $REF ==="
WORK=/tmp/det_rep_${PROTO}_${ENC}
rm -rf "$WORK"; mkdir -p "$WORK"
pids=()
for i in $(seq 1 "$N"); do
  g=${GPUS[$(( (i-1) % ${#GPUS[@]} ))]}
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_bench.py $REF --epochs 12 \
    --out "$WORK/r$i" >> "$LOG" 2>&1 &
  pids+=($!)
  if [ $(( i % ${#GPUS[@]} )) -eq 0 ]; then wait "${pids[@]}" || true; pids=(); say "  $i / $N"; fi
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}" || true; fi

$PY - "$WORK" "$STORED" "$OUT" "$PROTO / $ENC" <<'PYEOF'
import json, sys, glob, statistics
d, stored_p, out, cell = sys.argv[1:5]
xs = sorted(json.load(open(p))["test"]["macro_f1"]
            for p in glob.glob(f"{d}/r*/*__erm__s0.json"))
import os
stored = (json.load(open(stored_p))["test"]["macro_f1"]
          if os.path.exists(stored_p) else None)
rng = max(xs) - min(xs)
rec = {
    "what": f"{len(xs)} identical invocations of one cell under the current code",
    "cell": f"{cell} / erm / seed 0 --tta", "epochs": 12,
    "n_repeats": len(xs), "macro_f1_values": xs,
    "min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs),
    "range": rng, "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    "bit_reproducible": rng == 0.0,
    "stored_macro_f1": stored,
    "stored_inside_repeat_range": (None if stored is None
                                   else bool(min(xs) <= stored <= max(xs))),
    "stored_distance_to_nearest_repeat": (None if stored is None else
                                          min(abs(stored - v) for v in xs)),
    "note": ("The observed range over identical invocations bounds every "
             "same-seed comparison in this repository: two cells whose test "
             "macro-F1 differ by less than it have not been shown to differ at "
             "all, whatever their arguments. An earlier version of this file "
             "estimated the same quantity from a single pair of runs, which "
             "understated it."),
}
json.dump(rec, open(out, "w"), indent=2)
print(json.dumps(rec, indent=2))
PYEOF
say "=== determinism repeats done ==="
