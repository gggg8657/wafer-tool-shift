#!/usr/bin/env bash
# Re-measure the two floors that were truncated, in a file of their own.
#
# `determinism__iid__cnn_gn.json` was written with n_repeats = 1 and a "range"
# of 0.0000; `lot_time` got 4 of 6. Both were truncated because two stages ended
# up holding GPUs 0 and 1 at once, which happened because I appended a stage to
# `chain_after_size.sh` while bash was executing it. I reverted within a minute
# and assumed that had prevented it; `ps` shows `null_power_fix.sh` running as a
# child of `chain_after_size.sh`, so the appended bytes had already been read.
# Reverting a running script does not un-run it.
#
# The zero floor was the damaging part: `floors()` served it as the `iid`
# threshold, so every `iid` margin cleared it. `report.py` now refuses a floor
# from fewer than three repeats and `determinism_repeats.sh` refuses to write
# one, both asserted in tests.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/chain5.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== waiting for chain4 (dilation control) ==="
while tmux has-session -t chain4 2>/dev/null || tmux has-session -t chain3 2>/dev/null \
      || tmux has-session -t chain2 2>/dev/null; do sleep 30; done
say "=== lease free ==="

for proto in iid lot_time; do
  say "--- re-measuring the $proto floor, 6 repeats ---"
  rm -f "runs/determinism__${proto}__cnn_gn.json"
  PROTO=$proto ENC=cnn_gn bash scripts/determinism_repeats.sh 6 >> "$LOG" 2>&1 \
    || say "  ($proto floor returned nonzero -- it now refuses to write a short one)"
done
say "=== chain5 done ==="
