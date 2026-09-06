#!/usr/bin/env bash
# Queued behind `chain_after_size.sh`, in its own file rather than appended to
# a script that is currently executing.
#
# Bash reads a script by byte offset as it runs, so editing a running script
# makes it resume at the wrong place in the new text. That happened once
# already this weekend to `chain_rest.sh`, and I did it again to
# `chain_after_size.sh` while it was mid-sweep. Reverted within a minute and
# moved here, which is what should have happened the first time.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/chain3.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== waiting for chain_after_size.sh (scale-aware sweep + floors) ==="
while tmux has-session -t chain2 2>/dev/null; do sleep 30; done
say "=== lease free ==="

say "--- give the two underpowered nulls a test that could reject them (H68) ---"
bash scripts/null_power_fix.sh >> "$LOG" 2>&1 \
  || say "  (null power fix returned nonzero)"
say "=== chain3 done ==="
