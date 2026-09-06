#!/usr/bin/env bash
# Queued behind chain3, in its own file -- never appended to a running script.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/chain4.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== waiting for chain3 (null power fix) ==="
while tmux has-session -t chain3 2>/dev/null; do sleep 30; done
say "--- dilation control (H69) ---"
bash scripts/dilation_control.sh >> "$LOG" 2>&1 || say "  (returned nonzero)"
say "=== chain4 done ==="
