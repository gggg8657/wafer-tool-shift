#!/usr/bin/env bash
# Queued behind chain5, in its own file. Never appended to a running script.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/chain6.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== waiting for chain4 and chain5 ==="
while tmux has-session -t chain4 2>/dev/null || tmux has-session -t chain5 2>/dev/null; do sleep 30; done
say "--- focal completion (H70) ---"
bash scripts/focal_complete.sh >> "$LOG" 2>&1 || say "  (returned nonzero)"
say "=== chain6 done ==="
