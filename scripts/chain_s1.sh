#!/usr/bin/env bash
# Queue the weekend's stages onto the two leased GPUs, one stage at a time.
# Waits on a marker line in each stage's own log rather than on a PID, so the
# chain survives being started from a different shell than the stage was.
set -uo pipefail
cd "$(dirname "$0")/.."
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a logs/chain_s1.log; }
wait_for(){   # wait_for <logfile> <marker>
  local f=$1 m=$2
  while ! grep -q "$m" "$f" 2>/dev/null; do sleep 30; done
  say "saw '$m' in $f"
}
mkdir -p logs
wait_for logs/ablate_sigchannel.log "ablation done"
say "--- stage: ssl-init cells ---"
GPUS="0 1" bash scripts/ssl_init_cells.sh
say "--- stage: sinkhorn lambda sweep ---"
GPUS="0 1" bash scripts/sinkhorn_lambda.sh
say "=== chain_s1 done ==="
