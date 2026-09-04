#!/usr/bin/env bash
# Stage C on GPU 0, as soon as its two caches exist, in parallel with the
# stage A/B sweep on GPUs 2-3.
set -uo pipefail
cd "$(dirname "$0")/.."
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/chain_c.log; }
say "waiting for the RPCA cache and the SSL checkpoint"
while pgrep -f "rpca_prepare.py" >/dev/null || pgrep -f "pretrain_ssl.py" >/dev/null; do
  sleep 30
done
say "caches ready: rpca=$([ -f data/rpca.pt ] && echo yes || echo no) ssl=$([ -f runs/ssl_pretrain.pt ] && echo yes || echo no)"
GPUS="0" bash scripts/sweep_c.sh
say "stage C finished"
