#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "representation grid done" logs/repr_one_session.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] repr grid finished, starting pooling sweep" >> logs/chain_s1.log
GPUS="0 1" bash scripts/pooling_sweep.sh
echo "[$(date +%H:%M:%S)] === chain_s8 done ===" >> logs/chain_s1.log
