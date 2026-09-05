#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "pooling protocol seeds done" logs/pooling_protocols_seeds.log 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] protocol seeds done, extending the capacity control" >> logs/chain_s1.log
GPUS="0 1" bash scripts/pooling_control_seeds.sh
echo "[$(date +%H:%M:%S)] === chain_s16 done ===" >> logs/chain_s1.log
