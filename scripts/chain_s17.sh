#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "capacity control seeds done" logs/pooling_control_seeds.log 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] control done, starting the DG power check" >> logs/chain_s1.log
GPUS="0 1" bash scripts/dg_power_check.sh
echo "[$(date +%H:%M:%S)] === chain_s17 done ===" >> logs/chain_s1.log
