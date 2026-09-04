#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "size objective grid done" logs/size_objectives.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] size grid done, starting GN vs BN at 8 seeds" >> logs/chain_s1.log
GPUS="0 1" bash scripts/gn_vs_bn.sh
echo "[$(date +%H:%M:%S)] === chain_s10 done ===" >> logs/chain_s1.log
