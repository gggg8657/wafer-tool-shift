#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "pooling sweep done" logs/pooling.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] pooling done, starting size objective grid" >> logs/chain_s1.log
GPUS="0 1" bash scripts/size_objectives_seeds.sh
echo "[$(date +%H:%M:%S)] === chain_s9 done ===" >> logs/chain_s1.log
