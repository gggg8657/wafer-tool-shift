#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "chain_s5 done" logs/chain_s1.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] chain_s5 finished, starting long-tail sweep" >> logs/chain_s1.log
GPUS="0 1" bash scripts/longtail_sweep.sh
echo "[$(date +%H:%M:%S)] === chain_s6 done ===" >> logs/chain_s1.log
