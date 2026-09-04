#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "chain_s4 done" logs/chain_s1.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] chain_s4 finished, starting metric backfill" >> logs/chain_s1.log
GPUS="0 1" bash scripts/backfill_metrics.sh
echo "[$(date +%H:%M:%S)] === chain_s5 done ===" >> logs/chain_s1.log
