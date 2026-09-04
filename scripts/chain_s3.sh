#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "chain_s2 done" logs/chain_s1.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] chain_s2 finished, starting MIXED-SYNTH sweep" >> logs/chain_s1.log
GPUS="0 1" bash scripts/mixed_sweep.sh
echo "[$(date +%H:%M:%S)] === chain_s3 done ===" >> logs/chain_s1.log
