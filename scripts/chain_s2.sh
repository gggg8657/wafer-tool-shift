#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "chain_s1 done" logs/chain_s1.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] chain_s1 finished, starting domain-def sweep" >> logs/chain_s1.log
GPUS="0 1" bash scripts/domain_def_sweep.sh
echo "[$(date +%H:%M:%S)] === chain_s2 done ===" >> logs/chain_s1.log
