#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "protocols done" logs/pooling_protocols.log 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] protocol sweep done, resolving the size pooling effect" >> logs/chain_s1.log
GPUS="0 1" bash scripts/pooling_size_seeds.sh
echo "[$(date +%H:%M:%S)] === chain_s13 done ===" >> logs/chain_s1.log
