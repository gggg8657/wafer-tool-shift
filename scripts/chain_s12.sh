#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "clean RPCA ablation done" logs/rpca_clean.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] clean ablation done, starting hide-raw-fail" >> logs/chain_s1.log
GPUS="0 1" bash scripts/rpca_hide_raw.sh
echo "[$(date +%H:%M:%S)] === chain_s12 done ===" >> logs/chain_s1.log
