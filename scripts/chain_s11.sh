#!/usr/bin/env bash
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "GN vs BN done" logs/gn_vs_bn.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] gn_vs_bn done, starting clean RPCA ablation" >> logs/chain_s1.log
GPUS="0 1" bash scripts/rpca_ablation_clean.sh
echo "[$(date +%H:%M:%S)] === chain_s11 done ===" >> logs/chain_s1.log
