#!/usr/bin/env bash
# Score the second half of H50. The sweep's own permutation call tests macro-F1;
# the prediction was also about Scratch specifically, and the tool could not
# reach a per-class metric when the sweep was launched. It can now.
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
PY=$HOME/miniforge3/envs/pdeno/bin/python
while ! grep -q "lot pooling seeds done" logs/pooling_lot_seeds.log 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] lot pooling done, scoring the Scratch half of H50" >> logs/chain_s1.log
$PY scripts/gn_vs_bn.py --protocol lot \
  --arm-a cnn_gn:poolmeanmax --arm-b cnn_gn:poolmean \
  --label-a meanmax --label-b mean --metric class:Scratch \
  --out runs/pooling_lot_perm_scratch.json >> logs/pooling_lot_seeds.log 2>&1
$PY scripts/verify_stage.py --glob runs/pooling_lot_perm_scratch.json --expect 1 \
  --label "lot pooling Scratch permutation summary" | tee -a logs/pooling_lot_seeds.log || exit 1
echo "[$(date +%H:%M:%S)] === chain_s15 done ===" >> logs/chain_s1.log
