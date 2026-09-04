#!/usr/bin/env bash
# Run the permutation test the previous stage queued incorrectly.
# pooling_size_seeds.sh calls gn_vs_bn.py with --tag poolsize, which matches no
# files; that script was hardcoded to the lot protocol and two encoders. It has
# since been generalized, but pooling_size_seeds.sh was already executing and
# editing a running bash script is unsafe -- bash reads by byte offset -- so the
# correct call is made here instead.
set -uo pipefail
cd ~/Documents/workspace/wafer-tool-shift
while ! grep -q "size pooling seeds done" logs/pooling_size_seeds.log 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] size pooling seeds done, running the permutation test" >> logs/chain_s1.log
$HOME/miniforge3/envs/pdeno/bin/python scripts/gn_vs_bn.py \
  --protocol size --arm-a cnn_gn:poolmeanmax --arm-b cnn_gn:poolmean \
  --label-a meanmax --label-b mean \
  --out runs/pooling_size_perm.json >> logs/pooling_size_seeds.log 2>&1
echo "[$(date +%H:%M:%S)] === chain_s14 done ===" >> logs/chain_s1.log
