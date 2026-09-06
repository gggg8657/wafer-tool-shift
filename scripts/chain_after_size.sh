#!/usr/bin/env bash
# Everything queued behind `size_complete.sh`, in priority order.
#
#   tmux new-session -d -s chain2 "bash scripts/chain_after_size.sh"
#
# GPUs 0 and 1 are this loop's lease and `size_complete.sh` holds both. Rather
# than oversubscribe them or leave the lease idle when it finishes, this waits
# for that tmux session to end and then runs the queue.
#
#   1. scale_aware_sweep.sh   -- 16 cells, tests H66, the highest-value open
#                                question: does a receptive field dilated by
#                                round(64/w) recover the pooling gain on `size`?
#   2. the two missing floors -- `lot_time` and `iid`
#
# The floors matter more than their size suggests. This project's central
# methodological claim is that the run-to-run floor is not one number but a
# property of each protocol, and it is demonstrated with exactly two
# measurements: `lot` at 0.0054 and `size` at 0.0133, a factor of 2.5. The
# headline forward-only result lives on `lot_time`, whose floor has never been
# measured -- so every `lot_time` comparison in these documents is screened
# against a floor borrowed from a different protocol, which is the same kind of
# borrowing the paper criticises elsewhere. Two more measurements either
# support the claim across four protocols or complicate it.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/chain_after_size.log
mkdir -p logs runs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== waiting for size_complete.sh to release GPUs 0,1 ==="
while tmux has-session -t sizefin 2>/dev/null; do sleep 30; done
say "=== lease free ==="

say "--- 1/3 scale-aware sweep (H66) ---"
bash scripts/scale_aware_sweep.sh >> logs/chain_after_size.log 2>&1 \
  || say "  (scale-aware sweep returned nonzero; continuing to the floors)"

for proto in lot_time iid; do
  say "--- floor: $proto ---"
  PROTO=$proto ENC=cnn_gn bash scripts/determinism_repeats.sh 6 \
    >> logs/chain_after_size.log 2>&1 \
    || say "  ($proto floor returned nonzero)"
done

say "=== chain done ==="
