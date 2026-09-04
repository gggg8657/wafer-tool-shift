"""Assert a sweep produced the cells it launched, and say so loudly if not.

    python scripts/verify_stage.py --glob 'runs/lot__*__poolmean__s*.json' \
                                   --expect 3 --label 'pooling, mean arm'

Four times this weekend a stage logged completion and produced nothing, or less
than it claimed:

1. `sweep_c.sh` built a log path containing a `/` from `--init-from
   runs/ssl_pretrain.pt`, so bash failed the redirect before exec and three
   SSL-initialized cells never ran. The sweep logged "launch" for each.
2. `report.py` keyed cells without the seed, so a second seed of any cell would
   have silently replaced the first.
3. The first wafer-budget active-learning run was OOM-killed two strategies in.
   Its stage printed `=== al wafer budget done ===` and wrote no JSON.
4. `pooling_size_seeds.sh` invoked the permutation test with globs hardcoded to
   a different protocol. Zero files matched; it printed a note and returned 0.

Every one of them was found by an incidental check rather than by anything
failing. The shared shape is that a stage boundary converts an error into an
absence, and absences are invisible unless something is counting.

So: count. This is deliberately dumb -- a glob and an integer -- because the
failures it catches are not subtle, and a check that is hard to add will not be
added.
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, action="append",
                    help="pattern the stage should have produced; repeatable")
    ap.add_argument("--expect", required=True, type=int,
                    help="how many files each pattern should match")
    ap.add_argument("--label", default="stage")
    ap.add_argument("--at-least", action="store_true",
                    help="treat --expect as a lower bound rather than exact")
    a = ap.parse_args()

    bad = []
    for pat in a.glob:
        n = len(_glob.glob(pat))
        ok = n >= a.expect if a.at_least else n == a.expect
        print(f"  {'ok ' if ok else 'BAD'}  {n:4d} / {a.expect:<4d}  {pat}")
        if not ok:
            bad.append((pat, n))
    if bad:
        print(f"\nFAILED: {a.label} did not produce what it launched.")
        for pat, n in bad:
            print(f"  expected {a.expect} from {pat}, found {n}")
        print("A stage that logs completion having written nothing is the "
              "failure mode this check exists for; do not treat the missing "
              "results as 'no effect'.")
        return 1
    print(f"ok: {a.label} produced everything it launched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
