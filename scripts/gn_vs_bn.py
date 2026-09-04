"""Exact permutation test for GroupNorm against BatchNorm on `lot`.

    python scripts/gn_vs_bn.py        # -> runs/gn_vs_bn.json

Why not the range-overlap criterion used everywhere else in this repository:
the range of a sample grows with the sample size, so running more seeds to
settle a question makes non-overlap strictly harder to reach. That criterion is
the right conservative default for reading a table of three-seed cells; it is
the wrong tool for a sweep whose purpose is to resolve one comparison.

An exact permutation test on the group labels is assumption-free and does not
punish extra data. With eight seeds per arm there are C(16, 8) = 12,870
arrangements, so a two-sided p below 0.001 is reachable; with three seeds per
arm there are only 20, and the smallest attainable two-sided p is 0.1 -- which
is why no amount of care could have settled this at n = 3.

Both readings are reported. If they disagree, the disagreement is the result.
"""
from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path
import sys


def perm_p(a, b):
    """Two-sided exact permutation p for a difference in means."""
    pool = list(a) + list(b)
    n = len(a)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    total = hits = 0
    for idx in combinations(range(len(pool)), n):
        s = set(idx)
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in s]
        total += 1
        if abs(sum(x) / len(x) - sum(y) / len(y)) >= obs - 1e-12:
            hits += 1
    return hits / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="gnbn",
                    help="legacy: the GroupNorm/BatchNorm sweep's tag")
    ap.add_argument("--out", default="runs/gn_vs_bn.json")
    # Explicit arms, so this is a permutation test between any two cell
    # families rather than only between the two normalizations. The first
    # version hardcoded `lot__cnn_*__erm__<tag>`, which meant a queued call
    # asking it to compare pooling variants on `size` matched zero files and
    # returned quietly -- the fourth silent no-op of this weekend. A tool that
    # answers the wrong question loudly is fine; one that answers nothing
    # quietly is not.
    ap.add_argument("--protocol", default="lot")
    ap.add_argument("--objective", default="erm")
    ap.add_argument("--arm-a", default=None,
                    help="encoder[:tag] for the first arm, e.g. cnn_gn or "
                         "cnn_gn:poolmeanmax")
    ap.add_argument("--arm-b", default=None)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    a = ap.parse_args()

    def collect(spec, fallback_tag):
        enc, _, tg = spec.partition(":")
        tg = tg or fallback_tag
        out = {}
        for f in glob.glob(f"{a.runs}/{a.protocol}__{enc}__{a.objective}"
                           f"__{tg}__s*.json"):
            r = json.loads(Path(f).read_text())
            out[r["seed"]] = r["test"]["macro_f1"]
        return out, f"{a.protocol}/{enc}/{tg}"

    if a.arm_a and a.arm_b:
        ga, na = collect(a.arm_a, a.tag)
        gb, nb = collect(a.arm_b, a.tag)
    else:
        got = {"cnn_gn": {}, "cnn_bn": {}}
        for f in glob.glob(f"{a.runs}/lot__cnn_*__erm__{a.tag}__s*.json"):
            r = json.loads(Path(f).read_text())
            got[r["encoder"]][r["seed"]] = r["test"]["macro_f1"]
        ga, gb = got["cnn_gn"], got["cnn_bn"]
        na, nb = "cnn_gn", "cnn_bn"
    na, nb = (a.label_a or na), (a.label_b or nb)
    shared = sorted(set(ga) & set(gb))
    gn = [ga[s] for s in shared]
    bn = [gb[s] for s in shared]
    if len(gn) < 2:
        # loudly, and with a non-zero exit, so a queued caller cannot mistake
        # "matched nothing" for "found nothing to report"
        print(f"ERROR: need >=2 seeds present in BOTH arms; "
              f"{na} has {len(ga)}, {nb} has {len(gb)}, shared {len(shared)}")
        return 2

    F = json.loads(Path(a.runs, "determinism__lot__cnn_gn.json").read_text()) \
        if Path(a.runs, "determinism__lot__cnn_gn.json").exists() else None
    floor = F.get("range") if F else None

    p, total = perm_p(gn, bn)
    overlap = not (min(gn) > max(bn) or min(bn) > max(gn))
    margin = (0.0 if overlap else
              (min(gn) - max(bn) if min(gn) > max(bn) else max(gn) - min(bn)))
    rec = {
        "cell": f"{a.protocol} / {a.objective} / one session",
        "arm_a": na, "arm_b": nb,
        "n_per_arm": len(gn),
        "cnn_gn": gn, "cnn_bn": bn,
        "mean_gn": sum(gn) / len(gn), "mean_bn": sum(bn) / len(bn),
        "difference": sum(gn) / len(gn) - sum(bn) / len(bn),
        "range_test": {
            "ranges_overlap": overlap,
            "margin": margin,
            "floor": floor,
            "verdict": ("ranges overlap" if overlap else
                        ("clears the floor" if floor and abs(margin) > floor
                         else "below the floor")),
        },
        "permutation_test": {
            "arrangements": total,
            "p_two_sided": p,
            "smallest_attainable_p": 2.0 / total,
        },
        "note": ("The range test grows stricter with sample size, so it is the "
                 "right default for reading many three-seed cells and the wrong "
                 "tool for a sweep meant to settle one comparison. The "
                 "permutation test is exact and does not punish extra data. "
                 "Both are reported."),
    }
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
