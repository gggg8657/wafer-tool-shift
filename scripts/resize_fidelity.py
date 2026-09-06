"""What does `resize_nearest` delete, on the wafers it downsamples?

Two results in this project lean on the same fact about preprocessing, in
opposite directions. Input resolution was refuted as a lever because 97.7% of
wafers are *upsampled* to 64x64 and a finer grid has nothing to recover. The
pooling mechanism (H65) then showed that same upsampling makes a defect's
apparent width a function of its geometry, which is how it does harm.

Neither statement covers the wafers that are **downsampled**. `resize_nearest`
picks 64 row indices and 64 column indices and takes their outer product, so on
a wafer taller or wider than 64 it does not average or blur -- it *drops* the
rows and columns it did not pick. A failing die survives only if both its row
and its column were sampled.

That matters most for exactly the class this project's one positive result is
about. A `Scratch` is a thin connected line; if it runs along a dropped row, it
is gone, and the wafer arrives at the encoder labelled `Scratch` with little or
no scratch in it. That is label noise manufactured by the pipeline, and it would
be invisible to every metric here because the label is never re-derived from the
resized map.

H67, before the run: among downsampled wafers, a material fraction of failing
dies is dropped, and the loss is worse for `Scratch` than for classes whose
signature covers area (`Center`, `Edge-Ring`), because a thin line intersects
fewer sampled rows and columns than a blob of the same failing-die count.

Retention is computed exactly rather than estimated: the sampled cells are the
outer product of the two index vectors `resize_nearest` builds, so a die at
(r, c) survives if and only if r is among the sampled rows and c among the
sampled columns.

    python scripts/resize_fidelity.py     # writes runs/resize_fidelity.json
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import CLASSES, build_corpus  # noqa: E402


def retention(a: np.ndarray, n: int = 64):
    """Fraction of failing dies that survive `resize_nearest`, exactly.

    Also returns what a wafer of this size would retain if its failures were
    scattered uniformly -- |yi|/h * |xi|/w. That is the control: a bigger wafer
    drops more rows whatever is drawn on it, so a per-class difference in raw
    retention may be nothing but a per-class difference in wafer size. The
    ratio of the two is the part attributable to the *shape* of the defect.
    """
    h, w = a.shape
    yi = np.unique((np.arange(n) * h // n).clip(0, h - 1))
    xi = np.unique((np.arange(n) * w // n).clip(0, w - 1))
    fail = a == 2
    total = int(fail.sum())
    if total == 0:
        return None, None, 0
    kept = int(fail[np.ix_(yi, xi)].sum())
    expected = (len(yi) / h) * (len(xi) / w)
    return kept / total, expected, total


def main():
    C = build_corpus(cache="data/corpus.pt")
    by_class = defaultdict(list)
    by_class_exp = defaultdict(list)
    n_down = n_up = 0
    zero_left = defaultdict(int)

    for i in range(len(C)):
        a = C.maps[i]
        a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        h, w = a.shape
        down = h > 64 or w > 64
        if down:
            n_down += 1
        else:
            n_up += 1
            continue                      # upsampling keeps every die
        r, exp, total = retention(a)
        if r is None:
            continue
        cls = CLASSES[int(C.labels[i])]
        by_class[cls].append(r)
        by_class_exp[cls].append(exp)
        if r == 0.0:
            zero_left[cls] += 1

    per_class = {}
    for cls, rs in sorted(by_class.items()):
        ex = by_class_exp[cls]
        per_class[cls] = {
            "n_downsampled_wafers": len(rs),
            "mean_fail_die_retention": statistics.mean(rs),
            "median_fail_die_retention": statistics.median(rs),
            "min_fail_die_retention": min(rs),
            "n_losing_over_half": sum(1 for v in rs if v < 0.5),
            "n_losing_everything": zero_left.get(cls, 0),
            # the control: what a wafer of this size retains if its failures
            # are scattered uniformly, and how far the class departs from it
            "mean_expected_from_size_alone": statistics.mean(ex),
            "mean_ratio_actual_over_expected": statistics.mean(
                [a_ / e for a_, e in zip(rs, ex) if e > 0]),
            # The mean ratio is ~1 for ANY defect shape: a row is sampled with
            # probability |yi|/h whatever is drawn on it, so the expectation
            # carries no information about structure. What separates a thin
            # line from a scattered blob is the *spread* -- a horizontal
            # scratch confined to few rows retains nearly all or nearly none
            # depending on whether those rows survive, while scattered
            # failures average out. The dispersion is the discriminating
            # statistic and the mean is not.
            "stdev_ratio_actual_over_expected": (
                statistics.stdev([a_ / e for a_, e in zip(rs, ex) if e > 0])
                if len(rs) > 1 else None),
        }

    allr = [v for rs in by_class.values() for v in rs]
    res = {
        "what": "fraction of failing dies surviving resize_nearest, on the "
                "wafers it downsamples",
        "how": "resize_nearest samples 64 row and 64 column indices and takes "
               "their outer product, so a die at (r, c) survives iff r is a "
               "sampled row and c a sampled column. Computed exactly, not "
               "estimated.",
        "n_wafers_upsampled_or_exact": n_up,
        "n_wafers_downsampled": n_down,
        "frac_downsampled": n_down / max(1, n_up + n_down),
        "overall": {
            "mean_fail_die_retention": statistics.mean(allr) if allr else None,
            "median_fail_die_retention": (statistics.median(allr) if allr
                                          else None),
            "n_losing_over_half": sum(1 for v in allr if v < 0.5),
            "n_losing_everything": sum(zero_left.values()),
        },
        "per_class": per_class,
    }
    Path("runs/resize_fidelity.json").write_text(json.dumps(res, indent=2))

    print(f"{n_down:,} of {n_up + n_down:,} wafers are downsampled "
          f"({100 * res['frac_downsampled']:.1f}%)\n")
    print(f"{'class':12s} {'n':>6s} {'mean':>8s} {'exp(size)':>10s} "
          f"{'ratio':>7s} {'sd(ratio)':>10s} {'<50%':>6s}")
    for cls, d in sorted(per_class.items(),
                         key=lambda kv: kv[1]["mean_fail_die_retention"]):
        print(f"{cls:12s} {d['n_downsampled_wafers']:6d} "
              f"{d['mean_fail_die_retention']:8.4f} "
              f"{d['mean_expected_from_size_alone']:10.4f} "
              f"{d['mean_ratio_actual_over_expected']:7.3f} "
              + (f"{d['stdev_ratio_actual_over_expected']:10.4f} "
                 if d['stdev_ratio_actual_over_expected'] is not None
                 else f"{'-':>10s} ")
              + f"{d['n_losing_over_half']:6d}")
    o = res["overall"]
    print(f"\noverall mean retention {o['mean_fail_die_retention']:.4f}; "
          f"{o['n_losing_over_half']} wafers keep under half their failing "
          f"dies, {o['n_losing_everything']} keep none")
    print("wrote runs/resize_fidelity.json")


if __name__ == "__main__":
    main()
