"""What does meeting an unseen geometry actually cost, on `lot_time`?

`WEEKEND.md`'s decision 3 described the forward-only drop as part narrow-slice,
part drift, and "a further unmeasured share is geometries never trained on".
That share is measured -- `run_bench.py` records `test_seen_geometry` and
`test_unseen_geometry` per cell -- and it does not have the sign the sentence
implies.

The comparison needs one guard. The two halves do not always contain the same
classes: the unseen-geometry half of a `lot_time` test set is small and often
misses one of the nine, and macro-F1 over eight classes is not comparable to
macro-F1 over nine. Cells whose halves disagree on the class set are dropped
rather than averaged, which is why this is a script and not a table lookup.

    python scripts/unseen_geometry_cost.py    # writes runs/unseen_geometry_cost.json
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path


def main():
    matched, dropped = [], 0
    seen_files = set()
    for f in sorted(glob.glob("runs/lot_time__*.json")):
        if f in seen_files:
            continue
        seen_files.add(f)
        d = json.load(open(f))
        se = d.get("test_seen_geometry") or {}
        un = d.get("test_unseen_geometry") or {}
        if se.get("macro_f1") is None or un.get("macro_f1") is None:
            continue
        if sorted(se.get("classes_present", [])) != \
           sorted(un.get("classes_present", [])):
            dropped += 1
            continue
        matched.append({
            "file": Path(f).name, "encoder": d["encoder"],
            "objective": d["objective"], "tag": d.get("tag", ""),
            "seed": d["seed"], "n_classes": len(se.get("classes_present", [])),
            "seen": se["macro_f1"], "unseen": un["macro_f1"],
            "cost": un["macro_f1"] - se["macro_f1"],
        })

    res = {
        "what": "macro-F1 on the seen-geometry and unseen-geometry halves of "
                "the same lot_time test set, restricted to cells where both "
                "halves contain the same classes",
        "why_restricted": "the unseen-geometry half is small and often misses "
                          "a class; macro-F1 over eight classes is not "
                          "comparable to macro-F1 over nine",
        "n_cells_matched": len(matched), "n_cells_dropped_class_mismatch": dropped,
        "cells": matched,
    }
    if matched:
        c = [m["cost"] for m in matched]
        res["cost_mean"] = statistics.mean(c)
        res["cost_median"] = statistics.median(c)
        res["cost_min"], res["cost_max"] = min(c), max(c)
        res["n_negative"] = sum(1 for v in c if v < 0)
        res["sign"] = ("unseen geometry is easier" if statistics.mean(c) > 0
                       else "unseen geometry is harder")
    Path("runs/unseen_geometry_cost.json").write_text(json.dumps(res, indent=2))
    if matched:
        print(f"{len(matched)} matched cells ({dropped} dropped for class "
              f"mismatch)")
        print(f"  unseen minus seen: mean {res['cost_mean']:+.4f}, "
              f"median {res['cost_median']:+.4f}, "
              f"range {res['cost_min']:+.4f} to {res['cost_max']:+.4f}")
        print(f"  {res['n_negative']} of {len(matched)} cells negative — "
              f"{res['sign']}")
    else:
        print("no matched cells")
    print("wrote runs/unseen_geometry_cost.json")


if __name__ == "__main__":
    main()
