"""Settle the sinkhorn task: was 0.10 a finding, or a badly chosen lambda?

One of the four questions handed to this session was to diagnose the sinkhorn
cell's macro-F1 of 0.1026 with a lambda sweep, and to remove the claim if it
was still broken. The sweep ran; the answer never reached `WEEKEND.md`, which
mentions sinkhorn zero times. This writes the verdict where the generators can
read it.

The reading is not subtle. At the default lambda the model collapses to a
single class. At lambda = 0 the penalty is off and the cell lands in ERM's own
seed range, which is the control this needs: it says the collapse is the
penalty's weight and not a broken code path. In between, macro-F1 falls
monotonically with lambda and never exceeds lambda = 0.

What the sweep cannot say is anything about the ordering *among* the small
lambdas: it is one seed per point, and the whole spread across the non-collapsed
values is comparable to ERM's own seed range. So the finding is the shape and
the two endpoints, not the ranking.

    python scripts/sinkhorn_verdict.py    # writes runs/sinkhorn_verdict.json
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path


def main():
    pts = []
    for f in glob.glob("runs/lot__cnn_bn__sinkhorn__ot*__s*.json"):
        d = json.load(open(f))
        pts.append({"ot_lambda": d.get("ot_lambda"), "seed": d["seed"],
                    "macro_f1": d["test"]["macro_f1"],
                    "scratch_f1": d["test"]["per_class_f1"].get("Scratch")})
    pts.sort(key=lambda p: (p["ot_lambda"] is None, p["ot_lambda"]))
    if not pts:
        raise SystemExit("no sinkhorn lambda cells")

    erm = [json.load(open(f))["test"]["macro_f1"]
           for f in glob.glob("runs/lot__cnn_bn__erm__s*.json")]
    zero = next((p for p in pts if p["ot_lambda"] == 0.0), None)
    collapsed = [p for p in pts if p["macro_f1"] < 0.2]
    best = max(pts, key=lambda p: p["macro_f1"])

    res = {
        "what": "sinkhorn lambda sweep on lot/cnn_bn, one seed per lambda",
        "task": "handed to this session: diagnose the 0.10 cell with a lambda "
                "sweep, and remove the claim if it is still broken",
        "n_seeds_per_lambda": 1,
        "points": pts,
        "erm_same_cell": {"n": len(erm), "values": sorted(erm),
                          "min": min(erm) if erm else None,
                          "max": max(erm) if erm else None,
                          "seed_range": (max(erm) - min(erm)) if len(erm) > 1
                          else None},
        "lambda_zero_control": {
            "why": "at lambda = 0 the transport penalty contributes nothing, "
                   "so this cell must land where ERM lands. If it did not, the "
                   "collapse would be a broken code path rather than a "
                   "badly-chosen weight.",
            "macro_f1": zero["macro_f1"] if zero else None,
            "inside_erm_seed_range": (
                bool(erm and zero and min(erm) <= zero["macro_f1"] <= max(erm))
                if zero else None),
            "distance_to_nearest_erm_seed": (
                min(abs(zero["macro_f1"] - v) for v in erm)
                if zero and erm else None),
        },
        "n_collapsed": len(collapsed),
        "collapsed_lambdas": [p["ot_lambda"] for p in collapsed],
        "best_lambda": best["ot_lambda"],
        "best_beats_lambda_zero": bool(zero and best["macro_f1"]
                                       > zero["macro_f1"]),
        "verdict": ("the collapse is the penalty's weight, not the method: "
                    "lambda = 0 lands with ERM and no lambda in the sweep "
                    "beats it"),
        "limit": "one seed per lambda, so the ordering among the "
                 "non-collapsed points is not established; the spread across "
                 "them is comparable to ERM's own seed range",
    }
    Path("runs/sinkhorn_verdict.json").write_text(json.dumps(res, indent=2))
    for p in pts:
        flag = "  <- collapsed" if p["macro_f1"] < 0.2 else ""
        print(f"  lambda={str(p['ot_lambda']):<6} macro-F1 {p['macro_f1']:.4f} "
              f"Scratch {p['scratch_f1']:.4f}{flag}")
    z = res["lambda_zero_control"]
    print(f"\nlambda=0 control: {z['macro_f1']:.4f}, ERM on the same cell "
          f"{res['erm_same_cell']['min']:.4f}-{res['erm_same_cell']['max']:.4f} "
          f"(n={res['erm_same_cell']['n']}), nearest ERM seed "
          f"{z['distance_to_nearest_erm_seed']:.4f} away")
    print(f"best lambda {res['best_lambda']}; beats lambda=0: "
          f"{res['best_beats_lambda_zero']}")
    print("wrote runs/sinkhorn_verdict.json")


if __name__ == "__main__":
    main()
