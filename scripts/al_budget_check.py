"""Was active learning beaten by random, or beaten by a smaller training set?

    python scripts/al_budget_check.py

`runs/active_learning.json` reports macro-F1 against a budget counted in **whole
lots**, and on that axis random acquisition beats every heuristic. But the same
file records `wafers_mean`, the number of wafers each strategy actually ended up
labelling, and the strategies are not buying comparable amounts of data.

The mechanism is in `lot_scores`: a lot's score is the *mean* of its wafers'
scores. The maximum of noisy means favours small samples -- a 2-wafer lot's mean
is one or two draws from the score distribution and can land anywhere, while a
25-wafer lot's mean regresses to the pool average. So "take the top-scoring
lots" is partly "take the smallest lots", and random selection has no such bias.

This script does not re-run anything. It reads the stored curve, reports the
acquisition cost in wafers alongside the cost in lots, and re-plots accuracy
against wafers labelled -- the axis on which the strategies were actually being
compared without anyone saying so.

Comparisons at matched wafer counts are **linear interpolation between measured
budget points**, never extrapolation beyond the measured range, and every
interpolated figure is labelled as one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def interp(xs, ys, x):
    """Linear interpolation inside the measured range only; None outside it."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--al", default="runs/active_learning.json")
    ap.add_argument("--out", default="runs/al_budget_check.json")
    a = ap.parse_args()
    d = json.loads(Path(a.al).read_text())
    B = d["budgets"]
    R = d["results"]

    curves = {s: {"lots": B,
                  "wafers": [R[s][str(b)]["wafers_mean"] for b in B],
                  "macro_f1": [R[s][str(b)]["macro_f1_mean"] for b in B]}
              for s in R}

    out = {
        "source": a.al, "seeds": d["seeds"], "n_pool_lots": d["n_pool_lots"],
        "note": ("Budget is counted in lots. wafers_mean shows the strategies "
                 "do not buy comparable amounts of data at the same lot "
                 "budget, because lot score is the mean of the lot's wafer "
                 "scores and the max of noisy means favours small lots."),
        "curves": curves,
        "wafers_per_lot": {s: {str(b): R[s][str(b)]["wafers_mean"] / b for b in B}
                           for s in R},
        "wafers_relative_to_random": {
            s: {str(b): R[s][str(b)]["wafers_mean"] / R["random"][str(b)]["wafers_mean"]
                for b in B} for s in R},
    }

    # Re-compare on the axis that was implicitly being varied: wafers labelled.
    # Grid points are chosen inside every strategy's measured wafer range so no
    # comparison relies on extrapolating any curve.
    lo = max(min(c["wafers"]) for c in curves.values())
    hi = min(max(c["wafers"]) for c in curves.values())
    grid = [float(x) for x in np.linspace(lo, hi, 6)]
    matched = {}
    for s, c in curves.items():
        matched[s] = {f"{int(g)}": interp(c["wafers"], c["macro_f1"], g)
                      for g in grid}
    out["wafer_matched_range"] = {"min": lo, "max": hi,
                                  "method": "linear interpolation between "
                                            "measured budget points, no "
                                            "extrapolation"}
    out["macro_f1_at_matched_wafer_budget"] = matched

    wins = {}
    for g in grid:
        vals = {s: matched[s][f"{int(g)}"] for s in matched}
        vals = {k: v for k, v in vals.items() if v is not None}
        if vals:
            wins[f"{int(g)}"] = max(vals, key=vals.get)
    out["best_strategy_at_each_matched_wafer_budget"] = wins

    Path(a.out).write_text(json.dumps(out, indent=2))

    print("wafers actually labelled at each LOT budget")
    print("%-9s" % "strategy" + "".join("%9d" % b for b in B))
    for s in R:
        print("%-9s" % s + "".join("%9.0f" % R[s][str(b)]["wafers_mean"] for b in B))
    print("\nmacro-F1 at matched WAFER budgets (interpolated, no extrapolation)")
    print("%-9s" % "strategy" + "".join("%9d" % g for g in grid))
    for s in matched:
        print("%-9s" % s + "".join(
            ("%9.4f" % matched[s][f'{int(g)}']) if matched[s][f'{int(g)}'] is not None
            else "%9s" % "-" for g in grid))
    print("\nbest at each matched wafer budget:", wins)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
