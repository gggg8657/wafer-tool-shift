"""Correct the four-combination table for the fact that we picked the winner.

Entry 100 arranged the four normalisation-by-pooling combinations and tested
each protocol's best against the other three. It then noted, and argued past,
the obvious objection: the winner was chosen by the same data that scores it, so
those p-values are not family-wise. The argument was that the comparisons which
matter had been pre-specified as individual hypotheses. That is an argument, and
this is a measurement.

The correction is a max-statistic permutation test, and the design has to
respect the structure. All four arms ran the same eight seeds, so a seed fixes
the split, the initialisation and the batch order for all of them. The null is
therefore not "all thirty-two numbers are exchangeable" -- that would throw away
the pairing and understate the noise floor -- but "within each seed, which arm
produced which number is arbitrary". So each draw permutes the four values
*within* every seed independently, recomputes all six pairwise mean differences,
and records the largest.

A pairwise difference's family-wise p is the fraction of draws whose maximum
matches or exceeds it. 4!^8 is about 1.1e11 arrangements, so this is Monte Carlo
rather than exact, with a fixed seed and its standard error reported -- the
distinction this project has been strict about elsewhere and should not drop
here.

    python scripts/combination_maxt.py    # writes runs/combination_maxt.json
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import random
from pathlib import Path

COMBOS = [("cnn_gn", "poolmean"), ("cnn_gn", "poolmeanmax"),
          ("cnn_bn", "poolmean"), ("cnn_bn", "poolmeanmax")]
LABEL = {("cnn_gn", "poolmean"): "GroupNorm + mean",
         ("cnn_gn", "poolmeanmax"): "GroupNorm + mean-and-max",
         ("cnn_bn", "poolmean"): "BatchNorm + mean",
         ("cnn_bn", "poolmeanmax"): "BatchNorm + mean-and-max"}
DRAWS = 20000


def arm(proto, enc, tag, met):
    out = {}
    for f in glob.glob(f"runs/{proto}__{enc}__erm__{tag}__s*.json"):
        r = json.load(open(f))
        out[r["seed"]] = (r["test"]["per_class_f1"].get(met.split(":", 1)[1])
                          if met.startswith("class:") else r["test"][met])
    return out


def maxt(mat, draws, rng):
    """mat[seed][arm]. Returns the null distribution of the largest pairwise
    absolute mean difference, permuting arm labels within each seed."""
    n_seed, n_arm = len(mat), len(mat[0])
    pairs = list(itertools.combinations(range(n_arm), 2))
    out = []
    for _ in range(draws):
        cols = [[0.0] * n_arm for _ in range(n_seed)]
        for i, row in enumerate(mat):
            perm = list(row)
            rng.shuffle(perm)
            cols[i] = perm
        means = [sum(cols[i][j] for i in range(n_seed)) / n_seed
                 for j in range(n_arm)]
        out.append(max(abs(means[a] - means[b]) for a, b in pairs))
    return out


def main():
    res = {"what": "family-wise correction for having selected the best of "
                   "four combinations before testing it",
           "null": "within each seed, which arm produced which number is "
                   "arbitrary -- this keeps the seed pairing, which permuting "
                   "all thirty-two values would discard",
           "method": f"Monte Carlo max-statistic over {DRAWS:,} draws "
                     "(4!^8 is about 1.1e11, so not exact)",
           "draws": DRAWS, "protocols": {}}
    for proto in ("lot", "size", "iid", "lot_time"):
        for met in ("macro_f1", "class:Scratch"):
            arms = {}
            for enc, tag in COMBOS:
                a = arm(proto, enc, tag, met)
                sh = sorted(a)[:8]
                if len(sh) == 8 and not any(a[s] is None for s in sh):
                    arms[(enc, tag)] = [a[s] for s in sh]
            if len(arms) < 4:
                continue
            keys = list(arms)
            mat = [[arms[k][i] for k in keys] for i in range(8)]
            rng = random.Random(0)
            null = maxt(mat, DRAWS, rng)
            means = {k: sum(v) / 8 for k, v in arms.items()}
            best = max(means, key=means.get)
            rows = []
            for k in sorted(means, key=lambda z: -means[z]):
                if k == best:
                    continue
                d = means[best] - means[k]
                hits = sum(1 for v in null if v >= d - 1e-15)
                p = (hits + 1) / (DRAWS + 1)          # add-one, never zero
                se = math.sqrt(max(p * (1 - p), 1e-12) / DRAWS)
                rows.append({"rival": LABEL[k], "difference": d,
                             "p_family_wise": p, "monte_carlo_se": se,
                             "survives_05": p < 0.05})
            res["protocols"].setdefault(proto, {})[met] = {
                "best": LABEL[best],
                "n_rivals_beaten_family_wise": sum(1 for r in rows
                                                   if r["survives_05"]),
                "n_rivals": len(rows), "rows": rows,
            }
    Path("runs/combination_maxt.json").write_text(json.dumps(res, indent=2))
    for proto, e in res["protocols"].items():
        for met, v in e.items():
            print(f"{proto}/{met}: best = {v['best']}, beats "
                  f"{v['n_rivals_beaten_family_wise']} of {v['n_rivals']} "
                  "after family-wise correction")
            for r in v["rows"]:
                print(f"    vs {r['rival']:28s} {r['difference']:+.4f}  "
                      f"p_fw={r['p_family_wise']:.4f} "
                      f"(SE {r['monte_carlo_se']:.4f})"
                      f"{'  survives' if r['survives_05'] else ''}")
    print("\nwrote runs/combination_maxt.json")


if __name__ == "__main__":
    main()
