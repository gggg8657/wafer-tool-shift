"""Across every three-seed subset, not just the first: how often does the screen mislead?

Entry 96 counted what a three-seed experiment would have concluded, using seeds
0-2. That is one triple out of C(8,3) = 56, and the sharpest claim in that entry
-- that on `lot`/`cnn_bn` the range screen calls both the treatment and its
capacity control separated -- rested entirely on it. If seeds 0-2 happened to be
an unlucky draw, the claim is much weaker than it was stated.

So: enumerate all 56 triples, paired (the same three seeds for both arms, which
is how anyone would actually run three seeds), and score each with both
instruments. This turns a single anecdote into an operating characteristic:

  * for effects established at eight seeds, the fraction of triples on which the
    range screen would have called them separated -- its sensitivity;
  * for comparisons that are null at eight seeds, the fraction of triples on
    which it would have called them separated anyway -- its false-positive rate,
    which is the number that decides whether a control can be trusted.

The permutation test needs no enumeration: at three per arm its floor is 0.10 on
every triple, so its sensitivity at 0.05 is exactly zero by arithmetic.

    python scripts/triple_operating_curve.py   # writes runs/triple_operating_curve.json
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

SMALL = 3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    rep = load("rep", "scripts/report.py")
    cost = json.loads(Path("runs/three_seed_cost.json").read_text())
    A = load("tsc", "scripts/three_seed_cost.py").arms()
    F = rep.floors("runs")

    def series(proto, enc, arm, met):
        obj, _, tag = arm.partition("/")
        per = A.get((proto, enc, obj, tag), {})
        return {s: (r["test"]["per_class_f1"].get(met.split(":", 1)[1])
                    if met.startswith("class:") else r["test"][met])
                for s, r in per.items()}

    rows = []
    for c in cost["comparisons"]:
        a = series(c["protocol"], c["encoder"], c["arm"], c["metric"])
        b = series(c["protocol"], c["encoder"], c["baseline"], c["metric"])
        sh = sorted(set(a) & set(b))
        if len(sh) < 8 or any(a[s] is None or b[s] is None for s in sh):
            continue
        fl = rep.floor_for(F, c["protocol"])
        n_sep = 0
        triples = list(itertools.combinations(sh[:8], SMALL))
        for t in triples:
            v, _ = rep.separation([a[s] for s in t], [b[s] for s in t], fl)
            n_sep += v.startswith("**separated")
        rows.append({**{k: c[k] for k in ("protocol", "encoder", "arm",
                                          "baseline", "metric", "p_8_seeds",
                                          "sig_at_8")},
                     "n_triples": len(triples),
                     "n_triples_screen_calls_separated": n_sep,
                     "frac_separated": n_sep / len(triples)})

    est = [r for r in rows if r["sig_at_8"]]
    nul = [r for r in rows if not r["sig_at_8"]]
    res = {
        "what": "for every comparison at eight seeds, the fraction of all "
                f"C(8,{SMALL}) paired seed triples on which the range screen "
                "would have called it separated",
        "why": "entry 96 used seeds 0-2, one triple of fifty-six. This checks "
               "whether its claim survives the choice of triple.",
        "n_comparisons": len(rows),
        "n_triples_each": rows[0]["n_triples"] if rows else 0,
        "established_at_8": {
            "n": len(est),
            "mean_frac_screen_separates": (sum(r["frac_separated"] for r in est)
                                           / len(est)) if est else None,
            "n_never_separated_on_any_triple": sum(
                1 for r in est if r["n_triples_screen_calls_separated"] == 0),
        },
        "null_at_8": {
            "n": len(nul),
            "mean_frac_screen_separates": (sum(r["frac_separated"] for r in nul)
                                           / len(nul)) if nul else None,
            "n_separated_on_at_least_one_triple": sum(
                1 for r in nul if r["n_triples_screen_calls_separated"] > 0),
            "worst_frac": max((r["frac_separated"] for r in nul), default=None),
        },
        "permutation_sensitivity_at_3": 0.0,
        "note": "the permutation test needs no enumeration: its floor at three "
                "per arm is 0.10 on every triple, so nothing reaches 0.05.",
        "comparisons": rows,
    }
    Path("runs/triple_operating_curve.json").write_text(json.dumps(res, indent=2))

    e, n = res["established_at_8"], res["null_at_8"]
    print(f"{len(rows)} comparisons, {res['n_triples_each']} triples each\n")
    print(f"real effects (significant at 8 seeds), n={e['n']}:")
    print(f"  the range screen calls them separated on "
          f"{100 * e['mean_frac_screen_separates']:.0f}% of triples on average")
    print(f"  {e['n_never_separated_on_any_triple']} are missed on every triple")
    print(f"\nnull comparisons (not significant at 8 seeds), n={n['n']}:")
    print(f"  the screen calls them separated on "
          f"{100 * n['mean_frac_screen_separates']:.0f}% of triples on average")
    print(f"  {n['n_separated_on_at_least_one_triple']} of {n['n']} are called "
          f"separated on at least one triple; worst is "
          f"{100 * n['worst_frac']:.0f}% of triples")
    print("\nwrote runs/triple_operating_curve.json")


if __name__ == "__main__":
    main()
