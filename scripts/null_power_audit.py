"""For every null this project asserts, record what its test could have shown.

Twice now a null here has turned out to be a property of the seed budget rather
than of the methods. `size` was reported as "nothing separates from ERM" from
three seeds per arm, where the exact test cannot return below 0.10; at eight,
two objectives separate at p = 0.0019 and p = 0.0050. And a criterion invented
to replace that null -- an arm scattering wider than its effect is unstable
rather than worse -- was falsified by the same run.

The pattern is that a null gets stated with a difference and no statement of
what the instrument could resolve. So this records, for each null the documents
rest on, the seeds per arm and the smallest two-sided p that an exact
permutation test at that size can return. A null whose floor is above 0.05 is
not evidence of absence; it is absence of evidence, and the documents should say
which they mean.

Nothing here re-interprets a result. It attaches the power to it.

    python scripts/null_power_audit.py     # writes runs/null_power_audit.json
"""
from __future__ import annotations

import glob
import importlib.util
import json
import math
from pathlib import Path

# (label, glob template for the arm, treatment tags, control tag, metric)
FAMILIES = [
    ("RPCA fourth channel vs a channel of zeros",
     "runs/lot__rpca_cnn__erm__{t}__s*.json",
     ["rpca2_residual", "rpca2_failmask"], "rpca2_zeros", "macro_f1"),
    ("focal loss vs its bit-exact gamma = 0 control",
     "runs/lot__cnn_gn__focal__{t}__s*.json",
     ["focal0.5", "focal1.0", "focal2.0", "focal5.0"], "focal0.0", "macro_f1"),
    ("meanmean capacity control vs mean",
     "runs/lot__cnn_gn__erm__{t}__s*.json",
     ["poolmeanmean"], "poolmean", "macro_f1"),
]


def perm():
    spec = importlib.util.spec_from_file_location("g", "scripts/gn_vs_bn.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.perm_p


def arm(tmpl, tag, metric):
    out = {}
    for f in glob.glob(tmpl.format(t=tag)):
        r = json.load(open(f))
        v = (r["test"]["per_class_f1"].get(metric.split(":", 1)[1])
             if metric.startswith("class:") else r["test"][metric])
        if v is not None:
            out[r["seed"]] = v
    return out


def main():
    pp = perm()
    res = {"what": "seeds per arm and smallest attainable two-sided p for each "
                   "null the documents rest on",
           "why": "a null whose floor is above 0.05 is absence of evidence, "
                  "not evidence of absence",
           "families": []}
    for label, tmpl, treats, ctrl, metric in FAMILIES:
        base = arm(tmpl, ctrl, metric)
        entry = {"label": label, "control": ctrl, "metric": metric,
                 "comparisons": []}
        for t in treats:
            a = arm(tmpl, t, metric)
            shared = sorted(set(a) & set(base))
            if len(shared) < 2:
                entry["comparisons"].append(
                    {"treatment": t, "n_per_arm": len(shared),
                     "p_two_sided": None, "min_attainable_p": None})
                continue
            x = [a[s] for s in shared]
            b = [base[s] for s in shared]
            p, n = pp(x, b)
            entry["comparisons"].append({
                "treatment": t, "n_per_arm": len(shared),
                "difference": sum(x) / len(x) - sum(b) / len(b),
                "p_two_sided": p, "arrangements": n,
                "min_attainable_p": 2.0 / n,
                "could_reach_05": 2.0 / n <= 0.05,
            })
        ns = [c["n_per_arm"] for c in entry["comparisons"] if c["n_per_arm"]]
        fl = [c["min_attainable_p"] for c in entry["comparisons"]
              if c.get("min_attainable_p")]
        # Report the range and the count, not just the worst. Summarising a
        # family by its minimum said "focal: n=2, cannot reach 0.05" after
        # gamma = 2.0 had been taken to eight seeds and settled -- true of the
        # weakest comparison and false of the family, which is the same
        # collapsing-to-one-number error the per-protocol floor exists to
        # prevent.
        entry["n_per_arm"] = min(ns) if ns else None
        entry["n_per_arm_max"] = max(ns) if ns else None
        entry["min_attainable_p"] = max(fl) if fl else None
        entry["n_comparisons"] = len(entry["comparisons"])
        entry["n_powered"] = sum(1 for c in entry["comparisons"]
                                 if c.get("could_reach_05"))
        entry["could_reach_05"] = bool(fl) and max(fl) <= 0.05
        entry["partially_powered"] = (0 < entry["n_powered"]
                                      < entry["n_comparisons"])
        res["families"].append(entry)

    res["n_families"] = len(res["families"])
    res["n_underpowered"] = sum(1 for e in res["families"]
                                if not e["could_reach_05"])
    res["n_comparisons_total"] = sum(e["n_comparisons"] for e in res["families"])
    res["n_comparisons_powered"] = sum(e["n_powered"] for e in res["families"])
    Path("runs/null_power_audit.json").write_text(json.dumps(res, indent=2))

    print(f"{'null':46s} {'n/arm':>9s} {'worst floor':>12s}  powered")
    print("-" * 84)
    for e in res["families"]:
        f = e["min_attainable_p"]
        n = (str(e["n_per_arm"]) if e["n_per_arm"] == e["n_per_arm_max"]
             else f"{e['n_per_arm']}-{e['n_per_arm_max']}")
        print(f"{e['label'][:46]:46s} {n:>9s} "
              f"{(f'{f:.4f}' if f else '-'):>12s}  "
              f"{e['n_powered']}/{e['n_comparisons']}")
    print(f"\n{res['n_comparisons_powered']} of "
          f"{res['n_comparisons_total']} individual comparisons can return "
          f"p < 0.05; {res['n_underpowered']} of {res['n_families']} families "
          "still contain at least one that cannot.")
    print("wrote runs/null_power_audit.json")


if __name__ == "__main__":
    main()
