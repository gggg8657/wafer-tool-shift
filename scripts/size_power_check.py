"""Score the `size`-protocol objectives against ERM, and record the floor.

`paper_draft.md` said "the `size` half of the negative result stands as
measured" while the `lot` half was withdrawn for being underpowered. That
sentence was asserted from three seeds per arm, and three seeds per arm on a
two-sample exact permutation test admits C(6,3)/2 = 10 distinct arrangements,
so the smallest two-sided p obtainable is 0.1. Nothing on `size` could have
reached 0.05 no matter how large the effect was.

That is not the same failure as the `lot` half, where the domain vocabulary was
degenerate and the experiment could not have shown an effect. Here the effects
are large -- larger than anything measured on `lot` -- and the instrument simply
has no resolution to certify them. Both halves are unestablished; they are
unestablished for opposite reasons.

    python scripts/size_power_check.py            # writes runs/size_power_check.json
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path

OBJS = ("coral", "dann", "irm", "group_dro", "mixup_domain", "logit_adjust")


def load(spec_path="scripts/gn_vs_bn.py"):
    spec = importlib.util.spec_from_file_location("g", spec_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def arm(obj, tag):
    out = {}
    for f in glob.glob(f"runs/size__cnn_bn__{obj}__{tag}__s*.json"):
        r = json.load(open(f))
        out[r["seed"]] = r["test"]["macro_f1"]
    return out


def main():
    m = load()
    res = {}
    for tag in ("sizeseed",):
        base = arm("erm", tag)
        if len(base) < 2:
            continue
        for obj in OBJS:
            x = arm(obj, tag)
            shared = sorted(set(base) & set(x))
            if len(shared) < 2:
                continue
            a = [x[s] for s in shared]
            b = [base[s] for s in shared]
            p, n = m.perm_p(a, b)
            res[f"{obj}__macro_f1"] = {
                "objective": obj, "metric": "macro_f1", "protocol": "size",
                "tag": tag, "n_per_arm": len(a), "seeds": shared,
                "mean_objective": sum(a) / len(a), "mean_erm": sum(b) / len(b),
                "difference": sum(a) / len(a) - sum(b) / len(b),
                "p_two_sided": p, "arrangements": n,
                "min_attainable_p": 2.0 / n if n else None,
                "at_resolution_floor": bool(n and abs(p - 2.0 / n) < 1e-12),
                "ranges_overlap": not (min(a) > max(b) or min(b) > max(a)),
                # the fact that decides this section: an arm whose own seeds
                # spread further than its distance from ERM has not lost to
                # ERM, it is unstable. GroupDRO spans 0.1844 across three seeds
                # while sitting 0.1534 below ERM.
                "own_seed_range": max(a) - min(a),
                "erm_seed_range": max(b) - min(b),
                "range_exceeds_effect": (max(a) - min(a)) > abs(
                    sum(a) / len(a) - sum(b) / len(b)),
                # signed gap between the two ranges; negative means they overlap
                "range_gap": (min(b) - max(a) if min(b) > max(a)
                              else min(a) - max(b) if min(a) > max(b)
                              else -1.0),
            }
    if res:
        ns = {v["n_per_arm"] for v in res.values()}
        floors = {v["min_attainable_p"] for v in res.values()}
        res["_meta"] = {
            "n_per_arm": sorted(ns),
            "min_attainable_p": sorted(f for f in floors if f is not None),
            "n_range_exceeds_effect": sum(
                1 for k, v in res.items()
                if k != "_meta" and v["range_exceeds_effect"]),
            "n_at_floor": sum(1 for k, v in res.items()
                              if k != "_meta" and v["at_resolution_floor"]),
            "n_objectives": sum(1 for k in res if k != "_meta"),
            "erm_seed_range": (max(arm("erm", "sizeseed").values())
                               - min(arm("erm", "sizeseed").values()))
            if len(arm("erm", "sizeseed")) > 1 else None,
        }
    Path("runs/size_power_check.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if k == "_meta":
            continue
        flag = " AT FLOOR" if v["at_resolution_floor"] else ""
        print(f"{v['objective']:14s} n={v['n_per_arm']} "
              f"diff {v['difference']:+.4f} p={v['p_two_sided']:.4f}"
              f" (floor {v['min_attainable_p']:.4f}){flag}")
    print(f"\nwrote runs/size_power_check.json ({len(res) - 1} objectives)")


if __name__ == "__main__":
    main()
