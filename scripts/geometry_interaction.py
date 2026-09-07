"""Is the Scratch norm-by-pooling interaction driven by geometry exposure?

Entry 104 observed that the `Scratch` interaction -- `meanmax` shifting the
GroupNorm-BatchNorm gap -- is significant on `size` (p = 0.02344) and
`lot_time` (p = 0.02344) and not on `lot` (0.17969) or `iid` (0.23438). The two
where it fires are the two that hold geometry out entirely or test on a narrow
geometry slice. I recorded that as suggestive and explicitly not established,
because four protocols differ in many ways at once and a 2-of-4 pattern is weak
evidence for any one of them.

There is a much tighter test available and it needs no new runs. `run_bench.py`
scores each `lot_time` test set twice, on the wafers whose geometry appeared in
training and on those whose geometry did not -- 14.15% of that test set is
unseen -- and records per-class F1 for both halves. So the same models, the same
training data and the same eight seeds can be split by geometry exposure
directly.

H77, before computing: within `lot_time`, the `Scratch` interaction is larger on
the unseen-geometry half than on the seen half. If geometry exposure is what
makes the interaction appear, holding everything else fixed and varying only
that should reproduce the across-protocol pattern.

What would falsify it: the interaction being the same size on both halves, or
larger on the seen half. That would mean the 2-of-4 pattern across protocols is
driven by something else those two protocols share, or by nothing.

Two things this cannot escape and which are stated with the result: the unseen
half is small, so its estimates are noisier than the seen half's by construction;
and macro-F1 over one class set is not comparable to macro-F1 over another, so
cells whose two halves disagree on whether `Scratch` is present are dropped
rather than averaged -- the guard entry 80 needed.

    python scripts/geometry_interaction.py    # writes runs/geometry_interaction.json
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def arm(proto, enc, tag, half):
    """Scratch F1 on one geometry half, by seed. None where absent."""
    out = {}
    for f in glob.glob(f"runs/{proto}__{enc}__erm__{tag}__s*.json"):
        r = json.load(open(f))
        h = r.get(half) or {}
        if "Scratch" not in (h.get("classes_present") or []):
            continue
        v = (h.get("per_class_f1") or {}).get("Scratch")
        if v is not None:
            out[r["seed"]] = v
    return out


def main():
    fam = load("fam", "scripts/dg_family_test.py")
    res = {"what": "does the Scratch norm-by-pooling interaction depend on "
                   "whether the test wafer's geometry was seen in training",
           "how": "within lot_time, using the seen/unseen geometry "
                  "decomposition each cell already records, so the models, the "
                  "training data and the seeds are identical across the two "
                  "halves",
           "limits": ["the unseen half is 14.15% of the test set, so its "
                      "estimates are noisier by construction",
                      "cells whose halves disagree on whether Scratch is "
                      "present are dropped, not averaged"],
           "halves": {}}
    for half in ("test_seen_geometry", "test_unseen_geometry"):
        eff = {}
        for enc in ("cnn_gn", "cnn_bn"):
            a = arm("lot_time", enc, "poolmeanmax", half)
            b = arm("lot_time", enc, "poolmean", half)
            sh = sorted(set(a) & set(b))
            if len(sh) < 4:
                eff = {}
                break
            eff[enc] = {"seeds": sh, "per_seed": [a[s] - b[s] for s in sh]}
        if not eff:
            res["halves"][half] = {"n_usable_seeds": 0}
            continue
        shared = sorted(set(eff["cnn_gn"]["seeds"]) & set(eff["cnn_bn"]["seeds"]))
        gn = {s: v for s, v in zip(eff["cnn_gn"]["seeds"],
                                   eff["cnn_gn"]["per_seed"])}
        bn = {s: v for s, v in zip(eff["cnn_bn"]["seeds"],
                                   eff["cnn_bn"]["per_seed"])}
        d = [bn[s] - gn[s] for s in shared]
        p, n = fam.sign_flip_p(d)
        res["halves"][half] = {
            "n_usable_seeds": len(shared), "seeds": shared,
            "cnn_gn_effect": sum(gn[s] for s in shared) / len(shared),
            "cnn_bn_effect": sum(bn[s] for s in shared) / len(shared),
            "interaction_mean": sum(d) / len(d),
            "per_seed_interaction": d,
            "n_positive": sum(1 for v in d if v > 0),
            "p_two_sided": p, "arrangements": n,
            "min_attainable_p": 2.0 / n,
        }
    a, b = (res["halves"].get("test_seen_geometry") or {},
            res["halves"].get("test_unseen_geometry") or {})
    if a.get("n_usable_seeds") and b.get("n_usable_seeds"):
        res["unseen_minus_seen_interaction"] = (b["interaction_mean"]
                                                - a["interaction_mean"])
        res["larger_on_unseen"] = (abs(b["interaction_mean"])
                                   > abs(a["interaction_mean"]))
        # Establishing that the interaction is *specific* to one half needs the
        # halves compared to each other, not each to zero. Both halves come
        # from the same eight models, so this pairs by seed.
        sh = sorted(set(a["seeds"]) & set(b["seeds"]))
        if len(sh) >= 4:
            gi = {"seen": dict(zip(a["seeds"], a.get("per_seed_interaction",
                                                     []))),
                  "unseen": dict(zip(b["seeds"],
                                     b.get("per_seed_interaction", [])))}
            if gi["seen"] and gi["unseen"]:
                d = [gi["unseen"][s] - gi["seen"][s] for s in sh]
                pv, na = fam.sign_flip_p(d)
                res["halves_differ"] = {
                    "mean": sum(d) / len(d), "n_pairs": len(d),
                    "n_negative": sum(1 for v in d if v < 0),
                    "p_two_sided": pv, "arrangements": na,
                    "established": pv < 0.05,
                    "note": "if this is not established, the interaction is "
                            "not shown to be specific to either half -- only "
                            "established on one and not on the other, which "
                            "is a weaker statement",
                }
    Path("runs/geometry_interaction.json").write_text(json.dumps(res, indent=2))
    for half, v in res["halves"].items():
        if not v.get("n_usable_seeds"):
            print(f"{half}: no usable seeds")
            continue
        print(f"{half}: n={v['n_usable_seeds']}  gn {v['cnn_gn_effect']:+.4f}  "
              f"bn {v['cnn_bn_effect']:+.4f}  interaction "
              f"{v['interaction_mean']:+.4f} ({v['n_positive']}/"
              f"{v['n_usable_seeds']})  p={v['p_two_sided']:.5f} "
              f"(floor {v['min_attainable_p']:.5f})")
    if "larger_on_unseen" in res:
        print(f"\nunseen minus seen: "
              f"{res['unseen_minus_seen_interaction']:+.4f}; larger on "
              f"unseen: {res['larger_on_unseen']}")
    print("wrote runs/geometry_interaction.json")


if __name__ == "__main__":
    main()
