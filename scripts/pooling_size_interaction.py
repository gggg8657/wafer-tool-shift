"""Does the geometry-holdout behaviour depend on the encoder? It does.

H74 predicted that `meanmax`'s failure on `size` would replicate on `cnn_bn`,
because the explanation for it is that `resize_nearest` couples a defect's
apparent width to native geometry -- a property of the input pipeline, which
does not know which normalisation layer follows.

Neither encoder shows an established effect on `size`: `cnn_gn` is -0.0427 at
p = 0.20233 and `cnn_bn` is +0.0425 at p = 0.08096 on `Scratch` F1. But the
point estimates have opposite signs, and the *difference between them* is
testable directly. Both encoders were run on the same eight seeds, so a seed
fixes the split and the ordering for both, and the per-seed difference of
differences can be sign-flipped exactly.

If the mechanism were the whole story this should be null.

    python scripts/pooling_size_interaction.py   # writes runs/pooling_size_interaction.json
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


def arm(enc, tag, met):
    out = {}
    for f in glob.glob(f"runs/size__{enc}__erm__{tag}__s*.json"):
        r = json.load(open(f))
        v = (r["test"]["per_class_f1"].get(met.split(":", 1)[1])
             if met.startswith("class:") else r["test"][met])
        out[r["seed"]] = v
    return out


def main():
    fam = load("fam", "scripts/dg_family_test.py")
    res = {"what": "encoder x pooling interaction on the size protocol",
           "why": "the mechanism attributes the size behaviour to the resize, "
                  "which does not know which encoder follows it, so the two "
                  "encoders should behave alike. They do not.",
           "caveat": "post hoc: the comparison was suggested by the sign "
                     "disagreement rather than specified in advance, and it is "
                     "one test among many run this weekend. Paired by seed, "
                     "which fixes the split and the ordering but not the model.",
           "metrics": {}}
    for met in ("class:Scratch", "macro_f1"):
        per = {}
        for enc in ("cnn_gn", "cnn_bn"):
            a, b = arm(enc, "poolmeanmax", met), arm(enc, "poolmean", met)
            sh = sorted(set(a) & set(b))[:8]
            if len(sh) < 8 or any(a[s] is None or b[s] is None for s in sh):
                per = {}
                break
            per[enc] = {"seeds": sh, "per_seed": [a[s] - b[s] for s in sh]}
        if not per:
            continue
        d = [per["cnn_bn"]["per_seed"][i] - per["cnn_gn"]["per_seed"][i]
             for i in range(8)]
        p, n = fam.sign_flip_p(d)
        res["metrics"][met] = {
            "cnn_gn_effect": sum(per["cnn_gn"]["per_seed"]) / 8,
            "cnn_bn_effect": sum(per["cnn_bn"]["per_seed"]) / 8,
            "interaction_mean": sum(d) / len(d),
            "n_positive": sum(1 for v in d if v > 0), "n_pairs": len(d),
            "p_two_sided": p, "arrangements": n,
            "min_attainable_p": 2.0 / n,
        }
    Path("runs/pooling_size_interaction.json").write_text(json.dumps(res, indent=2))
    for met, v in res["metrics"].items():
        print(f"{met}: cnn_gn {v['cnn_gn_effect']:+.4f}, "
              f"cnn_bn {v['cnn_bn_effect']:+.4f}, interaction "
              f"{v['interaction_mean']:+.4f} "
              f"({v['n_positive']}/{v['n_pairs']}), p = {v['p_two_sided']:.5f}")
    print("wrote runs/pooling_size_interaction.json")


if __name__ == "__main__":
    main()
