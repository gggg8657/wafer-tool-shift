"""One of this paper's two surviving results is conditional on the other.

The two things that survived the weekend are: replacing global average pooling
with mean-and-max helps, and GroupNorm beats BatchNorm. They were measured
independently and never against each other.

The GN-vs-BN result was measured with `pool=mean` -- that is what the `gnbn`
arms carry. Under `meanmax`, which is the pooling this paper recommends on the
strength of its other result, the normalisation gap shrinks on every seed and
stops being established.

That is not a contradiction; both measurements are correct. It is a missing
conditional. A reader told "use mean-and-max" and "use GroupNorm" would
reasonably expect the second to hold once they had done the first, and here it
does not.

Reported with the change in the gap, which is the stronger quantity: whether GN
still beats BN under `meanmax` is a question about one comparison's power, but
whether the gap *changed* is paired across seeds and exact.

    python scripts/norm_pooling_interaction.py   # writes runs/norm_pooling_interaction.json
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


def arm(proto, enc, tag, met):
    out = {}
    for f in glob.glob(f"runs/{proto}__{enc}__erm__{tag}__s*.json"):
        r = json.load(open(f))
        out[r["seed"]] = (r["test"]["per_class_f1"].get(met.split(":", 1)[1])
                          if met.startswith("class:") else r["test"][met])
    return out


def main():
    fam = load("fam", "scripts/dg_family_test.py")
    g = load("g", "scripts/gn_vs_bn.py")
    res = {"what": "does the GroupNorm advantage survive the pooling this "
                   "paper recommends",
           "why": "the GN-vs-BN result was measured at pool=mean; the pooling "
                  "result recommends meanmax. The two were never measured "
                  "against each other.",
           "caveat": "post hoc -- the question was raised by an encoder x "
                     "pooling interaction noticed on `size`. The change in the "
                     "gap is paired by seed and exact; whether GN still beats "
                     "BN under meanmax is a statement about one comparison's "
                     "power, and 'not established' is not 'absent'.",
           "protocols": {}}
    for proto in ("lot", "size"):
        entry = {}
        for met in ("macro_f1", "class:Scratch"):
            gaps = {}
            ok = True
            for tag in ("poolmean", "poolmeanmax"):
                gn, bn = arm(proto, "cnn_gn", tag, met), arm(proto, "cnn_bn",
                                                             tag, met)
                sh = sorted(set(gn) & set(bn))[:8]
                if len(sh) < 8 or any(gn[s] is None or bn[s] is None
                                      for s in sh):
                    ok = False
                    break
                gaps[tag] = {"seeds": sh, "per_seed": [gn[s] - bn[s]
                                                       for s in sh],
                             "gn": [gn[s] for s in sh],
                             "bn": [bn[s] for s in sh]}
            if not ok:
                continue
            d = [gaps["poolmeanmax"]["per_seed"][i]
                 - gaps["poolmean"]["per_seed"][i] for i in range(8)]
            pc, nc = fam.sign_flip_p(d)
            p_mean, _ = g.perm_p(gaps["poolmean"]["gn"], gaps["poolmean"]["bn"])
            p_mmax, _ = g.perm_p(gaps["poolmeanmax"]["gn"],
                                 gaps["poolmeanmax"]["bn"])
            entry[met] = {
                "gap_under_mean": sum(gaps["poolmean"]["per_seed"]) / 8,
                "gap_under_meanmax": sum(gaps["poolmeanmax"]["per_seed"]) / 8,
                "p_gn_beats_bn_under_mean": p_mean,
                "p_gn_beats_bn_under_meanmax": p_mmax,
                "change_in_gap": sum(d) / len(d),
                "n_seeds_gap_shrinks": sum(1 for v in d if v < 0),
                "n_pairs": len(d),
                "p_change": pc, "arrangements": nc,
                "established_under_mean": p_mean < 0.05,
                "established_under_meanmax": p_mmax < 0.05,
            }
        if entry:
            res["protocols"][proto] = entry
    Path("runs/norm_pooling_interaction.json").write_text(json.dumps(res,
                                                                     indent=2))
    for proto, e in res["protocols"].items():
        for met, v in e.items():
            print(f"{proto}/{met}: gap {v['gap_under_mean']:+.4f} "
                  f"(p={v['p_gn_beats_bn_under_mean']:.4f}) -> "
                  f"{v['gap_under_meanmax']:+.4f} "
                  f"(p={v['p_gn_beats_bn_under_meanmax']:.4f}); change "
                  f"{v['change_in_gap']:+.4f}, "
                  f"{v['n_seeds_gap_shrinks']}/{v['n_pairs']} shrink, "
                  f"p={v['p_change']:.5f}")
    print("\nwrote runs/norm_pooling_interaction.json")


if __name__ == "__main__":
    main()
