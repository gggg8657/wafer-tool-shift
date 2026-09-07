"""How much does each encoder lose on Scratch when geometry is held out?

The comparison H78 asks for is a *degradation*, not a level. `spectral` is worse
than the CNN on `Scratch` everywhere in absolute terms, which is a fact about
capacity and inductive bias rather than about resampling. What the H65 mechanism
predicts is that an encoder which never resizes pays a smaller penalty when the
test wafers' geometries were never trained on.

Both encoders ran the same eight seeds on both protocols, so the per-seed
`lot`-to-`size` drop can be paired and the difference of drops sign-flipped
exactly.

One thing this cannot control for and which is stated with the result:
`spectral` differs from the CNN in the operator, the capacity and the inductive
bias, not only in the resampling. A confirmation is therefore weak evidence for
the mechanism and a falsification is strong evidence against it -- the asymmetry
entries 98 and 105 both leaned on. There is exactly one resize-free encoder in
this repository, and one is not a controlled comparison.

    python scripts/resize_free_degradation.py   # writes runs/resize_free_degradation.json
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path

ARMS = [("spectral", "spec8", "Fourier operator, no resampling"),
        ("cnn_gn", "poolmeanmax", "CNN + mean-and-max (resized to 64x64)"),
        ("cnn_gn", "poolmean", "CNN + mean (resized to 64x64)")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def arm(proto, enc, tag, met="class:Scratch"):
    out = {}
    for f in glob.glob(f"runs/{proto}__{enc}__erm__{tag}__s*.json"):
        r = json.load(open(f))
        out[r["seed"]] = (r["test"]["per_class_f1"].get(met.split(":", 1)[1])
                          if met.startswith("class:") else r["test"][met])
    return out


def main():
    fam = load("fam", "scripts/dg_family_test.py")
    res = {"what": "per-seed Scratch F1 drop from lot to size, by encoder",
           "why": "H65 says the resize couples a defect's apparent width to "
                  "geometry, so a resize-free encoder should lose less when "
                  "geometry is held out",
           "limit": "spectral differs from the CNN in operator, capacity and "
                    "inductive bias, not only in resampling: a confirmation is "
                    "weak evidence and a falsification is strong",
           "arms": {}}
    drops = {}
    for enc, tag, label in ARMS:
        a, b = arm("lot", enc, tag), arm("size", enc, tag)
        sh = sorted(set(a) & set(b))
        if len(sh) < 4 or any(a[s] is None or b[s] is None for s in sh):
            res["arms"][label] = {"n_seeds": len(sh), "usable": False}
            continue
        d = [b[s] - a[s] for s in sh]          # negative = worse on size
        drops[label] = {s: b[s] - a[s] for s in sh}
        res["arms"][label] = {
            "encoder": enc, "tag": tag, "n_seeds": len(sh), "usable": True,
            "lot_mean": sum(a[s] for s in sh) / len(sh),
            "size_mean": sum(b[s] for s in sh) / len(sh),
            "drop_mean": sum(d) / len(d),
            "n_seeds_worse_on_size": sum(1 for v in d if v < 0),
        }
    ref = "CNN + mean-and-max (resized to 64x64)"
    tgt = "Fourier operator, no resampling"
    if ref in drops and tgt in drops:
        sh = sorted(set(drops[ref]) & set(drops[tgt]))
        if len(sh) >= 4:
            d = [drops[tgt][s] - drops[ref][s] for s in sh]
            p, n = fam.sign_flip_p(d)
            res["spectral_minus_cnn_meanmax"] = {
                "mean": sum(d) / len(d), "n_pairs": len(d),
                "n_spectral_loses_less": sum(1 for v in d if v > 0),
                "p_two_sided": p, "arrangements": n,
                "min_attainable_p": 2.0 / n,
                "spectral_loses_less": sum(d) / len(d) > 0,
            }
    Path("runs/resize_free_degradation.json").write_text(json.dumps(res,
                                                                    indent=2))
    for label, v in res["arms"].items():
        if not v.get("usable"):
            print(f"{label}: only {v['n_seeds']} usable seeds")
            continue
        print(f"{label}:\n    lot {v['lot_mean']:.4f} -> size "
              f"{v['size_mean']:.4f}  drop {v['drop_mean']:+.4f} "
              f"({v['n_seeds_worse_on_size']}/{v['n_seeds']} worse)")
    s = res.get("spectral_minus_cnn_meanmax")
    if s:
        print(f"\nspectral drop minus CNN-meanmax drop: {s['mean']:+.4f}, "
              f"{s['n_spectral_loses_less']}/{s['n_pairs']} seeds where "
              f"spectral loses less, p = {s['p_two_sided']:.5f}")
    print("wrote runs/resize_free_degradation.json")


if __name__ == "__main__":
    main()
