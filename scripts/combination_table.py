"""Of the four normalisation-by-pooling combinations, which should anyone use?

This paper reports two positive results -- `meanmax` over `mean`, and GroupNorm
over BatchNorm -- and entry 99 showed they do not compose: the normalisation
advantage is established under `mean` and not under `meanmax`. A reader's actual
question is simpler than either finding: given four combinations, which one?

All four are at eight seeds on `lot` and `size`, so the question is answerable
without a new run. Each protocol's best combination is compared against the
other three with an exact permutation test.

The answer is protocol-dependent in a way that matters for deployment, and it is
not the answer either headline result implies on its own.

    python scripts/combination_table.py    # writes runs/combination_table.json
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path

COMBOS = [("cnn_gn", "poolmean"), ("cnn_gn", "poolmeanmax"),
          ("cnn_bn", "poolmean"), ("cnn_bn", "poolmeanmax")]
LABEL = {("cnn_gn", "poolmean"): "GroupNorm + mean",
         ("cnn_gn", "poolmeanmax"): "GroupNorm + mean-and-max",
         ("cnn_bn", "poolmean"): "BatchNorm + mean",
         ("cnn_bn", "poolmeanmax"): "BatchNorm + mean-and-max"}


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
    g = load("g", "scripts/gn_vs_bn.py")
    res = {"what": "which normalisation-by-pooling combination is best, per "
                   "protocol, with every rival tested against it",
           "why": "the two headline results do not compose, so neither answers "
                  "the question a reader actually has",
           "protocols": {}}
    for proto in ("lot", "size", "iid", "lot_time"):
        for met in ("macro_f1", "class:Scratch"):
            vals = {}
            for enc, tag in COMBOS:
                a = arm(proto, enc, tag, met)
                sh = sorted(a)[:8]
                if len(sh) == 8 and not any(a[s] is None for s in sh):
                    vals[(enc, tag)] = [a[s] for s in sh]
            if len(vals) < len(COMBOS):
                continue
            best = max(vals, key=lambda k: sum(vals[k]))
            rows = []
            for k, v in sorted(vals.items(), key=lambda kv: -sum(kv[1])):
                if k == best:
                    rows.append({"combination": LABEL[k], "mean": sum(v) / 8,
                                 "seed_range": max(v) - min(v),
                                 "is_best": True, "difference": 0.0,
                                 "p_vs_best": None})
                    continue
                p, _ = g.perm_p(vals[best], v)
                rows.append({"combination": LABEL[k], "mean": sum(v) / 8,
                             "seed_range": max(v) - min(v), "is_best": False,
                             "difference": sum(vals[best]) / 8 - sum(v) / 8,
                             "p_vs_best": p})
            res["protocols"].setdefault(proto, {})[met] = {
                "best": LABEL[best],
                "n_rivals_beaten_at_05": sum(
                    1 for r in rows
                    if r["p_vs_best"] is not None and r["p_vs_best"] < 0.05),
                "n_rivals": len(rows) - 1,
                "rows": rows,
            }
    Path("runs/combination_table.json").write_text(json.dumps(res, indent=2))
    for proto, e in res["protocols"].items():
        for met, v in e.items():
            print(f"{proto}/{met}: best = {v['best']}, beats "
                  f"{v['n_rivals_beaten_at_05']} of {v['n_rivals']} rivals at "
                  "p < 0.05")
            for r in v["rows"]:
                pv = ("     -" if r["p_vs_best"] is None
                      else f"{r['p_vs_best']:.5f}")
                print(f"    {r['combination']:28s} {r['mean']:.4f}  "
                      f"{r['difference']:+.4f}  p={pv}")
    print("\nwrote runs/combination_table.json")


if __name__ == "__main__":
    main()
