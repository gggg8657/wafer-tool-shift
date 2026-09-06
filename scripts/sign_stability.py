"""Does three seeds get the *sign* of an effect right, even when it gets the size wrong?

The abstract asserts it does. That claim has been in the paper since before any
arm went to eight seeds, and it is checkable now: several arms have both a
three-seed measurement (seeds 0-2) and an eight-seed one on the same cells, so
the three-seed estimate is a strict subset of the eight-seed one.

The claim matters because it is the one piece of comfort this project offers
about small seed budgets. Everything else here says three seeds mislead: they
get sizes wrong, they get *existence* wrong in both directions, and a null
asserted from them is not a null. If they also got signs wrong there would be
nothing left to salvage, and if they do not, that is worth stating precisely
rather than as a feeling.

    python scripts/sign_stability.py     # writes runs/sign_stability.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

FAMILIES = [
    ("size", "runs/size__cnn_bn__{o}__sizeseed__s*.json",
     "runs/size__cnn_bn__erm__sizeseed__s*.json",
     ("coral", "dann", "irm", "group_dro", "mixup_domain", "logit_adjust")),
    ("lot (production deciles)", "runs/lot__cnn_bn__{o}__dtime__s*.json",
     "runs/lot__cnn_bn__erm__dtime__s*.json",
     ("coral", "dann", "irm", "group_dro", "mixup_domain", "hsic")),
]
SMALL, LARGE = 3, 8


def arm(pat):
    out = {}
    for f in glob.glob(pat):
        r = json.load(open(f))
        out[r["seed"]] = r["test"]["macro_f1"]
    return out


def main():
    rows = []
    for label, tmpl, base_pat, objs in FAMILIES:
        base = arm(base_pat)
        for o in objs:
            a = arm(tmpl.format(o=o))
            if not (all(s in a for s in range(LARGE))
                    and all(s in base for s in range(LARGE))):
                continue
            d_small = (sum(a[s] for s in range(SMALL)) / SMALL
                       - sum(base[s] for s in range(SMALL)) / SMALL)
            d_large = (sum(a[s] for s in range(LARGE)) / LARGE
                       - sum(base[s] for s in range(LARGE)) / LARGE)
            rows.append({
                "family": label, "objective": o,
                "difference_at_3_seeds": d_small,
                "difference_at_8_seeds": d_large,
                "sign_preserved": (d_small >= 0) == (d_large >= 0),
                "magnitude_ratio": (abs(d_small) / abs(d_large)
                                    if d_large else None),
            })
    n_flip = sum(1 for r in rows if not r["sign_preserved"])
    mags = [r["magnitude_ratio"] for r in rows if r["magnitude_ratio"]]
    res = {
        "what": "for every arm measured at both three and eight seeds, whether "
                "the three-seed estimate of the difference from ERM has the "
                "same sign as the eight-seed one",
        "small_n": SMALL, "large_n": LARGE,
        "n_arms": len(rows), "n_sign_flips": n_flip,
        "magnitude_ratio_min": min(mags) if mags else None,
        "magnitude_ratio_max": max(mags) if mags else None,
        "note": "the three-seed set is a subset of the eight-seed set, so this "
                "is not two independent measurements; it is what an "
                "experimenter who stopped early would have concluded.",
        "arms": rows,
    }
    Path("runs/sign_stability.json").write_text(json.dumps(res, indent=2))
    print(f"{'arm':34s} {'3 seeds':>9s} {'8 seeds':>9s}  sign")
    for r in rows:
        print(f"{r['family'] + '/' + r['objective']:34s} "
              f"{r['difference_at_3_seeds']:+9.4f} "
              f"{r['difference_at_8_seeds']:+9.4f}  "
              f"{'same' if r['sign_preserved'] else 'FLIPPED'}")
    print(f"\n{n_flip} of {len(rows)} changed sign; magnitude ratio spans "
          f"{res['magnitude_ratio_min']:.2f}x to {res['magnitude_ratio_max']:.2f}x")
    print("wrote runs/sign_stability.json")


if __name__ == "__main__":
    main()
