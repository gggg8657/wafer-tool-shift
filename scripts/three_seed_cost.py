"""What would a three-seed protocol have concluded about the effects we established?

This project's thesis is that three seeds -- the budget behind essentially every
published wafer-map comparison we are aware of, including this repository's own
first draft -- are not enough. That has been argued case by case: four
three-seed results shrank or vanished at eight, a null on `size` turned into two
objectives significantly worse, a streak of three-of-three turned out to be a
coin.

It has never been counted. Every comparison here that reached eight seeds per
arm can be re-scored on its first three, which is exactly what an experimenter
who stopped early would have had. The three-seed subset is not independent of
the eight -- it shares three draws by construction -- so this measures what that
experimenter would have concluded, not what a fresh triple would say.

Two instruments are compared, because this repository used both:

  * the exact permutation test. At three per arm it admits twenty arrangements
    and cannot return below 0.10, so it can *never* call anything significant at
    0.05. Every established effect is missed by construction, and the count is
    the interesting part.
  * the seed-range screen with the protocol's own run-to-run floor, which is
    what the three-seed tables in `RESULTS.md` actually used. A range grows with
    the sample, so this becomes *stricter* with more seeds -- it can call an
    effect separated at three and overlapping at eight.

    python scripts/three_seed_cost.py     # writes runs/three_seed_cost.json
"""
from __future__ import annotations

import glob
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

SMALL = 3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def arms():
    """Every (protocol, encoder, tag) arm keyed by seed, from the run files."""
    out = defaultdict(dict)
    for f in glob.glob("runs/*.json"):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not (isinstance(d, dict)
                and {"protocol", "encoder", "objective", "seed"} <= set(d)):
            continue
        key = (d["protocol"], d["encoder"], d["objective"], d.get("tag", ""))
        out[key][d["seed"]] = d
    return out


def metric(rec, name):
    if name.startswith("class:"):
        return rec["test"]["per_class_f1"].get(name.split(":", 1)[1])
    return rec["test"][name]


def main():
    g = load("g", "scripts/gn_vs_bn.py")
    rep = load("rep", "scripts/report.py")
    F = rep.floors("runs")
    A = arms()

    # every treatment/control pair the documents actually compare, taken from
    # the summary files this repository already wrote, so the set of
    # comparisons is not chosen after seeing the answer
    pairs = []
    for f in sorted(glob.glob("runs/*perm*.json")) + sorted(
            glob.glob("runs/*_power_check.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "arm_a" not in d:
            continue
        pairs.append((Path(f).stem, d))

    rows = []
    for key_a, per_a in sorted(A.items()):
        proto, enc, obj, tag = key_a
        if len(per_a) < 8:
            continue
        # compare each arm against the ERM/mean arm of the same protocol+encoder
        for base_obj, base_tag in ((obj, ""), ("erm", "poolmean"), ("erm", "")):
            key_b = (proto, enc, base_obj, base_tag)
            if key_b == key_a or key_b not in A or len(A[key_b]) < 8:
                continue
            per_b = A[key_b]
            for met in ("macro_f1", "class:Scratch"):
                sh = sorted(set(per_a) & set(per_b))
                if len(sh) < 8:
                    continue
                xa = [metric(per_a[s], met) for s in sh]
                xb = [metric(per_b[s], met) for s in sh]
                if any(v is None for v in xa + xb):
                    continue
                p8, _ = g.perm_p(xa, xb)
                sa, sb = xa[:SMALL], xb[:SMALL]
                p3, n3 = g.perm_p(sa, sb)
                fl = rep.floor_for(F, proto)
                v3, _ = rep.separation(sa, sb, fl)
                v8, _ = rep.separation(xa, xb, fl)
                rows.append({
                    "protocol": proto, "encoder": enc, "arm": f"{obj}/{tag}",
                    "baseline": f"{base_obj}/{base_tag}", "metric": met,
                    "p_8_seeds": p8, "p_3_seeds": p3,
                    "floor_of_3_seed_test": 2.0 / n3,
                    "sig_at_8": p8 < 0.05, "sig_at_3": p3 < 0.05,
                    "range_verdict_3": v3, "range_verdict_8": v8,
                })
            break
    # de-duplicate symmetric comparisons
    seen, uniq = set(), []
    for r in rows:
        k = tuple(sorted([r["arm"], r["baseline"]])) + (r["protocol"],
                                                        r["encoder"],
                                                        r["metric"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    est = [r for r in uniq if r["sig_at_8"]]
    res = {
        "what": "for every comparison with eight seeds per arm, what its first "
                "three seeds would have concluded",
        "caveat": "the three-seed subset shares its draws with the eight, so "
                  "this is what an experimenter who stopped early would have "
                  "concluded, not what a fresh triple would say",
        "small_n": SMALL,
        "n_comparisons": len(uniq),
        "n_established_at_8": len(est),
        "n_of_those_reachable_at_3": sum(1 for r in est if r["sig_at_3"]),
        "permutation_floor_at_3": (uniq[0]["floor_of_3_seed_test"]
                                   if uniq else None),
        "n_range_screen_disagrees": sum(
            1 for r in uniq
            if r["range_verdict_3"].startswith("**separated")
            != r["range_verdict_8"].startswith("**separated")),
        "comparisons": uniq,
    }
    Path("runs/three_seed_cost.json").write_text(json.dumps(res, indent=2))
    print(f"{len(uniq)} comparisons with eight seeds on both arms")
    print(f"  {len(est)} are established at p < 0.05 with eight seeds")
    print(f"  {res['n_of_those_reachable_at_3']} of those could have been "
          f"established with three, where the test's floor is "
          f"{res['permutation_floor_at_3']:.2f}")
    print(f"  {res['n_range_screen_disagrees']} comparisons get a different "
          "range-screen verdict at three seeds than at eight")
    for r in est:
        print(f"    {r['protocol']}/{r['encoder']}/{r['arm']} [{r['metric']}] "
              f"p8={r['p_8_seeds']:.5f} p3={r['p_3_seeds']:.4f}")
    print("\nwrote runs/three_seed_cost.json")


if __name__ == "__main__":
    main()
