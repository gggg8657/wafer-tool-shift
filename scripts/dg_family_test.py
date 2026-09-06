"""Two questions the six per-objective tests do not answer.

At eight seeds under a domain vocabulary that carries real shift, all six
borrowed objectives sit below ERM and the p-values order monotonically with
effect size: group_dro -0.0201 (0.00171), mixup_domain -0.0180 (0.00513),
dann -0.0098 (0.0777), coral -0.0092 (0.0872), irm -0.0045 (0.338),
hsic -0.0029 (0.581).

**1. Six tests at 0.05.** Reporting two significant results out of six without
saying six were run is the oldest way to manufacture a finding. Holm-Bonferroni
is applied here, and the two survive it -- but that has to be shown, not
asserted.

**2. All six point the same way.** No single test asks whether the *family*
underperforms ERM, and the obvious way to ask is wrong: the six comparisons
share one ERM baseline and one corpus, so they are not independent and a sign
test across them would badly overstate its evidence.

The test that is valid pairs by seed. Every arm was run at seeds 0-7 against an
ERM arm at the same seeds, and a seed fixes the split, the initialisation and
the batch order. So for each seed take the mean of the six objectives minus ERM
at that same seed, giving eight paired differences, and test those by exact
sign-flipping: under the null that the family matches ERM the differences are
symmetric about zero, so all 2^8 = 256 sign assignments are equally likely.
That is exact, respects the pairing, and never treats the shared baseline as
six independent observations. Its smallest attainable two-sided p is 2/256 =
0.0078.

    python scripts/dg_family_test.py     # writes runs/dg_family_test.json
"""
from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

OBJS = ("coral", "dann", "irm", "group_dro", "mixup_domain", "hsic")


def arm(obj, tag="dtime", proto="lot"):
    out = {}
    for f in glob.glob(f"runs/{proto}__cnn_bn__{obj}__{tag}__s*.json"):
        r = json.load(open(f))
        out[r["seed"]] = r["test"]["macro_f1"]
    return out


# `size` holds its own DG arms at eight seeds under a domain vocabulary that
# was never degenerate (geometry hash, TV 0.2592). Running the same paired test
# there asks whether the family result is a fact about `lot` and production
# deciles, or about these objectives on this corpus. hsic was not run on `size`.
SIZE_OBJS = ("coral", "dann", "irm", "group_dro", "mixup_domain")


def family_on(objs, tag, proto):
    base = arm("erm", tag, proto)
    per = {o: arm(o, tag, proto) for o in objs}
    shared = sorted(set(base).intersection(*(set(per[o]) for o in objs)))
    if len(shared) < 2:
        return None
    d = [sum(per[o][s] for o in objs) / len(objs) - base[s] for s in shared]
    pv, na = sign_flip_p(d)
    return {"protocol": proto, "tag": tag, "objectives": list(objs),
            "seeds": shared, "n_pairs": len(d),
            "per_seed_difference": dict(zip(map(str, shared), d)),
            "mean_difference": sum(d) / len(d),
            "n_negative": sum(1 for v in d if v < 0),
            "p_two_sided": pv, "arrangements": na,
            "min_attainable_p": 2.0 / na}


def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        v = (len(pvals) - rank) * pvals[i]
        running = max(running, v)          # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def sign_flip_p(diffs):
    """Exact two-sided sign-flip test on paired differences."""
    n = len(diffs)
    obs = abs(sum(diffs))
    hits = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-15:
            hits += 1
    return hits / 2 ** n, 2 ** n


def main():
    base = arm("erm")
    per = {o: arm(o) for o in OBJS}

    # ---- 1. multiplicity over the six per-objective tests
    prev = json.loads(Path("runs/dg_power_check.json").read_text())
    names, ps = [], []
    for o in OBJS:
        k = f"{o}__macro_f1"
        if k in prev:
            names.append(o)
            ps.append(prev[k]["p_two_sided"])
    adj = holm(ps)
    multiplicity = {
        o: {"p_raw": p, "p_holm": a, "survives_holm_05": a < 0.05,
            "significant_raw_05": p < 0.05}
        for o, p, a in zip(names, ps, adj)
    }

    # ---- 2. family-level paired test on the seeds every arm shares
    shared = sorted(set(base).intersection(*(set(per[o]) for o in OBJS)))
    diffs = [sum(per[o][s] for o in OBJS) / len(OBJS) - base[s] for s in shared]
    p_fam, n_arr = sign_flip_p(diffs)

    # ---- is the family result just the two significant objectives?
    # A family test that only fires because of its largest members says
    # nothing a per-objective test did not already say. The sharp version of
    # the question is whether the objectives that individually *fail* to
    # separate are collectively worse than ERM.
    def family(objs):
        d = [sum(per[o][s] for o in objs) / len(objs) - base[s] for s in shared]
        pv, na = sign_flip_p(d)
        return {"objectives": list(objs), "n_pairs": len(d),
                "mean_difference": sum(d) / len(d),
                "n_negative": sum(1 for v in d if v < 0),
                "p_two_sided": pv, "arrangements": na,
                "min_attainable_p": 2.0 / na}

    insig = [o for o in OBJS if not multiplicity.get(o, {})
             .get("significant_raw_05")]
    robustness = {
        "individually_unestablished_only": family(insig) if insig else None,
        "leave_one_out": {o: family([x for x in OBJS if x != o]) for o in OBJS},
    }

    res = {
        "what": "multiplicity correction and a family-level paired test for "
                "the six borrowed DG objectives under domain-def time_decile",
        "protocol": "lot", "encoder": "cnn_bn", "tag": "dtime",
        "objectives": list(OBJS),
        "multiplicity": {
            "method": "Holm-Bonferroni over the six per-objective permutation "
                      "tests", "n_tests": len(ps),
            "per_objective": multiplicity,
            "n_significant_raw": sum(v["significant_raw_05"]
                                     for v in multiplicity.values()),
            "n_survives_holm": sum(v["survives_holm_05"]
                                   for v in multiplicity.values()),
        },
        "family_test": {
            "method": "exact two-sided sign-flip test on per-seed paired "
                      "differences (mean of six objectives minus ERM at the "
                      "same seed)",
            "why_paired": "the six comparisons share one ERM baseline and one "
                          "corpus, so they are not independent; pairing by "
                          "seed holds the split, the initialisation and the "
                          "batch order fixed and never counts the shared "
                          "baseline six times",
            "seeds": shared, "n_pairs": len(shared),
            "per_seed_difference": dict(zip(map(str, shared), diffs)),
            "mean_difference": sum(diffs) / len(diffs),
            "n_negative": sum(1 for d in diffs if d < 0),
            "p_two_sided": p_fam,
            "arrangements": n_arr,
            "min_attainable_p": 2.0 / n_arr,
        },
        "robustness": robustness,
    }

    # ---- the same test on `size`, which has a different domain vocabulary
    sz_all = family_on(SIZE_OBJS, "sizeseed", "size")
    if sz_all:
        _szsig = {"group_dro"}          # the only DG objective significant on
        _szrest = [o for o in SIZE_OBJS if o not in _szsig]
        res["size_replication"] = {
            "why": "the `lot` result uses production deciles as the domain "
                   "vocabulary. `size` holds geometry out and its hash was "
                   "never degenerate, so repeating the test there asks whether "
                   "this is a fact about one protocol or about these "
                   "objectives on this corpus.",
            "all": sz_all,
            "individually_unestablished_only": family_on(_szrest, "sizeseed",
                                                         "size"),
        }
    Path("runs/dg_family_test.json").write_text(json.dumps(res, indent=2))

    print("Holm-Bonferroni over six tests:")
    for o in names:
        v = multiplicity[o]
        print(f"  {o:14s} p={v['p_raw']:.5f}  holm={v['p_holm']:.5f}"
              f"  {'survives' if v['survives_holm_05'] else ''}")
    print(f"\nFamily paired sign-flip test on {len(shared)} seeds:")
    print(f"  mean difference {res['family_test']['mean_difference']:+.4f}, "
          f"{res['family_test']['n_negative']}/{len(diffs)} seeds negative")
    print(f"  p = {p_fam:.5f} over {n_arr} sign assignments "
          f"(floor {2.0 / n_arr:.5f})")
    io = robustness["individually_unestablished_only"]
    if io:
        print(f"\nDropping the objectives that reach p < 0.05 on their own "
              f"({', '.join(o for o in OBJS if o not in insig)}):")
        print(f"  {', '.join(io['objectives'])}")
        print(f"  mean {io['mean_difference']:+.4f}, "
              f"{io['n_negative']}/{io['n_pairs']} seeds negative, "
              f"p = {io['p_two_sided']:.5f}")
    lo = robustness["leave_one_out"]
    worst = max(lo.values(), key=lambda v: v["p_two_sided"])
    print(f"\nLeave-one-out: worst p over the six subsets is "
          f"{worst['p_two_sided']:.5f} "
          f"(dropping `{[k for k, v in lo.items() if v is worst][0]}`)")
    sz = res.get("size_replication")
    if sz:
        print("\nSame test on `size` (geometry holdout, its own vocabulary):")
        for lab, k in (("all five", "all"),
                       ("dropping group_dro", "individually_unestablished_only")):
            v = sz[k]
            if v:
                print(f"  {lab:22s} mean {v['mean_difference']:+.4f}, "
                      f"{v['n_negative']}/{v['n_pairs']} negative, "
                      f"p = {v['p_two_sided']:.5f}")
    print("\nwrote runs/dg_family_test.json")


if __name__ == "__main__":
    main()
