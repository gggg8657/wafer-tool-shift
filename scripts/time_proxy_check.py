"""Is the lot number actually production order? The assumption under `lot_time`.

    python scripts/time_proxy_check.py

`lot_time` is this repo's largest single result -- macro-F1 falls ~0.17 from the
lot-disjoint split to the purged forward-only one -- and it rests entirely on
`wts.data.lot_numbers`, which reads the integer out of each lot name and treats
it as a clock. WM-811K carries no timestamps, so the assumption cannot be
checked directly. It can be *refuted*, which is the next best thing.

If lot numbers were arbitrary identifiers, two lots would be no more alike for
being adjacent in the numbering than for being far apart. If they are issued in
production order, a fab's product mix and process health drift, so nearby lots
should resemble each other more than distant ones -- a monotone increase of
distributional distance with numbering gap.

The statistic is that relationship: bucket lots into deciles of lot number, take
the total-variation distance between each pair of deciles over a wafer-level
variable, and correlate it against the decile gap. The null is the same
computation after the lot->number assignment has been shuffled, which destroys
any ordering while preserving every marginal and every lot's contents.

**What this can and cannot show.** A positive result refutes "the numbers are
arbitrary". It does *not* prove the ordering is time: any administratively
contiguous grouping -- a product line numbered in a block -- would produce the
same signature. So a pass here downgrades the threat from "unfounded" to
"consistent with, and not distinguishable from, product blocking". That
distinction is stated in the output rather than glossed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import CLASSES, Corpus, lot_numbers          # noqa: E402


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def decile_profiles(vals, order_key, n_dec, n_bins):
    """Row-normalized histogram of `vals` within each decile of `order_key`."""
    edges = np.quantile(order_key, np.linspace(0, 1, n_dec + 1)[1:-1])
    dec = np.digitize(order_key, edges)
    P = np.zeros((n_dec, n_bins))
    for d in range(n_dec):
        m = dec == d
        if m.sum() == 0:
            continue
        h = np.bincount(vals[m], minlength=n_bins).astype(float)
        P[d] = h / max(h.sum(), 1)
    return P


def gap_vs_distance(P):
    """(decile gap, TV) for every pair, plus the Spearman correlation."""
    gaps, dists = [], []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            gaps.append(j - i)
            dists.append(tv(P[i], P[j]))
    gaps = np.asarray(gaps, float)
    dists = np.asarray(dists, float)
    return gaps, dists, spearman(gaps, dists)


def spearman(a, b):
    """Rank correlation without scipy (scipy is not in this environment)."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def rankdata(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties, so a variable with many equal gaps is not given a spurious
    # ordering by argsort's tie-breaking
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.pt")
    ap.add_argument("--out", default="runs/time_proxy.json")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c = Corpus.load(a.corpus)
    t = lot_numbers(c).numpy().astype(np.int64)
    lot = c.lot.numpy()
    lab = c.labels.numpy()
    _, geom = np.unique(c.size_id.numpy(), return_inverse=True)
    m = c.maps64
    rate = ((m == 2).float().sum((1, 2))
            / (m > 0).float().sum((1, 2)).clamp_min(1)).numpy()
    rate_bin = np.digitize(rate, np.quantile(rate, np.linspace(0, 1, 21)[1:-1]))

    variables = {
        "geometry": (geom, int(geom.max()) + 1),
        "defect_class": (lab, len(CLASSES)),
        "failed_die_rate_ventile": (rate_bin, 20),
    }

    rng = np.random.default_rng(a.seed)
    uniq_lots = np.unique(lot)
    lot_t = np.array([t[lot == l][0] for l in uniq_lots])   # one number per lot

    out = {"n_wafers": int(len(lot)), "n_lots": int(len(uniq_lots)),
           "deciles": a.deciles, "null_draws": a.null_draws,
           "lot_number_min": int(t.min()), "lot_number_max": int(t.max()),
           "variables": {}}

    for name, (vals, nb) in variables.items():
        P = decile_profiles(vals, t, a.deciles, nb)
        gaps, dists, rho = gap_vs_distance(P)

        # Null: permute which lot gets which lot number. Wafer contents, lot
        # membership and every marginal are untouched; only the ordering dies.
        null = []
        for _ in range(a.null_draws):
            perm = rng.permutation(len(uniq_lots))
            shuffled = np.zeros(len(uniq_lots), dtype=np.int64)
            shuffled[perm] = lot_t
            lut = dict(zip(uniq_lots.tolist(), shuffled.tolist()))
            t_null = np.array([lut[l] for l in lot], dtype=np.int64)
            Pn = decile_profiles(vals, t_null, a.deciles, nb)
            null.append(gap_vs_distance(Pn)[2])
        null = np.asarray(null)
        p = float((null >= rho).mean())
        out["variables"][name] = {
            "spearman_gap_vs_tv": rho,
            "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)),
            "p_value_one_sided": p,
            "mean_tv_adjacent_deciles": float(dists[gaps == 1].mean()),
            "mean_tv_farthest_deciles": float(dists[gaps == gaps.max()].mean()),
        }

    # How much of each protocol's test side is geometry the model never saw?
    # Model-free: a property of the split, computed from the split alone. This
    # is what tells you whether `lot_time` is a temporal protocol or a geometry
    # protocol wearing a clock.
    from wts.data import split                                    # noqa: E402
    out["split_geometry"] = {}
    for proto in ("iid", "lot", "size", "lot_time"):
        tr, te = split(c, proto, seed=a.seed)
        g_tr = set(np.unique(c.size_id.numpy()[tr.numpy()]).tolist())
        g_te_arr = c.size_id.numpy()[te.numpy()]
        g_te = set(np.unique(g_te_arr).tolist())
        unseen = g_te - g_tr
        n_unseen = int(np.isin(g_te_arr, list(unseen)).sum()) if unseen else 0
        out["split_geometry"][proto] = {
            "n_geometries_train": len(g_tr),
            "n_geometries_test": len(g_te),
            "n_geometries_test_unseen": len(unseen),
            "n_test_wafers": int(len(te)),
            "n_test_wafers_unseen_geometry": n_unseen,
            "frac_test_wafers_unseen_geometry": round(n_unseen / len(te), 6),
        }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    print("\nReading: a Spearman correlation well above the null's 95th "
          "percentile means adjacent lot numbers are more alike than distant "
          "ones, which refutes 'lot numbers are arbitrary identifiers'. It does "
          "NOT establish that the ordering is time rather than product "
          "blocking; both produce this signature.")


if __name__ == "__main__":
    main()
