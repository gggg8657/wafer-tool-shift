"""Every corpus-level figure the generated documents quote, computed once.

    python scripts/corpus_stats.py        # -> runs/corpus_stats.json

`report.py`, `paper.py` and `weekend.py` had ~50 measurements typed into their
prose: corpus sizes, the RPCA rank statistics, the label total-variation of each
domain vocabulary, the p10 quantization counts. Every one of them came from a
run in this repository, so none was invented -- but a number typed into prose
cannot go stale loudly. Change the corpus, the RPCA cache or a domain
definition, and the documents keep asserting the old figure while the tables
beside them move. That is the exact failure the "no number unless a run produced
it" rule exists to prevent, and half-obeying it is how it gets lost.

So the figures live here, in a JSON the generators interpolate, and
`tests/test_smoke.py` lints the generator sources for numeric literals that look
like measurements.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import CLASSES, Corpus, lot_numbers, split       # noqa: E402


def label_tv(labels, groups, min_n=200):
    """Mean pairwise total-variation between groups' class distributions."""
    d = np.asarray(groups)
    P = []
    for g in np.unique(d):
        m = d == g
        if m.sum() < min_n:
            continue
        h = np.bincount(labels[m], minlength=len(CLASSES)).astype(float)
        P.append(h / h.sum())
    P = np.asarray(P)
    if len(P) < 2:
        return None, len(P)
    v = [0.5 * np.abs(P[i] - P[j]).sum()
         for i in range(len(P)) for j in range(i + 1, len(P))]
    return float(np.mean(v)), len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.pt")
    ap.add_argument("--rpca", default="data/rpca.pt")
    ap.add_argument("--ssl", default="runs/ssl_pretrain.pt")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/corpus_stats.json")
    a = ap.parse_args()

    c = Corpus.load(a.corpus)
    y = c.labels.numpy()
    out = {
        "n_wafers": int(len(y)),
        "n_lots": int(torch.unique(c.lot).numel()),
        "n_geometries": int(torch.unique(c.size_id).numel()),
        "class_counts": {CLASSES[k]: int((y == k).sum())
                         for k in range(len(CLASSES))},
        "max_wafers_in_one_lot": int(np.bincount(c.lot.numpy()).max()),
    }

    # --- is 64x64 losing anything? The CNN path resamples every wafer to a
    # fixed 64x64 grid, and "a scratch is a few dies wide and 64x64 blurs it"
    # is an appealing explanation for the long tail. It is checkable in one
    # line and it is wrong on this corpus: almost every wafer is *up*sampled to
    # reach 64x64, so the resize adds pixels rather than removing detail. A
    # scratch is thin in units of dies, and no resampling changes that.
    hw = c.hw.numpy()
    mx = np.maximum(hw[:, 0], hw[:, 1])
    out["native_resolution"] = {
        "percentiles_max_dim": {str(q): int(np.percentile(mx, q))
                                for q in (1, 25, 50, 75, 95, 99)},
        "frac_downsampled_by_64": float((mx > 64).mean()),
        "frac_upsampled_by_64": float((mx < 64).mean()),
        "per_class": {
            CLASSES[k]: {
                "n": int((y == k).sum()),
                "frac_downsampled_by_64": float((mx[y == k] > 64).mean()),
                "median_max_dim": int(np.median(mx[y == k])),
            } for k in range(len(CLASSES)) if (y == k).sum()},
    }

    # --- the domain vocabularies, and how much shift each actually carries
    lot = c.lot.numpy()
    sid = c.size_id.numpy()
    t = lot_numbers(c).numpy().astype(float)
    m = c.maps64
    rate = ((m == 2).float().sum((1, 2))
            / (m > 0).float().sum((1, 2)).clamp_min(1)).numpy()
    uq, inv = np.unique(lot, return_inverse=True)
    lot_rate = np.zeros(len(uq))
    np.add.at(lot_rate, inv, rate)
    lot_rate /= np.maximum(np.bincount(inv), 1)
    dec = lambda v: np.digitize(v, np.quantile(v, np.linspace(0, 1, 11)[1:-1]))
    vocab = {
        "hash32_lot": lot % 32,
        "hash32_size": sid % 32,
        "time_decile": dec(t),
        "fail_decile": dec(lot_rate[inv]),
        "geometry": sid,
        "real_lots": lot,
    }
    out["domain_label_tv"] = {}
    for name, g in vocab.items():
        mn = 20 if name == "real_lots" else 200
        tv, n = label_tv(y, g, min_n=mn)
        out["domain_label_tv"][name] = {
            "mean_pairwise_label_tv": tv, "n_groups_scored": n,
            "min_wafers_per_group": mn,
            "n_groups_total": int(len(np.unique(g)))}

    # --- what the RPCA decomposition actually returns
    rp = Path(a.rpca)
    if rp.exists():
        b = torch.load(rp, map_location="cpu", weights_only=False)
        sig, res = b["signature"], b["residual"].float()
        fail = (c.maps64 == 2).float()
        dec_mask = sig[:, 5] > 0
        rank = sig[:, 4]
        per = (res - fail).abs().flatten(1).sum(1)
        out["rpca"] = {
            "n_wafers": int(len(sig)),
            "n_decomposed": int(dec_mask.sum()),
            "frac_rank0_of_decomposed": float((rank[dec_mask] == 0).float().mean()),
            "frac_residual_identical_to_failmask": float((per < 1e-3).float().mean()),
            "rank_histogram": {str(k): int(v) for k, v in
                               sorted(Counter(rank[dec_mask].int().tolist()).items())},
        }

    # --- how quantized the per-lot statistic is, counted rather than asserted
    vals = Counter()
    for p in Path(a.runs).glob("*.json"):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or "test" not in r:
            continue
        if r.get("protocol") != "lot":
            continue
        v = r["test"].get("p10_domain_macro_f1")
        if v is not None:
            vals[round(float(v), 6)] += 1
    if vals:
        top = vals.most_common(3)
        out["p10_quantization"] = {
            "n_lot_cells": int(sum(vals.values())),
            "n_distinct_values": int(len(vals)),
            "most_common": [{"value": v, "n_cells": n} for v, n in top],
        }
    # how many test domains the per-lot statistic is even computed over
    _, te = split(c, "lot", seed=0)
    sizes = np.bincount(lot[te.numpy()])
    out["p10_quantization"] = out.get("p10_quantization", {})
    out["p10_quantization"]["n_test_lots_scored_min12"] = int((sizes >= 12).sum())

    # --- the SSL checkpoint's weight scale against a fresh encoder
    sp = Path(a.ssl)
    if sp.exists():
        from wts.models import CnnResized                       # noqa: E402
        sd = torch.load(sp, map_location="cpu", weights_only=False)["model"]
        fresh = CnnResized(len(CLASSES), width=32, norm="gn").state_dict()
        k = "body.17.weight"
        out["ssl_checkpoint"] = {
            "deepest_conv": k,
            "ckpt_mean_abs_w": float(sd[k].abs().mean()),
            "fresh_mean_abs_w": float(fresh[k].abs().mean()),
            "ratio": float(sd[k].abs().mean() / fresh[k].abs().mean()),
            "head_ckpt_mean_abs_w": float(sd["head.weight"].abs().mean()),
            "head_fresh_mean_abs_w": float(fresh["head.weight"].abs().mean()),
        }
    sj = Path(a.runs) / "ssl_pretrain.json"
    if sj.exists():
        out["ssl_checkpoint"] = out.get("ssl_checkpoint", {})
        out["ssl_checkpoint"]["n_unlabeled"] = json.loads(sj.read_text())["n_unlabeled"]

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
