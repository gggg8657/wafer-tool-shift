"""Which lots should you pay to measure next?

    CUDA_VISIBLE_DEVICES=3 python scripts/active_learning.py

The scarce resource in a fab is not wafer maps, it is *labels*: metrology is
sampled, so somebody chooses which lots get inspected. That makes the labelling
budget a design variable, and the question "which lot next" a real one -- it is
the operational form of the open problem the industry states as sparse
metrology labels.

Budgets are spent in whole lots, never in wafers, because a fab inspects a lot,
and because sampling wafers from every lot would leak the domain structure the
benchmark exists to test.

| strategy   | borrowed from        | what it maximizes                               |
|------------|----------------------|-------------------------------------------------|
| `random`   | -                    | nothing; the baseline that is hard to beat      |
| `entropy`  | active learning      | the model's predictive entropy on the lot       |
| `coreset`  | k-center greedy      | coverage of the embedding space (k-center)      |
| `diverse`  | design of experiments| descriptor-space spread, needs *no model* -- so it works from a cold start, before any label exists |
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts import metrics                                    # noqa: E402
from wts.data import CLASSES, Corpus, split                # noqa: E402
from wts.models import FeatMlp                             # noqa: E402


def train_eval(X, y, tr_idx, te_idx, dev, epochs=40, hidden=256, seed=0):
    torch.manual_seed(seed)
    model = FeatMlp(X.shape[1], len(CLASSES), hidden=hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xt, yt = X[tr_idx].to(dev), y[tr_idx].to(dev)
    counts = torch.bincount(yt, minlength=len(CLASSES)).float().clamp_min(1)
    w = (1.0 / counts); w = (w / w.mean()).to(dev)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 256):
            sel = perm[i:i + 256]
            loss = F.cross_entropy(model(xt[sel]), yt[sel], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = model(X[te_idx].to(dev)).softmax(1).cpu().numpy()
    return model, p


def lot_scores(model, X, pool_idx, lot, dev, strategy, chosen_emb=None):
    """One score per candidate lot; higher means "label this next"."""
    with torch.no_grad():
        if strategy == "entropy":
            p = model(X[pool_idx].to(dev)).softmax(1)
            s = -(p * (p + 1e-12).log()).sum(1).cpu().numpy()
        elif strategy in ("coreset", "diverse"):
            e = (model.embed(X[pool_idx].to(dev)).cpu().numpy()
                 if strategy == "coreset" else X[pool_idx].numpy())
            if chosen_emb is None or len(chosen_emb) == 0:
                s = np.linalg.norm(e - e.mean(0, keepdims=True), axis=1)
            else:
                d = ((e[:, None, :] - chosen_emb[None, :, :]) ** 2).sum(-1)
                s = np.sqrt(d.min(1))          # k-center: distance to the set
        else:
            s = np.random.default_rng(0).random(len(pool_idx))
    lots = lot[pool_idx].numpy()
    out = {}
    for u in np.unique(lots):
        out[u] = float(s[lots == u].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.pt")
    ap.add_argument("--features", default="data/features.pt")
    ap.add_argument("--out", default="runs/active_learning.json")
    ap.add_argument("--budgets", default="20,50,100,200,400,800")
    ap.add_argument("--seed-lots", type=int, default=20)
    ap.add_argument("--strategies", default="random,entropy,coreset,diverse")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget-unit", default="lots", choices=["lots", "wafers"],
                    help="what the budget counts. The published curve used "
                         "lots, and the heuristics answered by buying lots "
                         "averaging 2.7 wafers against random's 15.7 -- so on "
                         "that axis they were compared while training on a "
                         "fifth of the data. `wafers` holds the supervision "
                         "volume fixed and measures acquisition quality alone.")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = Corpus.load(args.corpus)
    X = torch.load(args.features, map_location="cpu", weights_only=False)["X"]
    tr_all, te = split(c, "lot", seed=0)
    mu, sd = X[tr_all].mean(0, keepdim=True), X[tr_all].std(0, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd
    budgets = [int(b) for b in args.budgets.split(",")]
    lots_pool = np.unique(c.lot[tr_all].numpy())
    lot_size = {int(u): int(n) for u, n in
                zip(*np.unique(c.lot[tr_all].numpy(), return_counts=True))}
    results = {}

    def n_wafers(lots):
        return sum(lot_size.get(int(l), 0) for l in lots)

    def reached(lots, b):
        """Has this acquisition met the budget, in whichever unit is in force?"""
        return (len(lots) >= b if args.budget_unit == "lots"
                else n_wafers(lots) >= b)

    for strategy in args.strategies.split(","):
        curve = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            chosen = list(rng.choice(lots_pool, args.seed_lots, replace=False))
            chosen_emb = None
            per_budget = {}
            for b in budgets:
                # grow the labelled set to the budget with this strategy
                while not reached(chosen, b):
                    pool_lots = np.setdiff1d(lots_pool, np.array(chosen))
                    if len(pool_lots) == 0:
                        break
                    mask = np.isin(c.lot[tr_all].numpy(), pool_lots)
                    pool_idx = tr_all[torch.from_numpy(mask)]
                    # how many lots to take before re-scoring. Under a lot
                    # budget this is the shortfall directly; under a wafer
                    # budget the shortfall is in wafers, so it is converted at
                    # the pool's mean lot size and floored at one.
                    if args.budget_unit == "lots":
                        step = b - len(chosen)
                    else:
                        # Estimate the lots needed from the size of the lots
                        # *this strategy* has been buying, not the pool mean.
                        # The pool mean is ~15 and a heuristic that buys 2-wafer
                        # lots would then take seven times as many re-scoring
                        # rounds as the lot-budget run did, which is a different
                        # acquisition schedule and not a comparable experiment.
                        seen = [lot_size.get(int(l), 0) for l in chosen]
                        mean_size = max(float(np.mean(seen)) if seen else 0.0,
                                        1.0)
                        step = int(np.ceil((b - n_wafers(chosen)) / mean_size))
                    step = max(1, min(step, len(pool_lots)))
                    if strategy == "random":
                        pick = rng.choice(pool_lots, step, replace=False)
                        chosen += list(pick)
                        continue
                    model_now, _ = train_eval(
                        X, c.labels,
                        tr_all[torch.from_numpy(
                            np.isin(c.lot[tr_all].numpy(), np.array(chosen)))],
                        te[:2048], dev, epochs=15, seed=seed)
                    sc = lot_scores(model_now, X, pool_idx, c.lot, dev,
                                    strategy, chosen_emb)
                    order = sorted(sc, key=sc.get, reverse=True)
                    take = order[:step]
                    chosen += take
                    if strategy in ("coreset", "diverse"):
                        sel = tr_all[torch.from_numpy(
                            np.isin(c.lot[tr_all].numpy(), np.array(chosen)))]
                        with torch.no_grad():
                            chosen_emb = (model_now.embed(X[sel].to(dev)).cpu().numpy()
                                          if strategy == "coreset"
                                          else X[sel].numpy())
                # Under a lot budget the first b lots are the purchase. Under
                # a wafer budget the loop stops on the first lot that crosses
                # the line, which overshoots by up to one step, so the purchase
                # is trimmed to the longest prefix that fits -- you cannot buy
                # the lot that would exceed the budget. At least one lot is
                # always kept, or a budget below the first lot's size would
                # train on nothing.
                if args.budget_unit == "lots":
                    bought = chosen[:b]
                else:
                    bought, tot = [], 0
                    for l in chosen:
                        sz = lot_size.get(int(l), 0)
                        if bought and tot + sz > b:
                            break
                        bought.append(l); tot += sz
                labelled = tr_all[torch.from_numpy(
                    np.isin(c.lot[tr_all].numpy(), np.array(bought)))]
                _, p = train_eval(X, c.labels, labelled, te, dev, seed=seed)
                m = metrics.summarize(c.labels[te].numpy(), p,
                                      groups=c.lot[te].numpy())
                sizes = [lot_size.get(int(l), 0) for l in bought]
                per_budget[b] = {"macro_f1": m["macro_f1"],
                                 "p10_domain_macro_f1": m["p10_domain_macro_f1"],
                                 "n_wafers": int(len(labelled)),
                                 "n_lots": int(len(bought)),
                                 "mean_lot_size": float(np.mean(sizes)),
                                 "median_lot_size": float(np.median(sizes)),
                                 # the lots themselves, so the selection bias
                                 # can be diagnosed without re-running anything
                                 "lots": [int(l) for l in bought]}
                print(f"  [{strategy} seed{seed}] budget {b} "
                      f"{args.budget_unit}: {len(bought)} lots, "
                      f"{len(labelled)} wafers, mean lot size "
                      f"{np.mean(sizes):.2f}: macroF1 "
                      f"{m['macro_f1']:.4f}", flush=True)
            curve.append(per_budget)
        results[strategy] = {
            str(b): {
                "macro_f1_mean": float(np.mean([c_[b]["macro_f1"] for c_ in curve])),
                "macro_f1_std": float(np.std([c_[b]["macro_f1"] for c_ in curve])),
                "p10_mean": float(np.mean([c_[b]["p10_domain_macro_f1"]
                                           for c_ in curve])),
                "wafers_mean": float(np.mean([c_[b]["n_wafers"] for c_ in curve])),
                "lots_mean": float(np.mean([c_[b]["n_lots"] for c_ in curve])),
                "mean_lot_size": float(np.mean([c_[b]["mean_lot_size"]
                                                for c_ in curve])),
                "median_lot_size": float(np.mean([c_[b]["median_lot_size"]
                                                  for c_ in curve])),
            } for b in budgets}
    Path(args.out).write_text(json.dumps(
        {"budgets": budgets, "budget_unit": args.budget_unit,
         "seed_lots": args.seed_lots, "seeds": args.seeds,
         "n_pool_lots": int(len(lots_pool)), "results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
