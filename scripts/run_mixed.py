"""Multi-label training on the SYNTHETIC mixed-type set from `make_mixed.py`.

    python scripts/run_mixed.py --loss focal --data data/mixed.pt

Everything this writes carries `"synthetic": true` and the dataset name
`MIXED-SYNTH`. It is *not* MixedWM38 and it is not WM-811K: it is an overlay
construction whose patterns do not interact, so its F1 is an upper bound on the
real mixed-type task. Do not let a number from here into a sentence about
WM-811K.

Two protocol variants are meant to be run and reported side by side:

    data/mixed.pt      lot-disjoint sources  -- the honest protocol
    data/mixed_iid.pt  random split          -- the optimistic protocol

Thresholds are chosen per class on a **lot-disjoint validation split carved out
of training**, never on test. Both the fixed-0.5 and the tuned numbers are
reported, because tuning thresholds is worth several points of macro-F1 on a
long-tailed label set and a paper that reports only the tuned number without
saying so is reporting a different metric than one that does not.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.models import CnnResized, onehot_maps          # noqa: E402


def focal_bce(logits, y, gamma=2.0, alpha=None):
    """Multi-label focal loss (Lin et al., 2017), one sigmoid per class.

    `(1 - p_t)^gamma` down-weights the easy negatives that dominate a sparse
    multi-hot target -- here 8 classes with at most 2 positives, so ~78% of
    every target vector is a negative the model gets right immediately.
    """
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = ((1 - pt) ** gamma) * ce
    if alpha is not None:
        loss = loss * (alpha * y + (1 - alpha) * (1 - y))
    return loss.mean()


def f1_per_class(y_true, y_pred):
    tp = (y_true * y_pred).sum(0)
    fp = ((1 - y_true) * y_pred).sum(0)
    fn = (y_true * (1 - y_pred)).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)


def tune_thresholds(y, p, grid=np.arange(0.05, 0.96, 0.025)):
    """Per-class threshold maximizing that class's F1 on the given split."""
    th = np.full(y.shape[1], 0.5)
    for k in range(y.shape[1]):
        best, bt = -1.0, 0.5
        for t in grid:
            pred = (p[:, k] >= t).astype(np.float32)
            tp = (y[:, k] * pred).sum()
            f1 = 2 * tp / max(2 * tp + ((1 - y[:, k]) * pred).sum()
                              + (y[:, k] * (1 - pred)).sum(), 1e-9)
            if f1 > best:
                best, bt = f1, float(t)
        th[k] = bt
    return th


def summarize(y, p, th, classes):
    pred = (p >= th[None, :]).astype(np.float32)
    per = f1_per_class(y, pred)
    tp = (y * pred).sum(); fp = ((1 - y) * pred).sum(); fn = (y * (1 - pred)).sum()
    return {
        "macro_f1": float(per.mean()),
        "micro_f1": float(2 * tp / max(2 * tp + fp + fn, 1e-9)),
        "subset_accuracy": float((pred == y).all(1).mean()),
        "hamming_accuracy": float((pred == y).mean()),
        "per_class_f1": {c: float(v) for c, v in zip(classes, per)},
        "thresholds": {c: float(v) for c, v in zip(classes, th)},
    }


def inner_split(lot, frac=0.15, seed=0):
    """Lot-disjoint validation carved out of the training side."""
    rng = np.random.default_rng(seed + 1)
    u = rng.permutation(np.unique(lot))
    target = int(frac * len(lot))
    va = np.zeros(len(lot), dtype=bool)
    taken = 0
    for g in u:
        m = lot == g
        va |= m
        taken += int(m.sum())
        if taken >= target:
            break
    return ~va, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mixed.pt")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--loss", default="bce", choices=["bce", "focal", "posweight"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = torch.load(a.data, map_location="cpu", weights_only=False)
    classes = b["classes"]
    torch.manual_seed(a.seed)

    tr_m, va_m = inner_split(b["train_lot"].numpy(), seed=a.seed)
    Xtr = b["train_maps"].to(dev)
    Ytr = b["train_y"].to(dev)
    Xte = b["test_maps"].to(dev)
    Yte = b["test_y"].numpy()
    tr_i = torch.from_numpy(np.where(tr_m)[0]).to(dev)
    va_i = torch.from_numpy(np.where(va_m)[0]).to(dev)

    model = CnnResized(len(classes), width=a.width, norm="gn").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(1, (len(tr_i) + a.batch - 1) // a.batch) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    pos = Ytr[tr_i].mean(0).clamp(1e-4, 1 - 1e-4)
    pw = ((1 - pos) / pos).clamp(max=50.0)

    def loss_fn(logits, y):
        if a.loss == "focal":
            return focal_bce(logits, y, gamma=a.gamma)
        if a.loss == "posweight":
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        return F.binary_cross_entropy_with_logits(logits, y)

    @torch.no_grad()
    def probs(idx):
        model.eval()
        out = []
        for i in range(0, len(idx), 1024):
            sel = idx[i:i + 1024]
            out.append(torch.sigmoid(model(onehot_maps(Xtr[sel]))).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def probs_test():
        model.eval()
        out = []
        for i in range(0, len(Xte), 1024):
            out.append(torch.sigmoid(
                model(onehot_maps(Xte[i:i + 1024]))).float().cpu())
        return torch.cat(out).numpy()

    hist, best, best_state = [], -1.0, None
    t0 = time.time()
    gstep = 0
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    for ep in range(a.epochs):
        model.train()
        perm = tr_i[torch.randperm(len(tr_i), generator=g).to(dev)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), a.batch):
            sel = perm[i:i + a.batch]
            if len(sel) < 4:
                continue
            logits = model(onehot_maps(Xtr[sel]))
            loss = loss_fn(logits, Ytr[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            if gstep < steps:
                sched.step()
            tot += float(loss.detach()); nb += 1
        pv = probs(va_i)
        yv = Ytr[va_i].cpu().numpy()
        vm = float(f1_per_class(yv, (pv >= 0.5).astype(np.float32)).mean())
        hist.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1),
                     "val_macro_f1_at_0.5": vm})
        if vm > best:
            best = vm
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep+1:2d}/{a.epochs} loss {tot/max(nb,1):.4f} "
              f"val macroF1@0.5 {vm:.4f}", flush=True)

    model.load_state_dict(best_state)
    pv, yv = probs(va_i), Ytr[va_i].cpu().numpy()
    th = tune_thresholds(yv, pv)
    pt = probs_test()
    res = {
        "dataset": b["name"], "synthetic": True, "note": b["note"],
        "protocol": b["protocol"], "loss": a.loss, "gamma": a.gamma,
        "seed": a.seed, "tag": a.tag, "epochs": a.epochs,
        "n_train": int(len(tr_i)), "n_val": int(len(va_i)), "n_test": int(len(Yte)),
        "minutes": round((time.time() - t0) / 60, 2),
        "history": hist,
        "val_macro_f1_at_0.5": best,
        "test_at_0.5": summarize(Yte, pt, np.full(len(classes), 0.5), classes),
        "test_at_val_tuned_thresholds": summarize(Yte, pt, th, classes),
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suffix = f"__{a.tag}" if a.tag else ""
    name = f"mixedsynth__{b['protocol']}__{a.loss}{suffix}__s{a.seed}.json"
    (out / name).write_text(json.dumps(res, indent=2))
    print(f"wrote {out/name}")
    print(f"RESULT MIXED-SYNTH/{b['protocol']}/{a.loss}: "
          f"macroF1@0.5 {res['test_at_0.5']['macro_f1']:.4f}  "
          f"macroF1@tuned {res['test_at_val_tuned_thresholds']['macro_f1']:.4f}  "
          f"microF1@tuned {res['test_at_val_tuned_thresholds']['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
