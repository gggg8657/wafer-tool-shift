"""Self-supervised pretraining on the unlabelled 80% of WM-811K,
with the lot made deliberately unpredictable.

    CUDA_VISIBLE_DEVICES=2 python scripts/pretrain_ssl.py --epochs 8

Two objectives at once:

* **masked die modelling** -- blank out square patches of the wafer and predict
  each hidden die's true state (outside / pass / fail). The BERT/MAE recipe,
  which needs no labels and no negative sampling, and which forces the encoder
  to learn how failure propagates spatially because that is the only way to fill
  a hole.
* **adversarial nuisance heads** -- gradient-reversed classifiers that try to
  recover *properties of the lot* from the pooled embedding. Making the
  representation unable to recover them is domain invariance built during
  pretraining rather than bolted on during fine-tuning.

  The first version of this asked the adversary to name the lot itself, hashed
  into 64 buckets. Its loss sat at chance from the first epoch: 41,608 lots
  modulo 64 is a meaningless label, so there was no signal to remove and
  gradient reversal had nothing to reverse. A vacuous adversary is worse than
  none, because it looks like invariance in the logs. The nuisances used now are
  both genuinely predictable from a wafer map and both genuinely irrelevant to
  the defect class:

  - the **lot's failure-rate decile** -- a lot running hot is a lot property, not
    a defect pattern;
  - the **wafer geometry bucket** -- the product, which the benchmark holds out
    as its second shift axis.

Why this fits here: labels cover a fifth of the corpus, but lot membership and
geometry are free for every wafer, so the nuisance variables are fully observed
exactly where the labels are missing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import CLASSES, build_unlabeled                    # noqa: E402
from wts.methods import grad_reverse                             # noqa: E402
from wts.models import CnnResized, onehot_maps                   # noqa: E402

N_GEOM = 32          # top geometries, everything else pooled into one bucket


class MaskedDieModel(nn.Module):
    """Encoder, per-die decoder, and two nuisance heads behind a reversal layer."""

    def __init__(self, width=32, n_rate=10, n_geom=N_GEOM):
        super().__init__()
        self.enc = CnnResized(len(CLASSES), width=width, norm="gn")
        w = width * 4
        self.dec = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(w, w // 2, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(w // 2, w // 4, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(w // 4, 3, 1))
        self.rate_head = nn.Sequential(nn.Linear(w, 128), nn.GELU(),
                                       nn.Linear(128, n_rate))
        self.geom_head = nn.Sequential(nn.Linear(w, 128), nn.GELU(),
                                       nn.Linear(128, n_geom))

    def forward(self, x, lamb=0.3):
        h = self.enc.body(x)
        recon = self.dec(h)
        pooled = h.mean(dim=(-2, -1))
        rev = grad_reverse(pooled, lamb)
        return recon, self.rate_head(rev), self.geom_head(rev)


def mask_patches(x, n_patches=6, size=12):
    """Blank square patches; returns (masked input, boolean mask)."""
    B, _, H, W = x.shape
    m = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
    for _ in range(n_patches):
        y0 = torch.randint(0, H - size, (B,), device=x.device)
        x0 = torch.randint(0, W - size, (B,), device=x.device)
        for b in range(B):
            m[b, y0[b]:y0[b] + size, x0[b]:x0[b] + size] = True
    xm = x.clone()
    xm[m.unsqueeze(1).expand_as(x)] = 0.0        # all channels off = "unknown"
    return xm, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unlabeled", default="data/unlabeled.pt")
    p.add_argument("--out", default="runs/ssl_pretrain.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lot-lambda", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.unlabeled)
    if not path.exists():
        print("building the unlabelled cache (one-off) ...", flush=True)
        build_unlabeled(cache=str(path), limit=args.limit)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    maps = blob["maps64"].to(dev)
    n = maps.shape[0]

    # nuisance 1: the lot's failure-rate decile
    fail = (maps == 2).float().sum(dim=(-2, -1))
    inside = (maps > 0).float().sum(dim=(-2, -1)).clamp_min(1)
    rate = (fail / inside).cpu()
    lot = blob["lot"]
    lot_rate = torch.zeros(int(lot.max()) + 1)
    lot_rate.scatter_reduce_(0, lot, rate, reduce="mean", include_self=False)
    per_wafer_rate = lot_rate[lot]
    edges = torch.quantile(per_wafer_rate, torch.linspace(0, 1, 11)[1:-1])
    rate_bucket = torch.bucketize(per_wafer_rate, edges).clamp(0, 9).to(dev)

    # nuisance 2: the wafer geometry, top-31 shapes plus an "other" bucket
    hw = blob["hw"]
    key = hw[:, 0] * 1000 + hw[:, 1]
    uniq, counts = key.unique(return_counts=True)
    top = uniq[counts.argsort(descending=True)[: N_GEOM - 1]]
    geom = torch.full_like(key, N_GEOM - 1)
    for i, u in enumerate(top):
        geom[key == u] = i
    geom = geom.to(dev)
    print(f"unlabelled wafers: {n}, lots: {len(blob['lot_names'])}, "
          f"geometries kept: {len(top)} + other", flush=True)

    model = MaskedDieModel(width=args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = (n // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    hist = []
    t0 = time.time()
    gstep = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        rec_sum = lot_sum = nb = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            sel = perm[i:i + args.batch]
            raw = maps[sel]
            x = onehot_maps(raw)
            target = raw.long().clamp(0, 2)
            xm, m = mask_patches(x)
            recon, rate_logits, geom_logits = model(xm, args.lot_lambda)
            # score only the dies that were hidden
            rl = F.cross_entropy(recon.permute(0, 2, 3, 1)[m], target[m])
            nl = (F.cross_entropy(rate_logits, rate_bucket[sel])
                  + F.cross_entropy(geom_logits, geom[sel]))
            loss = rl + nl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if gstep + 1 < steps:
                sched.step()
            gstep += 1
            rec_sum += float(rl.detach()); lot_sum += float(nl.detach()); nb += 1
        import math
        chance = math.log(10) + math.log(N_GEOM)
        hist.append({"epoch": ep + 1, "recon_ce": rec_sum / nb,
                     "nuisance_ce": lot_sum / nb, "nuisance_ce_chance": chance})
        print(f"  ep {ep+1}/{args.epochs} recon CE {rec_sum/nb:.4f}  "
              f"nuisance CE {lot_sum/nb:.4f} (chance {chance:.4f}; rising "
              f"toward chance = the embedding is losing the nuisance)",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.enc.state_dict(), "history": hist,
                "args": vars(args), "minutes": round((time.time()-t0)/60, 2)},
               args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"history": hist, "args": vars(args),
         "minutes": round((time.time()-t0)/60, 2), "n_unlabeled": n}, indent=2))
    print(f"wrote {args.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
