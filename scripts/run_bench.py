"""Run one benchmark cell: (encoder, objective, protocol) -> runs/<name>.json

    CUDA_VISIBLE_DEVICES=2 python scripts/run_bench.py \
        --encoder cnn_bn --objective erm --protocol lot

Model selection uses a *domain-disjoint* validation split carved out of the
training domains, never the test domains. Selecting on an iid validation set is
the quiet way domain-generalization results get inflated, so it is not offered
as an option.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts import metrics, methods, tta                                # noqa: E402
from wts.data import CLASSES, Corpus, label_shift, split             # noqa: E402
from wts.models import CnnResized, FeatMlp, SpectralNet, onehot_maps  # noqa: E402

N_BUCKETS = 32          # domains are bucketed for the group-aware objectives


def domain_vector(corpus, protocol):
    """The domain label a protocol is defined by."""
    return corpus.size_id if protocol == "size" else corpus.lot


def make_loader_indices(idx, batch, shuffle, gen=None):
    idx = idx[torch.randperm(len(idx), generator=gen)] if shuffle else idx
    return [idx[i:i + batch] for i in range(0, len(idx), batch)]


def size_bucketed_batches(corpus, idx, batch, gen=None):
    """Batches whose wafers all share one geometry, for the spectral encoder.

    A size group still spans hundreds of lots, so domain-aware objectives keep
    a mixed batch even though the geometry is fixed.
    """
    sid = corpus.size_id[idx]
    out = []
    for s in torch.unique(sid):
        sub = idx[sid == s]
        if gen is not None:
            sub = sub[torch.randperm(len(sub), generator=gen)]
        out += [sub[i:i + batch] for i in range(0, len(sub), batch)]
    order = torch.randperm(len(out), generator=gen).tolist() if gen else range(len(out))
    return [out[i] for i in order]


class Runner:
    def __init__(self, args):
        self.a = args
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c = Corpus.load(args.corpus)
        self.feat = None
        if args.encoder == "feat":
            blob = torch.load(args.features, map_location="cpu", weights_only=False)
            self.feat_slices = blob["slices"]
            self.feat_raw = blob["X"]
        # Resident on the device: 172,948 resized maps are 708 MB as uint8 and
        # the descriptors are 86 MB, so the whole corpus fits in HBM many times
        # over. Doing the one-hot on the GPU rather than per batch on the host
        # is the difference between minutes and seconds per epoch.
        if args.encoder in ("cnn_bn", "cnn_gn"):
            self.maps64_dev = self.c.maps64.to(self.dev)
        self.dom = domain_vector(self.c, args.protocol)
        tr_all, te = split(self.c, args.protocol, seed=args.seed)
        # domain-disjoint validation carved out of the training domains
        self.tr, self.va = self._inner_split(tr_all, args.protocol, args.seed)
        self.te = te
        self.label_shift = label_shift(self.c, self.tr, self.te)

    def _inner_split(self, idx, protocol, seed):
        d = self.dom[idx].numpy()
        y = self.c.labels[idx].numpy()
        rng = np.random.default_rng(seed + 1)
        if protocol == "iid":
            perm = rng.permutation(len(idx))
            cut = int(0.85 * len(idx))
            return idx[torch.from_numpy(perm[:cut])], idx[torch.from_numpy(perm[cut:])]
        uniq = rng.permutation(np.unique(d))
        target = int(0.15 * len(idx))
        va = np.zeros(len(idx), dtype=bool)
        counts = np.bincount(y, minlength=len(CLASSES))
        taken = 0
        for g in uniq:
            sel = d == g
            cand = counts - np.bincount(y[sel], minlength=len(CLASSES))
            if (cand[counts > 0] < 1).any() or taken + sel.sum() > target * 1.2:
                continue
            va |= sel; counts = cand; taken += int(sel.sum())
            if taken >= target:
                break
        return idx[torch.from_numpy(~va)], idx[torch.from_numpy(va)]

    # ---------------------------------------------------------------- batches
    def batch_of(self, sel):
        y = self.c.labels[sel].to(self.dev)
        d = (self.dom[sel] % N_BUCKETS).to(self.dev)
        if self.a.encoder == "feat":
            x = (self.feat_dev[sel.to(self.dev)] - self.mu) / self.sd
            return {"x": x, "y": y, "d": d, "mask": None}
        if self.a.encoder == "spectral":
            maps = torch.stack([self.c.maps[i] for i in sel.tolist()])
            x = onehot_maps(maps).to(self.dev)
            mask = (maps > 0).to(self.dev)
            return {"x": x, "y": y, "d": d, "mask": mask}
        x = onehot_maps(self.maps64_dev[sel.to(self.dev)])
        return {"x": x, "y": y, "d": d, "mask": None}

    def loaders(self, idx, shuffle, batch=None):
        b = batch or self.a.batch
        gen = torch.Generator().manual_seed(self.a.seed) if shuffle else None
        if self.a.encoder == "spectral":
            return size_bucketed_batches(self.c, idx, b, gen)
        return make_loader_indices(idx, b, shuffle, gen)

    # ---------------------------------------------------------------- model
    def build(self):
        n = len(CLASSES)
        if self.a.encoder == "feat":
            m = FeatMlp(self.feat_raw.shape[1], n, hidden=self.a.width * 4)
        elif self.a.encoder == "spectral":
            m = SpectralNet(n, width=self.a.width, modes=self.a.modes)
        else:
            m = CnnResized(n, width=self.a.width,
                           norm="bn" if self.a.encoder == "cnn_bn" else "gn")
        return m.to(self.dev)

    def run(self):
        a = self.a
        torch.manual_seed(a.seed)
        if a.encoder == "feat":
            self.feat_dev = self.feat_raw.to(self.dev)
            tr_dev = self.tr.to(self.dev)
            self.mu = self.feat_dev[tr_dev].mean(0, keepdim=True)
            self.sd = self.feat_dev[tr_dev].std(0, keepdim=True).clamp_min(1e-6)
        model = self.build()
        counts = torch.bincount(self.c.labels[self.tr], minlength=len(CLASSES))
        prior = (counts.float() / counts.sum()).clamp_min(1e-9)
        st = {
            "cw": methods.class_weights(counts).to(self.dev) if a.class_weight else None,
            "log_prior": prior.log().to(self.dev),
            "tau": a.tau, "dro_eta": a.dro_eta,
            "q": torch.ones(N_BUCKETS, device=self.dev) / N_BUCKETS,
            "irm_lambda": a.irm_lambda, "coral_lambda": a.coral_lambda,
            "mix_alpha": a.mix_alpha, "lamb": a.dann_lambda,
            "masked": a.encoder == "spectral",
            "domain_index": torch.arange(N_BUCKETS, device=self.dev),
            "dummy": torch.tensor(1.0, device=self.dev, requires_grad=True),
        }
        if a.objective == "dann":
            st["domain_head"] = nn.Sequential(
                nn.Linear(model.feat_dim, 128), nn.ReLU(),
                nn.Linear(128, N_BUCKETS)).to(self.dev)
        params = list(model.parameters()) + (
            list(st["domain_head"].parameters()) if a.objective == "dann" else [])
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
        obj = methods.OBJECTIVES[a.objective]
        steps = max(1, len(self.loaders(self.tr, True))) * a.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)

        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        best, best_state, hist = -1.0, None, []
        t0 = time.time()
        gstep = 0
        for ep in range(a.epochs):
            model.train()
            run_loss, nb = 0.0, 0
            for sel in self.loaders(self.tr, True):
                if len(sel) < 4:
                    continue
                batch = self.batch_of(sel)
                loss, _ = obj(model, batch, st)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if gstep + 1 < steps:
                    sched.step()
                gstep += 1
                run_loss += float(loss.detach()); nb += 1
                if gstep % a.ema_every == 0:
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
            va = self.evaluate(model, self.va)
            hist.append({"epoch": ep + 1, "train_loss": run_loss / max(nb, 1),
                         "val_macro_f1": va["macro_f1"]})
            if va["macro_f1"] > best:
                best = va["macro_f1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  ep {ep+1:2d}/{a.epochs} loss {run_loss/max(nb,1):.4f} "
                  f"val macroF1 {va['macro_f1']:.4f}", flush=True)

        model.load_state_dict(best_state)
        res = {
            "encoder": a.encoder, "objective": a.objective, "protocol": a.protocol,
            "seed": a.seed, "epochs": a.epochs,
            "n_train": len(self.tr), "n_val": len(self.va), "n_test": len(self.te),
            "label_shift_tv": self.label_shift,
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "minutes": round((time.time() - t0) / 60, 2),
            "history": hist,
            "val_macro_f1": best,
        }
        cal = self.calibration(model)
        res["test"] = self.evaluate(model, self.te, groups=True, cal=cal)

        if a.tta and a.encoder != "feat":
            # unlabelled target batches: what a fab would actually have on a new
            # tool before anyone has measured it
            loader = self.loaders(self.te, False)[:40]
            tgt = [{"x": self.batch_of(s)["x"]} for s in loader]
            for name, fn in (("adabn", tta.adabn), ("tent", tta.tent)):
                mm, ok = fn(model, tgt, self.dev)
                if ok:
                    res[f"test_{name}"] = self.evaluate(mm, self.te, groups=True)
                    print(f"  TTA {name}: macroF1 "
                          f"{res[f'test_{name}']['macro_f1']:.4f}", flush=True)

        # EMA weights as a cheap SWA-style check
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype)
                               for k, v in ema.items()})
        res["test_ema"] = self.evaluate(model, self.te, groups=True)

        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        name = f"{a.protocol}__{a.encoder}__{a.objective}__s{a.seed}.json"
        (out / name).write_text(json.dumps(res, indent=2))
        print(f"wrote {out/name}")
        t = res["test"]
        print(f"RESULT {a.protocol}/{a.encoder}/{a.objective}: "
              f"macroF1 {t['macro_f1']:.4f}  defectF1 {t['defect_macro_f1']:.4f}  "
              f"worstDomain {t.get('worst_domain_macro_f1', float('nan')):.4f}  "
              f"AUROC {t['defect_auroc']:.4f}  ECE {t['ece']:.4f}")
        return res

    @torch.no_grad()
    def probs_for(self, model, batches):
        model.eval()
        out, ys = [], []
        for sel in batches:
            b = self.batch_of(sel)
            logits = model(b["x"], b["mask"]) if self.a.encoder == "spectral" \
                else model(b["x"])
            out.append(logits.softmax(1).float().cpu())
            ys.append(self.c.labels[sel])
        return torch.cat(out).numpy(), torch.cat(ys).numpy()

    def evaluate(self, model, idx, groups=False, cal=None):
        batches = self.loaders(idx, False, batch=max(self.a.batch, 512))
        p, y = self.probs_for(model, batches)
        # the spectral encoder groups batches by geometry, so evaluation order is
        # not index order; the domains are read back from the same batch list
        g = self.dom[torch.cat(batches)].numpy() if groups else None
        return metrics.summarize(y, p, groups=g, cal=cal)

    def calibration(self, model):
        """Conformal calibration on held-out *training* domains (the val split)."""
        return self.probs_for(model, self.loaders(self.va, False,
                                                   batch=max(self.a.batch, 512)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus.pt")
    p.add_argument("--features", default="data/features.pt")
    p.add_argument("--out", default="runs")
    p.add_argument("--encoder", default="cnn_bn",
                   choices=["cnn_bn", "cnn_gn", "spectral", "feat"])
    p.add_argument("--objective", default="erm", choices=list(methods.OBJECTIVES))
    p.add_argument("--protocol", default="lot", choices=["iid", "lot", "size"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--dro-eta", type=float, default=0.01)
    p.add_argument("--irm-lambda", type=float, default=1.0)
    p.add_argument("--coral-lambda", type=float, default=1.0)
    p.add_argument("--mix-alpha", type=float, default=0.2)
    p.add_argument("--dann-lambda", type=float, default=0.3)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--ema-every", type=int, default=8)
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()
    print(f"== {args.protocol} / {args.encoder} / {args.objective} ==", flush=True)
    Runner(args).run()


if __name__ == "__main__":
    main()
