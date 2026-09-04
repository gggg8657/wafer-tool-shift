"""Robustness objectives, and what each one is borrowed from.

The honest starting position is that domain-generalization methods often fail
to beat plain ERM once the evaluation is fair (the WILDS line of work, and the
"has any progress been made?" critiques). So each method here is implemented to
be *measured against ERM under the same budget*, not assumed to help.

| method          | field it comes from            | the assumption it makes                                  |
|-----------------|--------------------------------|----------------------------------------------------------|
| `erm`           | -                              | none; the number everything else has to beat             |
| `logit_adjust`  | long-tail recognition          | the label prior shifts, the class-conditional does not   |
| `group_dro`     | robust optimization / economics| the worst domain is what matters, not the average        |
| `dann`          | domain adaptation              | features that cannot predict the domain transfer better  |
| `irm`           | causal inference               | the optimal predictor is the same in every domain        |
| `coral`         | domain adaptation              | aligning second moments across domains is enough         |
| `mixup_domain`  | vicinal risk / LISA            | interpolating across domains fills the gap between them  |
| `ema`           | weight averaging (SWA/SWAD)    | flatter minima generalize across domains                 |

Every method takes the same signature so the runner can treat them uniformly:
`loss(model, batch, state) -> (loss, logs)`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb=1.0):
    return GradReverse.apply(x, lamb)


def class_weights(counts, beta=0.999):
    """Class-balanced weights (effective number of samples).

    From the long-tail literature: 1/n over-corrects when n spans three orders
    of magnitude (147,429 `none` against 149 `Near-full`), and the effective
    number (1 - beta^n)/(1 - beta) interpolates between 1/n and uniform.
    """
    c = counts.float().clamp_min(1)
    eff = (1 - torch.pow(beta, c)) / (1 - beta)
    w = 1.0 / eff
    return w / w.mean()


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #
def erm(model, batch, st):
    x, y = batch["x"], batch["y"]
    logits = model(x, batch["mask"]) if st.get("masked") else model(x)
    return F.cross_entropy(logits, y, weight=st.get("cw")), {}


def logit_adjust(model, batch, st):
    """Subtract tau * log(prior) from the logits during training.

    Balanced-softmax / logit adjustment: the Bayes-optimal correction when the
    label prior differs between train and test. This benchmark's `size` protocol
    moves the label marginal by design, so it is the right first thing to try --
    and unlike resampling it costs nothing and does not throw data away.
    """
    x, y = batch["x"], batch["y"]
    logits = model(x, batch["mask"]) if st.get("masked") else model(x)
    return F.cross_entropy(logits + st["tau"] * st["log_prior"], y), {}


def group_dro(model, batch, st):
    """Minimize the worst domain's loss via exponentiated-gradient weights."""
    x, y, d = batch["x"], batch["y"], batch["d"]
    logits = model(x, batch["mask"]) if st.get("masked") else model(x)
    per = F.cross_entropy(logits, y, weight=st.get("cw"), reduction="none")
    uniq = torch.unique(d)
    q = st["q"]
    losses = []
    for g in uniq:
        m = d == g
        gl = per[m].mean()
        q[g] = q[g] * torch.exp(st["dro_eta"] * gl.detach())
        losses.append((g, gl))
    q.clamp_(min=1e-8)
    q /= q.sum()
    loss = sum(q[g] * gl for g, gl in losses)
    return loss, {"worst_group_loss": max(gl.item() for _, gl in losses)}


def dann(model, batch, st):
    """Classification loss minus the domain head's ability to read the domain."""
    x, y, d = batch["x"], batch["y"], batch["d"]
    emb = model.embed(x, batch["mask"]) if st.get("masked") else model.embed(x)
    logits = model.head(emb)
    cls = F.cross_entropy(logits, y, weight=st.get("cw"))
    dom = st["domain_head"](grad_reverse(emb, st["lamb"]))
    dl = F.cross_entropy(dom, st["domain_index"][d])
    return cls + dl, {"domain_loss": dl.item()}


def irm(model, batch, st):
    """IRMv1: penalize the variance of the per-domain gradient of a dummy scale.

    From invariant causal prediction -- if one predictor is simultaneously
    optimal in every domain, the gradient of the loss with respect to a scaling
    of the logits vanishes in each domain separately.
    """
    x, y, d = batch["x"], batch["y"], batch["d"]
    logits = model(x, batch["mask"]) if st.get("masked") else model(x)
    scale = st["dummy"]
    pen, tot, n = 0.0, 0.0, 0
    for g in torch.unique(d):
        m = d == g
        if m.sum() < 2:
            continue
        l = F.cross_entropy(logits[m] * scale, y[m], weight=st.get("cw"))
        gr = torch.autograd.grad(l, [scale], create_graph=True)[0]
        pen = pen + (gr**2).sum()
        tot = tot + l
        n += 1
    n = max(n, 1)
    loss = tot / n + st["irm_lambda"] * pen / n
    return loss, {"irm_penalty": float(pen) / n}


def coral(model, batch, st):
    """Align the second moments of the embedding across domains in the batch."""
    x, y, d = batch["x"], batch["y"], batch["d"]
    emb = model.embed(x, batch["mask"]) if st.get("masked") else model.embed(x)
    cls = F.cross_entropy(model.head(emb), y, weight=st.get("cw"))
    covs = []
    for g in torch.unique(d):
        m = d == g
        if m.sum() < 4:
            continue
        e = emb[m] - emb[m].mean(0, keepdim=True)
        covs.append(e.T @ e / (m.sum() - 1))
    pen = emb.new_zeros(())
    if len(covs) > 1:
        for i in range(len(covs)):
            for j in range(i + 1, len(covs)):
                pen = pen + (covs[i] - covs[j]).pow(2).mean()
        pen = pen / (len(covs) * (len(covs) - 1) / 2)
    return cls + st["coral_lambda"] * pen, {"coral_penalty": float(pen)}


def mixup_domain(model, batch, st):
    """Mixup that deliberately pairs samples from *different* domains.

    Plain mixup interpolates random pairs; pairing across domains turns the
    convex path between two lots into training data, which is the LISA argument
    for why interpolation helps under shift rather than just regularizing.
    """
    x, y, d = batch["x"], batch["y"], batch["d"]
    perm = torch.randperm(x.shape[0], device=x.device)
    # prefer a partner from another domain where one exists
    diff = d[perm] != d
    if diff.any():
        idx = torch.arange(x.shape[0], device=x.device)
        fallback = perm.clone()
        perm = torch.where(diff, perm, fallback[torch.roll(idx, 1)])
    lam = float(torch.distributions.Beta(st["mix_alpha"], st["mix_alpha"]).sample())
    xm = lam * x + (1 - lam) * x[perm]
    logits = model(xm, batch["mask"]) if st.get("masked") else model(xm)
    loss = (lam * F.cross_entropy(logits, y, weight=st.get("cw"))
            + (1 - lam) * F.cross_entropy(logits, y[perm], weight=st.get("cw")))
    return loss, {}


OBJECTIVES = {
    "erm": erm,
    "logit_adjust": logit_adjust,
    "group_dro": group_dro,
    "dann": dann,
    "irm": irm,
    "coral": coral,
    "mixup_domain": mixup_domain,
}
NEEDS_DOMAIN = {"group_dro", "dann", "irm", "coral", "mixup_domain"}
NEEDS_EMBED = {"dann", "coral"}
