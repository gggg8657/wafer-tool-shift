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


# --------------------------------------------------------------------------- #
# independence and transport penalties
# --------------------------------------------------------------------------- #
def _rbf(x, sigma=None):
    d2 = torch.cdist(x, x).pow(2)
    if sigma is None:                      # median heuristic
        sigma = d2.detach().flatten().median().clamp_min(1e-6).sqrt()
    return torch.exp(-d2 / (2 * sigma**2))


def hsic_penalty(emb, dom, n_dom):
    """Hilbert-Schmidt independence criterion between features and domain.

    DANN asks a classifier whether the domain is predictable and can be fooled
    by a weak classifier; HSIC measures statistical dependence directly, with no
    adversary to under-train. From the kernel-methods literature, where it is
    the standard test for independence.
    """
    B = emb.shape[0]
    if B < 8:
        return emb.new_zeros(())
    K = _rbf(emb)
    D = F.one_hot(dom, n_dom).float()
    L = D @ D.T                            # delta kernel on a categorical label
    H = torch.eye(B, device=emb.device) - 1.0 / B
    return (K @ H @ L @ H).diagonal().sum() / (B - 1) ** 2


def sinkhorn_divergence(a, b, eps=0.1, n_iter=25):
    """Entropic optimal transport between two embedding clouds.

    CORAL matches second moments, which is blind to any difference the
    covariance cannot see. OT compares the distributions themselves; the
    entropic version is differentiable and cheap enough to put in a training
    loop. From computational optimal transport.
    """
    C = torch.cdist(a, b).pow(2)
    C = C / C.detach().max().clamp_min(1e-8)
    n, m = a.shape[0], b.shape[0]
    mu = a.new_full((n,), 1.0 / n)
    nu = b.new_full((m,), 1.0 / m)
    K = torch.exp(-C / eps)
    u = torch.ones_like(mu)
    for _ in range(n_iter):
        v = nu / (K.T @ u).clamp_min(1e-12)
        u = mu / (K @ v).clamp_min(1e-12)
    P = u[:, None] * K * v[None, :]
    return (P * C).sum()


def hsic(model, batch, st):
    x, y, d = batch["x"], batch["y"], batch["d"]
    emb = model.embed(x, batch["mask"]) if st.get("masked") else model.embed(x)
    cls = F.cross_entropy(model.head(emb), y, weight=st.get("cw"))
    pen = hsic_penalty(emb, d, st["n_dom"])
    return cls + st["hsic_lambda"] * pen, {"hsic": float(pen)}


def sinkhorn(model, batch, st):
    x, y, d = batch["x"], batch["y"], batch["d"]
    emb = model.embed(x, batch["mask"]) if st.get("masked") else model.embed(x)
    cls = F.cross_entropy(model.head(emb), y, weight=st.get("cw"))
    groups = [emb[d == g] for g in torch.unique(d)]
    groups = [g for g in groups if g.shape[0] >= 4]
    pen = emb.new_zeros(())
    pairs = 0
    for i in range(0, len(groups) - 1, 2):        # disjoint pairs, cost-capped
        pen = pen + sinkhorn_divergence(groups[i], groups[i + 1])
        pairs += 1
    if pairs:
        pen = pen / pairs
    return cls + st["ot_lambda"] * pen, {"ot": float(pen)}


def anchor(model, batch, st):
    """Anchor regression, adapted to a classifier.

    From econometrics and causal inference: with an anchor A that may affect
    both the covariates and the outcome, penalizing the component of the
    residual that A can explain buys distributional robustness against shifts
    generated through A -- exactly the setting here, where the anchor is the lot
    and the shift is a new lot. `gamma` interpolates between ERM (gamma = 1) and
    a hard invariance constraint (gamma -> infinity).

    The classifier version penalizes the per-domain mean of the residual
    softmax(logits) - onehot(y); a predictor whose errors are unbiased within
    every lot has nothing left for the anchor to explain.
    """
    x, y, d = batch["x"], batch["y"], batch["d"]
    logits = model(x, batch["mask"]) if st.get("masked") else model(x)
    cls = F.cross_entropy(logits, y, weight=st.get("cw"))
    resid = logits.softmax(1) - F.one_hot(y, logits.shape[1]).float()
    pen = logits.new_zeros(())
    B = x.shape[0]
    for g in torch.unique(d):
        m = d == g
        if m.sum() < 2:
            continue
        pen = pen + (m.sum() / B) * resid[m].mean(0).pow(2).sum()
    return cls + (st["anchor_gamma"] - 1.0) * pen, {"anchor_penalty": float(pen)}


def focal(model, batch, st):
    """Multi-class focal loss (Lin et al., 2017), single-label form.

    Aimed at this corpus's long tail rather than at its domain shift. The
    lot-disjoint per-class F1 is `none` 0.992 and `Edge-Ring` 0.984 against
    `Scratch` 0.747 and `Loc` 0.766, and 85% of the corpus is `none` predicted
    at high confidence -- so nearly all of the gradient mass comes from examples
    that are already right. Focal's `(1 - p_y)^gamma` factor removes that mass
    and leaves the hard minority.

    `gamma = 0` is exactly cross-entropy, which makes it the built-in control:
    the sweep runs it and it must reproduce ERM.

    Composes with `--class-weight`, which is a different correction (prior
    reweighting rather than difficulty reweighting), so the two are swept
    separately and never in the same cell.
    """
    x, y = batch["x"], batch["y"]
    logits = model(x, batch["mask"]) if st.get("masked") else model(x)
    logp = F.log_softmax(logits, dim=1)
    logpy = logp.gather(1, y[:, None]).squeeze(1)
    loss = -((1 - logpy.exp()) ** st["focal_gamma"]) * logpy
    if st.get("cw") is not None:
        loss = loss * st["cw"][y]
    return loss.mean(), {"focal_pt": float(logpy.exp().mean())}


OBJECTIVES.update({"hsic": hsic, "sinkhorn": sinkhorn, "anchor": anchor,
                   "focal": focal})
NEEDS_DOMAIN.update({"hsic", "sinkhorn", "anchor"})
NEEDS_EMBED.update({"hsic", "sinkhorn"})
