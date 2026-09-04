"""Test-time adaptation: adapt to the target lot using its *unlabelled* wafers.

This is the part a fab could actually deploy. A new tool or a new product comes
online, nobody has labels for it yet, but you do have its wafer maps -- so the
question is how much of the shift can be absorbed with unlabelled target data
and no retraining.

| method   | borrowed from            | what it changes at test time                       |
|----------|--------------------------|----------------------------------------------------|
| `adabn`  | domain adaptation        | recomputes BatchNorm statistics on the target lot  |
| `tent`   | test-time entropy min.   | one step on the norm layers' affine parameters     |
| `fda`    | Fourier domain adaptation| swaps low-frequency amplitude toward the target    |
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def adabn(model, loader, device):
    """Re-estimate BatchNorm running statistics on the target domain.

    Nothing is learned; the running mean and variance are simply recomputed
    where they belong. It costs one unlabelled pass and is the first thing to
    try whenever a BatchNorm model is shipped across a domain boundary.
    """
    m = copy.deepcopy(model)
    has_bn = any(isinstance(mod, nn.modules.batchnorm._BatchNorm)
                 for mod in m.modules())
    if not has_bn:
        return m, False
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats()
            mod.momentum = None          # cumulative average over the pass
    m.train()
    for batch in loader:
        m(batch["x"].to(device))
    m.eval()
    return m, True


def tent(model, loader, device, lr=1e-3, steps=1):
    """Entropy minimization on the normalization layers' affine parameters.

    The assumption is that confident predictions are more likely to be right, so
    only the scale/shift of the norm layers move and the features stay put. It
    can help and it can quietly collapse to one class -- which is why the
    benchmark reports per-class F1 after TTA, not just accuracy.
    """
    m = copy.deepcopy(model)
    params = []
    for mod in m.modules():
        if isinstance(mod, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm,
                            nn.LayerNorm)):
            for p in mod.parameters(recurse=False):
                p.requires_grad_(True)
                params.append(p)
    for p in m.parameters():
        if all(p is not q for q in params):
            p.requires_grad_(False)
    if not params:
        return m, False
    opt = torch.optim.Adam(params, lr=lr)
    m.train()
    for batch in loader:
        x = batch["x"].to(device)
        for _ in range(steps):
            logits = m(x)
            p = logits.softmax(1)
            ent = -(p * (p + 1e-12).log()).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            ent.backward()
            opt.step()
    m.eval()
    return m, True


def fda_amplitude_swap(x_src, x_tgt, beta=0.05):
    """Replace the source's low-frequency amplitude with the target's.

    In the Fourier domain amplitude carries "style" and phase carries content,
    so swapping a small low-frequency window moves a source wafer toward the
    target domain's look while leaving the defect pattern intact. Training on the
    swapped maps needs no adversarial machinery at all -- the appeal of FDA when
    it was introduced for segmentation.
    """
    Fs = torch.fft.fft2(x_src, norm="ortho")
    Ft = torch.fft.fft2(x_tgt, norm="ortho")
    As, Ps = Fs.abs(), Fs.angle()
    At = Ft.abs()
    H, W = x_src.shape[-2:]
    b = max(int(beta * min(H, W)), 1)
    m = torch.zeros(H, W, device=x_src.device, dtype=As.dtype)
    m[:b, :b] = 1; m[:b, -b:] = 1; m[-b:, :b] = 1; m[-b:, -b:] = 1
    A = As * (1 - m) + At * m
    return torch.fft.ifft2(A * torch.exp(1j * Ps), norm="ortho").real
