"""Three encoders, chosen so the geometry axis is treated three different ways.

* `CnnResized` — the conventional pipeline: resample every wafer to 64x64 and
  run a small CNN. Cheap, strong in-distribution, and its features depend on the
  resampling, which is exactly what the unseen-geometry protocol probes.
* `SpectralNet` — a Fourier neural-operator encoder. It multiplies learned
  weights against a fixed number of low-frequency coefficients, so the same
  weights apply to a 25x27 and a 53x58 wafer with no resizing at all:
  discretization invariance by construction, borrowed from operator learning
  (see the sibling repo `pde-neural-operator`).
* `FeatMlp` — an MLP over the hand-built descriptors in `wts.features`, which
  are size-invariant because each one is a rate, a moment or a spectrum rather
  than a pixel.

Normalization is a first-class switch, not a detail. BatchNorm mixes statistics
across whatever happens to be in the batch, which is a domain leak when batches
span lots; GroupNorm does not. The benchmark reports both.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(kind: str, ch: int):
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(min(8, ch), ch)
    return nn.Identity()


class CnnResized(nn.Module):
    """Conventional baseline: 3 conv blocks on a resampled 64x64 map."""

    def __init__(self, n_classes=9, width=32, norm="bn", in_ch=3):
        super().__init__()
        chs = [in_ch, width, width * 2, width * 4]
        blocks = []
        for a, b in zip(chs[:-1], chs[1:]):
            blocks += [nn.Conv2d(a, b, 3, padding=1), _norm(norm, b), nn.ReLU(),
                       nn.Conv2d(b, b, 3, padding=1), _norm(norm, b), nn.ReLU(),
                       nn.MaxPool2d(2)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(chs[-1], n_classes)
        self.feat_dim = chs[-1]

    def embed(self, x):
        return self.body(x).mean(dim=(-2, -1))

    def forward(self, x):
        return self.head(self.embed(x))


class SpectralConv2d(nn.Module):
    """Truncated Fourier multiply; weights stored real, viewed complex."""

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.out_ch, self.modes = out_ch, modes
        s = 1.0 / (in_ch * out_ch)
        self.w_lo = nn.Parameter(s * torch.randn(in_ch, out_ch, modes, modes, 2))
        self.w_hi = nn.Parameter(s * torch.randn(in_ch, out_ch, modes, modes, 2))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, _, H, W = x.shape
            xf = torch.fft.rfft2(x, norm="ortho")
            m1 = min(self.modes, H // 2)
            m2 = min(self.modes, W // 2 + 1)
            lo = torch.view_as_complex(self.w_lo.contiguous())
            hi = torch.view_as_complex(self.w_hi.contiguous())
            out = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                              dtype=xf.dtype, device=x.device)
            out[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", xf[:, :, :m1, :m2], lo[:, :, :m1, :m2])
            out[:, :, -m1:, :m2] = torch.einsum(
                "bixy,ioxy->boxy", xf[:, :, -m1:, :m2], hi[:, :, :m1, :m2])
            return torch.fft.irfft2(out, s=(H, W), norm="ortho")


class SpectralNet(nn.Module):
    """Resolution-invariant encoder: works on native wafer size, no resizing.

    Pooling is masked by the wafer outline so the padding introduced by batching
    same-size wafers never contributes, and the readout is a mean over dies --
    an average, not a sum, so wafers with different die counts stay comparable.
    """

    def __init__(self, n_classes=9, width=48, modes=12, n_layers=3, in_ch=3,
                 norm="gn"):
        super().__init__()
        self.lift = nn.Conv2d(in_ch + 2, width, 1)
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(n_layers)])
        self.pw = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.nm = nn.ModuleList([_norm(norm, width) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                  nn.Linear(width, n_classes))
        self.feat_dim = width

    @staticmethod
    def coords(B, H, W, device, dtype):
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([X, Y]).expand(B, 2, H, W)

    def embed(self, x, mask=None):
        B, _, H, W = x.shape
        h = self.lift(torch.cat([x, self.coords(B, H, W, x.device, x.dtype)], 1))
        for sp, pw, nm in zip(self.spec, self.pw, self.nm):
            h = h + F.gelu(nm(sp(h) + pw(h)))
        if mask is None:
            return h.mean(dim=(-2, -1))
        m = mask.unsqueeze(1).to(h.dtype)
        return (h * m).sum(dim=(-2, -1)) / m.sum(dim=(-2, -1)).clamp_min(1.0)

    def forward(self, x, mask=None):
        return self.head(self.embed(x, mask))


class FeatMlp(nn.Module):
    """MLP over the size-invariant descriptors."""

    def __init__(self, in_dim, n_classes=9, hidden=256, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                       nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)
        self.feat_dim = hidden

    def embed(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.embed(x))


def onehot_maps(u8: torch.Tensor) -> torch.Tensor:
    """(B, H, W) uint8 in {0,1,2} -> (B, 3, H, W) float one-hot.

    One-hot rather than a single scaled channel: 0/1/2 are categories (outside /
    pass / fail), and feeding them as one ordinal channel would tell the network
    that "outside" is closer to "pass" than to "fail", which is meaningless.
    """
    x = u8.long().clamp(0, 2)
    return F.one_hot(x, 3).permute(0, 3, 1, 2).float()
