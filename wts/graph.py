"""A graph neural network over dies, which is not the same thing as a CNN.

A convolution on the padded rectangle sees the region outside the wafer as
zeros and happily convolves across the boundary. A graph built on the dies
themselves cannot: nodes exist only where there is silicon, edges only between
neighbouring dies, and degree normalization means an edge die -- with three
neighbours instead of four -- is treated as an edge die rather than as a die
with one dead neighbour.

That distinction is the whole point for `Edge-Ring` and `Edge-Loc`, the two
classes defined by their relationship to the wafer boundary, and it is why this
encoder is worth having next to the CNN rather than instead of it. It is also
resolution-agnostic: the graph is whatever the wafer's die grid is.

Message passing is implemented with masked shifts rather than a sparse library,
which keeps it dependency-free and fast on a GPU: for a grid graph the neighbour
sum is four masked `roll`s.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_neighbor_sum(h: torch.Tensor, mask: torch.Tensor):
    """Sum over 4-neighbours that are *inside the wafer*, plus the degree.

    h: (B, C, H, W), mask: (B, H, W) bool. Shifts are zeroed at the array edge
    so the wafer never wraps around.
    """
    m = mask.unsqueeze(1).to(h.dtype)
    hm = h * m
    out = h.new_zeros(h.shape)
    deg = h.new_zeros(m.shape)
    for dim, shift in ((-2, 1), (-2, -1), (-1, 1), (-1, -1)):
        hs = torch.roll(hm, shift, dims=dim)
        ms = torch.roll(m, shift, dims=dim)
        if dim == -2:
            if shift == 1:
                hs[..., 0, :] = 0; ms[..., 0, :] = 0
            else:
                hs[..., -1, :] = 0; ms[..., -1, :] = 0
        else:
            if shift == 1:
                hs[..., 0] = 0; ms[..., 0] = 0
            else:
                hs[..., -1] = 0; ms[..., -1] = 0
        out = out + hs
        deg = deg + ms
    return out, deg


class DieGraphLayer(nn.Module):
    """One degree-normalized message-passing step on the die graph."""

    def __init__(self, ch):
        super().__init__()
        self.self_lin = nn.Conv2d(ch, ch, 1)
        self.nbr_lin = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(8, ch), ch)

    def forward(self, h, mask):
        nbr, deg = grid_neighbor_sum(h, mask)
        nbr = nbr / deg.clamp_min(1.0)
        out = self.self_lin(h) + self.nbr_lin(nbr)
        return h + F.gelu(self.norm(out)) * mask.unsqueeze(1).to(h.dtype)


class DieGraphNet(nn.Module):
    """Node features = die state + normalized position + boundary degree.

    The degree channel is what a CNN cannot see: it tells every node how much of
    the wafer surrounds it, so "on the rim" is available as a feature instead of
    something the model must infer from padded zeros.
    """

    def __init__(self, n_classes=9, width=48, n_layers=4, in_ch=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch + 3, width, 1)
        self.layers = nn.ModuleList([DieGraphLayer(width) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                  nn.Linear(width, n_classes))
        self.feat_dim = 2 * width

    def embed(self, x, mask=None):
        if mask is None:
            mask = x[:, 1:].sum(1) > 0
        B, _, H, W = x.shape
        y = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        _, deg = grid_neighbor_sum(x[:, :1], mask)
        h = self.lift(torch.cat([x, xx, y, deg / 4.0], dim=1))
        for lay in self.layers:
            h = lay(h, mask)
        m = mask.unsqueeze(1).to(h.dtype)
        denom = m.sum(dim=(-2, -1)).clamp_min(1.0)
        mean = (h * m).sum(dim=(-2, -1)) / denom
        # a max readout as well: a scratch is a few dies, and a mean over a
        # thousand dies averages it into nothing
        mx = (h.masked_fill(m == 0, float("-inf"))).amax(dim=(-2, -1))
        mx = torch.nan_to_num(mx, neginf=0.0)
        return torch.cat([mean, mx], dim=1)

    def forward(self, x, mask=None):
        return self.head(self.embed(x, mask))
