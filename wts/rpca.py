"""Robust PCA per lot: separate the tool signature from the wafer's own defect.

Borrowed from video background subtraction, where RPCA splits a matrix into a
low-rank background and a sparse foreground. Stack the wafers of one lot as
rows and the same split has a physical reading:

* **low-rank L** — what every wafer in the lot shares. A lot runs on one tool
  through one time window, so this is the tool/recipe signature: the systematic
  edge roll-off, the chamber's favourite corner.
* **sparse S** — what this wafer alone does. The defect the classifier is
  supposed to name.

That makes RPCA a *domain normalization* rather than a feature extractor: the
component being removed is precisely the nuisance the lot-disjoint protocol
punishes a model for memorizing. Two products come out of it, and the benchmark
uses both --

1. `residual_maps`, wafers with their lot signature subtracted, which any
   encoder can consume in place of the raw maps;
2. `signature_features`, a short description of the removed L, because "which
   tool signature is this" is itself useful when the label is a tool problem.

Solved with singular-value thresholding (inexact ALM). A lot holds at most 25
wafers, so each SVD is 25 x 4096 and the whole corpus takes seconds on a GPU.
"""
from __future__ import annotations

import torch


def svt(M: torch.Tensor, tau: float) -> torch.Tensor:
    """Singular-value thresholding: the proximal operator of the nuclear norm."""
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U * (S - tau).clamp_min(0)) @ Vh


def rpca(M: torch.Tensor, lam: float | None = None, n_iter: int = 60,
         tol: float = 1e-6):
    """M ~= L + S with L low-rank and S sparse (inexact ALM).

    `lam` defaults to the standard 1/sqrt(max(n, d)) choice, which is what makes
    the decomposition parameter-free in practice.
    """
    n, d = M.shape
    lam = lam if lam is not None else 1.0 / max(n, d) ** 0.5
    norm = torch.linalg.matrix_norm(M, 2).clamp_min(1e-8)
    mu = 1.25 / norm
    Y = M / max(float(norm), 1.0)
    L = torch.zeros_like(M)
    S = torch.zeros_like(M)
    m_norm = torch.linalg.matrix_norm(M, "fro").clamp_min(1e-8)
    for _ in range(n_iter):
        L = svt(M - S + Y / mu, 1.0 / mu)
        T = M - L + Y / mu
        S = torch.sign(T) * (T.abs() - lam / mu).clamp_min(0)   # soft threshold
        R = M - L - S
        Y = Y + mu * R
        mu = mu * 1.05
        if torch.linalg.matrix_norm(R, "fro") / m_norm < tol:
            break
    return L, S


@torch.no_grad()
def lot_decomposition(maps64: torch.Tensor, lot: torch.Tensor, device="cuda",
                      min_wafers: int = 12, n_iter: int = 30):
    """Per-lot RPCA over the whole corpus.

    Returns (residual, signature) where `residual` is the sparse part reshaped
    back to maps (float32, mean-zero-ish) and `signature` is a 6-number summary
    of the removed low-rank part per wafer. Lots with fewer than `min_wafers`
    wafers are passed through unchanged -- a rank-1 fit to two wafers would
    simply delete the defect.
    """
    N, H, W = maps64.shape
    fail = (maps64 == 2).float()
    inside = (maps64 > 0).float()
    resid = torch.zeros(N, H, W, dtype=torch.float32)
    sig = torch.zeros(N, 6, dtype=torch.float32)
    order = torch.argsort(lot)
    lots_sorted = lot[order]
    bounds = torch.nonzero(
        torch.cat([torch.tensor([True]), lots_sorted[1:] != lots_sorted[:-1]])
    ).flatten().tolist() + [N]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        F = fail[idx].reshape(len(idx), -1).to(device)
        if len(idx) < min_wafers:
            resid[idx] = fail[idx]
            continue
        L, S = rpca(F, n_iter=n_iter)
        resid[idx] = S.reshape(-1, H, W).cpu()
        lo = L.reshape(-1, H, W)
        rank = torch.linalg.matrix_rank(L.float(), rtol=1e-3).item()
        sig[idx] = torch.stack([
            lo.mean(dim=(-2, -1)),                       # signature strength
            lo.abs().amax(dim=(-2, -1)),                 # peak
            lo.std(dim=(-2, -1)),                        # spatial spread
            S.abs().sum(1) / F.abs().sum(1).clamp_min(1e-6),  # sparse share
            torch.full((len(idx),), float(rank), device=device),
            torch.full((len(idx),), float(len(idx)), device=device),
        ], dim=1).cpu()
    return resid, sig


def stack_channels(maps64: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """(B,H,W) uint8 + (B,H,W) residual -> (B,4,H,W): one-hot state + residual.

    The residual is handed to the encoder *alongside* the raw state rather than
    instead of it, so the model can still see the wafer outline and the pass/fail
    encoding; only the lot-shared component has been pulled into its own channel.
    """
    import torch.nn.functional as F
    x = F.one_hot(maps64.long().clamp(0, 2), 3).permute(0, 3, 1, 2).float()
    return torch.cat([x, resid.unsqueeze(1)], dim=1)
