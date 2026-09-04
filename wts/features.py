"""Wafer-map descriptors borrowed from other fields, all size-invariant.

The conventional pipeline resizes every wafer to a fixed grid and feeds a CNN.
That throws away the one thing this corpus has 344 of -- geometry -- and makes
the model's features a function of the resampling. Each descriptor here maps a
wafer of *any* shape to a fixed-length vector, so one model spans every
geometry, and the "unseen geometry" protocol stops being a resizing artifact.

Where each one comes from, and why it should work on a wafer:

| descriptor            | borrowed from                  | why it fits a wafer                                |
|-----------------------|--------------------------------|----------------------------------------------------|
| `zernike`             | optics / wavefront sensing     | the orthogonal basis *on a disk*; already the language lithography uses for aberration, and magnitudes are rotation-invariant |
| `betti`               | topological data analysis      | Donut is literally a 1-dimensional homology class; Scratch is a 1-D structure, Center a blob |
| `radial_angular`      | astronomy (surface photometry) | edge-ring vs center is a radial statement; sectors catch localized failure |
| `variogram`           | geostatistics / mining         | separates systematic spatial correlation from random speckle without any model |
| `power_spectrum`      | signal processing / PDE operators | normalizing frequency by Nyquist makes the descriptor grid-independent by construction |
| `moments`             | classical image analysis       | cheap anchors: fail rate, centroid radius, eccentricity, edge-band concentration |

Encoding convention throughout: 0 = outside the wafer, 1 = passing die,
2 = failing die.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def masks(a: np.ndarray):
    """(inside-wafer, failing) boolean masks."""
    return a > 0, a == 2


@lru_cache(maxsize=512)
def _polar_grid(h: int, w: int):
    """Normalized radius and angle for an h x w map, cached per geometry."""
    y = (np.arange(h) - (h - 1) / 2) / max((h - 1) / 2, 1e-9)
    x = (np.arange(w) - (w - 1) / 2) / max((w - 1) / 2, 1e-9)
    Y, X = np.meshgrid(y, x, indexing="ij")
    return np.sqrt(X**2 + Y**2), np.arctan2(Y, X)


# --------------------------------------------------------------------------- #
# optics: Zernike moments on the disk
# --------------------------------------------------------------------------- #
def _zernike_orders(n_max: int):
    return [(n, m) for n in range(n_max + 1)
            for m in range(-n, n + 1) if (n - abs(m)) % 2 == 0]


def _radial_poly(n: int, m: int, r: np.ndarray) -> np.ndarray:
    m = abs(m)
    out = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k)
             / (math.factorial(k) * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)))
        out += c * r ** (n - 2 * k)
    return out


@lru_cache(maxsize=256)
def _zernike_basis(h: int, w: int, n_max: int):
    """Basis evaluated on the unit disk of an h x w grid, cached per geometry."""
    r, th = _polar_grid(h, w)
    disk = r <= 1.0
    basis = []
    for n, m in _zernike_orders(n_max):
        R = _radial_poly(n, m, np.clip(r, 0, 1))
        z = R * (np.cos(m * th) if m >= 0 else np.sin(-m * th))
        basis.append(np.where(disk, z, 0.0))
    return np.stack(basis), disk


def zernike(a: np.ndarray, n_max: int = 8) -> np.ndarray:
    """Rotation-invariant Zernike magnitudes of the failure field.

    Pairs (n, m) and (n, -m) are combined into one magnitude, which removes the
    wafer's arbitrary rotational placement -- Edge-Ring should score the same
    whichever way the wafer sat in the cassette. Scratch orientation *is*
    information, so the anisotropy of the m != 0 terms is kept as a separate
    summary rather than discarded.
    """
    inside, fail = masks(a)
    h, w = a.shape
    B, disk = _zernike_basis(h, w, n_max)
    f = fail.astype(np.float64)
    area = max(disk.sum(), 1)
    coef = (B.reshape(len(B), -1) @ f.reshape(-1)) / area
    orders = _zernike_orders(n_max)
    mag, seen = [], {}
    for (n, m), c in zip(orders, coef):
        seen.setdefault((n, abs(m)), []).append(c)
    for key in sorted(seen):
        v = seen[key]
        mag.append(math.sqrt(sum(x * x for x in v)))
    aniso = sum(abs(c) for (n, m), c in zip(orders, coef) if m != 0)
    return np.asarray(mag + [aniso], dtype=np.float32)


# --------------------------------------------------------------------------- #
# TDA: Betti curves over a distance filtration
# --------------------------------------------------------------------------- #
def _euler_characteristic(S: np.ndarray) -> int:
    """chi = V - E + F on the cubical complex of a binary mask (4-connectivity).

    Cheap and exact, which is what makes beta_1 available without a persistent
    homology library: in 2-D, chi = beta_0 - beta_1.
    """
    V = int(S.sum())
    E = int((S[:, :-1] & S[:, 1:]).sum() + (S[:-1] & S[1:]).sum())
    F = int((S[:-1, :-1] & S[:-1, 1:] & S[1:, :-1] & S[1:, 1:]).sum())
    return V - E + F


_C4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def betti(a: np.ndarray, n_steps: int = 10) -> np.ndarray:
    """Betti curves of the failure *density*, not of the raw die mask.

    Filtering the binary mask directly does not work: partially-failing regions
    are full of single-die holes, so every class shows a dozen spurious loops
    and beta_1 carries noise instead of shape. The fix is the standard TDA
    construction for images -- take superlevel sets of a smooth function. Here
    the function is the local failure rate (Gaussian-smoothed at a fraction of
    the wafer radius, so the smoothing scales with geometry), and the levels are
    quantiles of that density, which makes the curve comparable across wafers
    with very different overall fail rates.

    Reading it: Donut holds beta_1 = 1 across many levels (a ring of high
    density around a low-density middle), Center is one component with no loop,
    Random starts as many components that merge, Edge-Ring is an annulus that
    also loops but sits at the rim -- which is why these features are used
    alongside the radial profile rather than instead of it.

    Prior art for TDA on this corpus: arXiv:2209.08945.
    """
    inside, fail = masks(a)
    n_fail = int(fail.sum())
    if n_fail == 0:
        return np.zeros(2 * n_steps, dtype=np.float32)
    radius = math.hypot(*a.shape) / 2
    sigma = max(0.06 * radius, 0.8)
    dens = ndimage.gaussian_filter(fail.astype(np.float64), sigma)
    dens[~inside] = 0.0
    vals = dens[inside]
    # quantile levels, descending: a superlevel set grows as the level drops
    qs = np.quantile(vals, np.linspace(0.98, 0.55, n_steps))
    b0, b1 = [], []
    for t in qs:
        S = (dens >= t) & inside
        n0 = ndimage.label(S, structure=_C4)[1]
        b0.append(n0)
        b1.append(max(n0 - _euler_characteristic(S), 0))
    return np.asarray(b0 + b1, dtype=np.float32)


# --------------------------------------------------------------------------- #
# astronomy: radial and angular profile
# --------------------------------------------------------------------------- #
def radial_angular(a: np.ndarray, n_r: int = 6, n_a: int = 8) -> np.ndarray:
    """Failure rate per radial ring x angular sector, plus two global rates.

    Rings answer "edge or center", sectors answer "one side or all round". Rates
    rather than counts, so a 25x27 wafer and a 53x58 wafer are comparable.
    """
    inside, fail = masks(a)
    r, th = _polar_grid(*a.shape)
    ri = np.clip((r * n_r).astype(int), 0, n_r - 1)
    ai = np.clip(((th + math.pi) / (2 * math.pi) * n_a).astype(int), 0, n_a - 1)
    key = ri * n_a + ai
    tot = np.bincount(key[inside], minlength=n_r * n_a).astype(np.float64)
    bad = np.bincount(key[inside & fail], minlength=n_r * n_a).astype(np.float64)
    rate = np.divide(bad, tot, out=np.zeros_like(bad), where=tot > 0)
    overall = fail.sum() / max(inside.sum(), 1)
    edge = r > 0.8
    edge_rate = (fail & edge & inside).sum() / max((edge & inside).sum(), 1)
    return np.concatenate([rate, [overall, edge_rate]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# geostatistics: variogram
# --------------------------------------------------------------------------- #
def variogram(a: np.ndarray, lags: int = 8) -> np.ndarray:
    """Semivariance of the failure indicator against lag, in wafer-radius units.

    A systematic pattern (tool signature) has variance that keeps growing with
    distance; random speckle saturates immediately. This is the mining
    industry's tool for exactly that distinction and needs no model fitted.
    """
    inside, fail = masks(a)
    f = fail.astype(np.float64)
    out = []
    for d in range(1, lags + 1):
        num = den = 0.0
        for ax in (0, 1):
            va = np.roll(f, -d, axis=ax)
            vm = inside & np.roll(inside, -d, axis=ax)
            if ax == 0:
                vm[-d:, :] = False
            else:
                vm[:, -d:] = False
            num += (((f - va) ** 2)[vm]).sum()
            den += vm.sum()
        out.append(0.5 * num / max(den, 1))
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# signal processing: radial power spectrum, normalized by Nyquist
# --------------------------------------------------------------------------- #
def power_spectrum(a: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """Radially binned 2-D power spectrum of the failure mask.

    Binning |k| as a *fraction of Nyquist* is what makes this grid-independent:
    the same physical pattern on a 33-die and a 58-die wafer lands in the same
    bin. Same reasoning as a spectral neural operator -- act on coefficients,
    not on samples.
    """
    inside, fail = masks(a)
    f = fail.astype(np.float64) - fail.sum() / max(inside.sum(), 1) * inside
    F = np.abs(np.fft.fft2(f)) ** 2
    h, w = a.shape
    ky = np.fft.fftfreq(h)[:, None] * 2      # -> [-1, 1] in Nyquist units
    kx = np.fft.fftfreq(w)[None, :] * 2
    k = np.sqrt(ky**2 + kx**2) / math.sqrt(2)
    ki = np.clip((k * n_bins).astype(int), 0, n_bins - 1)
    tot = np.bincount(ki.ravel(), weights=F.ravel(), minlength=n_bins)
    cnt = np.bincount(ki.ravel(), minlength=n_bins)
    ps = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    s = ps.sum()
    return (ps / s if s > 0 else ps).astype(np.float32)


# --------------------------------------------------------------------------- #
# classical moments
# --------------------------------------------------------------------------- #
def moments(a: np.ndarray) -> np.ndarray:
    inside, fail = masks(a)
    n_in = max(inside.sum(), 1)
    r, _ = _polar_grid(*a.shape)
    if fail.sum() == 0:
        return np.zeros(8, dtype=np.float32)
    ys, xs = np.nonzero(fail)
    cy, cx = ys.mean(), xs.mean()
    h, w = a.shape
    cov = np.cov(np.stack([ys - cy, xs - cx])) if fail.sum() > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov + 1e-9 * np.eye(2)))[::-1]
    lab, n_cc = ndimage.label(fail, structure=_C4)
    big = np.bincount(lab.ravel())[1:].max() if n_cc else 0
    return np.asarray([
        fail.sum() / n_in,                                   # fail rate
        r[fail].mean(),                                      # centroid radius
        r[fail].std(),                                       # radial spread
        math.sqrt(max(1 - ev[1] / max(ev[0], 1e-9), 0)),     # eccentricity
        n_cc / max(fail.sum(), 1),                           # fragmentation
        big / max(fail.sum(), 1),                            # largest-blob share
        inside.sum() / (h * w),                              # wafer coverage
        min(h, w) / max(h, w),                               # aspect ratio
    ], dtype=np.float32)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
BLOCKS = {
    "zernike": (zernike, 26),
    "betti": (betti, 20),
    "radial_angular": (radial_angular, 50),
    "variogram": (variogram, 8),
    "power_spectrum": (power_spectrum, 12),
    "moments": (moments, 8),
}
FEATURE_DIM = sum(d for _, d in BLOCKS.values())


def descriptor(a: np.ndarray, blocks=None) -> np.ndarray:
    """Concatenated descriptor for one wafer, independent of its size."""
    names = blocks or list(BLOCKS)
    return np.concatenate([BLOCKS[n][0](a) for n in names])


def block_slices(blocks=None):
    """{name: slice} into the concatenated vector, for ablations."""
    names = blocks or list(BLOCKS)
    out, o = {}, 0
    for n in names:
        d = BLOCKS[n][1]
        out[n] = slice(o, o + d)
        o += d
    return out
