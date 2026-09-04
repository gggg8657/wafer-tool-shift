"""CPU checks on the protocols, the descriptors and the encoders."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts import features, metrics
from wts.data import CLASSES, Corpus, label_shift, split
from wts.models import CnnResized, FeatMlp, SpectralNet, onehot_maps


def _synth(kind, h=45, w=48):
    """A wafer with a known signature, used to check the descriptors."""
    y = (np.arange(h) - (h - 1) / 2) / ((h - 1) / 2)
    x = (np.arange(w) - (w - 1) / 2) / ((w - 1) / 2)
    Y, X = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    a = np.where(r <= 1, 1, 0).astype(np.uint8)
    if kind == "donut":
        a[(r > 0.45) & (r < 0.7)] = 2
    elif kind == "center":
        a[r < 0.3] = 2
    elif kind == "edge":
        a[(r > 0.85) & (r <= 1)] = 2
    return a


def test_descriptors_are_size_invariant():
    """The same continuous pattern on two grids must give a close descriptor."""
    d1 = features.descriptor(_synth("edge", 45, 48))
    d2 = features.descriptor(_synth("edge", 70, 74))
    sl = features.block_slices()
    for name in ("radial_angular", "power_spectrum", "moments"):
        a, b = d1[sl[name]], d2[sl[name]]
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-8)
        assert rel < 0.35, f"{name} not size-invariant: {rel:.3f}"


def test_donut_has_a_persistent_loop():
    """The topological claim: only Donut keeps beta_1 alive across levels."""
    sl = features.block_slices()["betti"]
    out = {}
    for kind in ("donut", "center", "edge"):
        b = features.descriptor(_synth(kind))[sl]
        n = len(b) // 2
        out[kind] = (b[n:] >= 1).mean()
    assert out["donut"] > out["center"], out
    assert out["donut"] > 0.2, out


def test_zernike_is_rotation_invariant():
    a = _synth("center")
    z1 = features.zernike(a)
    z2 = features.zernike(np.rot90(a).copy())
    rel = np.linalg.norm(z1 - z2) / max(np.linalg.norm(z1), 1e-8)
    assert rel < 0.15, f"zernike moved under rotation: {rel:.3f}"


def test_spectral_encoder_accepts_any_resolution():
    m = SpectralNet(len(CLASSES), width=16, modes=6, n_layers=2)
    for h, w in ((25, 27), (45, 48), (53, 58)):
        x = onehot_maps(torch.randint(0, 3, (2, h, w), dtype=torch.uint8))
        assert m(x).shape == (2, len(CLASSES))


def test_encoders_run():
    x = onehot_maps(torch.randint(0, 3, (4, 64, 64), dtype=torch.uint8))
    assert CnnResized(len(CLASSES), width=8)(x).shape == (4, len(CLASSES))
    assert FeatMlp(features.FEATURE_DIM, len(CLASSES), hidden=32)(
        torch.randn(4, features.FEATURE_DIM)).shape == (4, len(CLASSES))


def test_worst_group_is_not_the_mean():
    y = np.array([0, 1] * 20)
    pred = y.copy()
    pred[:12] = 0                       # one domain is broken
    g = np.array([0] * 20 + [1] * 20)
    worst, n = metrics.worst_group(y, pred, g, min_n=12)
    assert n == 2 and worst < 0.9


def test_conformal_coverage_is_reported():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(len(CLASSES)), size=400)
    y = p.argmax(1)
    out = metrics.summarize(y, p, groups=np.zeros(400), cal=(p, y))
    assert 0.0 <= out["conformal_coverage"] <= 1.0
    assert "worst_domain_macro_f1" in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok ", name)
