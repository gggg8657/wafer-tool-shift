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


def test_rpca_only_separates_a_lot_that_shares_something():
    """The premise behind the `rpca_cnn` fourth channel, as a unit test.

    RPCA is sold here as removing the lot's shared tool signature. It can only
    do that when there *is* a shared component: on wafers whose failures are
    independent, the low-rank part is empty and the "residual" is the input
    unchanged -- which is what happens on 94.8% of the decomposed lots in
    WM-811K, and is why the fourth channel needs the failmask control in
    scripts/ablate_sigchannel.sh.
    """
    from wts.rpca import rpca

    g = torch.Generator().manual_seed(0)
    n, d = 20, 256
    sparse = (torch.rand(n, d, generator=g) < 0.02).float()

    L, S = rpca(sparse.clone(), n_iter=60)
    assert torch.linalg.matrix_rank(L, rtol=1e-3).item() == 0
    assert torch.allclose(S, sparse, atol=1e-4)

    shared = torch.zeros(d)
    shared[:40] = 1.0                       # the "chamber's favourite corner"
    L2, S2 = rpca(sparse + shared, n_iter=60)
    assert torch.linalg.matrix_rank(L2, rtol=1e-3).item() >= 1
    assert L2[:, :40].mean() > 5 * L2[:, 40:].abs().mean()


def _tiny_corpus(n_lots=640, per_lot=3):
    """A synthetic corpus with a *known* domain structure.

    Lot number carries the signal: the first half of production runs one class
    mix and the second half another, so a production-order domain vocabulary
    must see a difference and a modulo hash of the lot id must not.

    `n_lots` has to be well above the 32 buckets for the point to exist at all:
    the hash only erases the shift once each bucket averages many lots, which is
    exactly the regime the real corpus is in (10,762 lots, ~336 per bucket).
    """
    from wts.data import Corpus

    rng = np.random.default_rng(0)
    lot, labels, maps64 = [], [], []
    for l in range(n_lots):
        late = l >= n_lots // 2
        for _ in range(per_lot):
            lot.append(l)
            labels.append(int(rng.integers(0, 3) if late else 0))
            m = np.ones((64, 64), dtype=np.uint8)
            m[rng.random((64, 64)) < (0.3 if late else 0.05)] = 2
            maps64.append(m)
    n = len(lot)
    return Corpus(
        maps=[torch.from_numpy(m) for m in maps64],
        maps64=torch.from_numpy(np.stack(maps64)),
        labels=torch.tensor(labels), lot=torch.tensor(lot),
        size_id=torch.tensor([i % 4 for i in range(n)]),
        hw=torch.zeros(n, 2, dtype=torch.long),
        lot_names=[f"lot{i}" for i in range(n_lots)],
        size_names=["a", "b", "c", "d"])


def test_hash32_domains_erase_a_shift_that_production_order_keeps():
    """The bug that made every DG objective look like ERM on the `lot` protocol.

    `batch_of` used to hand the group-aware objectives `lot % 32`. Averaging
    hundreds of lots into each bucket makes the buckets near-identical, so an
    invariance penalty is already satisfied and the objective degenerates to
    ERM. This asserts the mechanism on a corpus where the shift is known by
    construction, so a future refactor cannot quietly reintroduce it.
    """
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "rb", _Path(__file__).resolve().parents[1] / "scripts" / "run_bench.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)

    c = _tiny_corpus()

    def label_tv(dom):
        d = dom.numpy()
        p = []
        for g in np.unique(d):
            h = np.bincount(c.labels.numpy()[d == g], minlength=3).astype(float)
            p.append(h / h.sum())
        p = np.array(p)
        return float(np.mean([0.5 * np.abs(p[i] - p[j]).sum()
                              for i in range(len(p)) for j in range(i + 1, len(p))]))

    hashed, n_hashed = rb.invariance_domain(c, "lot", "hash32")
    timed, n_timed = rb.invariance_domain(c, "lot", "time_decile")

    # hash32 must stay bit-identical to the expression it replaced, or every
    # already-published cell silently changes meaning
    assert torch.equal(hashed, c.lot % 32) and n_hashed == 32

    assert n_timed == 10
    assert label_tv(timed) > 5 * label_tv(hashed)

    # domain ids must be dense, since they index GroupDRO's weights and DANN's head
    for dom, n in ((hashed, n_hashed), (timed, n_timed)):
        assert int(dom.min()) >= 0 and int(dom.max()) < n


def test_focal_at_gamma_zero_is_exactly_cross_entropy():
    """The control built into the focal sweep, asserted rather than assumed.

    If `--focal-gamma 0` did not reproduce ERM, every delta in the long-tail
    sweep would be measured against a baseline that is not the baseline, and
    the sweep would look like it had found something.
    """
    import torch.nn.functional as F

    from wts import methods

    class Identity:
        def __call__(self, x, mask=None):
            return x

    torch.manual_seed(0)
    logits = torch.randn(128, len(CLASSES))
    y = torch.randint(0, len(CLASSES), (128,))
    batch = {"x": logits, "y": y, "mask": None}

    loss0, _ = methods.focal(Identity(), batch, {"focal_gamma": 0.0})
    assert torch.allclose(loss0, F.cross_entropy(logits, y), atol=1e-7)

    # and gamma > 0 must actually change the loss, or the knob does nothing
    loss2, aux = methods.focal(Identity(), batch, {"focal_gamma": 2.0})
    assert not torch.allclose(loss2, loss0, atol=1e-4)
    assert 0.0 <= aux["focal_pt"] <= 1.0


def test_rank_correlation_helpers_match_a_known_answer():
    """`time_proxy_check` rolls its own Spearman because scipy is not in this
    environment. A bug there would move a p-value silently, and that p-value is
    the only evidence offered for the lot-numbering-is-ordered claim.
    """
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "tpc", _Path(__file__).resolve().parents[1] / "scripts"
        / "time_proxy_check.py")
    tpc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpc)

    # perfectly monotone, and perfectly anti-monotone
    x = np.array([1.0, 2, 3, 4, 5])
    assert abs(tpc.spearman(x, np.array([2.0, 4, 6, 8, 10])) - 1.0) < 1e-12
    assert abs(tpc.spearman(x, np.array([10.0, 8, 6, 4, 2])) + 1.0) < 1e-12

    # ties must be averaged, not broken by argsort order: a constant vector has
    # no ordering and must correlate with nothing
    assert abs(tpc.spearman(x, np.ones(5))) < 1e-12
    # 0-based ranks with ties averaged: 1 -> 0, the two 5s share (1+2)/2
    assert list(tpc.rankdata(np.array([5.0, 5, 1, 9]))) == [1.5, 1.5, 0.0, 3.0]

    # against a hand-computed case: Pearson of the ranks
    a = np.array([1.0, 2, 3, 4, 5, 6])
    b = np.array([2.0, 1, 4, 3, 6, 5])
    ra, rb = tpc.rankdata(a) - 2.5, tpc.rankdata(b) - 2.5
    expected = (ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    assert abs(tpc.spearman(a, b) - expected) < 1e-12

    # tv is a proper total-variation distance on distributions
    assert abs(tpc.tv(np.array([1.0, 0]), np.array([0.0, 1]))) == 1.0
    assert tpc.tv(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0


def test_pooling_control_matches_the_treatment_in_capacity():
    """`meanmean` exists to be indistinguishable from `meanmax` except in
    information. If it ever stops matching in parameter count, it stops being a
    control and any win for `meanmax` becomes a win for a wider head.
    """
    from wts.models import CnnResized

    mean = CnnResized(len(CLASSES), pool="mean")
    treat = CnnResized(len(CLASSES), pool="meanmax")
    ctrl = CnnResized(len(CLASSES), pool="meanmean")

    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(treat) == n(ctrl)
    assert treat.feat_dim == ctrl.feat_dim == 2 * mean.feat_dim

    torch.manual_seed(0)
    x = torch.randn(6, 3, 64, 64)
    # the control must carry no information the mean does not already carry
    e = ctrl.embed(x)
    half = e.shape[1] // 2
    assert torch.allclose(e[:, :half], e[:, half:])
    # the treatment must carry something the mean does not
    et = treat.embed(x)
    assert not torch.allclose(et[:, :half], et[:, half:])


def test_permutation_test_is_exact_and_symmetric():
    """`gn_vs_bn.py` rolls its own permutation test because scipy is not in this
    environment, and that p-value is the evidence for or against a named claim.
    """
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "gnbn", _Path(__file__).resolve().parents[1] / "scripts" / "gn_vs_bn.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    # identical arms cannot be distinguished from each other
    p, n = m.perm_p([1.0, 2, 3], [1.0, 2, 3])
    assert p == 1.0 and n == 20

    # perfectly separated arms hit the smallest attainable p for that size
    p, n = m.perm_p([5.0, 6, 7, 8], [1.0, 2, 3, 4])
    assert abs(p - 2.0 / n) < 1e-12

    # swapping the arms cannot change a two-sided p
    a, b = [0.9, 1.1, 1.0], [0.5, 0.7, 0.6]
    assert m.perm_p(a, b)[0] == m.perm_p(b, a)[0]

    # and the reason this sweep needs eight seeds: at three per arm the
    # smallest two-sided p is 0.1, so n=3 could never have settled it
    assert 2.0 / m.perm_p([1.0, 2, 3], [4.0, 5, 6])[1] == 0.1


def test_hiding_the_fail_plane_actually_hides_it():
    """The first `--hide-raw-fail` was void and the failure was silent.

    The one-hot is over {outside, pass, fail} and its planes sum to 1
    everywhere, so zeroing the fail plane leaves it recoverable exactly as
    `1 - ch0 - ch1`. The experiment that used it produced a clean null that
    meant nothing. This asserts the property the flag is supposed to have:
    given the remaining planes, the failed-die mask must NOT be a linear
    function of them.
    """
    from wts.rpca import stack_channels

    maps = torch.tensor([[[0, 1, 2], [1, 2, 0], [2, 0, 1]]], dtype=torch.uint8)
    fail = (maps == 2).float()
    x = stack_channels(maps, fail)

    # the bug: the original one-hot leaks the fail plane through its complement
    assert torch.allclose(x[:, 0] + x[:, 1] + x[:, 2], torch.ones_like(x[:, 0]))
    assert torch.allclose(1.0 - x[:, 0] - x[:, 1], x[:, 2])

    # the fix: outside / inside / zero / switch, where inside does not
    # distinguish passing from failing dies
    hidden = torch.stack([(maps == 0).float(), (maps > 0).float(),
                          torch.zeros_like(x[:, 2]), x[:, 3]], dim=1)
    assert torch.allclose(hidden[:, 0] + hidden[:, 1], torch.ones_like(hidden[:, 0]))
    # every linear combination of the two visible planes is constant on the
    # inside, so none of them can equal the fail mask
    inside = maps > 0
    for a_ in (-1.0, 0.0, 1.0, 2.0):
        for b_ in (-1.0, 0.0, 1.0, 2.0):
            guess = a_ * hidden[:, 0] + b_ * hidden[:, 1]
            assert not torch.allclose(guess[inside], fail[inside])


def test_permutation_tool_reports_an_empty_match_loudly():
    """A queued call asked this tool to compare pooling variants on `size` and
    it matched zero files, because the globs were hardcoded to the `lot`
    protocol and two named encoders. It printed a note and returned 0. That is
    the fourth silent no-op of this project, so the contract is now asserted:
    an empty match is a non-zero exit, not a quiet return.
    """
    import importlib.util
    import io
    import contextlib
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "gnbn2", _Path(__file__).resolve().parents[1] / "scripts" / "gn_vs_bn.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    argv = sys.argv
    try:
        sys.argv = ["gn_vs_bn.py", "--protocol", "nosuchprotocol",
                    "--arm-a", "cnn_gn:nope", "--arm-b", "cnn_bn:nope",
                    "--out", "/tmp/_perm_should_not_exist.json"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = m.main()
    finally:
        sys.argv = argv
    assert rc == 2, "an empty match must not look like success"
    assert "ERROR" in buf.getvalue()
    assert not _Path("/tmp/_perm_should_not_exist.json").exists()


def test_permutation_tool_can_test_a_per_class_metric():
    """H50 predicted a smaller p on `Scratch` than on macro-F1, and the tool
    meant to score it could only read macro-F1. A prediction whose inconvenient
    half is unreachable by the instrument stops constraining anything, so the
    per-class path is pinned here.
    """
    import importlib.util
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "gnbn3", _Path(__file__).resolve().parents[1] / "scripts" / "gn_vs_bn.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    with tempfile.TemporaryDirectory() as d:
        for enc, tag, f1, sc in (("cnn_gn", "a", 0.90, 0.70),
                                 ("cnn_bn", "b", 0.80, 0.75)):
            for seed in range(3):
                cell = {"protocol": "p", "encoder": enc, "objective": "erm",
                        "tag": tag, "seed": seed,
                        "test": {"macro_f1": f1 + 0.001 * seed,
                                 "per_class_f1": {"Scratch": sc + 0.001 * seed}}}
                _Path(d, f"p__{enc}__erm__{tag}__s{seed}.json").write_text(
                    _json.dumps(cell))

        def run(metric, out):
            argv = sys.argv
            try:
                sys.argv = ["x", "--runs", d, "--protocol", "p",
                            "--arm-a", f"cnn_gn:a", "--arm-b", f"cnn_bn:b",
                            "--metric", metric, "--out", out]
                m.main()
            finally:
                sys.argv = argv
            return _json.loads(_Path(out).read_text())

        macro = run("macro_f1", str(_Path(d, "m.json")))
        scr = run("class:Scratch", str(_Path(d, "s.json")))

    # macro-F1: arm a is higher; Scratch: arm a is lower. The tool must follow
    # the metric it was asked for rather than always reading macro-F1.
    assert macro["difference"] > 0 and scr["difference"] < 0
    assert macro["metric"] == "macro_f1" and scr["metric"] == "class:Scratch"


def test_section_census_catches_a_vanished_section():
    """The guard for the failure in critique entry 62, tested against it.

    Its first version reported "21/21 present" when the file was removed,
    because it only matched `js("literal")` and the real call site passes
    filenames through a tuple. A guard that has never been made to fail is not
    a guard.
    """
    import importlib.util
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "census", _Path(__file__).resolve().parents[1] / "scripts"
        / "section_census.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    # outputs are not inputs; everything else under runs/ is
    assert m.normalise("RESULTS.md") is None
    assert m.normalise("runs/determinism.json") == "determinism.json"
    assert m.normalise('Path(a.runs) / "x.json"'.split('"')[1]) == "x.json"

    # the tuple form that defeated version one must be found
    src = 'for fn in ("a_summary.json", "b_summary.json"):\n    d = js(fn)\n'
    with tempfile.TemporaryDirectory() as d:
        f = _Path(d, "gen.py")
        f.write_text(src)
        found = set()
        for pat in m.PATTERNS:
            for raw in pat.findall(f.read_text()):
                n = m.normalise(raw)
                if n:
                    found.add(n)
        assert found == {"a_summary.json", "b_summary.json"}
