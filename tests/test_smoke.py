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


def test_duplicate_paragraph_detector_catches_a_section_rendered_twice():
    """The guard for critique entry 67, tested against the defect that caused it.

    `weekend.py` section 2.0 rendered four times for two commits. The cause was
    one indentation slip: the section body sat inside the `for proto in (...)`
    loop that was only meant to populate its inputs. Nothing detectable was
    wrong -- every number came from `runs/`, `section_census.py` saw all inputs
    present, `coverage_check.py` saw every JSON consumed. Those guards ask about
    content; this was a defect in shape, and the document was 145 lines where it
    should have been 41.

    `RESULTS.md` turned out to have the same defect independently, printing one
    caveat three times, which is how the detector earned its place rather than
    merely passing a test written for it.
    """
    import importlib.util
    import tempfile
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "lint", _Path(__file__).resolve().parents[1] / "scripts"
        / "prose_status_lint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    para = ("A paragraph long enough to count as prose rather than as a "
            "heading, which is the whole distinction the detector draws and "
            "so has to be exercised at realistic length.")
    with tempfile.TemporaryDirectory() as d:
        clean = _Path(d, "clean.md")
        clean.write_text(f"# t\n\n{para}\n\nSomething else entirely here, and "
                         "also long enough to clear the minimum length bar.\n")
        assert m.duplicated_blocks(clean) == [], "false positive on a clean doc"

        dirty = _Path(d, "dirty.md")
        dirty.write_text(f"# t\n\n{para}\n\n{para}\n\n{para}\n")
        assert len(m.duplicated_blocks(dirty)) == 2, "missed a tripled paragraph"

        # tables legitimately repeat rows across protocols; they are not prose
        rows = "| a | b |\n|---|---|\n| 1 | 2 |"
        tbl = _Path(d, "tbl.md")
        tbl.write_text(f"# t\n\n{rows}\n\n{rows}\n")
        assert m.duplicated_blocks(tbl) == [], "flagged a repeated table"

        # short repeated lines (headers, labels) are not prose either
        short = _Path(d, "short.md")
        short.write_text("# t\n\n**Verdict.**\n\n**Verdict.**\n")
        assert m.duplicated_blocks(short) == [], "flagged a repeated label"


def test_scale_aware_stem_is_exact_at_dilation_one_and_adds_no_parameters():
    """The controls for the scale-aware encoder, before it is ever trained.

    Measured with no model (`scripts/pooling_mechanism.py`): nearest-neighbour
    upsampling makes a one-die scratch arrive as a band ~64/w pixels wide, and
    native geometry then explains 0.7689 of the variance in a max-pooled line
    filter response against 0.1485 of the mean-pooled one. Correcting after the
    convolution does not work (0.7511); dilating the filter by round(64/w) does
    (0.0361).

    Two properties have to hold for the trained comparison to mean anything:

    1. Dilation adds no parameters, so `scale_aware` is compared against
       `meanmax` at identical capacity -- it is its own capacity control, the
       way `meanmean` is for `meanmax`.
    2. At w = 64 the dilation is 1 and the path must be *bit-identical* to the
       unscaled one, so any difference measured on real wafers comes from the
       geometries that are actually upsampled and not from a changed code path.

    The third check is for the grouping itself: samples are batched by dilation
    value and scattered back, and a bug there would make a wafer's output depend
    on who else is in its batch.
    """
    import torch as _t

    from wts.models import CnnResized

    _t.manual_seed(0)
    plain = CnnResized(pool="meanmax", scale_aware=False)
    scaled = CnnResized(pool="meanmax", scale_aware=True)
    scaled.load_state_dict(plain.state_dict())
    plain.eval()
    scaled.eval()

    assert (sum(p.numel() for p in plain.parameters())
            == sum(p.numel() for p in scaled.parameters())), \
        "dilation must not change the parameter count"

    x = _t.randn(8, 3, 64, 64)
    hw_native = _t.tensor([[64, 64], [32, 32], [26, 26], [45, 48],
                           [64, 64], [32, 32], [26, 26], [45, 48]])

    with _t.no_grad():
        hw_one = _t.full((8, 2), 64)
        assert _t.equal(plain(x), scaled(x, hw_one)), \
            "at dilation 1 the scale-aware path must be bit-identical"

        # on real geometries it must actually differ, or the flag does nothing
        assert not _t.allclose(plain(x), scaled(x, hw_native), atol=1e-6), \
            "mixed geometries must change the output"

        # batching by dilation must not make a wafer depend on its batchmates
        perm = _t.randperm(8)
        assert _t.equal(scaled(x, hw_native)[perm],
                        scaled(x[perm], hw_native[perm])), \
            "output must not depend on batch order"


def test_sign_flip_test_is_exact_and_matches_independent_enumeration():
    """The routine behind the headline family result, which had no test.

    `dg_family_test.py` reports that the six borrowed DG objectives are worse
    than ERM in aggregate at p = 0.00781, from an exact two-sided sign-flip
    test on eight per-seed paired differences. That number is the strongest
    negative claim in the project and it rested on a function written in one
    sitting and never checked -- exactly the gap `guard_audit.py` exists to
    close, left open in the same session that built it.

    Four properties, three of which have closed forms:

      * identical arms give p = 1, not p = 0;
      * n differences all of one sign give 2 / 2^n, the floor;
      * flipping every input sign changes nothing, since the test is two-sided;
      * and on random vectors it agrees with a brute-force enumeration written
        independently of the implementation.
    """
    import importlib.util
    import itertools
    import random
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "fam", _Path(__file__).resolve().parents[1] / "scripts"
        / "dg_family_test.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert m.sign_flip_p([0.0] * 8) == (1.0, 256), "no difference is not p=0"
    assert m.sign_flip_p([-0.01] * 8) == (2 / 256, 256), "floor at n=8"
    assert m.sign_flip_p([-0.01] * 4) == (2 / 16, 16), "floor at n=4"
    assert m.sign_flip_p([1, -1, 2, -2, 3, -3, 4, -4])[0] == 1.0

    d = [-0.03, -0.01, -0.02, 0.004, -0.011, -0.007, -0.02, -0.001]
    assert m.sign_flip_p(d) == m.sign_flip_p([-v for v in d]), \
        "a two-sided test must ignore the overall sign"

    def brute(diffs):
        obs = abs(sum(diffs))
        n = len(diffs)
        return sum(1 for s in itertools.product((1, -1), repeat=n)
                   if abs(sum(x * y for x, y in zip(s, diffs)))
                   >= obs - 1e-15) / 2 ** n

    rng = random.Random(0)
    for _ in range(50):
        v = [rng.gauss(0, 1) for _ in range(7)]
        assert abs(m.sign_flip_p(v)[0] - brute(v)) < 1e-12


def test_floors_refuses_a_floor_measured_from_too_few_repeats():
    """A range over one invocation is not a range, and zero is not a floor.

    `determinism_repeats.sh` wrote `determinism__iid__cnn_gn.json` with
    `n_repeats: 1` after two stages ended up sharing the GPU lease and the
    other five repeats were killed. Its `range` was 0.0000, and `floors()`
    served that as the `iid` threshold, so every `iid` margin cleared the floor
    and the screen was vacuous on that protocol.

    The failure direction matters. A *missing* floor falls back to the largest
    measured anywhere, which is conservative: it withdraws claims that might be
    true. A floor of zero does the opposite, and publishes claims that are not.
    """
    import importlib.util
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "rep", _Path(__file__).resolve().parents[1] / "scripts" / "report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    with tempfile.TemporaryDirectory() as d:
        _Path(d, "determinism__good__cnn_gn.json").write_text(
            _json.dumps({"range": 0.0100, "n_repeats": 6}))
        _Path(d, "determinism__thin__cnn_gn.json").write_text(
            _json.dumps({"range": 0.0000, "n_repeats": 1}))
        F = m.floors(d)

        assert "good" in F, "a properly measured floor must be used"
        assert "thin" not in F, "a one-repeat 'range' must not become a floor"
        assert m.floors.rejected["thin"]["n_repeats"] == 1
        # and the fallback must remain the largest *measured* one
        assert F["_fallback"] == 0.0100

        # the point of the rule: an unmeasured protocol is screened
        # conservatively, never trivially
        assert m.floor_for(F, "thin") == 0.0100
        assert m.floor_for(F, "thin") > 0.0


def test_check_all_reports_failure_when_any_check_fails():
    """The aggregator must not swallow a failing check.

    `check_all.py` exists because reading six separate check outputs is how the
    typed-decimal ratchet got past me: I read the `ok` from my own patch script
    instead of the guard's line and committed with the check red. An aggregator
    that reported PASS while a member failed would be strictly worse than the
    six tails it replaces, since it would carry more authority.

    Also asserts the non-strict path: `guard_audit.py` is informational (one
    known-vacuous entry) and must not fail the run, or the whole thing goes
    permanently red -- which is how a check stops being read at all.
    """
    import importlib.util
    import sys as _sys
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "chk", _Path(__file__).resolve().parents[1] / "scripts"
        / "check_all.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    ok_cmd = [_sys.executable, "-c", "print('fine')"]
    bad_cmd = [_sys.executable, "-c", "import sys; sys.exit(3)"]

    rows = m.run([("a passing check", ok_cmd, True),
                  ("a failing check", bad_cmd, True)])
    assert [r[1] for r in rows] == [True, False], "a nonzero exit must be FAIL"
    assert rows[1][2] == 3, "the exit code must be preserved for the report"

    # a check marked non-strict must be reported as PASS even when nonzero,
    # and its output must still be captured so the summary can quote it
    rows = m.run([("an informational check", bad_cmd, False)])
    assert rows[0][1] is True, "non-strict checks must not fail the run"

    # and the real check list must name questions, not scripts
    assert all("?" in q for q, _, _ in m.CHECKS), \
        "each check should state the question it answers"


def test_section_diff_catches_a_vanished_section_and_tolerates_growth():
    """The guard for critique entry 87, tested against the defect that caused it.

    Section 7.9 of `paper_draft.md` was guarded by
    `if npa and npa.get("n_underpowered")`. When the last underpowered null was
    fixed that count went to zero and the whole section disappeared, deleting
    the evidence the problem had been solved. All six existing checks passed on
    the shorter paper, because every one of them reasons about inputs and
    generator source rather than about the rendered document.

    Three properties. A removed heading must be reported; an added one must not
    fail, since documents are meant to grow; and a heading whose title contains
    a measured number must not read as removed when that number changes, or the
    guard cries wolf on every regeneration and stops being read -- the failure
    mode that made `number_provenance`'s traceability half worthless.
    """
    import importlib.util
    import tempfile
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "sd", _Path(__file__).resolve().parents[1] / "scripts"
        / "section_diff.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    with tempfile.TemporaryDirectory() as d:
        p = _Path(d, "doc.md")
        p.write_text("# Title\n\n## 7.9 How much each null could have shown\n\n"
                     "text\n\n## 8. Threats\n\ntext\n")
        before = m.headings(p)

        # the real failure: a section stops rendering
        p.write_text("# Title\n\n## 8. Threats\n\ntext\n")
        removed, added = m.compare(before, m.headings(p))
        assert len(removed) == 1 and "7.9" in removed[0], removed
        assert added == []

        # growth must not be an error
        p.write_text("# Title\n\n## 7.9 How much each null could have shown\n\n"
                     "t\n\n## 8. Threats\n\nt\n\n## 9. New section\n\nt\n")
        removed, added = m.compare(before, m.headings(p))
        assert removed == [] and len(added) == 1

        # a *measured* value in a heading must not look like a removal when
        # it moves, but a section number must still be part of its identity --
        # the first version stripped every digit, which made `7.9 Foo` and
        # `8.1 Foo` the same heading and hid renumbering entirely
        q = _Path(d, "num.md")
        q.write_text("## Result at p = 0.00781\n")
        b2 = m.headings(q)
        q.write_text("## Result at p = 0.02344\n")
        removed, added = m.compare(b2, m.headings(q))
        assert removed == [] and added == [], (removed, added)

        q.write_text("## 7.9 Power\n")
        b3 = m.headings(q)
        q.write_text("## 8.1 Power\n")
        removed, added = m.compare(b3, m.headings(q))
        assert len(removed) == 1 and len(added) == 1, "renumbering must show"
