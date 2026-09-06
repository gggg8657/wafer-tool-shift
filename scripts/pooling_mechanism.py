"""Test the mechanism claimed for the pooling result, with no model involved.

`meanmax` pooling improves `Scratch` F1 on every protocol where test wafers
share geometry with training (`iid` p=0.00093, `lot` 0.00047, `lot_time`
0.00031) and fails on `size`, which holds geometry out. Three documents explain
that with one sentence:

    "A max over a resampled feature map depends on the die density of the wafer
     it came from, so the statistic that recovers a thin structure is the one
     that does not transfer across geometry."

That is an explanation, not a measurement, and this project has spent the
weekend on the cost of those. It is also testable without a trained network,
because the claim is about the pooling operators and the resampling, not about
anything learned.

`resize_nearest` upsamples by *indexing*, so each native die becomes a block of
roughly 64/w pixels wide, and 97.7% of wafers are upsampled. A one-die-wide
scratch therefore arrives at the encoder as a band whose pixel width varies
across geometries by the ratio of their native widths. A fixed-size filter's
peak response should track that width; the mean of the same response is close
to the failure fraction, which nearest-neighbour replication preserves.

H65, before the run: for a fixed line-detecting filter, the fraction of variance
in the **max** response explained by native geometry is substantially larger
than for the **mean** response, on `Scratch` wafers, after removing the wafer's
failure fraction. If it is not, the sentence above is wrong and comes out of all
three documents.

Reported as eta-squared, the between-geometry share of total variance, over
geometries holding enough `Scratch` wafers to estimate a group mean.

    python scripts/pooling_mechanism.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import CLASSES, build_corpus  # noqa: E402

MIN_PER_GEOM = 30
CLASS = "Scratch"


def line_filters():
    """Four oriented 3x3 line detectors, zero-mean so flat regions give zero."""
    base = [
        [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],           # horizontal
        [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],           # vertical
        [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],           # diagonal
        [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],           # anti-diagonal
    ]
    return torch.tensor(base, dtype=torch.float32).unsqueeze(1) / 6.0


def eta_squared(values, groups):
    """Between-group share of total variance."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    gm = values.mean()
    ss_tot = ((values - gm) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (v.mean() - gm) ** 2
    return float(ss_between / ss_tot)


def residualise(y, x):
    """Remove the least-squares linear fit of x from y."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def main():
    C = build_corpus(cache="data/corpus.pt")
    if CLASS not in CLASSES:
        raise SystemExit(f"{CLASS} not in {CLASSES}")
    ci = CLASSES.index(CLASS)

    sel = (C.labels == ci).nonzero(as_tuple=True)[0]
    maps64 = C.maps64[sel].float()
    hw = C.hw[sel]
    geom = [f"{int(h)}x{int(w)}" for h, w in hw.tolist()]

    # fail fraction from the *native* map, which resampling should preserve
    fail_frac = []
    for i in sel.tolist():
        a = C.maps[i]
        a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        die = a > 0
        fail_frac.append(float((a == 2).sum()) / max(1, int(die.sum())))
    fail_frac = np.array(fail_frac)

    # binary fail plane at 64x64, exactly what the encoder's channel 2 carries
    x = (maps64 == 2).float().unsqueeze(1)
    resp = F.conv2d(x, line_filters(), padding=1)          # (N, 4, 64, 64)
    stat_mean = resp.mean(dim=(-2, -1)).amax(dim=1).numpy()
    stat_max = resp.amax(dim=(-2, -1)).amax(dim=1).numpy()

    # geometries with enough wafers of this class to estimate a mean
    geom = np.array(geom)
    keep_g = {g for g in np.unique(geom) if (geom == g).sum() >= MIN_PER_GEOM}
    m = np.array([g in keep_g for g in geom])
    if m.sum() < 2 * MIN_PER_GEOM:
        raise SystemExit("not enough wafers per geometry")

    g, ff = geom[m], fail_frac[m]
    sm, sx = stat_mean[m], stat_max[m]

    # ---- the control that decides whether the resize causes this.
    # Geometry correlates with fab, product and era, so scratches on different
    # geometries may simply differ. If the effect is created by nearest-
    # neighbour upsampling then it must be much weaker at native resolution,
    # where a one-die scratch is one die wide on every geometry. If it survives
    # there, the geometry dependence is intrinsic to the wafers and the
    # explanation that blames the resize is wrong -- however well H65 read.
    nat_mean, nat_max = [], []
    filt = line_filters()
    for i in sel.tolist():
        a = C.maps[i]
        a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        t = torch.from_numpy((a == 2).astype(np.float32))[None, None]
        r = F.conv2d(t, filt, padding=1)
        nat_mean.append(float(r.mean(dim=(-2, -1)).amax()))
        nat_max.append(float(r.amax()))
    nat_mean = np.array(nat_mean)[m]
    nat_max = np.array(nat_max)[m]

    # ---- the proposed fix, tested at the same level before any GPU is spent.
    # If the resize creates the dependence by replicating each die into a
    # 64/w-wide block, then average-pooling the response back onto the native
    # die grid before taking the max should undo exactly that replication.
    # A fix that cannot pass the cheap test does not deserve a training run.
    fix_max = []
    for j, i in enumerate(sel.tolist()):
        h_i, w_i = int(hw[j, 0]), int(hw[j, 1])
        r = resp[j:j + 1]                       # (1, 4, 64, 64)
        r = F.adaptive_avg_pool2d(r, (max(1, h_i), max(1, w_i)))
        fix_max.append(float(r.amax()))
    fix_max = np.array(fix_max)[m]

    # ---- second candidate: scale the *filter*, not the pooling.
    # Pooling after the fact cannot help, because convolution and downsampling
    # do not commute: the filter has already responded to a band of width 64/w
    # with a magnitude set by that width, and averaging preserves the
    # magnitude. Dilating the filter by round(64/w) makes its receptive field
    # cover the same number of native dies on every geometry, which is the
    # thing the native-resolution control actually holds fixed.
    dil_max = []
    for j, i in enumerate(sel.tolist()):
        w_i = max(1, int(hw[j, 1]))
        d = max(1, int(round(64 / w_i)))
        t = maps64[j:j + 1].unsqueeze(1)
        t = (t == 2).float()
        r = F.conv2d(t, filt, padding=d, dilation=d)
        dil_max.append(float(r.amax()))
    dil_max = np.array(dil_max)[m]

    res = {
        "what": "does native geometry explain more of the max-pooled filter "
                "response than of the mean-pooled one, with no model involved",
        "class": CLASS, "n_wafers": int(m.sum()),
        "n_geometries": int(len(keep_g)),
        "min_wafers_per_geometry": MIN_PER_GEOM,
        "filter": "four oriented zero-mean 3x3 line detectors, max over "
                  "orientation, on the binary fail plane at 64x64",
        "raw": {
            "eta_sq_geometry_on_mean_pool": eta_squared(sm, g),
            "eta_sq_geometry_on_max_pool": eta_squared(sx, g),
        },
        "after_removing_fail_fraction": {
            "eta_sq_geometry_on_mean_pool": eta_squared(residualise(sm, ff), g),
            "eta_sq_geometry_on_max_pool": eta_squared(residualise(sx, ff), g),
        },
        "native_resolution_control": {
            "why": "if nearest-neighbour upsampling causes the geometry "
                   "dependence, it must be much weaker at native resolution, "
                   "where a one-die scratch is one die wide on every "
                   "geometry. If it survives here the dependence is intrinsic "
                   "to the wafers and the resize explanation is wrong.",
            "eta_sq_geometry_on_mean_pool": eta_squared(
                residualise(nat_mean, ff), g),
            "eta_sq_geometry_on_max_pool": eta_squared(
                residualise(nat_max, ff), g),
        },
        "proposed_fix_2_dilated_filter": {
            "what": "dilate the filter by round(64/w) so its receptive field "
                    "spans the same number of native dies on every geometry, "
                    "instead of correcting after the convolution",
            "eta_sq_geometry_on_max_pool": eta_squared(
                residualise(dil_max, ff), g),
        },
        "proposed_fix": {
            "what": "average-pool the response back onto the native die grid "
                    "(adaptive_avg_pool2d to h x w) before taking the max, "
                    "which undoes the nearest-neighbour replication",
            "eta_sq_geometry_on_max_pool": eta_squared(
                residualise(fix_max, ff), g),
        },
        "note": "eta-squared is the between-geometry share of total variance. "
                "The mean-pooled response is close to the failure fraction, "
                "which nearest-neighbour upsampling preserves; the max-pooled "
                "response is a peak whose width in pixels is set by 64/w.",
    }
    a = res["after_removing_fail_fraction"]
    res["ratio_max_over_mean_residualised"] = (
        a["eta_sq_geometry_on_max_pool"]
        / a["eta_sq_geometry_on_mean_pool"]
        if a["eta_sq_geometry_on_mean_pool"] > 0 else None)
    res["h65_supported"] = bool(
        a["eta_sq_geometry_on_max_pool"] > a["eta_sq_geometry_on_mean_pool"])

    Path("runs/pooling_mechanism.json").write_text(json.dumps(res, indent=2))
    print(f"{CLASS}: {res['n_wafers']} wafers over {res['n_geometries']} "
          f"geometries with >= {MIN_PER_GEOM} each\n")
    for k, lab in (("raw", "raw"),
                   ("after_removing_fail_fraction", "fail-fraction removed")):
        d = res[k]
        print(f"{lab}:")
        print(f"   mean-pooled  eta^2(geometry) = "
              f"{d['eta_sq_geometry_on_mean_pool']:.4f}")
        print(f"   max-pooled   eta^2(geometry) = "
              f"{d['eta_sq_geometry_on_max_pool']:.4f}")
    n = res["native_resolution_control"]
    print("native-resolution control (fail-fraction removed):")
    print(f"   mean-pooled  eta^2(geometry) = "
          f"{n['eta_sq_geometry_on_mean_pool']:.4f}")
    print(f"   max-pooled   eta^2(geometry) = "
          f"{n['eta_sq_geometry_on_max_pool']:.4f}")
    fx = res["proposed_fix"]
    fx2 = res["proposed_fix_2_dilated_filter"]
    print("candidate fix 1 -- max after pooling back to the native die grid:")
    print(f"   max-pooled   eta^2(geometry) = "
          f"{fx['eta_sq_geometry_on_max_pool']:.4f}")
    print("candidate fix 2 -- dilate the filter by round(64/w):")
    print(f"   max-pooled   eta^2(geometry) = "
          f"{fx2['eta_sq_geometry_on_max_pool']:.4f}")
    res["fix_recovers_native_behaviour"] = bool(
        fx["eta_sq_geometry_on_max_pool"]
        < 0.5 * (a["eta_sq_geometry_on_max_pool"]
                 + n["eta_sq_geometry_on_max_pool"]))
    r = res["ratio_max_over_mean_residualised"]
    print(f"\nratio at 64x64 (residualised) = {r:.2f}x" if r
          else "\nratio undefined")
    res["max_pool_eta_sq_drop_at_native"] = (
        a["eta_sq_geometry_on_max_pool"] - n["eta_sq_geometry_on_max_pool"])
    res["resize_explains_it"] = bool(
        n["eta_sq_geometry_on_max_pool"] < a["eta_sq_geometry_on_max_pool"])
    print(f"max-pool eta^2 falls by "
          f"{res['max_pool_eta_sq_drop_at_native']:+.4f} at native resolution")
    print(f"H65 {'supported' if res['h65_supported'] else 'NOT supported'}; "
          f"resize explanation "
          f"{'survives its control' if res['resize_explains_it'] else 'FAILS its control'}")
    Path("runs/pooling_mechanism.json").write_text(json.dumps(res, indent=2))
    print("\nwrote runs/pooling_mechanism.json")


if __name__ == "__main__":
    main()
