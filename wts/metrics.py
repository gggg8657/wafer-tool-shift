"""Metrics chosen so a model cannot look good by ignoring the hard part."""
from __future__ import annotations

import numpy as np
import torch

from .data import CLASSES, DEFECT_IDS


def per_class_f1(y_true, y_pred, n_classes=len(CLASSES)):
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        f1[c] = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    return f1


def auroc(scores, labels):
    """Rank-based AUROC, ties averaged. `labels` is boolean."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            m = (s >= su[i]) & (s <= su[j])
            ranks[m] = (i + j + 2) / 2
        i = j + 1
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(probs, y_true, n_bins=15):
    """Expected calibration error of the top-1 confidence."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y_true).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def worst_group(y_true, y_pred, groups, min_n=12, quantile=0.0):
    """Macro-F1 of the worst test domain (optionally a low quantile).

    Averages hide the failure mode this benchmark exists to expose: a model can
    post a fine mean while being useless on a particular tool. Domains with
    fewer than `min_n` wafers are excluded, because a 3-wafer lot's F1 is noise.

    A lot holds at most 25 wafers, so the single worst lot is itself a noisy
    statistic; the 10th percentile across domains is the more honest headline
    and both are reported.
    """
    g = np.asarray(groups)
    scores = []
    for u in np.unique(g):
        m = g == u
        if m.sum() < min_n:
            continue
        present = np.unique(y_true[m])
        f1 = per_class_f1(y_true[m], y_pred[m])
        scores.append(f1[present].mean())
    if not scores:
        return float("nan"), 0
    scores = np.sort(np.asarray(scores))
    if quantile > 0:
        return float(np.quantile(scores, quantile)), len(scores)
    return float(scores[0]), len(scores)


def classwise_conformal(cal_probs, cal_y, test_probs, alpha=0.1):
    """Class-conditional conformal sets; returns (coverage, mean set size).

    Class-conditional rather than marginal, because a marginal guarantee on this
    corpus is satisfied by covering `none` and abandoning `Near-full`. The
    calibration split is drawn from the *training* domains, so the coverage
    measured on test is coverage under shift -- which is the number that tells
    you whether the guarantee survives a new tool.
    """
    n_classes = cal_probs.shape[1]
    qs = np.ones(n_classes)
    for c in range(n_classes):
        m = cal_y == c
        if m.sum() < 10:
            continue
        s = 1.0 - cal_probs[m, c]
        k = int(np.ceil((m.sum() + 1) * (1 - alpha)))
        qs[c] = np.sort(s)[min(k, m.sum()) - 1]
    keep = (1.0 - test_probs) <= qs[None, :]
    return keep, qs


def summarize(y_true, probs, groups=None, cal=None, alpha=0.1):
    y_pred = probs.argmax(1)
    f1 = per_class_f1(y_true, y_pred)
    present = np.unique(y_true)
    out = {
        "macro_f1": float(f1[present].mean()),
        "defect_macro_f1": float(np.mean([f1[c] for c in DEFECT_IDS
                                          if c in present])),
        "accuracy": float((y_pred == y_true).mean()),
        "per_class_f1": {CLASSES[c]: float(f1[c]) for c in range(len(CLASSES))},
        "defect_auroc": float(auroc(1.0 - probs[:, 0], y_true != 0)),
        "ece": ece(probs, y_true),
    }
    if groups is not None:
        w, n = worst_group(y_true, y_pred, groups)
        out["worst_domain_macro_f1"] = w
        out["p10_domain_macro_f1"] = worst_group(y_true, y_pred, groups,
                                                 quantile=0.10)[0]
        out["n_domains_scored"] = n
    if cal is not None:
        keep, _ = classwise_conformal(cal[0], cal[1], probs, alpha)
        hit = keep[np.arange(len(y_true)), y_true]
        out["conformal_coverage"] = float(hit.mean())
        out["conformal_set_size"] = float(keep.sum(1).mean())
        out["conformal_target"] = 1 - alpha
    return out


def weighted_conformal(cal_probs, cal_y, test_probs, cal_w, test_w, alpha=0.1):
    """Conformal prediction under covariate shift, with importance weights.

    Standard split conformal assumes calibration and test are exchangeable. Under
    a tool change they are not, and the coverage guarantee quietly fails. The
    weighted version (Tibshirani et al.) restores it given the likelihood ratio
    p_test(x) / p_train(x), estimated here by a logistic probe on the embeddings
    -- so the guarantee is only as good as that probe, which is why the probe's
    own AUC is reported alongside and the unweighted coverage is kept next to it.

    Exact weighted conformal recomputes the quantile for every test point, since
    the point's own weight enters the denominator. With ~19k calibration points
    that term is negligible, so one quantile per class is computed against
    `sum(cal_w) + median(test_w)`. The approximation is stated rather than
    hidden; it is what makes the metric affordable inside a sweep.
    """
    n_classes = cal_probs.shape[1]
    cal_w = np.asarray(cal_w, dtype=np.float64)
    test_w = np.asarray(test_w, dtype=np.float64)
    w_extra = float(np.median(test_w))
    qs = np.ones(n_classes)
    for c in range(n_classes):
        m = cal_y == c
        if m.sum() < 10:
            continue
        s_c = 1.0 - cal_probs[m, c]
        w_c = cal_w[m]
        order = np.argsort(s_c)
        s_sorted, w_sorted = s_c[order], w_c[order]
        cum = np.cumsum(w_sorted) / (w_sorted.sum() + w_extra)
        j = int(np.searchsorted(cum, 1 - alpha))
        qs[c] = s_sorted[min(j, len(s_sorted) - 1)]
    return (1.0 - test_probs) <= qs[None, :]


def coverage_of(keep, y_true):
    return float(keep[np.arange(len(y_true)), y_true].mean())
