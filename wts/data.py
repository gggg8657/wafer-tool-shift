"""WM-811K as a domain-shift benchmark.

The public wafer-map corpus carries two shift axes that most published work
erases:

* **lot** (`lotName`) — 10,762 labelled lots, ~16 wafers each. A lot shares
  equipment, timing and process condition, so it is the closest public proxy for
  the tool-to-tool matching problem fabs actually have. 4,204 lots contain more
  than one defect class, so splitting by lot does not trivially split by label.
* **wafer geometry** — 346 distinct map sizes (product / technology node). The
  standard pipeline resizes everything to 32x32 or 64x64 and the axis disappears.
  Held out, it is a real covariate shift.

Three protocols are built from those axes:

| protocol   | train / test disjoint on | what it measures                       |
|------------|--------------------------|----------------------------------------|
| `iid`      | nothing (random wafers)  | the optimistic number papers report    |
| `lot`      | lotName                  | generalization to unseen tools/time    |
| `size`     | (H, W)                   | generalization to unseen geometry      |

`iid` minus `lot` is the headline: how much of the reported accuracy was lot
leakage.
"""
from __future__ import annotations

import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CLASSES = ("none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch")
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
DEFECT_IDS = tuple(i for c, i in CLASS_ID.items() if c != "none")


def _install_legacy_pandas_shim():
    """LSWMD.pkl was written by pandas 0.x under Python 2.

    Two incompatibilities: the pre-0.20 module layout (`pandas.indexes`) and
    py2 string pickling. Both are fixable without pinning an ancient pandas.
    """
    import pandas.core.indexes as pci

    shim = types.ModuleType("pandas.indexes")
    shim.base = pci.base
    sys.modules.setdefault("pandas.indexes", shim)
    sys.modules.setdefault("pandas.indexes.base", pci.base)
    sys.modules.setdefault("pandas.indexes.range", pci.range)


def load_raw(path="data/LSWMD.pkl"):
    """The full 811,457-row frame, labels unwrapped, unlabelled rows kept."""
    _install_legacy_pandas_shim()
    with open(path, "rb") as f:
        df = pickle.load(f, encoding="latin1")
    df["label"] = df["failureType"].apply(
        lambda x: x[0][0] if isinstance(x, (list, np.ndarray)) and len(x) else "")
    return df


@dataclass
class Corpus:
    """Labelled wafer maps plus the metadata the protocols need.

    `maps` keeps every wafer at its native resolution (a list, because the sizes
    differ); `maps64` is the resized tensor the conventional CNN path uses. Both
    are uint8 with the raw encoding 0 = outside the wafer, 1 = passing die,
    2 = failing die.
    """
    maps: list
    maps64: torch.Tensor          # (N, 64, 64) uint8
    labels: torch.Tensor          # (N,) int64
    lot: torch.Tensor             # (N,) int64 lot code
    size_id: torch.Tensor         # (N,) int64 (H, W) code
    hw: torch.Tensor              # (N, 2) int64
    lot_names: list
    size_names: list

    def __len__(self):
        return self.labels.shape[0]

    def save(self, path):
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path):
        return Corpus(**torch.load(path, map_location="cpu", weights_only=False))


def resize_nearest(a: np.ndarray, n=64) -> np.ndarray:
    """Nearest-neighbour resize that preserves the 0/1/2 encoding.

    Interpolating would invent die states that do not exist, so this indexes
    instead of averaging.
    """
    h, w = a.shape
    yi = (np.arange(n) * h // n).clip(0, h - 1)
    xi = (np.arange(n) * w // n).clip(0, w - 1)
    return a[yi][:, xi]


def build_corpus(path="data/LSWMD.pkl", min_dim=8, cache=None):
    """Labelled subset as a `Corpus`. Wafers smaller than `min_dim` are dropped."""
    df = load_raw(path)
    df = df[df["label"] != ""]
    maps, maps64, labels, lots, hs, ws = [], [], [], [], [], []
    for wm, lab, lot in zip(df["waferMap"], df["label"], df["lotName"]):
        a = np.asarray(wm, dtype=np.uint8)
        if a.ndim != 2 or min(a.shape) < min_dim:
            continue
        maps.append(torch.from_numpy(a))
        maps64.append(torch.from_numpy(resize_nearest(a, 64)))
        labels.append(CLASS_ID[lab])
        lots.append(str(lot))
        hs.append(a.shape[0]); ws.append(a.shape[1])

    lot_names = sorted(set(lots))
    lot_code = {n: i for i, n in enumerate(lot_names)}
    size_names = sorted({(h, w) for h, w in zip(hs, ws)})
    size_code = {s: i for i, s in enumerate(size_names)}
    c = Corpus(
        maps=maps,
        maps64=torch.stack(maps64),
        labels=torch.tensor(labels, dtype=torch.long),
        lot=torch.tensor([lot_code[n] for n in lots], dtype=torch.long),
        size_id=torch.tensor([size_code[(h, w)] for h, w in zip(hs, ws)],
                             dtype=torch.long),
        hw=torch.tensor(list(zip(hs, ws)), dtype=torch.long),
        lot_names=lot_names,
        size_names=size_names,
    )
    if cache:
        c.save(cache)
    return c


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def _stratified_group_split(groups, y, frac, seed, min_per_class=1):
    """Hold out whole groups, trying to keep every class present on both sides.

    Groups are shuffled and assigned to the test side until the target fraction
    is reached, but a group is skipped if moving it would leave a class with
    fewer than `min_per_class` training examples. Without that guard the rare
    classes (Near-full lives in 137 lots) can vanish from one side entirely.
    """
    g = np.asarray(groups); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(g))
    target = int(frac * len(g))
    counts = np.bincount(y, minlength=len(CLASSES))
    test_mask = np.zeros(len(g), dtype=bool)
    taken = 0
    for gid in uniq:
        sel = g == gid
        if taken + sel.sum() > target * 1.15:
            continue
        cand = counts - np.bincount(y[sel], minlength=len(CLASSES))
        if (cand[counts > 0] < min_per_class).any():
            continue
        test_mask |= sel
        counts = cand
        taken += int(sel.sum())
        if taken >= target:
            break
    return ~test_mask, test_mask


def split(corpus: Corpus, protocol="lot", frac=0.25, seed=0):
    """(train_idx, test_idx) for one protocol. Indices are LongTensors."""
    y = corpus.labels.numpy()
    if protocol == "iid":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        cut = int((1 - frac) * len(y))
        tr, te = perm[:cut], perm[cut:]
        return torch.from_numpy(tr), torch.from_numpy(te)
    if protocol == "lot":
        tr, te = _stratified_group_split(corpus.lot.numpy(), y, frac, seed)
    elif protocol == "size":
        # Hold out whole geometries at random. Taking the *rarest* geometries
        # first looked tempting but maximizes an artifact: geometry and defect
        # class are entangled in this corpus (a product fails the way its
        # design fails), so rarest-first pushed 96% of Edge-Ring into the test
        # side and the split stopped measuring covariate shift. Random choice
        # keeps the entanglement at its natural level, and `label_shift()`
        # reports whatever remains.
        tr, te = _stratified_group_split(corpus.size_id.numpy(), y, frac, seed)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    return (torch.from_numpy(np.where(tr)[0]), torch.from_numpy(np.where(te)[0]))


def class_counts(corpus, idx):
    return torch.bincount(corpus.labels[idx], minlength=len(CLASSES))


def label_shift(corpus, train_idx, test_idx):
    """Total-variation distance between train and test class distributions.

    A protocol that holds out groups cannot hold the label marginal fixed, so
    the honest thing is to measure how far it moved. `iid` sits near zero;
    anything larger is label shift riding along with the covariate shift, and a
    drop in accuracy has to be attributed carefully.
    """
    p = class_counts(corpus, train_idx).float()
    q = class_counts(corpus, test_idx).float()
    p, q = p / p.sum(), q / q.sum()
    return 0.5 * (p - q).abs().sum().item()
