"""Build a *synthetic* mixed-type wafer benchmark by overlaying WM-811K patterns.

    python scripts/make_mixed.py --out data/mixed.pt

WM-811K is single-label with nine classes, so "multi-label F1 on WM-811K" is not
a defined quantity. The real mixed-type benchmark is MixedWM38 (38 mixed-type
classes, Wang et al.), which this session could not obtain: the only copies
reachable from this box are HuggingFace `wafervqaanon/MixedWM38-VQA`, which is
gated behind manual approval, and Zenodo record 10.5281/zenodo.20061545, whose
files return HTTP 403 (restricted). Neither can be acquired unattended.

So this constructs the alternative the brief allows, and every artefact it
produces is named `MIXED-SYNTH` so that no number computed on it can be mistaken
for a MixedWM38 result or for a WM-811K result:

* a mixed wafer is the union of the failed dies of two real WM-811K wafers of
  the **same geometry**, so the die grid and the wafer outline are physically
  consistent rather than resampled together;
* its label is the multi-hot union of the two source labels;
* single-defect and defect-free wafers are carried through unchanged, so the
  label cardinality distribution spans 0, 1 and 2 rather than being all-2.

**The one thing this construction cannot give you.** Real mixed-type defects
interact: a scratch crossing an edge ring changes both signatures where they
meet. A pixel-wise OR of two independently produced wafers has no interaction
term, so it is an *easier* problem than MixedWM38 in a way that no amount of
sample size fixes. Any F1 measured here is an upper bound on the real
mixed-type task and is labelled as such wherever it is reported.

The split is built **before** the overlay and both sources of a mixed wafer are
drawn from the same side, so no real wafer contributes to both train and test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import CLASSES, Corpus, split          # noqa: E402

DEFECTS = CLASSES[1:]          # the 8 defect classes; 'none' is the empty label


def build_side(c, idx, rng, n_mixed, min_geo=40):
    """Mixed wafers from `idx` only.

    Returns (maps64, multi-hot labels, source index pairs, lot ids).
    """
    lab = c.labels.numpy()[idx.numpy()]
    sid = c.size_id.numpy()[idx.numpy()]
    fail = (c.maps64[idx] == 2)
    inside = (c.maps64[idx] > 0)

    # candidate pool: defect wafers, grouped by geometry, only geometries with
    # enough of them that two *different* classes are actually available
    by_geo = {}
    for g in np.unique(sid):
        m = (sid == g) & (lab > 0)
        if m.sum() >= min_geo and len(np.unique(lab[m])) >= 2:
            by_geo[int(g)] = np.where(m)[0]
    if not by_geo:
        return None
    geos = np.array(sorted(by_geo))
    weights = np.array([len(by_geo[g]) for g in geos], dtype=float)
    weights /= weights.sum()

    maps, ys, pairs, lots = [], [], [], []
    for _ in range(n_mixed):
        g = int(rng.choice(geos, p=weights))
        pool = by_geo[g]
        a = int(rng.choice(pool))
        # force a *different* class, or the "mixed" wafer has one label
        for _ in range(20):
            b = int(rng.choice(pool))
            if lab[b] != lab[a]:
                break
        else:
            continue
        m = torch.where(fail[a] | fail[b], 2,
                        torch.where(inside[a], 1, 0)).to(torch.uint8)
        y = np.zeros(len(DEFECTS), dtype=np.float32)
        y[lab[a] - 1] = 1.0
        y[lab[b] - 1] = 1.0
        maps.append(m)
        ys.append(y)
        pairs.append((int(idx[a]), int(idx[b])))
        # the mixed wafer inherits the lot of its first source, so an inner
        # validation split can still be lot-disjoint downstream
        lots.append(int(c.lot[idx[a]]))

    # carry the originals through: cardinality 1 for defects, 0 for 'none'
    orig_y = np.zeros((len(idx), len(DEFECTS)), dtype=np.float32)
    for i, l in enumerate(lab):
        if l > 0:
            orig_y[i, l - 1] = 1.0
    return (torch.cat([c.maps64[idx], torch.stack(maps)]),
            torch.from_numpy(np.concatenate([orig_y, np.stack(ys)])),
            pairs,
            torch.cat([c.lot[idx], torch.tensor(lots)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.pt")
    ap.add_argument("--out", default="data/mixed.pt")
    ap.add_argument("--protocol", default="lot", choices=["lot", "iid", "size"])
    ap.add_argument("--n-mixed-train", type=int, default=40000)
    ap.add_argument("--n-mixed-test", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c = Corpus.load(a.corpus)
    tr, te = split(c, a.protocol, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    train = build_side(c, tr, rng, a.n_mixed_train)
    test = build_side(c, te, rng, a.n_mixed_test)
    if train is None or test is None:
        raise SystemExit("no geometry had two defect classes on one side")

    # a source wafer must never appear on both sides
    src_tr = {i for p in train[2] for i in p} | set(tr.tolist())
    src_te = {i for p in test[2] for i in p} | set(te.tolist())
    assert not (src_tr & src_te), "a source wafer reached both splits"

    blob = {
        "name": "MIXED-SYNTH",
        "note": ("SYNTHETIC. Mixed-type wafers constructed by OR-ing the failed "
                 "dies of two same-geometry WM-811K wafers. Not MixedWM38. No "
                 "interaction between overlaid patterns, so metrics on this set "
                 "are an upper bound on the real mixed-type task."),
        "classes": list(DEFECTS),
        "protocol": a.protocol, "seed": a.seed,
        "train_maps": train[0], "train_y": train[1], "train_lot": train[3],
        "test_maps": test[0], "test_y": test[1], "test_lot": test[3],
        "n_train_real": len(tr), "n_train_mixed": len(train[2]),
        "n_test_real": len(te), "n_test_mixed": len(test[2]),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, a.out)

    card = lambda y: np.bincount(y.sum(1).numpy().astype(int), minlength=3)
    summary = {
        "name": "MIXED-SYNTH", "synthetic": True, "protocol": a.protocol,
        "n_train": int(len(blob["train_y"])), "n_test": int(len(blob["test_y"])),
        "n_train_mixed": blob["n_train_mixed"], "n_test_mixed": blob["n_test_mixed"],
        "train_label_cardinality_0_1_2": card(blob["train_y"]).tolist(),
        "test_label_cardinality_0_1_2": card(blob["test_y"]).tolist(),
        "train_positives_per_class": dict(zip(
            DEFECTS, blob["train_y"].sum(0).int().tolist())),
        "test_positives_per_class": dict(zip(
            DEFECTS, blob["test_y"].sum(0).int().tolist())),
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/mixed_synth_dataset.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
