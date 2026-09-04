"""Cache the labelled WM-811K subset with its lot and geometry metadata.

    python scripts/prepare.py            # -> data/corpus.pt

Runs once (~3 min); everything downstream loads the cache.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import (CLASSES, build_corpus, class_counts, label_shift,  # noqa: E402
                      split)

t0 = time.time()
c = build_corpus(cache="data/corpus.pt")
print(f"corpus: {len(c)} wafers, {len(c.lot_names)} lots, "
      f"{len(c.size_names)} geometries  ({time.time()-t0:.0f}s)")
print("class counts:", dict(zip(CLASSES, class_counts(c, slice(None)).tolist())))
for p in ("iid", "lot", "size"):
    tr, te = split(c, p)
    ntr = class_counts(c, tr); nte = class_counts(c, te)
    print(f"\n[{p}] train {len(tr)} / test {len(te)}  "
          f"label shift (TV) {label_shift(c, tr, te):.4f}")
    print("  train:", dict(zip(CLASSES, ntr.tolist())))
    print("  test :", dict(zip(CLASSES, nte.tolist())))
    if p == "lot":
        overlap = set(c.lot[tr].tolist()) & set(c.lot[te].tolist())
        print("  lot overlap:", len(overlap))
    if p == "size":
        overlap = set(c.size_id[tr].tolist()) & set(c.size_id[te].tolist())
        print("  geometry overlap:", len(overlap),
              "| unseen geometries in test:", len(set(c.size_id[te].tolist())))
