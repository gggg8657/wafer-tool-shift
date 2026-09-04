"""Compute the size-invariant descriptors for every labelled wafer.

    python scripts/extract.py --workers 16      # -> data/features.pt

~4 ms per wafer single-core, so this is worth parallelizing once and caching.
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import Corpus  # noqa: E402
from wts.features import FEATURE_DIM, block_slices, descriptor  # noqa: E402

_MAPS = None


def _init(maps):
    global _MAPS
    _MAPS = maps


def _one(i):
    return descriptor(_MAPS[i].numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus.pt")
    p.add_argument("--out", default="data/features.pt")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    c = Corpus.load(args.corpus)
    n = len(c)
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(c.maps,)) as pool:
        feats = pool.map(_one, range(n), chunksize=256)
    X = torch.from_numpy(np.stack(feats).astype(np.float32))
    bad = ~torch.isfinite(X)
    if bad.any():
        print(f"  {int(bad.sum())} non-finite entries zeroed")
        X[bad] = 0.0
    torch.save({"X": X, "dim": FEATURE_DIM, "slices": block_slices()}, args.out)
    print(f"wrote {args.out}  {tuple(X.shape)}  ({time.time()-t0:.0f}s, "
          f"{1000*(time.time()-t0)/n:.2f} ms/wafer effective)")
    print("per-block std (should be non-degenerate):")
    for name, sl in block_slices().items():
        print(f"  {name:16s} mean|x| {X[:, sl].abs().mean():.4f}  "
              f"std {X[:, sl].std():.4f}")


if __name__ == "__main__":
    main()
