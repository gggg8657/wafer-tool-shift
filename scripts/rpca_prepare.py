"""Per-lot RPCA over the corpus -> data/rpca.pt (residual maps + signatures).

    CUDA_VISIBLE_DEVICES=2 python scripts/rpca_prepare.py

Runs once; `--encoder rpca_cnn` and the `rpca` descriptor block read the cache.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import Corpus                       # noqa: E402
from wts.rpca import lot_decomposition            # noqa: E402

c = Corpus.load("data/corpus.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
resid, sig = lot_decomposition(c.maps64, c.lot, device=dev)
torch.save({"residual": resid.half(), "signature": sig}, "data/rpca.pt")
print(f"wrote data/rpca.pt  residual {tuple(resid.shape)} "
      f"signature {tuple(sig.shape)}  ({time.time()-t0:.0f}s)")
fail = (c.maps64 == 2).float()
kept = resid.abs().sum() / fail.abs().sum().clamp_min(1)
print(f"sparse share of the failure mass kept: {kept:.3f} "
      f"(the rest was the lot's shared signature)")
print("signature columns: strength, peak, spread, sparse_share, rank, lot_size")
print("means:", sig.mean(0).tolist())
