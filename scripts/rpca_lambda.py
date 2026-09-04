"""Record the lambda sweep behind the RPCA choice.

    CUDA_VISIBLE_DEVICES=0 python scripts/rpca_lambda.py

The decomposition has one knob and it decides everything, so the choice is
documented rather than asserted: at the standard 1/sqrt(max(n, d)) the low-rank
part carries about a third of the failure energy on a 20+ wafer lot; 4x that and
the background absorbs the defect too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import Corpus          # noqa: E402
from wts.rpca import rpca            # noqa: E402

c = Corpus.load("data/corpus.pt")
fail = (c.maps64 == 2).float()
order = torch.argsort(c.lot)
ls = c.lot[order]
bounds = torch.nonzero(
    torch.cat([torch.tensor([True]), ls[1:] != ls[:-1]])).flatten().tolist() + [len(ls)]
big = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 20][:40]
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
    ranks, shares, energy = [], [], []
    for a, b in big:
        idx = order[a:b]
        F = fail[idx].reshape(len(idx), -1).to(dev)
        L, S = rpca(F, lam=scale / max(F.shape) ** 0.5, n_iter=30)
        ranks.append(torch.linalg.matrix_rank(L, rtol=1e-3).item())
        shares.append(float(S.abs().sum() / F.abs().sum().clamp_min(1e-6)))
        energy.append(float(L.pow(2).sum() / F.pow(2).sum().clamp_min(1e-9)))
    rows.append({"lambda_scale": scale, "mean_rank": float(np.mean(ranks)),
                 "sparse_share": float(np.mean(shares)),
                 "lowrank_energy_share": float(np.mean(energy))})
    print(f"  lambda x{scale:<5} rank {np.mean(ranks):6.2f}  "
          f"sparse share {np.mean(shares):.3f}  "
          f"low-rank energy {np.mean(energy):.3f}", flush=True)
Path("runs/rpca_lambda.json").write_text(json.dumps(
    {"n_lots": len(big), "min_lot_size": 20, "sweep": rows}, indent=2))
print("wrote runs/rpca_lambda.json")
