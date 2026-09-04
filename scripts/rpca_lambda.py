"""Record the lambda sweep behind the RPCA choice -- on a representative sample.

    python scripts/rpca_lambda.py [--sample random|first] [--n-lots 200]

The decomposition has one knob and it decides everything, so the choice was
documented rather than asserted. The documentation was wrong about the corpus.

The original sweep took `[:40]` of the lots with at least 20 wafers, sorted by
lot id, and reported a mean rank of 0.475 at the standard
`1/sqrt(max(n, d))` -- from which the docstring concluded that "the low-rank
part carries about a third of the failure energy on a 20+ wafer lot". Measured
against the shipped decomposition:

    first 40 by lot id   mean rank 0.475   rank 0 for 52.5% of lots
    random 40            mean rank 0.050   rank 0 for 95.0%
    all 6,504 lots       mean rank 0.050   rank 0 for 95.1%

`[:40]` of a lot-id-sorted list is not a sample, it is a slice, and lot id is
not arbitrary on this corpus -- geometry and failed-die rate both drift with it
(`scripts/time_proxy_check.py`). The knob that decides everything about the RPCA
channel was chosen on evidence describing 0.6% of the lots, and the most
favourable 0.6% available.

This version samples at random with a recorded seed, sweeps a wider grid, and
reports the biased slice alongside so the two are comparable. The question it
should have answered from the start: **is there any lambda at which the
decomposition is non-trivial on a representative sample without the low-rank
part absorbing the defect?**
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
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--sample", default="random", choices=["random", "first"])
_ap.add_argument("--n-lots", type=int, default=200)
_ap.add_argument("--seed", type=int, default=0)
_ap.add_argument("--out", default="runs/rpca_lambda.json")
_a = _ap.parse_args()

_all = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 20]
if _a.sample == "first":
    big = _all[:_a.n_lots]
else:
    _rng = np.random.default_rng(_a.seed)
    big = [_all[i] for i in _rng.choice(len(_all),
                                        min(_a.n_lots, len(_all)),
                                        replace=False)]
print(f"{len(big)} lots, sample={_a.sample}, seed={_a.seed}, "
      f"out of {len(_all)} with >=20 wafers", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for scale in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
    ranks, shares, energy = [], [], []
    for a, b in big:
        idx = order[a:b]
        F = fail[idx].reshape(len(idx), -1).to(dev)
        L, S = rpca(F, lam=scale / max(F.shape) ** 0.5, n_iter=30)
        ranks.append(torch.linalg.matrix_rank(L, rtol=1e-3).item())
        shares.append(float(S.abs().sum() / F.abs().sum().clamp_min(1e-6)))
        energy.append(float(L.pow(2).sum() / F.pow(2).sum().clamp_min(1e-9)))
    rows.append({"lambda_scale": scale, "mean_rank": float(np.mean(ranks)),
                 "frac_rank_zero": float(np.mean([r == 0 for r in ranks])),
                 "sparse_share": float(np.mean(shares)),
                 "lowrank_energy_share": float(np.mean(energy))})
    print(f"  lambda x{scale:<5} rank {np.mean(ranks):6.2f}  "
          f"sparse share {np.mean(shares):.3f}  "
          f"low-rank energy {np.mean(energy):.3f}", flush=True)
Path(_a.out).write_text(json.dumps(
    {"n_lots": len(big), "min_lot_size": 20, "sample": _a.sample,
     "seed": _a.seed, "n_lots_available": len(_all), "sweep": rows,
     "note": ("The original run of this script took the first 40 lots of a "
              "lot-id-sorted list, which on this corpus is a biased slice: it "
              "gives mean rank 0.475 at lambda x1.0 against 0.050 for a random "
              "sample and 0.050 for all 6,504 lots.")}, indent=2))
print(f"wrote {_a.out}")
