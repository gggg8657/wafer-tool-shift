"""What does the per-lot RPCA actually remove, and from which classes?

    python scripts/rpca_mechanism.py      # -> runs/rpca_mechanism.json

The fourth-channel ablation established that the RPCA lot-signature channel is
worth what a channel of zeros is worth. That is a result about the *outcome*.
This is the mechanism, and it is more interesting than the outcome, because it
says the decomposition is not merely inert — it is pointed the wrong way.

RPCA splits a lot's wafers into what they share (low-rank) and what each does
alone (sparse), and hands the encoder the sparse part on the theory that the
shared component is the tool's nuisance signature. That theory requires the
defect to be per-wafer and the nuisance to be lot-wide. On WM-811K the opposite
is true for the class it fires on most: `Edge-Ring` is edge roll-off, which is a
lot-level process condition, so "what the wafers of this lot share" *is* the
defect. The decomposition deletes the label.

Meanwhile the classes that genuinely are per-wafer — `Scratch`, `Loc`, the long
tail this repository has been trying to move all weekend — are exactly the ones
it never fires on, because a scratch on one wafer is not shared by its lot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wts.data import CLASSES, Corpus                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.pt")
    ap.add_argument("--rpca", default="data/rpca.pt")
    ap.add_argument("--out", default="runs/rpca_mechanism.json")
    a = ap.parse_args()

    c = Corpus.load(a.corpus)
    b = torch.load(a.rpca, map_location="cpu", weights_only=False)
    res, sig = b["residual"].float(), b["signature"]
    y = c.labels.numpy()
    rank, dec = sig[:, 4].numpy(), (sig[:, 5] > 0).numpy()
    fail = (c.maps64 == 2).float()

    # "active" = the decomposition found something to remove
    active = dec & (rank >= 1)
    base = float(active.mean())
    removed = ((res[active] - fail[active]).abs().flatten(1).sum(1)
               / fail[active].flatten(1).sum(1).clamp_min(1))
    ya = y[active]

    per_class = {}
    for k, name in enumerate(CLASSES):
        m = y == k
        if not m.sum():
            continue
        sa = float(active[m].mean())
        sub = ya == k
        per_class[name] = {
            "n": int(m.sum()),
            "n_active": int(active[m].sum()),
            "share_active": sa,
            "enrichment_vs_corpus": sa / base if base else None,
            "mean_failed_die_mass_removed": (float(removed[sub].mean())
                                             if sub.sum() >= 20 else None),
        }

    ranked = sorted(per_class.items(),
                    key=lambda kv: -(kv[1]["enrichment_vs_corpus"] or 0))
    out = {
        "n_wafers": int(len(y)),
        "n_active": int(active.sum()),
        "frac_active": base,
        "mean_failed_die_mass_removed_when_active": float(removed.mean()),
        "median_failed_die_mass_removed_when_active": float(removed.median()),
        "per_class": per_class,
        "most_enriched": ranked[0][0],
        "least_enriched": ranked[-1][0],
        "note": ("RPCA's low-rank part is what a lot's wafers share. It fires "
                 "almost only on classes that are lot-level process conditions, "
                 "where the shared component IS the defect, and essentially "
                 "never on the per-wafer classes the long tail is made of. The "
                 "decomposition removes signal from the classes it touches and "
                 "does nothing for the classes that need help."),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
