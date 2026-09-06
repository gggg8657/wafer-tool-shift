"""Check that every decimal typed into generator prose exists somewhere in `runs/`.

The brief's central rule is that no number appears in any file unless a run in
this repository produced it. The generators enforce that for *tables* by
construction -- every cell is substituted from a JSON. Prose is not protected
the same way, and two defects this weekend were exactly that:

  "25 separate `lot` cells report that identical 0.4898 and 11 more report
   exactly 0.5000"                      -- true at ~40 cells, 111 and 75 at 208
  "TV 0.0208, against 0.1666 ... 0.1822 ... 0.2592"
                                        -- all four in runs/corpus_stats.json,
                                           all four typed by hand into the text

A number typed into prose decays exactly as fast as one typed into a table, and
is harder to see because prose is not where anyone looks for stale figures.

This does not check that a number is used *correctly* -- only that some run in
`runs/` produced a value that rounds to it.

**And that turns out to be almost no evidence at all.** `runs/` holds tens of
thousands of distinct values, so measured against a random draw the check
accepts 100% of three-digit decimals and 86% of four-digit ones. A guard that
passes most random garbage is not a guard, and "43 of 44 traceable" would have
been a reassuring sentence meaning nothing. The tool therefore measures its own
false-negative rate on every run and prints it beside the result, so the number
cannot be read without its own worthlessness attached.

What survives is the *inventory*. Forty-odd measurements typed into prose is
itself the finding, whatever their provenance, because every one of them is
frozen against a `runs/` that keeps moving -- which is how "25 cells report
0.4898" became wrong by a factor of four while still being perfectly traceable.
`--strict` fails on unmatched literals only, and unmatched is a strictly weaker
signal than the count above it.

    python scripts/number_provenance.py
    python scripts/number_provenance.py --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SOURCES = ("scripts/report.py", "scripts/paper.py", "scripts/weekend.py")
RATCHET = Path("runs/typed_decimal_ratchet.json")
# >=3 fraction digits: 0.85 is a rounded restatement, 0.8523 is a measurement
LITERAL = re.compile(r"(?<![\d.])(\d{1,2}\.\d{3,})(?![\d])")
# things that are not measurements
ALLOW = {"10.5281"}          # the Zenodo DOI prefix


def run_values(run_dir="runs"):
    """Every float anywhere in every run JSON, at several roundings."""
    seen = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            for nd in (3, 4, 5):
                seen.add(f"{abs(float(o)):.{nd}f}")

    for p in Path(run_dir).glob("*.json"):
        try:
            walk(json.loads(p.read_text()))
        except Exception:
            continue
    return seen


def literals(src):
    """Decimals inside string literals in a generator, with line numbers."""
    out = []
    for i, line in enumerate(Path(src).read_text().split("\n"), 1):
        if line.lstrip().startswith("#"):
            continue
        for lit in re.findall(r'"([^"]*)"', line):
            for m in LITERAL.findall(lit):
                if m in ALLOW:
                    continue
                out.append((i, m, lit.strip()[:78]))
    return out


def traceable(num, vals):
    """Does some run value round to this literal, at the literal's precision?"""
    nd = len(num.split(".")[1])
    if f"{float(num):.{nd}f}" in vals:
        return True
    # the literal may be a rounding of a value stored at more precision
    return any(v.startswith(num) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--show-matched", action="store_true")
    a = ap.parse_args()

    vals = run_values(a.runs)
    n_ok = n_bad = 0
    for src in SOURCES:
        if not Path(src).exists():
            continue
        ok, bad = [], []
        for i, m, ctx in literals(src):
            (ok if traceable(m, vals) else bad).append((i, m, ctx))
        n_ok += len(ok)
        n_bad += len(bad)
        if bad:
            print(f"\n{Path(src).name}: {len(bad)} decimal(s) with no run "
                  f"producing them ({len(ok)} traceable)")
            for i, m, ctx in bad:
                print(f"  L{i:<5} {m:<10} {ctx}")
        if a.show_matched and ok:
            print(f"\n{Path(src).name}: {len(ok)} traceable")
            for i, m, ctx in ok:
                print(f"  L{i:<5} {m:<10} {ctx}")

    print(f"\n{n_ok} typed decimals traceable, {n_bad} not, over {len(vals)} "
          f"distinct values in {a.runs}/.")
    print("\nHow much that is worth, measured rather than assumed:")
    import random
    rng = random.Random(0)
    for nd in (3, 4, 5):
        k = sum(1 for _ in range(5000)
                if traceable(f"{rng.random():.{nd}f}", vals))
        print(f"  a random {nd}-digit decimal in [0,1) is 'traceable' "
              f"{100 * k / 5000:5.1f}% of the time")
    print("So traceability is close to vacuous below five digits. The count "
          "that matters is the first one: every typed decimal is frozen "
          "against a runs/ that moves, whether or not it matches today.")

    # ---- the part of this script that can actually fail.
    # `guard_audit.py` reports the traceability check as VACUOUS, correctly: it
    # accepts ~86% of random four-digit decimals, so a green result from it
    # means nothing. What *is* checkable is the inventory. Every typed decimal
    # is a number frozen against a `runs/` that keeps moving, so the count is
    # a debt and it should only ever go down. This ratchet can genuinely fail:
    # add one hand-typed measurement to a generator and it does.
    total_typed = n_ok + n_bad
    prev = None
    if RATCHET.exists():
        prev = json.loads(RATCHET.read_text()).get("count")
    breach = prev is not None and total_typed > prev
    print(f"\nRatchet: {total_typed} decimals typed into generator prose"
          + (f" (was {prev})" if prev is not None else " (baseline set)"))
    if breach:
        print(f"  RAISED by {total_typed - prev}. A number typed into a "
              "sentence decays exactly as fast as one typed into a table and "
              "is harder to see. Compute it from `runs/` instead.")
    elif prev is not None and total_typed < prev:
        print(f"  lowered by {prev - total_typed}")
    if prev is None or total_typed < prev:
        RATCHET.write_text(json.dumps(
            {"count": total_typed, "sources": list(SOURCES),
             "what": "decimals with >=3 fraction digits typed into generator "
                     "prose; a debt that should only decrease",
             "why": "the traceability check above accepts ~86% of random "
                    "four-digit decimals and is not evidence; this is the "
                    "part of the script that can fail"}, indent=2))
    if n_bad:
        print("Unmatched is not automatically wrong: a ratio, a percentage or "
              "a difference between two measured values is derived rather than "
              "stored. But a derived number should be computed in the "
              "generator, not typed, for the same reason a measured one "
              "should -- the inputs move.")
    return 1 if (a.strict and (n_bad or breach)) else 0


if __name__ == "__main__":
    sys.exit(main())
