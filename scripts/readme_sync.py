"""Does the README still describe the benchmark this repository contains?

The README is the first thing anyone picking this up reads, and it is the one
document here that is *not* generated -- so nothing regenerates it when the code
grows and no guard noticed when it stopped being true. Found by reading it
against `run_bench.py`:

  * "The three protocols", with a three-row table, when there are four. The
    missing one was `lot_time`, which carries the largest drop in the project.
  * "Four representations", with a three-row table, when there are six. The
    missing ones were `graph` and `rpca_cnn` -- the second being this repo's own
    withdrawn contribution.
  * "Seven objectives", when `methods.OBJECTIVES` holds eleven.

None of that is a typo. Each is a place where the code grew and the prose that
introduces it did not, and a reader would have formed a wrong picture of the
benchmark's scope before reaching a single number.

This checks the counts the README states against the ones the code defines. It
cannot check that the *descriptions* are right -- that still needs reading --
but a count is exactly the part that drifts silently.

    python scripts/readme_sync.py
    python scripts/readme_sync.py --strict
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12}


def code_counts():
    """Read the choice lists out of the source rather than importing torch."""
    src = Path("scripts/run_bench.py").read_text()
    out = {}
    for flag, key in (("--protocol", "protocols"), ("--encoder", "encoders")):
        m = re.search(re.escape(f'"{flag}"') + r".*?choices=(\[.*?\])",
                      src, re.S)
        if m:
            out[key] = len(ast.literal_eval(m.group(1)))
    # OBJECTIVES is built in two places: a dict literal, then an `.update()`
    # further down that adds four more. The first version of this parser read
    # only the literal, reported 7 against the README's 11, and would have
    # flagged a correct README forever -- the same missing-a-member error the
    # power audit made when it summarised a family by its weakest comparison.
    # Counting every key assigned to the name, wherever it is assigned.
    meth = Path("wts/methods.py").read_text()
    keys = set()
    for m in re.finditer(r"OBJECTIVES(?:\s*=\s*|\.update\()\s*\{(.*?)\}",
                         meth, re.S):
        keys.update(re.findall(r'"([a-z_]+)"\s*:', m.group(1)))
    if keys:
        out["objectives"] = len(keys)
    return out


def readme_counts():
    """The counts the README's own headings claim."""
    txt = Path("README.md").read_text()
    out = {}
    for pat, key in ((r"##\s+The\s+(\w+)\s+protocols", "protocols"),
                     (r"##\s+(\w+)\s+representations", "encoders"),
                     (r"##\s+(\w+)\s+objectives", "objectives")):
        m = re.search(pat, txt, re.I)
        if m:
            w = m.group(1).lower()
            out[key] = WORDS.get(w, int(w) if w.isdigit() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    code, doc = code_counts(), readme_counts()
    bad = 0
    print(f"{'thing':14s} {'README says':>12s} {'code has':>10s}")
    for k in ("protocols", "encoders", "objectives"):
        c, d = code.get(k), doc.get(k)
        ok = (c is not None and c == d)
        bad += 0 if ok else 1
        print(f"{k:14s} {str(d):>12s} {str(c):>10s}   "
              f"{'ok' if ok else 'MISMATCH'}")
    if bad:
        print(f"\n{bad} count(s) in README.md no longer match the code. The "
              "README is the only document here that is not generated, so "
              "nothing else will notice.")
    else:
        print("\nREADME's stated counts match the code")
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
