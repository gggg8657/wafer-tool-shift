"""Flag a recorded prediction that covers both outcomes it is meant to choose between.

`hypothesis_ledger.py` verifies that every finished hypothesis was given a
verdict. Entry 102 named what it cannot do: check that the verdict was honest.
H74 is the case — "the point estimate is at or below zero, *or* if positive it
does not reach p < 0.05" — a disjunction wide enough that a sign flip satisfies
it. I reported it as a failure anyway, but only because I noticed; nothing would
have stopped me claiming the literal reading.

The defect has a shape: a prediction that names a direction and then adds a
clause covering the opposite direction. That is checkable, and it is worth
checking *before* the run rather than after, because a prediction phrased this
way cannot be wrong and therefore cannot be evidence.

Auditing all recorded hypotheses, H74 is the only one of twenty-two with it.
The others state a direction and a threshold — "separates at p < 0.05 and is
worse", "the control stays null", "no gamma separates" — and most name what
would falsify them. So this lints for one specific failure rather than trying to
score prediction quality in general, which is not something a regular expression
can do.

    python scripts/hypothesis_sharpness.py
    python scripts/hypothesis_sharpness.py --strict
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# a prediction sentence: from the marker to the first full stop that ends it
HEAD = re.compile(r"H(\d{2,3}),?\s*(?:before the run|on record)[^:]*:", re.I)
# "at or below zero, or if positive ..." -- a direction plus its negation
DISJUNCTION = re.compile(
    r"(at or (?:below|above)|"
    r"or if (?:positive|negative|it (?:is|does))|"
    r"or,? if (?:positive|negative))", re.I)
# these read as disjunctions and are not: a falsification clause, or an adverb
NOT_A_HEDGE = re.compile(r"if either does|either way|does not .* either", re.I)


def blocks():
    out = []
    for p in sorted(Path("scripts").glob("*.sh")) + sorted(
            Path("scripts").glob("*.py")):
        if p.name.startswith("hypothesis_"):
            continue
        t = p.read_text()
        for m in HEAD.finditer(t):
            raw = t[m.end():m.end() + 500]
            stop = raw.find("#\n")
            raw = raw[:stop if stop > 0 else 400]
            text = " ".join(raw.replace("\n#", " ").replace("#", " ").split())
            # the prediction is the first sentence or two, before the reasoning
            pred = re.split(r"(?<=\.)\s+(?=The reason|Why|What would|This is)",
                            text)[0]
            out.append({"hypothesis": f"H{m.group(1)}", "file": p.name,
                        "prediction": pred})
    return out


def flagged(b):
    if NOT_A_HEDGE.search(b["prediction"]):
        return False
    return bool(DISJUNCTION.search(b["prediction"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    bs = blocks()
    bad = [b for b in bs if flagged(b)]
    print(f"{len(bs)} recorded predictions; {len(bad)} state a direction and "
          "then cover its opposite")
    for b in bad:
        print(f"  {b['hypothesis']} ({b['file']})")
        print(f"    {b['prediction'][:150]}")
    if bad:
        print("\nA prediction that covers both directions cannot be wrong, so "
              "it cannot be evidence. Phrase it as one direction with a "
              "threshold, and say separately what would falsify it.")
    else:
        print("no recorded prediction covers both directions")
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
