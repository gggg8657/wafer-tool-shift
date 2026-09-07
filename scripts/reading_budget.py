"""Is the five-minute version still five minutes?

The brief specifies a hand-off "readable in five minutes". `WEEKEND.md` reached
5,184 words -- twenty-two minutes -- before anyone measured it, because every
result that landed was appended to the section it belonged in and nothing was
ever compressed. Entry 74 split it into a core above a stop marker and a body
below, and measured the core into the text.

Measuring it once does not keep it measured. Since that split the core has taken
a clause per turn -- the sinkhorn verdict, the `size` replication, the RPCA
tie -- each of them a reasonable local addition, and it drifted from 949 words
to 1,237. That is the same mechanism as the original overrun, running slower.

So the budget is a check rather than a habit. It fails when the core exceeds
the brief's five minutes at 230 words per minute, which is a specification, not
a preference -- the number comes from the brief and the reading rate is stated
so it can be argued with.

    python scripts/reading_budget.py
    python scripts/reading_budget.py --strict     # exit 1 if over budget
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DOC = "WEEKEND.md"
MARKER = "Stop here if five minutes is what you have"
WPM = 230
BUDGET_MIN = 5.0


def split_core(text):
    cut = text.find(MARKER)
    if cut < 0:
        return None, None
    return text[:cut], text[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = Path(DOC)
    if not p.exists():
        print(f"{DOC} not found")
        return 0
    core, rest = split_core(p.read_text())
    if core is None:
        print(f"no stop marker in {DOC}; the five-minute split is gone")
        return 1 if a.strict else 0

    n_core, n_rest = len(core.split()), len(rest.split())
    m_core, m_rest = n_core / WPM, n_rest / WPM
    over = m_core > BUDGET_MIN

    print(f"{DOC}: core {n_core:,} words = {m_core:.1f} min "
          f"(budget {BUDGET_MIN:.0f}); body {n_rest:,} words = {m_rest:.1f} min")
    if over:
        # ceil, not int: at 1,151 words the core is 5.0043 minutes, and
        # int((5.0043 - 5) * 230) rounds the overage to zero -- so the check
        # failed while reporting "OVER by 0 words", which reads like a bug in
        # the check rather than a fact about the document. An off-by-one in a
        # guard's *message* costs the guard its authority.
        import math
        excess = max(1, math.ceil((m_core - BUDGET_MIN) * WPM))
        print(f"  OVER by {excess} words. The brief asks for a hand-off "
              f"readable in five minutes. This drifts a clause at a time, "
              "each one locally reasonable; the fix is to move detail below "
              "the marker, not to delete a finding.")
    else:
        print(f"  within budget, {int((BUDGET_MIN - m_core) * WPM)} words spare")
    return 1 if (a.strict and over) else 0


if __name__ == "__main__":
    sys.exit(main())
