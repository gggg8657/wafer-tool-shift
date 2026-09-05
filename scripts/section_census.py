"""Which sections did a generator decline to render, and why?

    python scripts/section_census.py            # report
    python scripts/section_census.py --strict   # exit 1 if any section skipped

Three guards exist in this repository and none of them catches this.
`verify_stage.py` counts files, so it cannot see a wrong number.
`coverage_check.py` finds results no generator reads, so it cannot see a
generator that read a result and emitted nothing. The numeric lint finds
hand-typed constants, so it cannot see a live computation with a stale divisor.
Every one detects *an absent file* or *a bad literal*.

What none of them detects is a document that is quietly shorter than it should
be. That happened: a survivors row in `WEEKEND.md` asserting the pooling gain is
not extra parameters was guarded by `if pc:`, the file it read had a trailing
underscore in its name, the guard was false, and the row vanished. I regenerated
the document, read the table, and committed — because a section that omits
itself looks exactly like a section that was never written.

So this walks each generator's source for the optional inputs it reads
(`js("x.json")`, `Path(a.runs) / "x.json"`, and f-string families) and reports
which are present and which are missing. A missing input is a section that did
not render. It does not prove the document is complete; it does turn one class
of silent omission into a printed line.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GENERATORS = ("scripts/report.py", "scripts/paper.py", "scripts/weekend.py")

# Every string literal naming a .json, wherever it appears.
#
# The first version of this matched only `js("literal")`, `Path(a.runs) /
# "literal"` and a hand-picked f-string family. Tested against the bug it was
# built for -- a file renamed so a section would vanish -- it reported
# "21/21 present" instead of flagging the miss, because the real call site
# passes the names through a tuple:
#
#     for fn in ("pooling_lot_control_perm_macro_f1.json", ...):
#         d_ = js(fn)
#
# so the name never appears inside `js(...)`. A guard that has never been made
# to fail is not a guard; this one was, and this is the second version.
PATTERNS = (
    re.compile(r'"((?:[A-Za-z0-9_./{}-]+)\.json)"'),
    re.compile(r"'((?:[A-Za-z0-9_./{}-]+)\.json)'"),
)
# The generators name inputs three ways -- "x.json", "runs/x.json", and
# Path(a.runs) / "x.json" -- so normalise before deciding. Only the documents
# themselves are outputs; everything else under runs/ is an input.
OUTPUTS = ("RESULTS.md", "paper_draft.md", "WEEKEND.md")


def normalise(n):
    n = n.split("/")[-1]
    return None if n in OUTPUTS or not n.endswith(".json") else n


def expand(name, runs):
    """An f-string family expands to whatever files match its literal parts."""
    if "{" not in name:
        return [name]
    frags = [f for f in re.split(r"\{[^}]*\}", name) if f]
    out = []
    for p in Path(runs).glob("*.json"):
        if all(fr in p.name for fr in frags):
            out.append(p.name)
    return out or [name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    missing = []
    for g in GENERATORS:
        src = Path(g)
        if not src.exists():
            continue
        names = set()
        for pat in PATTERNS:
            for raw in pat.findall(src.read_text()):
                n = normalise(raw)
                if n:
                    names.add(n)
        rows = []
        for n in sorted(names):
            for real in expand(n, a.runs):
                present = (Path(a.runs) / real).exists()
                rows.append((present, real))
                if not present:
                    missing.append((g, real))
        print(f"{src.name}: {sum(1 for p, _ in rows if p)}/{len(rows)} optional "
              f"inputs present")
        for present, real in rows:
            if not present:
                print(f"    MISSING  {real}  -> a section did not render")

    print()
    if missing:
        print(f"{len(missing)} optional input(s) missing. Each one is a section "
              f"the generator skipped without saying so:")
        for g, n in missing:
            print(f"  - {Path(g).name} wanted runs/{n}")
        print("A section that omits itself looks exactly like a section that "
              "was never written. Check before trusting the document's length.")
        return 1 if a.strict else 0
    print("every optional input a generator reads is present; no section "
          "silently skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
