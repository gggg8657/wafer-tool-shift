"""Are the generated documents' section numbers unique and in order?

`section_diff.py` tracks headings by text, so it cannot see two sections
sharing a number. Renumbering five subsections into document order last turn
produced **two sections numbered 7.1** -- one pre-existing, one moved -- and
every check passed on the result. The numbering had also drifted into `4.1`,
`4.2` as `###` beside `4.3` as `##`, and a section 5 whose only children were
`5.5` and `5.6`, both artefacts of inserting sections where they were easiest
to insert rather than where they belonged.

None of that is a wrong number in a table. It is a document a reader cannot
navigate, which is the same class of defect as the twenty-two-minute hand-off:
locally reasonable edits, globally incoherent shape.

Checks, per document: no number appears twice; numbers increase in document
order; a subsection's parent exists; and heading level matches depth (`x.y` is
`###`, `x` is `##`).

    python scripts/heading_numbers.py
    python scripts/heading_numbers.py --strict
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCS = ("RESULTS.md", "paper_draft.md", "WEEKEND.md")
# The number may be followed by a period ("## 2. What actually shifts") or
# by a letter ("### 2.3a The sinkhorn cell"). The first version required
# whitespace immediately after the digits, so it matched none of the
# top-level headings and then reported every subsection as parentless --
# fourteen problems in a document whose numbering was correct.
H = re.compile(r"^(#{2,6})\s+(\d+(?:\.\d+)*)([a-z]?)\.?\s+(.*)$")


def headings(path):
    out = []
    for line in path.read_text().split("\n"):
        m = H.match(line)
        if m:
            out.append({"level": len(m.group(1)), "number": m.group(2),
                        "suffix": m.group(3), "title": m.group(4).strip()})
    return out


def problems(hs):
    bad = []
    seen = {}
    prev = None
    for h in hs:
        n = h["number"] + h.get("suffix", "")
        if n in seen:
            bad.append(f"number {n} used twice: '{seen[n]}' and '{h['title']}'")
        seen[n] = h["title"]
        depth = h["number"].count(".") + 1
        want = depth + 1                      # "7" -> ##, "7.1" -> ###
        if h["level"] != want:
            bad.append(f"{n} '{h['title'][:40]}' is at level "
                       f"{'#' * h['level']} but its depth wants "
                       f"{'#' * want}")
        if depth > 1:
            parent = h["number"].rsplit(".", 1)[0]
            if parent not in seen:
                bad.append(f"{n} has no parent section {parent}")
        # Compare the numeric parts at their shared depth, using the letter
        # suffix only to break a tie -- "2.3a" comes after "2.3". Two earlier
        # attempts put the suffix in the same tuple as the integers, which made
        # `(2, 0) <= (1, "")` compare an int against a str.
        nums = tuple(int(x) for x in h["number"].split("."))
        suf = h.get("suffix") or ""
        if prev is not None:
            pn, ps = prev
            d = min(len(nums), len(pn))
            if nums[:d] < pn[:d] or (nums[:d] == pn[:d]
                                     and len(nums) == len(pn)
                                     and suf <= ps):
                bad.append(f"{n} does not increase on "
                           f"{'.'.join(map(str, pn))}{ps}")
        prev = (nums, suf)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    total = 0
    for d in DOCS:
        p = Path(d)
        if not p.exists():
            continue
        bad = problems(headings(p))
        if bad:
            print(f"\n{d}: {len(bad)} numbering problem(s)")
            for b in bad:
                print(f"  {b}")
        total += len(bad)
    print(f"\n{total} numbering problem(s) across {len(DOCS)} documents"
          + ("" if total else " -- numbers unique, ordered, and at the right "
                             "heading level"))
    return 1 if (a.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
