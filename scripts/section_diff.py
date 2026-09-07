"""Warn when regenerating a document *removes* a section it used to have.

Every other guard here reasons about inputs and generator source: whether a JSON
is present (`section_census`), whether it is read by anything (`coverage_check`),
whether a sentence asserts a stale status or a paragraph repeats
(`prose_status_lint`). None of them compares the rendered document against what
it contained last time, and that gap has a demonstrated cost.

Section 7.9 of `paper_draft.md` -- the section reporting how much each of this
paper's nulls could have shown -- was guarded by
`if npa and npa.get("n_underpowered")`. When the last underpowered null was
taken to eight seeds that count became zero and **the entire section vanished**,
deleting the evidence that the problem had been fixed. All six checks passed on
the shorter paper: the JSON was present, it was read, nothing was duplicated,
no sentence was stale.

The shape worth naming is that the section was conditioned on the *presence of a
defect*, so it could only disappear on success -- which means no amount of
testing against a broken repository would ever have shown it. A document-shape
comparison catches it because it does not care why a heading went away.

    python scripts/section_diff.py            # compare against the snapshot
    python scripts/section_diff.py --accept   # record the current shape
    python scripts/section_diff.py --strict   # exit 1 if a heading vanished

Added headings are reported and never fail: documents are meant to grow. Only
removal is an error, and it is an error even when deliberate -- in which case
`--accept` is the way to say so, which leaves a diff in git rather than silence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DOCS = ("RESULTS.md", "paper_draft.md", "WEEKEND.md")
SNAPSHOT = Path("state/section_shape.json")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def headings(path: Path):
    """Heading text by level, in order.

    Only *measured* values are normalised, not section numbers. The first
    version stripped every digit so that a heading carrying a number would not
    read as removed when the number moved -- which also erased the numbering,
    making `## 7.9 Foo` and `## 8.1 Foo` indistinguishable and hiding exactly
    the kind of renumbering a reader would want flagged. No heading in these
    three documents currently carries a measured value, so the normalisation is
    narrow: a decimal with three or more fraction digits, or a percentage.
    """
    out = []
    for line in path.read_text().split("\n"):
        m = HEADING.match(line)
        if m:
            text = re.sub(r"[-+]?\d+\.\d{3,}|\d+(?:\.\d+)?%", "#",
                          m.group(2)).strip()
            out.append(f"{len(m.group(1))}:{text}")
    return out


def compare(prev, cur):
    """Removed, added, and renamed — with renames not counted as removals.

    Some headings embed a data-derived label: "Per-class F1, best `lot` cell
    (CNN + RPCA lot-signature channel, erm)" names whichever cell is currently
    best, so it changes whenever the ranking does. Reporting that as a section
    disappearing plus another appearing is technically true and practically
    noise, and a guard that cries wolf on ordinary regeneration teaches its
    reader to pass `--accept` without looking — which is how a check stops
    being a check.

    A removed and an added heading at the same level sharing a long prefix are
    treated as one heading renamed. Still reported, never a failure.
    """
    removed = [h for h in prev if h not in cur]
    added = [h for h in cur if h not in prev]
    renamed, still_removed = [], []
    for h in removed:
        lvl, _, text = h.partition(":")
        match = None
        for g in added:
            glvl, _, gtext = g.partition(":")
            if glvl != lvl:
                continue
            common = len(os.path.commonprefix([text, gtext]))
            if common >= 20 and common >= 0.5 * min(len(text), len(gtext)):
                match = g
                break
        if match:
            renamed.append((h, match))
            added.remove(match)
        else:
            still_removed.append(h)
    return still_removed, added, renamed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    cur = {d: headings(Path(d)) for d in DOCS if Path(d).exists()}
    if a.accept or not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(
            {"what": "headings each generated document contained when last "
                     "accepted, so a regeneration that removes one is visible",
             "docs": cur}, indent=2))
        print(f"recorded {sum(len(v) for v in cur.values())} headings across "
              f"{len(cur)} documents"
              + ("" if a.accept else " (no snapshot existed)"))
        return 0

    prev = json.loads(SNAPSHOT.read_text()).get("docs", {})
    n_removed = 0
    for doc in DOCS:
        removed, added, renamed = compare(prev.get(doc, []), cur.get(doc, []))
        if removed or added or renamed:
            print(f"\n{doc}:")
        for h in removed:
            print(f"  REMOVED  {h.split(':', 1)[1]}")
        for a_, b_ in renamed:
            print(f"  renamed  {a_.split(':', 1)[1]}")
            print(f"        -> {b_.split(':', 1)[1]}")
        for h in added:
            print(f"  added    {h.split(':', 1)[1]}")
        n_removed += len(removed)

    if n_removed:
        print(f"\n{n_removed} heading(s) disappeared. A section can vanish "
              "because its guard was conditioned on a problem that has since "
              "been fixed, which deletes the record of the fix. If the removal "
              "is intended, `--accept` records it and leaves a diff in git.")
    else:
        print("no heading removed since the last accepted shape")
    return 1 if (a.strict and n_removed) else 0


if __name__ == "__main__":
    sys.exit(main())
