"""Does every family of runs appear in at least one generated document?

    python scripts/coverage_check.py        # exit 1 if anything is orphaned

The three generators substitute numbers from `runs/` into prose that lives in
the scripts. That makes the *tables* impossible to leave stale -- regenerate and
they are current. It does nothing for *structure*: a section is written in the
turn that produces its result, and a result that arrives after a section was
written does not retroactively appear in it.

That failed twice in two turns, both times on the same result. The pooling
sweep -- the only positive finding of the weekend -- was missing from
`WEEKEND.md` while five paragraphs of withdrawals were not, and missing from
`paper_draft.md`, whose only mentions of pooling listed it as an untried lever.
Both documents were regenerating perfectly and reporting numbers that were
current and complete for every section that existed.

So this enumerates the distinct experiment families in `runs/` and checks each
one is referenced somewhere. It is deliberately crude -- a substring search for
the tag and for the script that produced it -- because the failure it catches is
not subtle. Something that ran and is discussed nowhere is the thing to find.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DOCS = ("RESULTS.md", "paper_draft.md", "WEEKEND.md")
GENERATORS = ("scripts/report.py", "scripts/paper.py", "scripts/weekend.py")


def families(run_dir):
    """Distinct experiment families: cells grouped by tag, plus standalone JSONs."""
    tagged, standalone = defaultdict(int), {}
    for p in sorted(Path(run_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(r, dict) and {"protocol", "encoder", "objective"} <= set(r):
            tagged[r.get("tag", "") or "(untagged)"] += 1
        else:
            standalone[p.stem] = p
    return tagged, standalone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    text = {d: Path(d).read_text() for d in DOCS if Path(d).exists()}
    gen = {g: Path(g).read_text() for g in GENERATORS if Path(g).exists()}
    tagged, standalone = families(a.runs)

    # The first version of this check searched the *documents* for each run
    # file's name and reported 24 orphans, nearly all of them false: a document
    # discusses MIXED-SYNTH at length without ever writing the string
    # "mixedsynth__iid__bce__s0". The question that matters is whether any
    # generator *consumes* the result -- if none reads it, nothing can report
    # it, and that is the failure worth catching.
    def covered(key):
        return ([g for g, t in gen.items() if key in t]
                or [d for d, t in text.items() if key in t])

    orphans = []
    print(f"{'family':34s} {'cells':>6s}  read by")
    for tag, n in sorted(tagged.items(), key=lambda kv: -kv[1]):
        if tag == "(untagged)":
            continue
        stem = re.sub(r"[0-9._]+$", "", tag)
        where = covered(tag) or (covered(stem) if len(stem) > 3 else [])
        print(f"  {tag:32s} {n:6d}  "
              f"{', '.join(Path(w).name for w in where) if where else '** NOWHERE **'}")
        if not where:
            orphans.append(f"tag '{tag}' ({n} cells)")
    for stem, p in sorted(standalone.items()):
        # per-cell result files belong to their family, named by prefix
        fam = stem.split("__")[0]
        where = covered(stem) or covered(fam)
        print(f"  {stem:32s} {'json':>6s}  "
              f"{', '.join(Path(w).name for w in where) if where else '** NOWHERE **'}")
        if not where:
            orphans.append(f"standalone result '{stem}'")

    print()
    if orphans:
        print(f"{len(orphans)} orphaned result families -- they ran and no "
              f"generator reads them, so no document can report them:")
        for o in orphans:
            print(f"  - {o}")
        return 1
    print("every family of runs is referenced in at least one document")
    return 0


if __name__ == "__main__":
    sys.exit(main())
