"""Flag prose in the generators that asserts a *status* rather than a number.

    python scripts/prose_status_lint.py          # report
    python scripts/prose_status_lint.py --strict # exit 1 if anything is flagged

The generators guarantee that no number is typed by hand. Every failure this
project has had in its documents was in the prose *around* the numbers, and
specifically in sentences that assert a state of the work rather than a
measurement:

  "the `size` half of the negative result stands as measured"   -- falsified
  "the size of the resulting bias is [not measured]"            -- since measured
  "in a proportion that section 2.1 will quantify and currently does not"
                                                                -- since quantified
  "its two columns must agree exactly"                          -- they no longer do
  "we do not attach a p-value to it"                            -- we now do
  "`size` and `lot_time` are still running"                     -- they finished

Six of these in one reading of `paper_draft.md`, plus two in `WEEKEND.md`. They
share a shape: a claim whose truth depends on the state of `runs/` at some past
moment, written into a script that regenerates against `runs/` as it is now.

Nothing can check them automatically -- deciding whether "stands as measured"
is still true requires knowing what was measured. What *can* be done is find
them, so that re-verifying is a short list rather than a re-read of 8,000 words.
This is a reminder, not a verifier, and it is deliberately noisy in the
direction of flagging too much.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SOURCES = ("scripts/report.py", "scripts/paper.py", "scripts/weekend.py")
DOCS = ("RESULTS.md", "paper_draft.md", "WEEKEND.md")

# phrases asserting the state of the work rather than a measured quantity
PHRASES = [
    (r"not measured", "a metric declared unmeasured -- has it since been measured?"),
    (r"still running|is running|are running|in flight|queued",
     "a claim about a job's state -- has it finished?"),
    (r"will (?:quantify|measure|answer|settle|show|tell)",
     "a promise about future work -- has it been done?"),
    (r"stands as measured|survives as measured|remains? (?:true|valid)",
     "a claim that a prior result still holds -- re-check it"),
    (r"must agree (?:exactly|precisely)|identical by construction",
     "an assertion of exact equality -- verify against the table beside it"),
    (r"do(?:es)? not (?:attach|report|use) a p-value|no p-value",
     "a methodological disclaimer -- is it still the method?"),
    (r"cannot be (?:computed|measured|obtained)|could not be (?:acquired|obtained)",
     "an impossibility claim -- still impossible?"),
    (r"pending|to be (?:run|measured|decided)",
     "deferred work -- still deferred?"),
]


def duplicated_blocks(path: Path, minlen: int = 120):
    """Paragraphs that render more than once in the same document.

    The generators are loops over protocols and objectives, and an indentation
    slip puts a section body inside the loop that was meant to populate its
    inputs. That is not hypothetical: section 2.0 of `WEEKEND.md` rendered four
    times for two commits. Every table in it was correct, every number came
    from `runs/`, `section_census.py` saw all its files present, and
    `coverage_check.py` saw every JSON consumed -- because each of those asks
    about content, and this is a defect in *shape*. Nothing was wrong except
    that the document was 145 lines where it should have been 41.

    A paragraph long enough to be prose should appear once.
    """
    seen, dups = {}, []
    for para in path.read_text().split("\n\n"):
        t = " ".join(para.split())
        if len(t) < minlen or t.startswith("|"):
            continue
        if t in seen:
            dups.append((t[:88], seen[t]))
        else:
            seen[t] = t[:88]
    return dups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    total = 0
    for src in SOURCES:
        p = Path(src)
        if not p.exists():
            continue
        hits = []
        for i, line in enumerate(p.read_text().split("\n"), 1):
            s = line.strip()
            # prose only: a quoted string that is not obviously code
            if '"' not in s and "'" not in s:
                continue
            for pat, why in PHRASES:
                if re.search(pat, s, re.I):
                    hits.append((i, why, s[:96]))
                    break
        if hits:
            print(f"\n{p.name}: {len(hits)} status assertion(s) to re-verify")
            for i, why, s in hits:
                print(f"  L{i:<5} {why}")
                print(f"         {s}")
        total += len(hits)

    # ---- document side: what actually shipped, not what the source contains.
    # The scan above reads generator source, so it cannot tell a sentence that
    # renders from one sitting in an `else:` fallback that never fires. Three
    # of its flags are exactly that. This pass reads the rendered documents.
    shipped = n_dups = 0
    for doc in DOCS:
        d = Path(doc)
        if not d.exists():
            continue
        hits = []
        for i, line in enumerate(d.read_text().split("\n"), 1):
            for pat, why in PHRASES:
                if re.search(pat, line, re.I):
                    hits.append((i, why, line.strip()[:96]))
                    break
        dups = duplicated_blocks(d)
        if hits or dups:
            print(f"\n{doc}: {len(hits)} shipped status assertion(s)"
                  + (f", {len(dups)} duplicated paragraph(s)" if dups else ""))
            for i, why, ln in hits:
                print(f"  L{i:<5} {why}")
                print(f"         {ln}")
            for t, _ in dups:
                print(f"  DUP    paragraph renders more than once -- a section "
                      f"body inside its own input loop?")
                print(f"         {t}")
        shipped += len(hits)
        n_dups += len(dups)

    print(f"\n{total} status assertion(s) across {len(SOURCES)} generators, "
          f"{shipped} in the {len(DOCS)} rendered documents, "
          f"{n_dups} duplicated paragraph(s).")
    print("None of these can be checked automatically: deciding whether "
          "'stands as measured' is still true needs to know what was measured. "
          "The point is that re-verifying is now a short list rather than a "
          "re-read of 8,000 words.")
    if n_dups:
        print("A duplicated paragraph is a defect with no judgement in it: a "
              "section body left inside the loop that populates its inputs. "
              "--strict fails on those and only those; the status assertions "
              "above are a re-read list, not errors.")
    return 1 if (a.strict and n_dups) else 0


if __name__ == "__main__":
    sys.exit(main())
