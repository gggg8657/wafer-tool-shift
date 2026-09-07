"""Every hypothesis stated before a run, and whether it was ever scored.

Entry 101 ended on the only defensible reason to trust one of two p-values: a
post-hoc correction cannot tell an effect that was specified in advance from one
that was noticed, and only the record can. This project keeps that record in
script headers -- each sweep states its prediction before it launches -- and
scores it afterwards in `critique_log.md`.

A record like that has one failure mode, and it is silent: a hypothesis stated
before a run that goes badly and is never mentioned again. Nothing breaks, no
number is wrong, and the surviving prose is a list of predictions that came
true. Entry 85 already noted the risk in words -- "a loop that only records the
predictions it gets right is not keeping a record, it is keeping a highlight
reel" -- and words are what drifts.

So: extract every `H<n>` from the scripts that state hypotheses, extract every
`H<n>` the critique log scores, and report any stated-but-never-scored. That is
mechanically checkable and it is the property that matters. Whether a scored
hypothesis was scored *correctly* still needs reading; whether it was scored at
all does not.

    python scripts/hypothesis_ledger.py
    python scripts/hypothesis_ledger.py --strict   # exit 1 on an unscored one
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

H = re.compile(r"\bH(\d{2,3})\b")
LEDGER = Path("runs/hypothesis_ledger.json")


def stated():
    """Hypotheses stated in a sweep's header, with the script that states it."""
    out = defaultdict(list)
    for p in sorted(Path("scripts").glob("*.sh")) + sorted(
            Path("scripts").glob("*.py")):
        if p.name == "hypothesis_ledger.py":
            continue          # this file names hypotheses to explain itself
        text = p.read_text()
        for m in H.finditer(text):
            # a statement, not a back-reference: the sweep headers say
            # "H63, before the run:" or "H70, on record:"
            tail = text[m.end():m.end() + 40]
            if re.match(r"\s*(,|\s)\s*(before|on record|,)", tail):
                out[int(m.group(1))].append(p.name)
    return {k: sorted(set(v)) for k, v in out.items()}


# A mention is not an outcome. The first version of this counted any `H<n>`
# in any entry, so H76 -- stated in entry 100 and still running -- read as
# scored by the very entry that launched it. An outcome needs a verdict word
# near the mention.
STATEMENT = re.compile(
    r"\s*(,\s*)?(on record|before (?:the|any) run|before computing|"
    r"before the data|predicted)", re.I)
VERDICT = re.compile(
    r"\b(confirmed|falsifi\w+|scored|right|wrong|holds|held|survives?|"
    r"survived|does not|did not|withdraw\w*|reverses?|dissolv\w+|"
    r"replicates?|replicated|of (?:four|six|two)|correct)\b", re.I)


def scored():
    """Hypotheses the critique log reports an *outcome* for, not merely names."""
    text = Path("critique_log.md").read_text()
    out = defaultdict(list)
    for block in re.split(r"^### ", text, flags=re.M)[1:]:
        head = block.split("\n", 1)[0]
        n = re.match(r"(\d+)\.", head)
        entry = int(n.group(1)) if n else None
        for m in H.finditer(block):
            # A prediction's own text contains outcome words, because it
            # predicts an outcome: entry 100 says "H76 on record: the
            # interaction replicates". So a mention followed by a statement
            # marker is a statement, whatever words surround it, and only the
            # rest can be a scoring.
            tail = block[m.end():m.end() + 30]
            if STATEMENT.match(tail):
                continue
            lo, hi = max(0, m.start() - 200), min(len(block), m.end() + 400)
            if VERDICT.search(block[lo:hi]):
                out[int(m.group(1))].append(entry)
    return {k: sorted({v for v in vs if v is not None})
            for k, vs in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    st, sc = stated(), scored()
    rows = []
    for n in sorted(set(st) | set(sc)):
        rows.append({"hypothesis": f"H{n}",
                     "stated_in": st.get(n, []),
                     "scored_in_entries": sc.get(n, []),
                     "stated": n in st, "scored": bool(sc.get(n))})
    # A hypothesis whose sweep is still running is not a gap in the record.
    # The failure mode is a sweep that *finished* and was never scored, so
    # completion is read from the stage's own log rather than assumed.
    def sweep_done(scripts):
        if not scripts:
            return False
        for name in scripts:
            log = Path("logs") / (Path(name).stem + ".log")
            if log.exists() and "done ===" in log.read_text():
                return True
        return False

    for r in rows:
        r["sweep_finished"] = sweep_done(r["stated_in"])
    unscored = [r for r in rows
                if r["stated"] and not r["scored"] and r["sweep_finished"]]
    in_flight = [r for r in rows
                 if r["stated"] and not r["scored"] and not r["sweep_finished"]]
    res = {"what": "every hypothesis stated before a run, and whether the "
                   "critique log ever reports its outcome",
           "why": "a prediction that went badly and was never mentioned again "
                  "leaves a record of predictions that came true",
           "limit": "this checks that a hypothesis was scored, not that it was "
                    "scored correctly -- that still needs reading",
           "n_stated": sum(1 for r in rows if r["stated"]),
           "n_scored": sum(1 for r in rows if r["scored"]),
           "n_stated_never_scored": len(unscored),
           "n_in_flight": len(in_flight),
           "hypotheses": rows}
    LEDGER.write_text(json.dumps(res, indent=2))

    print(f"{res['n_stated']} hypotheses stated in sweep headers, "
          f"{res['n_scored']} scored somewhere in the critique log")
    for r in rows:
        mark = ("  " if r["scored"]
                else ("running " if not r["sweep_finished"] else "UNSCORED"))
        where = ", ".join(r["stated_in"]) or "-"
        ent = ("entry " + ", ".join(str(e) for e in r["scored_in_entries"])
               if r["scored_in_entries"] else "never")
        print(f"  {mark} {r['hypothesis']:5s} {where[:34]:34s} {ent}")
    if unscored:
        print(f"\n{len(unscored)} hypothesis(es) whose sweep finished and "
              "which were never scored. That is the one failure mode this "
              "record has: the surviving prose becomes a list of predictions "
              "that came true.")
    else:
        print("\nevery hypothesis whose sweep has finished has an outcome "
              "in the log"
              + (f"; {len(in_flight)} still running" if in_flight else ""))
    return 1 if (a.strict and unscored) else 0


if __name__ == "__main__":
    sys.exit(main())
