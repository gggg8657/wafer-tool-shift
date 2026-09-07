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


# Detecting "was this scored?" from prose is not reliably automatable, and I
# spent three attempts proving it. Counting any mention marked H76 as scored by
# the entry that launched it. Requiring a verdict word nearby failed too, since
# a prediction's own text contains outcome words. Tightening the window then
# failed on entry 102, which is *about* this ledger -- prose discussing the
# record is indistinguishable from prose scoring it.
#
# So the verdict is recorded explicitly instead. `state/hypothesis_outcomes.json`
# holds one line per hypothesis with a verdict and the entry that argues it, and
# this script's job is to enforce that none is missing. The judgement stays
# human, which it has to be -- H74 was satisfied on a literal reading and
# reported as a failure, and no pattern would have made that call. What a
# machine can do is refuse to let a stated hypothesis have no recorded verdict.
OUTCOMES = Path("state/hypothesis_outcomes.json")
VERDICTS = {"confirmed", "falsified", "partly", "superseded", "in_flight"}


def scored():
    """Explicit verdicts, keyed by hypothesis number."""
    if not OUTCOMES.exists():
        return {}
    d = json.loads(OUTCOMES.read_text()).get("outcomes", {})
    return {int(k.lstrip("Hh")): v for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    st, sc = stated(), scored()
    rows = []
    for n in sorted(set(st) | set(sc)):
        o = sc.get(n) or {}
        rows.append({"hypothesis": f"H{n}",
                     "stated_in": st.get(n, []),
                     "verdict": o.get("verdict"),
                     "entry": o.get("entry"),
                     "stated": n in st,
                     "scored": bool(o.get("verdict"))})
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
    in_flight = [r for r in rows if r["verdict"] == "in_flight"
                 or (r["stated"] and not r["scored"]
                     and not r["sweep_finished"])]
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
        v = r["verdict"] or "-"
        ent = f"entry {r['entry']}" if r.get("entry") else ""
        print(f"  {mark} {r['hypothesis']:5s} {where[:30]:30s} "
              f"{v:11s} {ent}")
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
