"""Do the three generators still run, and does each still write its document?

`check_all.py` had nine checks and none of them noticed when `paper.py` crashed.
An indentation error in a patch made it raise before writing, the previous
`paper_draft.md` stayed on disk, and every check passed on the stale file:
its inputs were present, its runs were referenced, nothing was duplicated, no
number had been typed, no heading had vanished, the core was under budget. All
true, and all about a document the generator could no longer produce.

That is the third time in this session a patch has died before writing while
surrounding commands printed successes. The first two were caught by reading the
rendered document for the new sentence; this one was caught only because the
`IndentationError` happened to print above a `PASS`.

So: run each generator to a temporary output and require a zero exit and a
non-empty file. It is the cheapest possible check and it covers the failure that
made the other nine meaningless.

    python scripts/generators_run.py
    python scripts/generators_run.py --strict
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

PY = sys.executable
GENERATORS = [
    ("scripts/report.py", "RESULTS.md"),
    ("scripts/paper.py", "paper_draft.md"),
    ("scripts/weekend.py", "WEEKEND.md"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for script, doc in GENERATORS:
            out = Path(d) / doc
            r = subprocess.run([PY, script, "--out", str(out)],
                               capture_output=True, text=True)
            wrote = out.exists() and out.stat().st_size > 0
            ok = r.returncode == 0 and wrote
            bad += 0 if ok else 1
            n = f"{out.stat().st_size:,} bytes" if wrote else "wrote nothing"
            print(f"  {'ok  ' if ok else 'FAIL'} {script:22s} {n}")
            if not ok:
                tail = (r.stderr or r.stdout).strip().split("\n")[-3:]
                for line in tail:
                    print(f"         {line}")

    if bad:
        print(f"\n{bad} generator(s) cannot produce their document. Every "
              "other check in this repository reads the file on disk, which "
              "is the *last successful* output — so they all keep passing "
              "while the generator is broken.")
    else:
        print(f"\nall {len(GENERATORS)} generators run and write")
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
