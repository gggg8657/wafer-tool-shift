"""Run every check and print one verdict, because reading six is how I missed one.

The previous commit shipped with the typed-decimal ratchet failing. I ran the
check, saw 45 against a baseline of 44, applied a fix, saw it *still* read 45,
and committed — because the output I read was the `ok` printed by my own patch
script rather than the guard's line, several hundred characters earlier in a
combined stdout. Nothing was wrong with the guard.

Six checks with six tails is a reading problem, and reading problems do not get
solved by being more careful. This runs all of them, prints one line each, and
ends with a single verdict and a single exit code.

    python scripts/check_all.py
    python scripts/check_all.py --quiet    # one line per check, no detail

Each entry names the *question* the check answers, not the script, because
"section_census passed" tells a reader nothing they can act on.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (question the check answers, argv, treat nonzero as failure)
CHECKS = [
    ("do the unit tests pass?",
     [PY, "tests/run_tests.py"], True),
    ("do the guards themselves catch a defect they claim to catch?",
     [PY, "scripts/guard_audit.py"], False),          # 1 known-vacuous entry
    ("is every input a generator reads present?",
     [PY, "scripts/section_census.py"], True),
    ("is every family of runs reported in some document?",
     [PY, "scripts/coverage_check.py"], True),
    ("does any document duplicate a section?",
     [PY, "scripts/prose_status_lint.py", "--strict"], True),
    ("has anyone typed a new number into generator prose?",
     [PY, "scripts/number_provenance.py", "--strict"], True),
    ("did a regeneration silently remove a section?",
     [PY, "scripts/section_diff.py", "--strict"], True),
]


def run(checks, cwd=ROOT):
    rows = []
    for question, argv, strict in checks:
        r = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
        ok = (r.returncode == 0) if strict else True
        rows.append((question, ok, r.returncode, (r.stdout + r.stderr)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rows = run(CHECKS)
    w = max(len(q) for q, *_ in rows)
    print()
    for question, ok, code, out in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {question:{w}s}")
        if not ok and not a.quiet:
            for line in out.strip().split("\n")[-6:]:
                print(f"        {line}")
    n_bad = sum(1 for _, ok, _, _ in rows if not ok)
    print()
    if n_bad:
        print(f"  {n_bad} of {len(rows)} checks FAILED — do not commit.")
    else:
        print(f"  all {len(rows)} checks pass.")
    # the one check that is informational rather than pass/fail
    audit = next((o for q, _, _, o in rows if "guards themselves" in q), "")
    for line in audit.split("\n"):
        if "of" in line and "guards demonstrably" in line:
            print(f"  ({line.strip()})")
    print()
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
