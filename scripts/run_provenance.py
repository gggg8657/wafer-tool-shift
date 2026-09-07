"""Which version of the runner produced each cell? Currently: unrecorded.

Every table in this project compares cells against each other, and the runner
that produced them changed thirteen times over the weekend -- new flags, an
`invariance_domain` refactor, a geometry decomposition, a batch dict that gained
a field. Two of those changes were asserted bit-identical on the default path
and the rest were additive, which is the *intent*; nothing in any cell records
which version actually wrote it, so the assertion cannot be checked per cell.

This project has already established that provenance of this kind matters. Seed
spread understates total variability because a cell's seeds run back to back,
and the Friday cells measured on different GPUs sit outside the range of six
repeats of the same configuration. A code version is a stronger form of the same
concern than a session or a device.

What can be measured after the fact is only an upper bound on the exposure: file
modification times say which commits to `scripts/run_bench.py` preceded each
cell. That is a proxy -- a file can be touched without being rewritten, and a
commit can change nothing a given cell uses -- so this reports the spread and
does not claim any cell is wrong.

The fix is forward-looking and cheap: record the commit in each run. That is
queued rather than applied here, because a sweep is in flight and changing the
runner mid-sweep would leave half its cells labelled and half not.

    python scripts/run_provenance.py     # writes runs/run_provenance.json
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

RUNNER = "scripts/run_bench.py"


def commits_for(path):
    out = subprocess.run(["git", "log", "--format=%H %ct", "--", path],
                         capture_output=True, text=True).stdout.strip()
    rows = [(h, int(t)) for h, t in (l.split() for l in out.split("\n") if l)]
    return sorted(rows, key=lambda r: r[1])


def main():
    runs = sorted(glob.glob("runs/*.json"))
    commits = commits_for(RUNNER)
    if not commits:
        raise SystemExit("no git history for " + RUNNER)

    n_recording = 0
    buckets = Counter()
    for f in runs:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if isinstance(d, dict) and ("git_sha" in d or "commit" in d):
            n_recording += 1
        m = os.path.getmtime(f)
        buckets[sum(1 for _, t in commits if t <= m)] += 1

    states = [{"n_preceding_commits": n,
               "commit": commits[n - 1][0] if n else None,
               "committed_at": (dt.datetime.fromtimestamp(commits[n - 1][1])
                                .isoformat() if n else None),
               "n_cells": c}
              for n, c in sorted(buckets.items())]

    res = {
        "what": "how many distinct states of the runner the stored cells span, "
                "and how many cells record the version that produced them",
        "runner": RUNNER,
        "n_runner_commits": len(commits),
        "n_run_files": len(runs),
        "n_cells_recording_their_commit": n_recording,
        "n_distinct_code_states_spanned": len(states),
        "method": "file mtime against commit time -- an upper bound on "
                  "exposure, not a claim that any cell is wrong. A commit may "
                  "change nothing a given cell uses.",
        "states": states,
    }
    Path("runs/run_provenance.json").write_text(json.dumps(res, indent=2))
    print(f"{len(runs)} run files span {len(states)} states of {RUNNER} "
          f"({len(commits)} commits total)")
    print(f"{n_recording} of {len(runs)} record the commit that produced them")
    for st in states:
        print(f"  {st['n_cells']:4d} cells after "
              f"{st['n_preceding_commits']:2d} commits"
              + (f" ({st['commit'][:8]})" if st["commit"] else ""))
    print("\nwrote runs/run_provenance.json")


if __name__ == "__main__":
    main()
