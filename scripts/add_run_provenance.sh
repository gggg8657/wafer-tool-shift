#!/usr/bin/env bash
# Record the commit and wall-clock time in every future run JSON.
#
# `scripts/run_provenance.py` reports that 625 stored cells span twelve distinct
# states of `run_bench.py` and that none of them records which one produced it.
# Every table in this project compares cells against each other, and this repo
# has already measured that provenance of a weaker kind -- which session, which
# GPU -- moves numbers by about twice the within-session seed spread.
#
# The change is two fields and cannot affect any measurement. It is applied by
# this script rather than by hand because a sweep was in flight when the gap was
# found, and editing the runner mid-sweep would have left half its cells
# labelled and half not -- an inhomogeneity introduced by the act of recording
# provenance, which would be a poor joke.
#
#   bash scripts/add_run_provenance.sh     # waits for the lease, then patches
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/add_run_provenance.log
mkdir -p logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== waiting for any sweep to release the lease ==="
while tmux has-session -t p2e 2>/dev/null; do sleep 30; done
say "=== lease free ==="

PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
$PY - <<'PYEOF' | tee -a "$LOG"
from pathlib import Path
p = Path("scripts/run_bench.py")
s = p.read_text()
anchor = '            "pool": a.pool,\n'
if "git_sha" in s:
    print("already records provenance; nothing to do")
    raise SystemExit(0)
assert anchor in s, "anchor for the record dict not found"
add = ('            # which code and when. 625 cells were written without this\n'
       '            # across twelve states of this file, and every table here\n'
       '            # compares cells to each other.\n'
       '            "git_sha": _git_sha(), "written_at": _now_iso(),\n')
s = s.replace(anchor, anchor + add, 1)

helper = '''

def _git_sha():
    """Short commit of the working tree, with a marker if it is dirty."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return (sha + ("+dirty" if dirty else "")) or None
    except Exception:
        return None


def _now_iso():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
'''
marker = "\nREF_SEED" if "\nREF_SEED" in s else "\ndef main("
s = s.replace(marker, helper + marker, 1)
p.write_text(s)
print("patched run_bench.py to record git_sha and written_at")
PYEOF

$PY -c "import ast;ast.parse(open('scripts/run_bench.py').read());print('run_bench.py parses')" | tee -a "$LOG"
$PY scripts/run_provenance.py | tail -2 | tee -a "$LOG"
say "=== done ==="
