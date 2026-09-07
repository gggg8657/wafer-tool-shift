"""Feed every guard a defect it claims to catch, and record which ones do.

Twice in one session I built a check whose pass condition was close to vacuous:

  * `number_provenance.py` reported "43 of 44 traceable" and, measured against
    random draws, accepts 100% of three-digit decimals and 86% of four-digit
    ones. `runs/` is simply a large enough haystack that almost any number
    matches something.
  * `resize_fidelity.py`'s first control compared per-class retention against
    what a uniform scatter of the same wafer size would give. Every class came
    in at 0.995-1.006, which looked decisive and could not have come out any
    other way -- a row survives with probability |yi|/h whatever is drawn on
    it, so the mean ratio is 1 for *any* defect shape by construction.

Both had the same shape: an observation compared against a quantity that does
not depend on the thing being tested. Neither was caught by running the guard;
both were caught by asking afterwards what the guard would have said if the
defect had been present.

This makes that question routine. Each entry below constructs a specific,
realistic defect -- the ones this project actually shipped -- runs the guard on
it, and asserts the guard objects. A guard that passes its own broken input is
reported as **VACUOUS**, which is a stronger complaint than a failing test:
it means every green result that guard has ever produced was uninformative.

    python scripts/guard_audit.py
    python scripts/guard_audit.py --strict     # exit 1 if any guard is vacuous
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------- guards

def audit_verify_stage():
    """A stage that produced fewer cells than it launched must fail."""
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            (Path(d) / f"c__s{i}.json").write_text("{}")
        r = subprocess.run(
            [PY, str(ROOT / "scripts/verify_stage.py"),
             "--glob", str(Path(d) / "c__s*.json"),
             "--expect", "8", "--label", "audit"],
            capture_output=True, text=True)
        caught = r.returncode != 0
        # and the honest case must pass, or it is merely always-failing
        r2 = subprocess.run(
            [PY, str(ROOT / "scripts/verify_stage.py"),
             "--glob", str(Path(d) / "c__s*.json"),
             "--expect", "3", "--label", "audit"],
            capture_output=True, text=True)
        return caught and r2.returncode == 0, "3 cells where 8 were expected"


def audit_duplicated_blocks():
    """A section rendered twice must be flagged; a repeated table must not."""
    m = load("lint", "scripts/prose_status_lint.py")
    para = ("A paragraph long enough to count as prose and not as a heading, "
            "which is the distinction the detector draws, so it must be "
            "exercised at a realistic length.")
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d, "bad.md")
        bad.write_text(f"# t\n\n{para}\n\n{para}\n")
        good = Path(d, "good.md")
        good.write_text("# t\n\n| a | b |\n|---|---|\n\n| a | b |\n|---|---|\n")
        return (len(m.duplicated_blocks(bad)) == 1
                and m.duplicated_blocks(good) == []), \
            "a section body left inside its own input loop"


def audit_section_census():
    """A vanished input must be reported, not silently skipped."""
    m = load("census", "scripts/section_census.py")
    src = 'for fn in ("audit_alpha.json", "audit_beta.json"):\n    d = js(fn)\n'
    with tempfile.TemporaryDirectory() as d:
        f = Path(d, "gen.py")
        f.write_text(src)
        found = set()
        for pat in m.PATTERNS:
            for hit in pat.findall(f.read_text()):
                n = m.normalise(hit if isinstance(hit, str) else hit[0])
                if n:
                    found.add(n)
        # the tuple form is the one that defeated version one of this guard
        return found == {"audit_alpha.json", "audit_beta.json"}, \
            "an input referenced through a tuple, which v1 could not see"


def audit_floor_screen():
    """Two cells closer than the floor must not be called separated."""
    m = load("rep", "scripts/report.py")
    floor = 0.0133
    # ranges disjoint but by less than the floor -- the group_dro/size case
    a = [0.7600, 0.7602, 0.7604]
    b = [0.7612, 0.7620, 0.7640]
    # `separation` returns (verdict, margin). The first version of this audit
    # applied bool() to that tuple, which is always truthy, and reported a
    # correct guard as VACUOUS -- an untested test, inside the script whose
    # entire purpose is to catch untested tests.
    near, _ = m.separation(a, b, floor)
    far, _ = m.separation([0.60, 0.61, 0.62], b, floor)
    overlap, _ = m.separation([0.7610, 0.7650], b, floor)
    return (not near.startswith("**separated**")
            and far.startswith("**separated**")
            and overlap == "ranges overlap"), \
        "disjoint ranges separated by less than the floor"


def audit_permutation_test():
    """The permutation test must return 1.0 on identical arms."""
    m = load("gn", "scripts/gn_vs_bn.py")
    same = [0.80, 0.81, 0.82, 0.83]
    p_same, _ = m.perm_p(same, list(same))
    apart = [0.90, 0.91, 0.92, 0.93]
    p_apart, n = m.perm_p(apart, same)
    return (p_same == 1.0 and abs(p_apart - 2.0 / n) < 1e-12), \
        "identical arms must give p = 1, maximally separated arms the floor"


def audit_number_provenance_traceability():
    """The traceability half. Known vacuous; measured, not assumed."""
    m = load("np_", "scripts/number_provenance.py")
    vals = m.run_values(str(ROOT / "runs"))
    rng = random.Random(0)
    rate4 = sum(1 for _ in range(3000)
                if m.traceable(f"{rng.random():.4f}", vals)) / 3000
    return rate4 < 0.5, (f"a random 4-digit decimal is accepted "
                         f"{100 * rate4:.0f}% of the time")


def audit_number_provenance_ratchet():
    """The half that can fail: adding a typed decimal must be detected."""
    m = load("np2", "scripts/number_provenance.py")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d, "gen.py")
        f.write_text('W("a sentence quoting 0.1234 and 0.98765 by hand")\n')
        n_before = len(m.literals(f))
        f.write_text('W("a sentence quoting 0.1234 and 0.98765 by hand")\n'
                     'W("and now one more, 0.4321, typed in")\n')
        n_after = len(m.literals(f))
        # the DOI prefix must not be counted as a measurement
        f.write_text('W("Zenodo 10.5281/zenodo.20061545 returns 403")\n')
        n_doi = len(m.literals(f))
        return (n_before == 2 and n_after == 3 and n_doi == 0), \
            "one more decimal typed into prose raises the count"


def audit_coverage_check():
    """A run family no generator reads must be reported as an orphan."""
    orphan = ROOT / "runs" / "zz_guard_audit_orphan.json"
    created = not orphan.exists()
    if created:
        orphan.write_text(json.dumps({"what": "temporary guard-audit probe"}))
    try:
        r = subprocess.run([PY, str(ROOT / "scripts/coverage_check.py")],
                           capture_output=True, text=True, cwd=ROOT)
        out = r.stdout + r.stderr
        caught = ("zz_guard_audit_orphan" in out
                  or "orphan" in out.lower() and r.returncode != 0)
        return caught, "a JSON no document ever reads"
    finally:
        if created and orphan.exists():
            orphan.unlink()


def audit_sign_flip_test():
    """The exact test behind the headline family claim must not always fire."""
    m = load("fam", "scripts/dg_family_test.py")
    none, _ = m.sign_flip_p([0.0] * 8)
    allneg, n = m.sign_flip_p([-0.01] * 8)
    mixed, _ = m.sign_flip_p([1, -1, 2, -2, 3, -3, 4, -4])
    return (none == 1.0 and abs(allneg - 2.0 / n) < 1e-12 and mixed == 1.0), \
        "identical arms, and a perfectly balanced set, must give p = 1"


def audit_floor_min_repeats():
    """A floor from too few repeats must be refused, not served as zero."""
    import json as _json
    m = load("rep2", "scripts/report.py")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "determinism__good__x.json").write_text(
            _json.dumps({"range": 0.01, "n_repeats": 6}))
        Path(d, "determinism__thin__x.json").write_text(
            _json.dumps({"range": 0.0, "n_repeats": 1}))
        F = m.floors(d)
        return ("thin" not in F and "good" in F
                and m.floor_for(F, "thin") == 0.01), \
            "a one-repeat range of 0.0000 offered as a protocol's floor"


def audit_section_diff():
    """A section that stops rendering must be reported; growth must not fail."""
    m = load("sd", "scripts/section_diff.py")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "doc.md")
        p.write_text("# T\n\n## 7.9 Power\n\nx\n\n## 8. Threats\n\nx\n")
        before = m.headings(p)
        p.write_text("# T\n\n## 8. Threats\n\nx\n")
        gone, _ = m.compare(before, m.headings(p))
        p.write_text("# T\n\n## 7.9 Power\n\nx\n\n## 8. Threats\n\nx\n"
                     "\n## 9. New\n\nx\n")
        grew_removed, grew_added = m.compare(before, m.headings(p))
        return (len(gone) == 1 and grew_removed == []
                and len(grew_added) == 1), \
            "a section whose guard was conditioned on a problem that got fixed"


def audit_reading_budget():
    """An over-long core must fail; a short one must pass."""
    m = load("rb", "scripts/reading_budget.py")
    long_core = ("word " * 2000) + m.MARKER + " tail"
    short_core = ("word " * 100) + m.MARKER + " tail"
    lc, _ = m.split_core(long_core)
    sc, _ = m.split_core(short_core)
    over = len(lc.split()) / m.WPM > m.BUDGET_MIN
    under = len(sc.split()) / m.WPM <= m.BUDGET_MIN
    missing = m.split_core("a document with no stop marker")[0] is None
    return (over and under and missing), \
        "a core that has drifted past the brief's five minutes"


AUDITS = [
    ("verify_stage.py", audit_verify_stage),
    ("prose_status_lint.py duplicated_blocks", audit_duplicated_blocks),
    ("section_census.py", audit_section_census),
    ("report.py separation (floor screen)", audit_floor_screen),
    ("report.py floors (min repeats)", audit_floor_min_repeats),
    ("gn_vs_bn.py perm_p", audit_permutation_test),
    ("dg_family_test.py sign_flip_p", audit_sign_flip_test),
    ("coverage_check.py", audit_coverage_check),
    ("section_diff.py", audit_section_diff),
    ("reading_budget.py", audit_reading_budget),
    ("number_provenance.py traceability", audit_number_provenance_traceability),
    ("number_provenance.py ratchet", audit_number_provenance_ratchet),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    rows, n_vacuous = [], 0
    for name, fn in AUDITS:
        try:
            ok, defect = fn()
        except Exception as e:                      # a guard that crashes is
            ok, defect = False, f"raised {type(e).__name__}: {e}"
        rows.append((name, ok, defect))
        if not ok:
            n_vacuous += 1

    w = max(len(n) for n, _, _ in rows)
    print(f"{'guard':{w}s}  verdict   defect it was fed")
    print("-" * (w + 40))
    for name, ok, defect in rows:
        print(f"{name:{w}s}  {'catches ' if ok else 'VACUOUS '}  {defect}")

    print(f"\n{len(rows) - n_vacuous} of {len(rows)} guards demonstrably reject "
          f"a defect they claim to catch.")
    if n_vacuous:
        print("A VACUOUS guard is worse than a failing one: every green result "
              "it has ever produced was uninformative. Fix it or say in the "
              "documents that it is not evidence.")
    return 1 if (a.strict and n_vacuous) else 0


if __name__ == "__main__":
    sys.exit(main())
