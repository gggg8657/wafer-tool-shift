"""Which verdicts depend on the floor, and which would survive any of them?

Measuring `iid`'s own run-to-run floor changed the threshold it is judged
against: it had been screened with the fallback -- the largest floor measured
anywhere -- and now has its own, which is smaller. A smaller floor is a weaker
test, so this is a change in criterion, and the brief is explicit that numbers
before and after a criterion change are not comparable and both must be
reported.

Rather than report two tables, this reports the *sensitivity*: for every
comparison the documents make, whether its verdict is the same under every floor
measured anywhere in the project. A verdict that flips depending on which
protocol's floor you borrow was never resting on the data.

    python scripts/floor_sensitivity.py     # writes runs/floor_sensitivity.json
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path


def load_report():
    spec = importlib.util.spec_from_file_location("rep", "scripts/report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    m = load_report()
    F = m.floors("runs")
    measured = {k: v for k, v in F.items() if k != "_fallback"}
    if not measured:
        raise SystemExit("no measured floors")
    lo, hi = min(measured.values()), max(measured.values())

    # load() is keyed by (protocol, encoder, objective, tag, seed); by_cell
    # regroups it to {(protocol, encoder, objective, tag): {seed: record}}.
    cells = m.by_cell(m.load("runs"))
    by_cell = defaultdict(dict)
    for (proto, enc, obj, tag), per_seed in cells.items():
        by_cell[(proto, enc, tag)][obj] = per_seed

    rows, flips = [], 0
    for (proto, enc, tag), objs in sorted(by_cell.items()):
        base = objs.get("erm")
        if not base or len(base) < 2:
            continue
        b = sorted(r["test"]["macro_f1"] for r in base.values())
        for obj, runs in sorted(objs.items()):
            if obj == "erm" or len(runs) < 2:
                continue
            x = sorted(r["test"]["macro_f1"] for r in runs.values())
            v_lo, _ = m.separation(x, b, lo)
            v_hi, _ = m.separation(x, b, hi)
            v_now, _ = m.separation(x, b, m.floor_for(F, proto))
            same = v_lo == v_hi
            if not same:
                flips += 1
            rows.append({
                "protocol": proto, "encoder": enc, "objective": obj,
                "tag": tag, "n_treatment": len(x), "n_baseline": len(b),
                "verdict_at_smallest_floor": v_lo,
                "verdict_at_largest_floor": v_hi,
                "verdict_at_this_protocols_floor": v_now,
                "stable_across_all_measured_floors": same,
            })

    res = {
        "what": "does each comparison's verdict survive every floor measured "
                "anywhere in this project",
        "why": "measuring iid's own floor replaced a borrowed threshold with a "
               "smaller one, which is a change of criterion. A verdict that "
               "depends on which protocol's floor you borrow was not resting "
               "on the data.",
        "floors_measured": measured,
        "smallest": lo, "largest": hi,
        "n_comparisons": len(rows),
        "n_flipping": flips,
        "n_stable": len(rows) - flips,
        "comparisons": rows,
    }
    Path("runs/floor_sensitivity.json").write_text(json.dumps(res, indent=2))
    print(f"floors measured: "
          + ", ".join(f"{k} {v:.4f}" for k, v in sorted(measured.items())))
    print(f"\n{len(rows)} comparisons with >=2 seeds on both arms")
    print(f"  {res['n_stable']} give the same verdict at every measured floor")
    print(f"  {flips} change verdict depending on which floor is applied")
    for r in rows:
        if not r["stable_across_all_measured_floors"]:
            print(f"    {r['protocol']}/{r['encoder']}/{r['objective']}"
                  f"{('/' + r['tag']) if r['tag'] else ''}: "
                  f"{r['verdict_at_smallest_floor']}  ->  "
                  f"{r['verdict_at_largest_floor']}")
    print("\nwrote runs/floor_sensitivity.json")


if __name__ == "__main__":
    main()
