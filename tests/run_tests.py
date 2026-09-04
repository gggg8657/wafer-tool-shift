"""Run the test functions without pytest.

pytest is not installed in the `pdeno` env this repo runs in, and that env is
shared, so it does not get a new package just to collect eight functions.

    python tests/run_tests.py
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    fails = 0
    for mod_path in sorted(Path(__file__).parent.glob("test_*.py")):
        mod = __import__(f"tests.{mod_path.stem}", fromlist=["*"])
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                print(f"PASS {mod_path.stem}::{name}")
            except Exception:
                fails += 1
                print(f"FAIL {mod_path.stem}::{name}")
                traceback.print_exc()
    print(f"\n{'FAILED' if fails else 'OK'} ({fails} failure(s))")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
