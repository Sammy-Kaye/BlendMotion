"""Headless test runner:  blender -b --python tests/run.py [-- --no-real] [-- -k substring]

Plain asserts, no pytest. Discovers tests/test_*.py, runs every `test_*`
function. The real-rig test (test_real_rig.py) runs last because it opens a
large .blend; skip it with `-- --no-real`.
"""
import os
import sys
import time
import traceback
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    skip_real = "--no-real" in argv
    pattern = argv[argv.index("-k") + 1] if "-k" in argv else None
    mods = sorted(f[:-3] for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    last = [m for m in mods if m == "test_real_rig"]
    mods = [m for m in mods if m not in last] + ([] if skip_real else last)
    passed, failed = 0, []
    t_all = time.perf_counter()
    for mn in mods:
        mod = importlib.import_module(mn)
        for fn_name in sorted(n for n in dir(mod) if n.startswith("test_")):
            full = f"{mn}.{fn_name}"
            if pattern and pattern not in full:
                continue
            t0 = time.perf_counter()
            try:
                getattr(mod, fn_name)()
            except BaseException as e:  # noqa: BLE001 - report everything
                failed.append(full)
                print(f"FAIL {full}  ({type(e).__name__}: {e})")
                traceback.print_exc(file=sys.stdout)
            else:
                passed += 1
                print(f"PASS {full}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
    print(f"\n{passed} passed, {len(failed)} failed in {time.perf_counter() - t_all:.1f} s")
    if failed:
        print("FAILED: " + ", ".join(failed))
    sys.stdout.flush()
    os._exit(1 if failed else 0)


main()
