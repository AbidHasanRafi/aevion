"""Run the benchmark suite and write machine-readable results.

python tools/bench.py                       # run everything, print a table
python tools/bench.py -k physics            # only benchmarks whose name contains 'physics'
python tools/bench.py --json out.json       # also write results (input for tools/bench_check.py)
python tools/bench.py --update-baseline     # accept the current numbers as benchmarks/baseline.json

Timings are the *minimum* over several repeats (the least-noisy estimator on a shared machine). Every result is also
expressed relative to a small fixed NumPy calibration workload, which makes the numbers comparable across machines.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "benchmarks" / "baseline.json"


def calibrate(repeats: int = 9) -> float:
    """Seconds taken by a fixed workload that mimics the suite: small-array NumPy calls, a Python loop and a little BLAS."""
    import numpy as np

    rng = np.random.default_rng(1)
    a = rng.normal(size=(96, 96))
    small = rng.normal(size=(8, 3))
    v = rng.normal(size=100_000)

    def work() -> None:
        x = small
        for _ in range(3000):  # interpreter + tiny-array overhead dominates, like a simulation step
            x = x * 0.999 + np.cross(x, small) * 1e-3
            x = x / (1.0 + np.abs(x).max())
        acc = 0.0
        for i in range(60000):
            acc += (i * 0.5) % 7.0
        (a @ a).sum()
        np.sort(v)

    work()
    return best_of(work, repeats, min_total=0.4)[0]


def best_of(fn: Callable[[], Any], repeats: int, min_total: float = 0.25) -> tuple:
    """(best seconds, number of runs). Repeats until both ``repeats`` runs and ``min_total`` seconds are reached."""
    best, runs, total = float("inf"), 0, 0.0
    gc_was = gc.isenabled()
    gc.disable()
    try:
        while runs < repeats or total < min_total:
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            best, runs, total = min(best, dt), runs + 1, total + dt
            if runs >= 50 or total > 20:
                break
    finally:
        if gc_was:
            gc.enable()
    return best, runs


def environment() -> Dict[str, Any]:
    import numpy as np

    from aevion import __version__

    info: Dict[str, Any] = dict(
        aevion=__version__, python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), machine=platform.machine()
    )
    try:
        from importlib import metadata

        info["numba"] = metadata.version("numba")
    except Exception:
        info["numba"] = None
    return info


def run_suite(select: str = "", repeats: int = 7, verbose: bool = True) -> Dict[str, Any]:
    from benchmarks import suite

    calib = calibrate()
    results: Dict[str, Any] = {}
    if verbose:
        print(f"calibration workload: {calib * 1e3:.2f} ms\n")
        print(f"{'benchmark':<26}{'best ms':>10}{'runs':>6}{'throughput':>22}{'relative':>10}")
    for name, b in suite.SUITE.items():
        if select and select not in name:
            continue
        if not suite.available(b):
            if verbose:
                print(f"{name:<26}  skipped (needs {b.requires})")
            continue
        run = b.setup()
        best, runs = best_of(run, repeats)
        rate = b.units / best
        results[name] = dict(seconds=best, runs=runs, units=b.units, unit=b.unit, per_second=rate, relative=best / calib)
        if verbose:
            print(f"{name:<26}{best * 1e3:>10.2f}{runs:>6}{rate:>14,.0f} {b.unit.split()[0]:<7}{best / calib:>10.3f}")
    return dict(calibration_s=calib, environment=environment(), results=results)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="substring filter")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--update-baseline", action="store_true", help=f"write {BASELINE.relative_to(ROOT)}")
    args = ap.parse_args(argv)
    out = run_suite(args.k, args.repeats)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    if args.update_baseline:
        if args.k:
            print("refusing to write a partial baseline (drop -k)", file=sys.stderr)
            return 2
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
