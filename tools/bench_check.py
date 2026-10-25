"""Fail when a benchmark got meaningfully slower than the stored baseline.

python tools/bench_check.py                      # run the suite now and compare with benchmarks/baseline.json
python tools/bench_check.py --current out.json   # compare a previous run instead
python tools/bench_check.py --tol 0.30           # allowed slow-down (default 0.25 locally, 0.35 when $CI is set)

Comparison uses the *relative* time (benchmark time / calibration time), so a faster or slower machine does not matter,
only a change in the code does. A benchmark that looks slower is re-measured once before it is reported (shared CI
machines are noisy); only a regression that persists fails the check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B


def compare(baseline: Dict[str, Any], current: Dict[str, Any], tol: float) -> Dict[str, List[str]]:
    """Return {'regressions': [...], 'improvements': [...], 'missing': [...], 'new': [...]} (human-readable lines)."""
    out: Dict[str, List[str]] = dict(regressions=[], improvements=[], missing=[], new=[], ok=[])
    base, cur = baseline["results"], current["results"]
    for name, b in base.items():
        if name not in cur:
            out["missing"].append(name)
            continue
        ratio = cur[name]["relative"] / b["relative"]
        line = f"{name:<26} {ratio:6.2f}x  ({b['relative']:.3f} -> {cur[name]['relative']:.3f} calibration units)"
        if ratio > 1.0 + tol:
            out["regressions"].append(line)
        elif ratio < 1.0 / (1.0 + tol):
            out["improvements"].append(line)
        else:
            out["ok"].append(line)
    out["new"] = [n for n in cur if n not in base]
    return out


def default_tol() -> float:
    return 0.35 if os.environ.get("CI") else 0.25


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=str(B.BASELINE))
    ap.add_argument("--current", help="results JSON from tools/bench.py --json (default: run the suite now)")
    ap.add_argument("--tol", type=float, default=default_tol())
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args(argv)

    bp = Path(args.baseline)
    if not bp.exists():
        print(f"no baseline at {bp}; create one with: python tools/bench.py --update-baseline", file=sys.stderr)
        return 2
    baseline = json.loads(bp.read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8")) if args.current else B.run_suite(repeats=args.repeats, verbose=False)

    rep = compare(baseline, current, args.tol)
    if rep["regressions"] and not args.current:
        # confirm: re-run only the benchmarks that looked slower and keep the better measurement
        for line in list(rep["regressions"]):
            name = line.split()[0]
            again = B.run_suite(select=name, repeats=args.repeats + 3, verbose=False)
            if name in again["results"] and again["results"][name]["relative"] < current["results"][name]["relative"]:
                current["results"][name] = again["results"][name]
        rep = compare(baseline, current, args.tol)

    print(f"benchmark comparison (tolerance {args.tol:.0%}):")
    for key, title in (("regressions", "REGRESSED"), ("improvements", "improved"), ("ok", "ok")):
        for line in rep[key]:
            print(f"  {title:<10} {line}")
    for name in rep["missing"]:
        print(f"  MISSING    {name} (in the baseline but not measured; optional dependency absent?)")
    for name in rep["new"]:
        print(f"  new        {name} (not in the baseline yet)")
    if rep["regressions"]:
        print("\nPerformance regression. If it is intended, refresh the baseline: python tools/bench.py --update-baseline", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
