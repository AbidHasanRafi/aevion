"""Run the CI gates on this machine, in the same way and the same order as the workflows.

python tools/ci_local.py              # lint, types, API snapshot, generated docs, security, tests + coverage
python tools/ci_local.py --fast       # skip the test-suite (seconds): lint, types, API, docs-sync, static security
python tools/ci_local.py --full       # everything: + benchmarks, docs build, wheel build + clean-environment smoke test
python tools/ci_local.py --only lint types
python tools/ci_local.py --keep-going # run every gate even if one fails

Exit code 0 only if every selected gate passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass
class Gate:
    name: str
    cmd: List[str]
    tier: str  # fast | default | full
    help: str
    env: dict | None = None


def gates(jobs: str) -> List[Gate]:
    cov = ["--cov=aevion", "--cov-report=term:skip-covered", "--cov-fail-under=85"]
    return [
        Gate("lint", [PY, "-m", "ruff", "check", "."], "fast", "ruff lint"),
        Gate("format", [PY, "-m", "ruff", "format", "--check", "."], "fast", "ruff format (no changes needed)"),
        Gate("types", [PY, "-m", "mypy", "src/aevion"], "fast", "mypy (strict for the foundation modules)"),
        Gate("api", [PY, "tools/api_snapshot.py"], "fast", "public API matches tests/golden/public_api.json"),
        Gate("docs-sync", [PY, "tools/gen_config_docs.py", "--check"], "fast", "generated config reference is current"),
        Gate(
            "security-static",
            [PY, "-m", "pytest", "tests/test_security_static.py", "-p", "no:xdist"],
            "fast",
            "no pickle/eval/shell/cleartext URLs; bandit clean",
        ),
        Gate("tests", [PY, "-m", "pytest", "-n", jobs, *cov], "default", "full test-suite with the 85 % coverage gate"),
        Gate("bench", [PY, "tools/bench_check.py"], "full", "no performance regression against benchmarks/baseline.json"),
        Gate(
            "docs",
            [PY, "-m", "mkdocs", "build", "--strict", "-d", str(ROOT / "site")],
            "full",
            "documentation builds in strict mode",
            {"NO_MKDOCS_2_WARNING": "true"},
        ),
        Gate("wheel", [PY, "tools/ci_local.py", "--_wheel"], "full", "wheel builds, twine check, clean-venv smoke test"),
    ]


def build_and_smoke() -> int:
    dist = ROOT / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    for cmd in ([PY, "-m", "build", "--outdir", str(dist)], [PY, "-m", "twine", "check", *map(str, dist.glob("*"))]):
        if cmd[2] == "twine":
            cmd = [PY, "-m", "twine", "check", *map(str, sorted(dist.glob("*")))]
        if subprocess.run(cmd, cwd=ROOT, check=False).returncode:
            return 1
    wheel = next(dist.glob("*.whl"))
    return subprocess.run([PY, "tools/smoke_wheel.py", str(wheel), "--offline"], cwd=ROOT, check=False).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="GATE")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--jobs", default="auto", help="pytest-xdist workers (default: auto)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--_wheel", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args._wheel:
        return build_and_smoke()

    allg = gates(args.jobs)
    if args.list:
        for g in allg:
            print(f"{g.name:<16}{g.tier:<9}{g.help}")
        return 0
    wanted = {"fast": {"fast"}, "default": {"fast", "default"}, "full": {"fast", "default", "full"}}[
        "fast" if args.fast else "full" if args.full else "default"
    ]
    selected = [g for g in allg if g.name in args.only] if args.only else [g for g in allg if g.tier in wanted]
    unknown = set(args.only or []) - {g.name for g in allg}
    if unknown:
        print(f"unknown gate(s): {', '.join(sorted(unknown))}; see --list", file=sys.stderr)
        return 2

    results = []
    for g in selected:
        print(f"\n=== {g.name}: {g.help}\n$ {' '.join(g.cmd)}", flush=True)
        t0 = time.perf_counter()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(g.env or {})}
        rc = subprocess.run(g.cmd, cwd=ROOT, env=env, check=False).returncode
        results.append((g.name, rc, time.perf_counter() - t0))
        if rc and not args.keep_going:
            break

    print("\n" + "-" * 44)
    for name, rc, dt in results:
        print(f"{'PASS' if rc == 0 else 'FAIL':<6}{name:<18}{dt:7.1f}s")
    skipped = [g.name for g in selected if g.name not in {r[0] for r in results}]
    if skipped:
        print(f"not run (stopped at first failure): {', '.join(skipped)}")
    return 0 if results and all(rc == 0 for _, rc, _ in results) and not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
