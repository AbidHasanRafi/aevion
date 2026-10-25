"""Regenerate golden files after an intentional change (review the diff before committing!).

python tools/update_goldens.py            # figures
python tools/update_goldens.py figures    # same
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"figures": ["tests/test_figure_goldens.py", "-k", "golden"]}


def main(argv: list[str]) -> int:
    names = argv or list(TARGETS)
    for n in names:
        if n not in TARGETS:
            print(f"unknown target {n!r}; choose from {sorted(TARGETS)}", file=sys.stderr)
            return 2
        cmd = [sys.executable, "-m", "pytest", *TARGETS[n], "--update-goldens", "-p", "no:xdist"]
        print("+", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=ROOT, check=False).returncode
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
