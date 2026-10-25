"""List mypy errors with the offending source line, grouped by file (helper for annotation work).

python tools/mypy_todo.py src/aevion/engine/faults.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(paths: list[str]) -> None:
    out = subprocess.run([sys.executable, "-m", "mypy", *paths, "--no-error-summary"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if not any(line.replace("\\", "/").startswith(p.replace("\\", "/")) for p in paths):
            continue
        try:
            file, lineno, rest = line.split(":", 2)
            src = Path(file).read_text(encoding="utf-8").splitlines()[int(lineno) - 1].strip()
        except Exception:
            print(line)
            continue
        print(f"{file}:{lineno}:{rest}\n      {src[:140]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["src/aevion"])
