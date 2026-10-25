"""MkDocs hook: generate the derived pages before every build (see tools/gen_site.py)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def on_pre_build(config, **kwargs):
    cmd = [sys.executable, str(ROOT / "tools" / "gen_site.py")]
    if os.environ.get("AEVION_DOCS_NO_GALLERY"):
        cmd.append("--no-gallery")
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
