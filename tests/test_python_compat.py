"""``requires-python = ">=3.10"`` is a promise: no syntax or standard-library feature newer than 3.10 anywhere."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_sources_need_nothing_newer_than_python_3_10():
    exe = shutil.which("vermin")
    if exe is None:
        pytest.skip("vermin not installed (pip install vermin)")
    res = subprocess.run(
        [exe, "-t=3.10-", "--no-tips", "--eval-annotations", "--violations", "src", "tools", "benchmarks", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout[-3000:]
