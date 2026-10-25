"""The wireless example script and scenario file run and validate."""

import subprocess
import sys
from pathlib import Path

from aevion.cli import main
from aevion.config import load_config
from aevion.engine import Simulation

ROOT = Path(__file__).resolve().parents[1]


def test_wireless_scenario_file_validates_and_runs():
    cfg = load_config(ROOT / "examples" / "scenarios" / "wireless_city.yaml")
    assert cfg.network is not None and cfg.network.type == "nr_cellular" and len(cfg.faults) == 2
    cfg.duration = 4.0
    cfg.faults = []
    arch = Simulation(cfg).run()
    assert "link_sinr_db" in arch and "sensors/gps" in arch


def test_cli_validates_the_wireless_scenario():
    import pytest

    with pytest.raises(SystemExit) as e:
        main(["validate", str(ROOT / "examples" / "scenarios" / "wireless_city.yaml")])
    assert e.value.code == 0


def test_example_script_runs_end_to_end(tmp_path):
    res = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "11_wireless_handover.py"), "--out", str(tmp_path), "--duration", "20"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"},
    )
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
    assert "handovers:" in res.stdout
    for name in ("link_stack.png", "rem.png", "sinr_cdf.png", "link_stack.pdf"):
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 5_000
    assert (tmp_path / "run.aevd" / "manifest.json").exists()
