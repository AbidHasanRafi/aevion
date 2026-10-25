"""The multi-agent, learning and experiment example scripts run end to end and print what their documentation says."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_example(name, *args):
    res = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name), *args],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-2500:]
    return res.stdout


def test_multiagent_tour(tmp_path):
    out = run_example("15_multiagent_tour.py", "--out", str(tmp_path), "--duration", "11")
    assert "predicted rate" in out and "conflict-free True" in out and "formation_keeping" in out
    assert (tmp_path / "swarm_replay.html").stat().st_size > 500_000


def test_learning_pipeline(tmp_path):
    out = run_example("16_learning_pipeline.py", "--out", str(tmp_path), "--iterations", "1")
    assert "DAgger 1" in out and "ONNX Runtime output differs" in out
    assert (tmp_path / "landing_policy.onnx").exists() and (tmp_path / "demonstrations.npz").exists() and (tmp_path / "dagger.jsonl").exists()


def test_experiment_pipeline(tmp_path):
    out = run_example("17_experiment_pipeline.py", "--out", str(tmp_path), "--samples", "4")
    assert "sensitivity (Sobol" in out and "a repeat simulated 0" in out and "mean [95% CI]" in out
    assert "## Simulation" in (tmp_path / "methods.md").read_text(encoding="utf-8") and (tmp_path / "results.db").exists()
