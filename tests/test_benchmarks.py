"""The benchmark suite must keep running, and the regression gate must judge correctly."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from benchmarks import suite  # noqa: E402


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench_check = load_tool("bench_check")


def result(**rel):
    return {"results": {k: {"relative": v} for k, v in rel.items()}}


@pytest.mark.parametrize("name", suite.names())
def test_every_benchmark_runs(name):
    b = suite.SUITE[name]
    if not suite.available(b):
        pytest.skip(f"needs {b.requires}")
    run = b.setup()
    run()
    assert b.units > 0 and b.unit


def test_baseline_covers_the_suite_and_is_well_formed():
    base = json.loads((ROOT / "benchmarks" / "baseline.json").read_text(encoding="utf-8"))
    assert base["calibration_s"] > 0 and base["environment"]["aevion"]
    for name, r in base["results"].items():
        assert name in suite.SUITE and r["relative"] > 0 and r["per_second"] > 0
    required = {n for n, b in suite.SUITE.items() if not b.requires}
    assert required <= set(base["results"]), f"benchmarks missing from the baseline: {sorted(required - set(base['results']))}"


def test_compare_flags_regressions_and_improvements_only_beyond_tolerance():
    base = result(a=1.0, b=1.0, c=1.0, d=1.0)
    cur = result(a=1.10, b=1.40, c=0.60, e=2.0)
    rep = bench_check.compare(base, cur, tol=0.25)
    assert [x.split()[0] for x in rep["regressions"]] == ["b"]
    assert [x.split()[0] for x in rep["improvements"]] == ["c"]
    assert [x.split()[0] for x in rep["ok"]] == ["a"]
    assert rep["missing"] == ["d"] and rep["new"] == ["e"]


def test_default_tolerance_is_looser_on_ci(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    local = bench_check.default_tol()
    monkeypatch.setenv("CI", "true")
    assert bench_check.default_tol() > local


def test_check_exit_codes(tmp_path, capsys):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(result(a=1.0)))
    ok, slow = tmp_path / "ok.json", tmp_path / "slow.json"
    ok.write_text(json.dumps(result(a=1.1)))
    slow.write_text(json.dumps(result(a=2.0)))
    assert bench_check.main(["--baseline", str(base), "--current", str(ok), "--tol", "0.25"]) == 0
    assert bench_check.main(["--baseline", str(base), "--current", str(slow), "--tol", "0.25"]) == 1
    assert bench_check.main(["--baseline", str(tmp_path / "nope.json"), "--current", str(ok)]) == 2
