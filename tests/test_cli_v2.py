import json

import pytest

from aevion.cli import main


def run_cli(*argv):
    return main(list(argv))


def test_init_validate_run_inspect_diff_check_cycle(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AEVION_HOME", str(tmp_path / "home"))
    cfg = tmp_path / "sim.yaml"
    run_cli("init", str(cfg), "--name", "cli_test")
    assert cfg.exists() and "schema_version: 1" in cfg.read_text()
    with pytest.raises(SystemExit) as e:
        run_cli("validate", str(cfg))
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "OK" in out and "hash" in out

    # shorten the starter config so the test is quick
    text = cfg.read_text().replace("duration: 15.0", "duration: 3.0").replace("t_start: 6.0", "t_start: 1.0").replace("t_end: 9.0", "t_end: 2.0")
    cfg.write_text(text)
    out_dir = tmp_path / "out"
    run_cli("run", str(cfg), "--out", str(out_dir), "--no-report", "--register", "--profile")
    printed = capsys.readouterr().out
    assert "registered as run" in printed and "kernel:" in printed and "faults" in printed
    arch = out_dir / "run.aevd"
    assert (arch / "manifest.json").exists()

    run_cli("check", "--quick", str(arch)) if False else None
    with pytest.raises(SystemExit) as e:
        run_cli("check", str(arch))
    assert e.value.code == 0
    run_cli("inspect", str(arch))
    assert "gps_outage" in json.dumps(json.loads((arch / "manifest.json").read_text())["meta"]["faults"])

    # stream mode gives an identical result
    out2 = tmp_path / "out2"
    run_cli("run", str(cfg), "--out", str(out2), "--no-report", "--stream")
    with pytest.raises(SystemExit) as e:
        run_cli("diff", str(arch), str(out2 / "run.aevd"))
    assert e.value.code == 0
    assert "IDENTICAL" in capsys.readouterr().out

    run_cli("runs", "list")
    assert "cli_test" in capsys.readouterr().out
    run_cli(
        "runs",
        "tag",
        "cli_test"[:0] or __import__("aevion.datasets", fromlist=["RunRegistry"]).RunRegistry().list()[0].run_id,
        "--tags",
        "ci",
        "good",
    )
    run_cli("runs", "list", "--tag", "ci")
    assert "cli_test" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        run_cli("verify", str(arch))
    assert e.value.code == 0


def test_validate_reports_every_problem_and_exits_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nduration: -3\ncontroller: geometrik\nsensors: [imu, lidr]\nengine: {dt: 1.0}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        run_cli("validate", str(bad))
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "duration" in err and "did you mean 'geometric'" in err and "sensors[1].type" in err and "engine.dt" in err


def test_run_with_bad_config_prints_clean_error_not_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "duration": 0}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        run_cli("run", str(bad), "--out", str(tmp_path / "o"))
    assert e.value.code == 2
    assert "invalid configuration" in capsys.readouterr().err


def test_v01_scenario_json_still_runs_from_the_cli(tmp_path, capsys):
    from aevion.scenario import Scenario

    p = tmp_path / "old.json"
    p.write_text(Scenario(duration=1.0, trajectory=dict(kind="hover", pos=[0, 0, 2.0])).to_json(), encoding="utf-8")
    run_cli("run", str(p), "--out", str(tmp_path / "o"), "--no-report")
    assert (tmp_path / "o" / "run.aevd").exists()


def test_registry_plugins_schema_convert_pack(tmp_path, capsys):
    run_cli("registry", "controllers")
    assert "geometric" in capsys.readouterr().out
    run_cli("registry")
    assert "faults" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run_cli("registry", "nonsense")
    run_cli("plugins")
    run_cli("schema", "-o", str(tmp_path / "s.json"))
    assert json.loads((tmp_path / "s.json").read_text())["title"].startswith("Aevion")

    from aevion.config import SimulationConfig
    from aevion.engine import Simulation

    a = Simulation(SimulationConfig.from_dict({"duration": 1.0})).run()
    v2 = a.save(tmp_path / "a.aevd")
    run_cli("convert", str(v2), str(tmp_path / "a.aev"))
    run_cli("convert", str(tmp_path / "a.aev"), str(tmp_path / "b.aevd"))
    run_cli("pack", str(tmp_path / "b.aevd"))
    assert (tmp_path / "b.aevz").exists()
    run_cli("unpack", str(tmp_path / "b.aevz"), "--out", str(tmp_path / "c.aevd"))
    with pytest.raises(SystemExit) as e:
        run_cli("diff", str(v2), str(tmp_path / "c.aevd"))
    assert e.value.code == 0


def test_check_detects_corruption(tmp_path, capsys):
    from aevion.config import SimulationConfig
    from aevion.engine import Simulation

    p = Simulation(SimulationConfig.from_dict({"duration": 1.0})).run().save(tmp_path / "a.aevd")
    chunk = next((p / "channels" / "pos").glob("*.npy"))
    raw = bytearray(chunk.read_bytes())
    raw[-5] ^= 0x55
    chunk.write_bytes(bytes(raw))
    with pytest.raises(SystemExit) as e:
        run_cli("check", str(p))
    assert e.value.code == 1 and "hash mismatch" in capsys.readouterr().out
