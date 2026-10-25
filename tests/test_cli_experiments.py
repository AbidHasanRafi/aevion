"""Command-line interface: experiments, suites, vision datasets and the methods appendix."""

import json

import pytest

from aevion.cli import main
from aevion.experiments import ConfigExperiment, ResultCache, ResultDB

BASE = {"duration": 2.0, "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.7}}}


def spec(tmp_path):
    exp = ConfigExperiment(
        "cli",
        BASE,
        {"geometric": {"controller": "geometric"}, "cascaded": {"controller": "cascaded"}},
        None,
        seeds=[0, 1],
        metric_params={"skip": 0.5},
    )
    return exp.save_spec(tmp_path / "exp.json")


def test_experiment_run_caches_stores_and_exports(tmp_path, capsys):
    path = spec(tmp_path)
    main(["experiment", "run", str(path), "--cache", str(tmp_path / "cache"), "--db", str(tmp_path / "r.db"), "--csv", str(tmp_path / "rows.csv")])
    out = capsys.readouterr().out
    assert "4 trials (0 cached, 4 computed, 0 failed)" in out and "geometric" in out and "rmse_pos" in out
    assert len(ResultCache(tmp_path / "cache")) == 4 and (tmp_path / "rows.csv").read_text(encoding="utf-8").startswith("key,")
    with ResultDB(tmp_path / "r.db") as db:
        assert db.experiments() == ["cli"] and db.counts("cli") == {"ok": 4}
    main(["experiment", "run", str(path), "--cache", str(tmp_path / "cache")])
    assert "4 cached, 0 computed" in capsys.readouterr().out
    main(["experiment", "run", str(path), "--cache", str(tmp_path / "cache"), "--shard", "1", "--of", "2"])
    assert "2 trials" in capsys.readouterr().out


def test_experiment_slurm_writes_a_job_array_script(tmp_path, capsys):
    path = spec(tmp_path)
    main(["experiment", "slurm", str(path), "--cache", "/scratch/c", "--shards", "6", "--partition", "short", "-o", str(tmp_path / "job.sh")])
    text = (tmp_path / "job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-5" in text and "#SBATCH --partition=short" in text and "--of 6" in text and "\r" not in text
    main(["experiment", "slurm", str(path), "--cache", "/scratch/c", "--shards", "2"])
    assert "#SBATCH --array=0-1" in capsys.readouterr().out


def test_suite_list_and_run(tmp_path, capsys):
    main(["suite", "list"])
    assert "tracking" in capsys.readouterr().out
    main(
        ["suite", "run", "tracking", "--controllers", "geometric", "cascaded", "--cache", str(tmp_path / "c"), "--out", str(tmp_path / "board.json")]
    )
    out = capsys.readouterr().out
    assert "[circle]" in out and "[overall: mean rank over scenarios]" in out
    board = json.loads((tmp_path / "board.json").read_text(encoding="utf-8"))
    assert board["suite"] == "tracking" and len(board["overall"]) == 2


def test_vision_dataset_and_methods_commands(tmp_path, capsys):
    cfg = tmp_path / "cam.json"
    cfg.write_text(
        json.dumps(
            {
                "duration": 1.5,
                "world": {"type": "scene", "params": {"boxes": [{"center": [10, 0, 2], "size": [2, 3, 4]}], "bounds": [-5, -10, 0, 30, 10, 15]}},
                "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2.0]}},
                "sensors": [{"type": "camera", "params": {"rate": 10.0, "width": 48, "height": 36, "modes": ["depth", "rgb", "seg"], "far": 30.0}}],
            }
        ),
        encoding="utf-8",
    )
    main(["vision-dataset", str(cfg), "--out", str(tmp_path / "ds"), "--max-frames", "5"])
    assert "wrote 5 labelled frames" in capsys.readouterr().out
    coco = json.loads((tmp_path / "ds" / "annotations.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 5 and len(coco["annotations"]) >= 5
    main(["methods", str(cfg)])
    md = capsys.readouterr().out
    assert "## Simulation" in md and "camera" in md and "## Reproducibility" in md
    main(["methods", str(cfg), "--format", "latex"])
    assert "\\subsection*{Simulation}" in capsys.readouterr().out


def test_bad_experiment_specs_fail_cleanly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an Aevion experiment"):
        main(["experiment", "run", str(bad)])
