"""Result cache, runners, the SQLite result database, config experiments, leaderboards and benchmark suites."""

import functools
import json
import operator
import os
import subprocess
import sys

import numpy as np
import pytest

from aevion.experiments import (
    SUITES,
    BenchmarkSuite,
    ConfigExperiment,
    ExperimentResult,
    ParameterSpace,
    ProcessRunner,
    ResultCache,
    ResultDB,
    SerialRunner,
    canonical_json,
    deep_merge,
    format_leaderboard,
    get_suite,
    leaderboard,
    nest,
    run_suite,
    shard_indices,
    slurm_script,
    trial_key,
)
from aevion.experiments.config_experiment import get_path

BASE = {"duration": 2.5, "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.7}}}
METHODS = {"geometric": {"controller": "geometric"}, "cascaded": {"controller": "cascaded"}}
CONDITIONS = {"calm": {}, "windy": {"wind": {"type": "constant", "velocity": [4.0, 0.0, 0.0]}}}


def experiment(**kw):
    kw.setdefault("seeds", [0, 1])
    kw.setdefault("metric_params", {"skip": 0.5})
    return ConfigExperiment("t", BASE, METHODS, CONDITIONS, **kw)


# ------------------------------------------------------------------------------------------------ keys and cache
def test_trial_keys_are_deterministic_content_hashes():
    a = trial_key({"x": 1, "y": [1, 2]})
    assert a == trial_key({"y": [1, 2], "x": 1}) and len(a) == 32 and all(c in "0123456789abcdef" for c in a)
    assert a != trial_key({"x": 1, "y": [1, 3]}) and a != trial_key({"x": 1, "y": [1, 2]}, salt="v2")
    assert canonical_json({"a": np.float64(1.5), "b": np.arange(3), "c": float("nan")}) == '{"a":1.5,"b":[0,1,2],"c":"nan"}'
    with pytest.raises(TypeError, match="plain data"):
        trial_key({"f": lambda: 0})


def test_the_cache_stores_atomically_survives_damage_and_merges(tmp_path):
    c = ResultCache(tmp_path / "c")
    k = trial_key({"a": 1})
    assert c.get(k) is None and c.misses == 1 and not c.has(k)
    c.put(k, {"rmse": 0.25, "n": np.int64(3)})
    assert c.get(k) == {"rmse": 0.25, "n": 3} and c.hits == 1 and c.has(k) and len(c) == 1 and list(c.keys()) == [k]
    assert not list((tmp_path / "c").rglob("*.tmp"))  # no temporary files left behind
    c.path(k).write_text("{not json", encoding="utf-8")  # a crash left garbage
    assert c.get(k) is None and c.corrupt == 1
    k2 = trial_key({"a": 2})
    c.put(k2, {"v": 1})
    other = ResultCache(tmp_path / "o")
    other.put(k2, {"v": 99})
    other.put(trial_key({"a": 3}), {"v": 3})
    assert c.merge_from(other) == 1 and c.get(k2) == {"v": 1}  # existing entries are never overwritten
    assert c.stats()["entries"] == 3 and c.clear() == 3 and len(c) == 0
    with pytest.raises(ValueError, match="cache key"):
        c.path("../../etc/passwd")
    wrong = trial_key({"a": 4})
    c.put(wrong, {"v": 1})
    c.path(wrong).write_text(json.dumps({"key": "0" * 32, "row": {"v": 2}}), encoding="utf-8")  # an entry filed under the wrong key
    assert c.get(wrong) is None


# ------------------------------------------------------------------------------------------------ helpers
def test_configuration_helpers():
    assert deep_merge({"a": {"b": 1, "c": 2}, "l": [1, 2]}, {"a": {"b": 5}, "l": [3]}) == {"a": {"b": 5, "c": 2}, "l": [3]}
    same = deep_merge({"t": {"type": "circle", "params": {"r": 1, "w": 2}}}, {"t": {"type": "circle", "params": {"r": 5}}})
    assert same == {"t": {"type": "circle", "params": {"r": 5, "w": 2}}}  # same component: parameters merge
    other = deep_merge({"t": {"type": "circle", "params": {"r": 1, "w": 2}}}, {"t": {"type": "hover", "params": {"pos": [0, 0, 1]}}})
    assert other == {"t": {"type": "hover", "params": {"pos": [0, 0, 1]}}}  # a different component replaces the block
    assert nest({"controller.params.wn_pos": 5, "seed": 1}) == {"controller": {"params": {"wn_pos": 5}}, "seed": 1}
    with pytest.raises(ValueError):
        nest({"a": 1, "a.b": 2})
    assert get_path({"a": {"b": 3}}, "a.b") == 3 and get_path({"a": 1}, "a.b.c", "d") == "d"
    assert shard_indices(10, 1, 3) == [1, 4, 7] and sorted(functools.reduce(operator.iadd, (shard_indices(10, k, 3) for k in range(3)), [])) == list(
        range(10)
    )
    with pytest.raises(ValueError):
        shard_indices(5, 3, 3)


def test_the_slurm_script_runs_one_shard_per_array_task_and_refuses_shell_metacharacters():
    s = slurm_script("exp.json", "/scratch/cache", 8, partition="short", cpus=2, concurrent=4)
    assert "#SBATCH --array=0-7%4" in s and "#SBATCH --partition=short" in s and "--cpus-per-task=2" in s
    assert 'python -m aevion.experiments.worker "exp.json" --cache "/scratch/cache" --shard "$SLURM_ARRAY_TASK_ID" --of 8' in s
    for bad in ("a; rm -rf /", "x`id`", "y$(z)"):
        with pytest.raises(ValueError, match="metacharacters"):
            slurm_script(bad, "c", 2)
    with pytest.raises(ValueError):
        slurm_script("e.json", "c", 0)


# ------------------------------------------------------------------------------------------------ the experiment
def test_a_config_experiment_runs_the_grid_and_reduces_each_trial_to_metrics():
    exp = experiment()
    assert exp.n_trials() == 2 * 2 * 2 and len({t.key for t in exp.trials()}) == 8
    res = exp.run()
    assert len(res.rows) == 8 and all(r["status"] == "ok" for r in res.rows) and res.meta["failed"] == 0
    assert res.methods == ["geometric", "cascaded"] and res.conditions == ["calm", "windy"]
    calm, windy = res.values("rmse_pos", condition="calm"), res.values("rmse_pos", condition="windy")
    assert len(calm) == 4 and windy.mean() > calm.mean()  # a 4 m/s wind makes tracking worse
    assert {"rmse_pos", "energy_wh", "max_tilt_deg"} <= set(res.metric_names())
    assert res.summary("rmse_pos")[0]["n"] == 2
    again = experiment().run()
    assert [r["rmse_pos"] for r in again.rows] == [r["rmse_pos"] for r in res.rows]  # bit-reproducible


def test_the_cache_makes_a_second_run_free_and_a_changed_trial_gets_a_new_key(tmp_path):
    cache = ResultCache(tmp_path / "cache")
    first = experiment().run(cache=cache)
    assert first.meta["computed"] == 8 and first.meta["cached"] == 0 and len(cache) == 8
    second = experiment().run(cache=cache)
    assert second.meta["computed"] == 0 and second.meta["cached"] == 8
    assert [r["rmse_pos"] for r in second.rows] == [r["rmse_pos"] for r in first.rows]
    more = ConfigExperiment("t", BASE, {**METHODS, "lqr": {"controller": "lqr"}}, CONDITIONS, seeds=[0, 1], metric_params={"skip": 0.5})
    third = more.run(cache=cache)
    assert third.meta["computed"] == 4 and third.meta["cached"] == 8  # only the new method is simulated
    changed = ConfigExperiment("t", {**BASE, "duration": 3.0}, METHODS, CONDITIONS, seeds=[0, 1], metric_params={"skip": 0.5})
    assert not {t.key for t in changed.trials()} & {t.key for t in experiment().trials()}  # any change to a trial is a new key


class Crash(Exception):
    pass


class CrashingRunner(SerialRunner):
    """Dies after ``n`` trials, like a killed job."""

    def __init__(self, n):
        self.n = n

    def imap(self, fn, jobs):
        for i, j in enumerate(jobs):
            if i == self.n:
                raise Crash()
            yield i, fn(j)


def test_an_interrupted_run_resumes_from_the_cache_and_equals_an_uninterrupted_one(tmp_path):
    cache = ResultCache(tmp_path / "cache")
    with pytest.raises(Crash):
        experiment().run(runner=CrashingRunner(3), cache=cache)
    assert len(cache) == 3  # every finished trial was stored the moment it finished
    resumed = experiment().run(cache=cache)
    assert resumed.meta["cached"] == 3 and resumed.meta["computed"] == 5
    clean = experiment().run()
    assert [r["rmse_pos"] for r in resumed.rows] == [r["rmse_pos"] for r in clean.rows]


def test_shards_partition_the_trials_and_their_union_is_the_whole_experiment(tmp_path):
    cache = ResultCache(tmp_path / "cache")
    parts = [experiment().run(cache=cache, shard=(k, 3)) for k in range(3)]
    assert sum(p.meta["trials"] for p in parts) == 8 and all(p.meta["shard"] == [k, 3] for k, p in enumerate(parts))
    whole = experiment().run(cache=cache)
    assert whole.meta["cached"] == 8 and whole.meta["computed"] == 0
    flat = {r["key"]: r["rmse_pos"] for p in parts for r in p.rows}
    assert flat == {r["key"]: r["rmse_pos"] for r in whole.rows}


def test_a_process_pool_gives_the_same_numbers_as_serial_execution():
    pool = experiment(seeds=[0]).run(runner=ProcessRunner(2))
    serial = experiment(seeds=[0]).run()
    assert [r["rmse_pos"] for r in pool.rows] == [r["rmse_pos"] for r in serial.rows] and pool.meta["runner"] == "process"
    with pytest.raises(ValueError):
        ProcessRunner(0)


def test_failing_trials_are_recorded_not_raised_and_not_cached(tmp_path):
    # the goal lies inside a building: valid configuration, but the planner finds no route when the simulation is built
    blocked = {
        "duration": 1.0,
        "world": {"type": "scene", "params": {"boxes": [{"center": [15, 0, 6], "size": [4, 14, 12]}], "bounds": [-5, -15, 0, 35, 15, 20]}},
        "trajectory": {"type": "planned", "params": {"start": [0, 0, 3], "goal": [15, 0, 6], "resolution": 1.0}},
    }
    exp = ConfigExperiment("f", BASE, {"ok": {}, "blocked": blocked}, None, seeds=[0], metric_params={"skip": 0.2})
    cache = ResultCache(tmp_path / "c")
    res = exp.run(cache=cache)
    ok = [r for r in res.rows if r["method"] == "ok"][0]
    bad = [r for r in res.rows if r["method"] == "blocked"][0]
    assert ok["status"] == "ok" and bad["status"] == "failed" and "no theta_star path" in bad["error"]
    assert res.meta["failed"] == 1 and len(cache) == 1  # the failure is not cached, so a re-run retries it
    assert res.values("rmse_pos", "blocked").size == 0 and res.values("rmse_pos", "ok").size == 1
    with pytest.raises(RuntimeError, match="failed"):
        exp.run(raise_on_error=True)
    again = exp.run(cache=cache)
    assert again.meta["cached"] == 1 and again.meta["computed"] == 1


def test_configuration_mistakes_are_caught_before_any_trial_runs():
    with pytest.raises(ValueError, match="method 'bad'"):
        ConfigExperiment("e", BASE, {"bad": {"controller": "no_such_controller"}}).run()
    with pytest.raises(ValueError, match="unknown metric"):
        ConfigExperiment("e", BASE, metric="nope").validate()
    with pytest.raises(ValueError, match="seed"):
        ConfigExperiment("e", BASE, seeds=[])
    with pytest.raises(TypeError):
        ConfigExperiment("e", "not a dict")


def test_specifications_round_trip_through_json_and_the_worker_command(tmp_path):
    exp = experiment(seeds=[0])
    path = exp.save_spec(tmp_path / "exp.json")
    back = ConfigExperiment.load_spec(path)
    assert [t.key for t in back.trials()] == [t.key for t in exp.trials()]
    with pytest.raises(ValueError, match="not an Aevion experiment"):
        ConfigExperiment.from_spec({"name": "x"})
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for k in range(2):
        r = subprocess.run(
            [sys.executable, "-m", "aevion.experiments.worker", str(path), "--cache", str(tmp_path / "cc"), "--shard", str(k), "--of", "2"],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
        assert r.returncode == 0 and "shard" in r.stdout, r.stdout + r.stderr
    merged = ConfigExperiment.load_spec(path).run(cache=ResultCache(tmp_path / "cc"))
    assert merged.meta["computed"] == 0 and merged.meta["cached"] == 4  # the two shards together covered everything


def test_design_experiments_map_a_parameter_space_to_configuration_paths():
    space = ParameterSpace({"wn": ("log", 2.0, 14.0), "radius": (1.0, 3.0)})
    exp = ConfigExperiment.from_design(
        "d", BASE, space, {"wn": "controller.params.wn_pos", "radius": "trajectory.params.radius"}, n=4, seeds=[0], metric_params={"skip": 0.5}
    )
    assert len(exp.conditions) == 4
    cond = next(iter(exp.conditions.values()))
    assert set(cond) == {"controller", "trajectory"} and 2.0 <= cond["controller"]["params"]["wn_pos"] <= 14.0
    res = exp.run()
    X, y, names = exp.design_table(res, "rmse_pos")
    assert X.shape == (4, 2) and y.shape == (4,) and names == ["wn", "radius"] and np.all(y > 0)
    with pytest.raises(ValueError, match="no configuration path"):
        ConfigExperiment.from_design("d", BASE, space, {"wn": "controller.params.wn_pos"}, n=2)
    with pytest.raises(ValueError, match="from_design"):
        experiment().design_table(res, "rmse_pos")


# ------------------------------------------------------------------------------------------------ database and leaderboard
def rows_for(method_means, n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [
        dict(method=m, condition="c", seed=i, key=f"{m}{i}", status="ok", rmse=float(rng.normal(mu, 0.05)))
        for m, mu in method_means.items()
        for i in range(n)
    ]


def test_the_database_stores_queries_merges_and_updates_without_duplicating(tmp_path):
    res = experiment().run()
    with ResultDB(tmp_path / "r.db") as db:
        assert db.add_result(res) == 8 and db.experiments() == ["t"]
        assert db.add_result(res) == 8 and sum(db.counts("t").values()) == 8  # same keys: updated, not duplicated
        back = db.result("t")
        assert [r["rmse_pos"] for r in back.rows] == [r["rmse_pos"] for r in res.rows] and back.meta["metric"] == "flight"
        assert len(db.rows("t", method="cascaded", condition="windy")) == 2
        v = db.metric_values("t", "rmse_pos", method="geometric")
        assert len(v) == 4 and np.allclose(sorted(v), sorted(res.values("rmse_pos", "geometric")))
        best = db.best("t", "rmse_pos")
        assert best["rmse_pos"] == pytest.approx(res.values("rmse_pos").min())
        assert db.best("t", "energy_wh", minimize=False)["energy_wh"] == pytest.approx(res.values("energy_wh").max())
        with pytest.raises(KeyError):
            db.best("t", "no_such_metric")
        with pytest.raises(KeyError):
            db.result("missing")
        other = ResultDB()
        other.add_result(ExperimentResult("o", rows_for({"a": 1.0}, 3)))
        other.add_result(
            ExperimentResult("t", res.rows[:2] + [dict(method="new", condition="c", seed=9, key="zzz", status="ok", rmse_pos=1.0)], res.meta)
        )
        assert db.merge(other) == 3 + 1 and sorted(db.experiments()) == ["o", "t"]  # three new rows of 'o' and one new row of 't'
        db.delete("o")
        assert db.experiments() == ["t"]
    with ResultDB(tmp_path / "r.db") as reopened:  # persisted
        assert reopened.experiments() == ["t"]


def test_the_database_keeps_failed_trials_out_of_the_statistics():
    db = ResultDB()
    rows = rows_for({"a": 1.0}, 4) + [dict(method="a", condition="c", seed=99, key="bad", status="failed", error="boom")]
    db.add_result(ExperimentResult("e", rows))
    assert db.counts("e") == {"ok": 4, "failed": 1} and len(db.rows("e")) == 4 and db.failures("e")[0]["error"] == "boom"
    assert len(db.metric_values("e", "rmse")) == 4


def test_the_leaderboard_ranks_by_the_mean_and_flags_what_cannot_be_distinguished_from_the_leader():
    board = leaderboard(rows_for({"best": 1.0, "close": 1.01, "far": 1.5, "worst": 2.0}, n=15), "rmse")
    assert [b["name"] for b in board] == ["best", "close", "far", "worst"] and [b["rank"] for b in board] == [1, 2, 3, 4]
    assert board[0]["p_vs_best"] is None and board[0]["tied_with_best"]
    assert board[1]["tied_with_best"] and not board[2]["tied_with_best"] and board[3]["p_vs_best"] < 1e-3  # 'close' is within the noise of 'best'
    assert all(b["ci_lo"] <= b["mean"] <= b["ci_hi"] and b["n"] == 15 for b in board)
    hi = leaderboard(rows_for({"a": 1.0, "b": 2.0}), "rmse", lower_is_better=False)
    assert hi[0]["name"] == "b"
    text = format_leaderboard(board, "rmse")
    assert "best" in text and "*" in text.splitlines()[2]
    with pytest.raises(ValueError, match="no finite values"):
        leaderboard(rows_for({"a": 1.0}, 2), "missing")
    skipped = leaderboard(rows_for({"a": 1.0, "b": 2.0}, 4) + [dict(method="c", status="failed", rmse=0.0)], "rmse")
    assert [b["name"] for b in skipped] == ["a", "b"]  # a failed trial never ranks


# ------------------------------------------------------------------------------------------------ benchmark suites
def test_a_benchmark_suite_ranks_methods_per_scenario_and_overall(tmp_path):
    suite = BenchmarkSuite(
        "mini",
        {
            "circle": {"duration": 2.5, "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.7}}},
            "windy": {
                "duration": 2.5,
                "wind": {"type": "constant", "velocity": [5.0, 0.0, 0.0]},
                "trajectory": {"type": "hover", "params": {"pos": [0, 0, 3.0]}},
            },
        },
        seeds=(0, 1),
        metric_params={"skip": 0.5},
    )
    cache = ResultCache(tmp_path / "c")
    db = ResultDB()
    methods = {"geometric": {"controller": "geometric"}, "cascaded": {"controller": "cascaded"}}
    out = run_suite(suite, methods, cache=cache, db=db)
    assert set(out.boards) == {"circle", "windy"} and out.n_trials == 8 and out.failed == 0
    assert (
        sorted(o["name"] for o in out.overall) == ["cascaded", "geometric"]
        and out.overall[0]["rank"] == 1
        and all(len(o["ranks"]) == 2 for o in out.overall)
    )
    assert set(db.experiments()) == {"mini/circle", "mini/windy"}
    assert "mean rank" in out.table() and json.loads(out.to_json())["suite"] == "mini"
    assert out.save(tmp_path / "board.json").exists()
    again = run_suite(suite, {**methods, "lqr": {"controller": "lqr"}}, cache=cache)
    assert again.n_trials == 12 and len(cache) == 12  # only the new method was simulated
    with pytest.raises(ValueError):
        run_suite(suite, {})
    with pytest.raises(KeyError):
        suite.experiment("nope", methods)
    only = run_suite(suite, methods, scenarios=["circle"], cache=cache)
    assert list(only.boards) == ["circle"]


def test_the_built_in_suites_are_complete_and_valid():
    assert {"tracking", "disturbance"} <= set(SUITES) and get_suite("tracking").primary == "rmse_pos"
    with pytest.raises(KeyError, match="unknown suite"):
        get_suite("nope")
    for s in SUITES.values():
        assert len(s.scenarios) >= 3 and s.description
        for sc in s.scenarios:
            s.experiment(sc, {"geometric": {"controller": "geometric"}}).validate()
