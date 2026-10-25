"""Datasets, behaviour cloning and DAgger, policy export, curricula, tracking, tuning, parallel rollouts, the Gymnasium vector env."""

import sys
import types
from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_support import expert_policy, make_env, zero_policy  # noqa: E402

from aevion.envs import QuadVectorEnv, VecQuadEnv  # noqa: E402
from aevion.envs.experts import make_expert  # noqa: E402
from aevion.ml import (  # noqa: E402
    CsvLogger,
    Curriculum,
    Dataset,
    JsonlLogger,
    MLflowLogger,
    MLPPolicy,
    MultiLogger,
    TensorBoardLogger,
    WandbLogger,
    behavior_cloning,
    collect,
    dagger,
    evaluate_policy_rollouts,
    export_onnx,
    export_torchscript,
    load_policy,
    optuna_search,
    parallel_collect,
    read_jsonl,
)
from aevion.ml.policies import verify_onnx  # noqa: E402


def landing_env(seed=0, n=16, secs=14.0):
    return VecQuadEnv(n, task="landing", action_mode="velocity", seed=seed, max_episode_s=secs, control_dt=0.05, physics_dt=0.01)


# ------------------------------------------------------------------------------------------------ dataset
def test_collect_records_transitions_with_the_right_flags_and_next_observations():
    env = landing_env(n=4, secs=2.0)
    d = collect(env, make_expert(env), 60, seed=1)
    assert len(d) == 4 * 60 and d.obs_dim == env.obs_dim and d.act_dim == 4
    assert d.observations.dtype == np.float32 and set(np.unique(d.env_ids)) == {0, 1, 2, 3}
    assert d.timeouts.sum() + d.terminals.sum() >= 4  # 2 s episodes are 40 steps long: every environment finished at least once
    # within an episode the next observation of step k is the observation of step k + 1
    for e in range(4):
        rows = np.nonzero(d.env_ids == e)[0]
        inside = rows[:-1][~(d.terminals[rows[:-1]] | d.timeouts[rows[:-1]])]
        pos = {r: i for i, r in enumerate(rows)}
        assert all(np.array_equal(d.next_observations[r], d.observations[rows[pos[r] + 1]]) for r in inside[:20])
    assert np.all(np.abs(d.actions) <= 1.0) and len(d.returns()) >= 4


def test_labelling_stores_the_expert_action_while_the_learner_drives():
    env = landing_env(n=2, secs=3.0)
    exp = make_expert(env)
    zero = lambda o: env.B.zeros(env.n, 4)
    d = collect(env, zero, 40, labeler=exp, beta=0.0, seed=2)
    assert np.abs(d.actions).max() > 0.05 and d.meta["labeled"]  # expert labels, not the zero actions the learner executed
    plain = collect(landing_env(n=2, secs=3.0), zero, 40, seed=2)
    assert np.all(plain.actions == 0.0)
    mixed = collect(env, zero, 40, labeler=exp, beta=1.0, seed=2)  # beta = 1: the expert drives, so the visited states are the expert's
    assert not np.array_equal(mixed.observations, d.observations)


def test_datasets_round_trip_through_npz_and_the_d4rl_layout(tmp_path):
    env = landing_env(n=2, secs=2.0)
    d = collect(env, make_expert(env), 50, seed=3)
    path = d.save(tmp_path / "demo")
    assert path.suffix == ".npz"
    back = Dataset.load(path)
    for name in ("observations", "actions", "rewards", "terminals", "timeouts", "next_observations", "env_ids"):
        assert np.array_equal(getattr(d, name), getattr(back, name))
    assert back.meta["steps"] == 50
    d4 = d.to_d4rl()
    assert set(d4) == {"observations", "actions", "rewards", "terminals", "timeouts", "next_observations"}
    again = Dataset.from_d4rl(d4)
    assert len(again) == len(d) and np.array_equal(again.actions, d.actions)
    no_next = Dataset.from_d4rl({k: v for k, v in d4.items() if k != "next_observations"})
    assert np.array_equal(no_next.next_observations[:-1], d.observations[1:])
    bad = dict(np.load(path))
    bad.pop("rewards")
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="not an Aevion dataset"):
        Dataset.load(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="rows"):
        Dataset(d.observations, d.actions[:-1], d.rewards, d.terminals, d.timeouts, d.next_observations, d.env_ids)


def test_episodes_concatenation_and_the_normaliser_floor():
    env = landing_env(n=3, secs=1.5)
    d = collect(env, make_expert(env), 90, seed=4)
    eps = list(d.episodes())
    assert sum(len(e) for e in eps) == len(d)
    assert len({int(i) for i in d.episode_ids()}) == len(eps) and all(len(set(e.env_ids)) == 1 for e in eps)
    finished = [e.rewards.sum() for e in eps if e.terminals[-1] or e.timeouts[-1]]
    assert np.allclose(sorted(d.returns()), sorted(finished), atol=1e-4)
    both = d.concat(d)
    assert len(both) == 2 * len(d)
    nz = d.normalizer()
    assert nz.std.min() >= 0.05
    with pytest.raises(ValueError):
        d.concat(Dataset(d.observations[:, :3], d.actions, d.rewards, d.terminals, d.timeouts, d.next_observations[:, :3], d.env_ids))


# ------------------------------------------------------------------------------------------------ imitation
def test_behaviour_cloning_learns_a_known_mapping_and_does_not_amplify_constant_features():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(2000, 6)).astype(np.float32)
    obs[:, 5] = 0.0  # a feature that never varies (like an unused action channel fed back as an input)
    act = np.tanh(obs[:, :2] @ np.array([[0.8, -0.3], [0.2, 0.9]]))
    d = Dataset(obs, act, np.zeros(2000), np.zeros(2000, bool), np.zeros(2000, bool), obs, np.zeros(2000, int))
    res = behavior_cloning(d, hidden=(32, 32), epochs=60, val_fraction=0.2, seed=1)
    assert res.losses[-1] < 0.2 * res.losses[0] and res.val_loss < 5e-3
    probe = obs[:5].copy()
    probe[:, 5] = (
        0.05  # the learner nudges the constant feature a little (with an unfloored standard deviation of ~1e-6 this would be a 50000-sigma input)
    )
    assert np.abs(res.policy(probe) - res.policy(obs[:5])).max() < 0.3
    cont = behavior_cloning(d, epochs=5, policy=res.policy)
    assert cont.policy is res.policy
    with pytest.raises(ValueError):
        behavior_cloning(d.take([0]))
    with pytest.raises(ValueError, match="disagree"):
        behavior_cloning(d, policy=MLPPolicy(3, 2))


def test_dagger_repairs_the_covariate_shift_that_limits_plain_cloning():
    env = landing_env(seed=0)
    expert = make_expert(env)
    ev_env = landing_env(seed=5)
    res = dagger(env, expert, iterations=2, steps_per_iter=250, hidden=(64, 64), epochs=40, eval_env=ev_env, eval_episodes=32, seed=0)
    h = res.history
    assert [r["iteration"] for r in h] == [0, 1, 2] and h[0]["dataset"] == 16 * 250 and h[2]["dataset"] == 3 * 16 * 250
    assert h[0]["success_rate"] < 0.85  # cloning the expert's own trajectories alone is not enough
    assert max(r["success_rate"] for r in h[1:]) >= 0.9 and h[-1]["crash_rate"] <= 0.1
    ev = landing_env(seed=6)
    stats = evaluate_policy_rollouts(ev, lambda o: ev.B.asarray(res.policy(ev.B.to_numpy(o))), 16, seed=3)
    assert stats["success_rate"] >= 0.85


# ------------------------------------------------------------------------------------------------ export
def test_policies_save_load_and_export_to_onnx_and_torchscript_with_identical_outputs(tmp_path):
    rng = np.random.default_rng(1)
    pol = MLPPolicy(10, 4, (16, 16), rng.normal(size=10), rng.uniform(0.5, 2, 10)).eval()
    x = rng.normal(size=(7, 10)).astype(np.float32)
    ref = pol(x)
    assert ref.shape == (7, 4) and np.all(np.abs(ref) <= 1.0)
    back = load_policy(pol.save(tmp_path / "p.pt"))
    assert np.array_equal(back(x), ref) and back.hidden == (16, 16)
    onnx_path = export_onnx(pol, tmp_path / "p.onnx")
    pytest.importorskip("onnxruntime")
    assert verify_onnx(pol, onnx_path) < 1e-5
    ts = torch.jit.load(str(export_torchscript(pol, tmp_path / "p.ts")))
    assert np.allclose(ts(torch.as_tensor(x)).detach().numpy(), ref, atol=1e-6)
    tens = pol(torch.as_tensor(x, dtype=torch.float64))
    assert isinstance(tens, torch.Tensor) and tens.dtype == torch.float64
    with pytest.raises(ValueError, match="activation"):
        MLPPolicy(2, 2, activation="swish")
    torch.save({"config": pol.config(), "state": pol.state_dict(), "extra": types.SimpleNamespace(a=1)}, tmp_path / "bad.pt")
    with pytest.raises(Exception):
        load_policy(tmp_path / "bad.pt")  # weights_only loading refuses arbitrary pickled objects


def test_verify_onnx_catches_a_graph_that_differs_from_the_policy(tmp_path):
    pytest.importorskip("onnxruntime")
    pol = MLPPolicy(4, 2, (8,)).eval()
    path = export_onnx(pol, tmp_path / "a.onnx")
    with torch.no_grad():
        pol.net[0].weight += 0.5
    with pytest.raises(AssertionError, match="differs"):
        verify_onnx(pol, path)


# ------------------------------------------------------------------------------------------------ curriculum
def test_curriculum_promotes_after_a_full_window_and_never_skips_a_stage():
    c = Curriculum([0.0, 0.5, 1.0], promote_at=0.8, window=10)
    assert c.level == 0 and not c.finished
    assert not c.report([1] * 9)  # not a full window yet
    assert c.report([1]) and c.level == 1 and c.history[-1]["success_rate"] == 1.0
    assert not c.report([1] * 9) and c.level == 1  # the window restarted at the new stage
    assert c.report([1]) and c.level == 2 and c.finished
    assert not c.report([1] * 50) and c.level == 2  # nothing above the last stage
    d = Curriculum([0.0, 1.0], promote_at=0.9, window=5, demote_at=0.2, start=1)
    assert d.report([0] * 5) and d.level == 0
    low = Curriculum([0.0, 1.0], promote_at=0.9, window=5)
    assert not low.report([1, 0, 1, 0, 0]) and low.level == 0 and low.success_rate() == pytest.approx(0.4)
    for bad in (dict(stages=[]), dict(stages=[0.0], promote_at=0.0), dict(stages=[0.0], window=0), dict(stages=[0.0], demote_at=0.9)):
        with pytest.raises(ValueError):
            Curriculum(**bad)


def test_curriculum_changes_the_task_and_trains_against_a_real_success_signal():
    env = VecQuadEnv(8, task="gate_racing", action_mode="velocity", seed=0, max_episode_s=14.0, control_dt=0.05, physics_dt=0.01)
    cur = Curriculum([0.0, 0.5, 1.0], promote_at=0.9, window=8)
    cur.apply(env)
    r_easy = env.task.radius
    pol = make_expert(env)
    obs = env.reset(seed=1)
    for _ in range(env.max_steps * 3):
        obs, r, te, tr, info = env.step(pol(obs))
        d = np.asarray(env.B.to_numpy(info["done"])).astype(bool)
        if d.any() and cur.report(np.asarray(env.B.to_numpy(info["success"]))[d]):
            cur.apply(env)
    assert cur.level >= 1 and env.task.radius < r_easy  # the expert is good enough to be promoted, and the gates got narrower
    stage_dict = Curriculum([{"radius": 1.2}, {"radius": 0.4}])
    stage_dict.apply(env.task)
    assert env.task.radius == 1.2
    with pytest.raises(AttributeError):
        Curriculum([{"colour": 1}]).apply(env)
    with pytest.raises(AttributeError, match="set_difficulty"):
        Curriculum([0.5]).apply(VecQuadEnv(2, task="hover"))


# ------------------------------------------------------------------------------------------------ tracking
def test_jsonl_and_csv_loggers_record_every_call_and_survive_a_truncated_file(tmp_path):
    with JsonlLogger(tmp_path / "run.jsonl") as lg:
        lg.log(1, loss=np.float32(0.5), note="a")
        lg.log(2, loss=0.25, extra=np.array(3.0))
    rows = read_jsonl(tmp_path / "run.jsonl")
    assert [r["step"] for r in rows] == [1, 2] and rows[0]["loss"] == 0.5 and rows[1]["extra"] == 3.0
    with open(tmp_path / "run.jsonl", "a", encoding="utf-8") as f:
        f.write('{"step": 3, "lo')  # a crash mid-write
    assert len(read_jsonl(tmp_path / "run.jsonl")) == 2
    with CsvLogger(tmp_path / "run.csv") as c:
        c.log(1, a=1.0)
        c.log(2, a=2.0, b=5)
    text = (tmp_path / "run.csv").read_text(encoding="utf-8").splitlines()
    assert text[0] == "step,a,b" and text[1].startswith("1,1.0,") and text[2] == "2,2.0,5"
    multi = MultiLogger([JsonlLogger(tmp_path / "m.jsonl"), CsvLogger(tmp_path / "m.csv")])
    multi.log(7, x=1)
    multi.close()
    assert read_jsonl(tmp_path / "m.jsonl")[0]["step"] == 7 and (tmp_path / "m.csv").exists()


def test_optional_back_ends_explain_what_to_install_and_call_the_documented_api(tmp_path, monkeypatch):
    for mod in ("mlflow", "wandb", "tensorboardX", "torch.utils.tensorboard"):
        monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(ImportError, match="mlflow"):
        MLflowLogger()
    with pytest.raises(ImportError, match="wandb"):
        WandbLogger("p")
    with pytest.raises(ImportError, match="tensorboard"):
        TensorBoardLogger(tmp_path)
    calls = []
    fake_mlflow = types.SimpleNamespace(
        set_experiment=lambda e: calls.append(("exp", e)),
        start_run=lambda run_name=None: calls.append(("start", run_name)) or object(),
        log_params=lambda p: calls.append(("params", p)),
        log_metric=lambda k, v, step: calls.append(("metric", k, v, step)),
        end_run=lambda: calls.append(("end",)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    lg = MLflowLogger("r", "e", {"lr": 0.1})
    lg.log(5, loss=np.float32(0.5), text="skip")
    lg.close()
    assert ("metric", "loss", 0.5, 5) in calls and ("params", {"lr": 0.1}) in calls and calls[-1] == ("end",)
    assert not any(c[0] == "metric" and c[1] == "text" for c in calls)
    logged = []
    fake_wandb = types.SimpleNamespace(
        init=lambda **kw: logged.append(("init", kw)), log=lambda d, step: logged.append(("log", d, step)), finish=lambda: logged.append(("finish",))
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    w = WandbLogger("proj", "n", {"a": 1}, mode="offline")
    w.log(3, r=1.5)
    w.close()
    assert logged[0][1]["project"] == "proj" and logged[0][1]["mode"] == "offline" and ("log", {"r": 1.5}, 3) in logged and logged[-1] == ("finish",)
    scalars = []

    class Writer:
        def __init__(self, d):
            self.d = d

        def add_scalar(self, k, v, step):
            scalars.append((k, v, step))

        def close(self):
            scalars.append("closed")

    monkeypatch.setitem(sys.modules, "tensorboardX", types.SimpleNamespace(SummaryWriter=Writer))
    tb = TensorBoardLogger(tmp_path)
    tb.log(2, a=1.0, b="no")
    tb.close()
    assert scalars == [("a", 1.0, 2), "closed"]


def test_the_sb3_callback_logs_episode_statistics(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from aevion.envs import make_sb3_vec_env
    from aevion.ml import LoggingCallback

    venv = make_sb3_vec_env(4, task="hover", max_episode_s=0.6, seed=0)
    path = tmp_path / "ppo.jsonl"
    with JsonlLogger(path) as lg:
        model = sb3.PPO("MlpPolicy", venv, n_steps=64, batch_size=64, n_epochs=1, device="cpu", verbose=0, seed=0)
        model.learn(total_timesteps=4 * 64 * 2, callback=LoggingCallback(lg))
    rows = read_jsonl(path)
    assert rows and {"ep_return", "ep_length", "success_rate"} <= set(rows[0]) and rows[-1]["step"] > rows[0]["step"]


# ------------------------------------------------------------------------------------------------ tuning
def test_optuna_search_finds_the_minimum_of_a_known_function_and_is_seeded():
    f = lambda p: (p["x"] - 1.5) ** 2 + (np.log10(p["lr"]) + 2) ** 2 + (0 if p["kind"] == "b" else 1) + 0.1 * abs(p["n"] - 3)
    space = {"x": ("float", -5.0, 5.0), "lr": ("float", 1e-5, 1e-1, "log"), "kind": ("categorical", ["a", "b"]), "n": ("int", 1, 8)}
    a = optuna_search(f, space, n_trials=60, seed=0)
    b = optuna_search(f, space, n_trials=60, seed=0)
    assert a.best_value == b.best_value and a.best_params == b.best_params and len(a.trials) == 60
    assert a.best_value < 0.5 and a.best_params["kind"] == "b" and abs(a.best_params["x"] - 1.5) < 0.7
    rand = optuna_search(f, space, n_trials=60, seed=0, sampler="random")
    assert a.best_value <= rand.best_value + 0.5
    top = optuna_search(lambda p: -((p["x"] - 2) ** 2), {"x": ("float", -4.0, 4.0)}, n_trials=40, direction="maximize", seed=1)
    assert abs(top.best_params["x"] - 2) < 0.5 and top.best_value <= 0.0
    for bad in (
        dict(space={}),
        dict(space={"x": ("weird", 0, 1)}),
        dict(space={"x": ("float", 0, 1)}, direction="up"),
        dict(space={"x": ("float", 0, 1)}, sampler="grid"),
    ):
        with pytest.raises(ValueError):
            optuna_search(lambda p: 0.0, **bad)


def test_optuna_search_logs_trials_skips_failing_ones_and_tunes_a_controller_gain(tmp_path):
    def flaky(p):
        if p["x"] > 0.9:
            raise ValueError("diverged")
        return abs(p["x"] - 0.4)

    with JsonlLogger(tmp_path / "t.jsonl") as lg:
        res = optuna_search(flaky, {"x": ("float", 0.0, 1.0)}, n_trials=25, logger=lg, seed=0)
    assert 0 < len(read_jsonl(tmp_path / "t.jsonl")) <= 25 and res.best_value < 0.2

    def always_fails(p):
        raise ValueError("no")

    with pytest.raises(RuntimeError, match=r"every trial failed.*no"):
        optuna_search(always_fails, {"x": ("float", 0.0, 1.0)}, n_trials=3)

    from aevion.engine import Simulation

    def tracking_error(p):
        cfg = {
            "duration": 4.0,
            "seed": 1,
            "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.8}},
            "controller": {"type": "geometric", "params": {"wn_pos": p["wn"]}},
        }
        a = Simulation(cfg).run()
        late = a.t > 1.0
        return float(np.linalg.norm(a["pos"][late, 0] - a["ref_pos"][late, 0], axis=1).mean())

    best = optuna_search(tracking_error, {"wn": ("float", 1.0, 15.0)}, n_trials=8, seed=0)
    assert best.best_value < tracking_error({"wn": 1.0})  # a tuned bandwidth beats the sluggish one


# ------------------------------------------------------------------------------------------------ parallel rollouts
def test_parallel_collect_matches_the_sequential_result_and_is_deterministic():
    kw = dict(make_env=make_env, make_policy=expert_policy, steps=30, seed=5)
    two = parallel_collect(n_workers=2, **kw)
    again = parallel_collect(n_workers=2, **kw)
    assert len(two) == 2 * 4 * 30 and set(np.unique(two.env_ids)) == set(range(8)) and two.meta["workers"] == 2
    assert np.array_equal(two.observations, again.observations) and np.array_equal(two.actions, again.actions)
    inline = [parallel_collect(n_workers=1, **{**kw, "seed": 5 + w}) for w in range(2)]
    assert np.array_equal(two.observations[: len(inline[0])], inline[0].observations)  # worker 0 = the same rollout in-process
    assert np.array_equal(two.observations[len(inline[0]) :], inline[1].observations)
    labeled = parallel_collect(make_env, zero_policy, 20, n_workers=2, seed=1, make_labeler=expert_policy)
    assert np.abs(labeled.actions).max() > 0.05
    with pytest.raises(ValueError):
        parallel_collect(make_env, expert_policy, 5, n_workers=0)
    assert partial(make_env, 1)  # factories may be partials of top-level callables


# ------------------------------------------------------------------------------------------------ Gymnasium vector env
def test_the_gymnasium_vector_env_follows_the_same_step_autoreset_contract():
    gym = pytest.importorskip("gymnasium")
    env = QuadVectorEnv(4, task="hover", max_episode_s=0.5, seed=0)
    assert isinstance(env, gym.vector.VectorEnv) and env.num_envs == 4
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and obs.dtype == np.float32 and info == {}
    seen_final = False
    for _ in range(30):
        obs, r, term, trunc, info = env.step(np.zeros((4, 4), np.float32))
        assert (
            obs.shape == (4, env.single_observation_space.shape[0]) and r.shape == (4,) and term.dtype == bool and env.observation_space.contains(obs)
        )
        if (term | trunc).any():
            seen_final = True
            done = term | trunc
            assert np.array_equal(info["_final_obs"], done) and all(info["final_obs"][i] is not None for i in np.nonzero(done)[0])
            assert info["final_obs"][np.nonzero(done)[0][0]].shape == (env.single_observation_space.shape[0],)
            break
    assert seen_final
    a, b = QuadVectorEnv(3, task="hover", seed=2), QuadVectorEnv(3, task="hover", seed=2)
    oa, _ = a.reset(seed=4)
    ob, _ = b.reset(seed=4)
    assert np.array_equal(oa, ob)
    env.close()
