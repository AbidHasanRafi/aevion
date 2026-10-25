"""Gate racing, landing, evasive-target tracking, inspection and area coverage - semantics and scripted experts."""

import numpy as np
import pytest

from aevion.envs import VecQuadEnv, make_task
from aevion.envs.experts import EXPERTS, make_expert
from aevion.envs.tasks_ext import AreaCoverageTask, GateRacingTask, InspectionTask, LandingTask, TargetTrackingTask

TASKS = ["gate_racing", "landing", "target_tracking", "inspection", "area_coverage"]


def make(task, n=8, seed=0, **kw):
    return VecQuadEnv(
        n, task=task, action_mode="velocity", seed=seed, max_episode_s=kw.pop("max_episode_s", 20.0), control_dt=0.05, physics_dt=0.01, **kw
    )


def episodes(env, policy, seed=1, factor=2):
    B = env.B
    obs = env.reset(seed=seed)
    succ, crash = [], []
    for _ in range(env.max_steps * factor):
        obs, r, te, tr, info = env.step(policy(obs))
        d = np.asarray(B.to_numpy(info["done"])).astype(bool)
        if d.any():
            succ += np.asarray(B.to_numpy(info["success"]))[d].astype(float).tolist()
            crash += np.asarray(B.to_numpy(info["crashed"]))[d].astype(float).tolist()
    return np.array(succ), np.array(crash)


def put(env, pos=None, vel=None):
    B = env.B
    if pos is not None:
        env.state.pos = B.asarray(np.asarray(pos, float))
    if vel is not None:
        env.state.vel = B.asarray(np.asarray(vel, float))


# ------------------------------------------------------------------------------------------------ registry and shapes
@pytest.mark.parametrize("name", TASKS)
def test_the_tasks_are_registered_and_the_observation_has_room_for_their_features(name):
    task = make_task(name)
    assert task.name == name and task.to_dict()["name"] == name
    env = make(name, n=4)
    obs = env.reset()
    assert obs.shape == (4, 18 + 4 + task.extra_obs_dim) and np.all(np.isfinite(obs))
    for _ in range(10):
        obs, r, te, tr, info = env.step(np.zeros((4, 4)))
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(obs))
    assert name in EXPERTS


@pytest.mark.parametrize(
    "name,minimum", [("gate_racing", 0.95), ("landing", 0.95), ("target_tracking", 0.7), ("inspection", 0.9), ("area_coverage", 0.9)]
)
def test_each_scripted_expert_solves_its_task(name, minimum):
    env = make(name, n=16)
    succ, crash = episodes(env, make_expert(env))
    assert len(succ) >= 16 and succ.mean() >= minimum and crash.mean() <= 0.05


def test_a_hovering_vehicle_does_not_solve_anything():
    for name in ("gate_racing", "landing", "inspection", "area_coverage"):
        env = make(name, n=8, max_episode_s=6.0)
        succ, _ = episodes(env, lambda o, env=env: env.B.zeros(env.n, 4), factor=1)
        assert succ.mean() < 0.2, name


def test_experts_check_the_task_and_the_action_mode():
    with pytest.raises(ValueError, match="no scripted expert"):
        make_expert(VecQuadEnv(2, task="hover", action_mode="velocity"))
    with pytest.raises(ValueError, match="velocity"):
        make_expert(VecQuadEnv(2, task="landing", action_mode="thrust_rates"))


def test_the_experts_work_on_the_torch_backend():
    pytest.importorskip("torch")
    env = VecQuadEnv(
        8, task="landing", action_mode="velocity", backend="torch", dtype="float64", seed=0, max_episode_s=12.0, control_dt=0.05, physics_dt=0.01
    )
    succ, crash = episodes(env, make_expert(env))
    assert succ.mean() >= 0.95 and crash.mean() == 0.0


# ------------------------------------------------------------------------------------------------ gate racing
def test_a_gate_counts_only_when_crossed_forwards_inside_its_radius():
    env = make("gate_racing", n=1, seed=3)
    env.reset()
    t = env.task
    c, nrm = env._gates[0, 0].copy(), env._gnorm[0, 0].copy()
    side = np.cross(nrm, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    # fly straight through the middle
    env._gprev = (c - 0.5 * nrm)[None]
    put(env, (c + 0.5 * nrm)[None])
    t.update(env)
    assert env._gidx[0] == 1 and env._gpass[0] == 1.0
    # the same crossing, wide of the gate
    env.reset()
    c, nrm = env._gates[0, 0].copy(), env._gnorm[0, 0].copy()
    side = np.cross(nrm, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    wide = c + (t.radius + 0.5) * side
    env._gprev = (wide - 0.5 * nrm)[None]
    put(env, (wide + 0.5 * nrm)[None])
    t.update(env)
    assert env._gidx[0] == 0 and env._gmiss[0] == 1.0
    # backwards through the middle does not count either
    env._gprev = (c + 0.5 * nrm)[None]
    put(env, (c - 0.5 * nrm)[None])
    t.update(env)
    assert env._gidx[0] == 0 and env._gmiss[0] == 0.0


def test_gate_rewards_and_difficulty():
    env = make("gate_racing", n=1, seed=3)
    env.reset()
    t = env.task
    c, nrm = env._gates[0, 0].copy(), env._gnorm[0, 0].copy()
    env._gprev = (c - 0.5 * nrm)[None]
    put(env, (c + 0.5 * nrm)[None])
    t.update(env)
    r = t.reward(env, env.B.zeros(1, 4), env.B.zeros(1, 4))
    assert r[0] > 5.0  # the gate bonus dominates
    t.reward(env, env.B.zeros(1, 4), env.B.zeros(1, 4))  # same pose, no new gate: no bonus again after the next update
    t.update(env)
    assert env._gpass[0] == 0.0
    t.set_difficulty(0.0)
    easy = t.radius
    t.set_difficulty(1.0)
    assert t.radius < easy and GateRacingTask().n_gates == 4


# ------------------------------------------------------------------------------------------------ landing
def test_landing_succeeds_only_on_the_platform_and_slowly():
    env = make("landing", n=4, seed=2)
    env.reset()
    t = env.task
    tgt = env.B.to_numpy(env.target)
    tv = env.B.to_numpy(env.target_vel)
    low = 0.04
    pos = tgt + [[0, 0, low], [2.0, 0, low], [0, 0, low], [0, 0, 1.5]]
    vel = tv + np.array([[0, 0, -0.3], [0, 0, -0.3], [0, 0, -2.5], [0, 0, 0.0]])
    put(env, pos, vel)
    ok = env.B.to_numpy(t.success(env)).astype(bool)
    bad = env.B.to_numpy(t.extra_crash(env)).astype(bool)
    assert list(ok) == [True, False, False, False]  # on target and slow / off target / too fast / still in the air
    assert list(bad) == [False, True, True, False]  # off the platform or too fast are failures; in the air is neither
    t.set_difficulty(0.0)
    assert t.platform_speed[1] == 0.0
    t.set_difficulty(1.0)
    assert t.platform_speed[1] > 1.0


def test_the_platform_moves_with_its_velocity():
    env = make("landing", n=2, seed=4)
    env.reset()
    p0 = env.B.to_numpy(env.target).copy()
    v = env.B.to_numpy(env.target_vel).copy()
    for _ in range(20):
        env.step(np.zeros((2, 4)))
    dt = env.sim_time - env._pf_t0
    assert np.allclose(env.B.to_numpy(env.target)[:, :2], p0[:, :2] + v[:, :2] * dt[:, None], atol=1e-9)
    assert LandingTask().zone > 0


# ------------------------------------------------------------------------------------------------ target tracking
def test_the_evasive_target_stays_in_its_box_and_obeys_its_speed_and_smoothness_limits():
    env = make("target_tracking", n=6, seed=1)
    env.reset()
    t = env.task
    lo, hi = np.asarray(t.box[0]), np.asarray(t.box[1])
    prev_v = None
    max_jerk = 0.0
    for _ in range(380):  # one episode (20 s = 400 control steps): a new episode legitimately restarts the target
        env.step(np.zeros((6, 4)))
        p, v = env.B.to_numpy(env.target), env.B.to_numpy(env.target_vel)
        assert np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9) and np.linalg.norm(v, axis=1).max() <= t.max_speed + 1e-9
        if prev_v is not None:
            max_jerk = max(max_jerk, np.linalg.norm(v - prev_v, axis=1).max() / env.ctrl_dt)
        prev_v = v
    assert max_jerk < 12.0  # no instantaneous velocity reversals: the target turns back smoothly at the walls
    t.set_difficulty(0.0)
    slow = t.max_speed
    t.set_difficulty(1.0)
    assert t.max_speed > 3 * slow and TargetTrackingTask().agility > 0


# ------------------------------------------------------------------------------------------------ inspection
def test_inspection_counts_visited_points_and_crashes_into_the_structure():
    env = make("inspection", n=3, seed=1)
    env.reset()
    t = env.task
    pts = t._surface()
    put(env, [pts[0] * [1.3, 1.3, 1.0], [0.1, 0.0, 1.0], [10.0, 10.0, 2.0]])
    t.update(env)
    assert env._ins_seen[0, 0] and env._ins_new[0] >= 1
    assert env._ins_seen[2].sum() == 0 and not env._ins_seen[1, :].all()
    crashed = env.B.to_numpy(t.extra_crash(env)).astype(bool)
    assert list(crashed) == [False, True, False]  # the second vehicle is inside the cylinder
    # visiting every point completes the task
    for p in pts:
        put(env, np.tile(p * [1.3, 1.3, 1.0], (3, 1)))
        t.update(env)
    assert env.B.to_numpy(t.success(env)).all() and t.n_points == 24
    t.set_difficulty(1.0)
    hard = t.sense
    t.set_difficulty(0.0)
    assert t.sense > hard and InspectionTask().sense > 0


# ------------------------------------------------------------------------------------------------ area coverage
def test_coverage_only_grows_and_completes_at_the_threshold():
    env = make("area_coverage", n=2, seed=1)
    env.reset()
    t = env.task
    frac = []
    c = t._centres()
    for k in range(len(c)):
        put(env, np.array([[c[k, 0], c[k, 1], t.altitude]] * 2))
        t.update(env)
        frac.append(env._cov_seen.mean(axis=1).copy())
        if k == 0:
            assert env._cov_new.min() >= 1
    f = np.array(frac)
    assert np.all(np.diff(f, axis=0) >= -1e-12) and f[-1].min() == 1.0
    assert env.B.to_numpy(t.success(env)).all()
    t.set_difficulty(1.0)
    assert t.footprint < 1.0 and AreaCoverageTask().cells == 6
