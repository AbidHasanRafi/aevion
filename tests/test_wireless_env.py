"""Communication-aware learning environment: link features, rewards and the stateless snapshot."""

import numpy as np
import pytest

from aevion.core.backend import torch_available
from aevion.environments.world import World
from aevion.envs import ConnectedNavigationTask, EnvConfig, VecQuadEnv, make_task
from aevion.wireless.mac_network import NrCellularNetwork

NETWORK = {"sites": [{"pos": [-100.0, 0.0, 25.0], "sectors": 1}], "los_mode": "geometric"}


def building():
    w = World(bounds=(-250, -150, 0, 250, 150, 150))
    w.add_box_corners([-20, -60, 0], [20, 60, 90])  # a wide tower between the cell and the goal
    return w


def make_env(n=6, backend="numpy", **task_kw):
    task = ConnectedNavigationTask(network=NETWORK, **task_kw)
    cfg = EnvConfig(task=task, backend=backend, **ConnectedNavigationTask.env_kwargs())
    return VecQuadEnv(n, cfg, world=building()), task


def test_snapshot_is_stateless_deterministic_and_well_formed():
    net = NrCellularNetwork(world=building(), seed=3, **NETWORK)
    pos = np.array([[-60.0, 0, 50], [60.0, 0, 50], [60.0, 100, 50]])
    a = net.snapshot(pos, rng=np.random.default_rng(0))
    b = net.snapshot(pos, rng=np.random.default_rng(0))
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    assert a["sinr_db"].shape == (3,) and a["direction"].shape == (3, 3) and np.allclose(np.linalg.norm(a["direction"], axis=1), 1.0)
    assert np.all(a["rate_bps"] >= 0) and set(a["serving"]) == {0}
    # no hidden state: a different batch composition gives the same answer for the same UAV
    solo = net.snapshot(pos[1:2], rng=np.random.default_rng(0), shadowing=False)
    full = net.snapshot(pos, rng=np.random.default_rng(0), shadowing=False)
    assert solo["sinr_db"][0] == pytest.approx(full["sinr_db"][1])
    # behind the tower (NLoS) is clearly worse than the same distance in the open
    far_shadow, far_open = (
        net.snapshot(np.array([[60.0, 0, 50]]), shadowing=False),
        net.snapshot(np.array([[-250.0 + 0.0, 100, 50]]), shadowing=False),
    )
    assert not far_shadow["los"][0] and far_shadow["sinr_db"][0] < 40.0
    assert far_open["los"][0]


def test_task_registers_and_adds_six_observation_features():
    assert isinstance(make_task("connected_navigation"), ConnectedNavigationTask)
    env, task = make_env()
    assert env.obs_dim == 3 + 3 + 9 + 3 + env.act_dim + 6 == 28
    obs = env.reset()
    assert obs.shape == (6, 28) and np.all(np.isfinite(obs))
    link = obs[:, -6:]
    assert np.allclose(np.linalg.norm(link[:, 2:5], axis=1), 1.0, atol=1e-6)  # unit vector to the serving cell
    assert np.all(link[:, 5] > 0)
    info = env.info()
    assert info["obs_dim"] == 28 and info["task"]["name"] == "connected_navigation"


def test_stepping_produces_finite_rewards_and_link_state():
    env, task = make_env(link_every=2)
    env.reset()
    for _ in range(12):
        obs, r, term, trunc, info = env.step(np.zeros((env.n, env.act_dim)))
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(obs)) and obs.shape == (6, 28)
    assert env._link["sinr_db"].shape == (6,)
    assert np.all((env._link_quality >= 0) & (env._link_quality <= 1))


def test_reward_prefers_connected_positions_and_penalises_outage():
    env, task = make_env()
    env.reset()
    # place all vehicles identically but compare the link terms directly
    env._link_quality = env.B.asarray(np.array([1.0, 0.0, 0.5, 1.0, 0.0, 0.5]))
    env._link_outage = env.B.asarray(np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))
    env._prev_dist = env.B.norm(env.state.pos - env.target, axis=-1)
    zero = env.B.zeros(env.n, env.act_dim)
    r = task.reward(env, zero, zero)
    assert r[0] > r[2] > r[1]  # connected > half connected > outage
    assert r[0] - r[1] == pytest.approx(task.w_link + task.w_outage, rel=1e-6)


def test_environment_with_the_task_is_reproducible():
    def run(seed):
        task = ConnectedNavigationTask(network={**NETWORK, "seed": 1})
        env = VecQuadEnv(4, EnvConfig(task=task, seed=seed, **ConnectedNavigationTask.env_kwargs()), world=building())
        obs = [env.reset().copy()]
        rng = np.random.default_rng(0)
        for _ in range(10):
            o, *_ = env.step(rng.uniform(-1, 1, (4, env.act_dim)))
            obs.append(o.copy())
        return np.array(obs)

    assert np.array_equal(run(5), run(5)) and not np.array_equal(run(5), run(6))


@pytest.mark.skipif(not torch_available(), reason="needs torch")
def test_task_runs_on_the_torch_backend():
    env, _ = make_env(4, backend="torch")
    obs = env.reset()
    obs, r, *_ = env.step(np.zeros((4, env.act_dim)))
    assert tuple(obs.shape) == (4, 28) and bool(np.all(np.isfinite(env.B.to_numpy(r))))


def test_gym_wrapper_sees_the_extended_observation_space():
    gym = pytest.importorskip("gymnasium")
    from aevion.envs import QuadrotorEnv

    env = QuadrotorEnv(task=ConnectedNavigationTask(network=NETWORK), **ConnectedNavigationTask.env_kwargs())
    obs, _ = env.reset(seed=0)
    assert isinstance(env.observation_space, gym.spaces.Box) and env.observation_space.shape == obs.shape == (28,)
    env.close()
