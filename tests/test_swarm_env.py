"""The cooperative multi-agent environment, its scripted experts, the PettingZoo-style API and the parameter-sharing view."""

import numpy as np
import pytest

from aevion.envs.swarm_env import SWARM_TASKS, ParallelSwarmEnv, SwarmEnvConfig, VecSwarmEnv, make_swarm_sb3_vec_env, parallel_env
from aevion.envs.swarm_policies import SwarmScriptedPolicy


def episodes(env, policy, n_steps=None):
    obs = env.reset(seed=1)
    succ, crash, ret = [], [], []
    for _ in range(n_steps or 2 * env.max_steps):
        obs, r, te, tr, info = env.step(policy(obs))
        d = info["done"]
        if d.any():
            succ += info["success"][d].tolist()
            crash += info["crashed"][d].tolist()
            ret += info["final_return"][d].tolist()
    return np.array(succ), np.array(crash), np.array(ret)


# ------------------------------------------------------------------------------------------------ shapes and bookkeeping
@pytest.mark.parametrize("task", SWARM_TASKS)
def test_observation_action_and_reward_shapes(task):
    env = VecSwarmEnv(3, task=task, n_agents=5, k_neighbors=3, seed=1)
    obs = env.reset()
    assert obs.shape == (3, 5, env.obs_dim) == (3, 5, 18 + 3 + 3 * 7) and env.spaces() == (env.obs_dim, 4)
    obs, r, te, tr, info = env.step(np.zeros((3, 5, 4)))
    assert obs.shape == (3, 5, env.obs_dim) and r.shape == (3, 5) and te.shape == tr.shape == (3,)
    assert np.all(np.isfinite(obs)) and np.all(np.isfinite(r))
    assert {"success", "crashed", "min_separation", "final_observation", "done"} <= set(info) and env.info()["n_agents"] == 5


def test_neighbour_lists_are_sorted_by_distance_and_respect_the_communication_range():
    env = VecSwarmEnv(1, task="rendezvous", n_agents=4, k_neighbors=3, seed=2)
    env.reset()
    st = env._state()
    pos = st["pos"][0]
    nb = env._obs()[0, :, 21:].reshape(4, 3, 7)
    for i in range(4):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        order = np.argsort(d)[:3]
        assert np.allclose(nb[i, :, :3] * env.cfg.arena, pos[order] - pos[i], atol=1e-9) and np.all(nb[i, :, 6] == 1.0)
    short = VecSwarmEnv(1, task="rendezvous", n_agents=4, k_neighbors=3, comm_range=0.5, seed=2)
    short.reset()
    z = short._obs()[0, :, 21:].reshape(4, 3, 7)
    assert np.all(z == 0.0)  # nobody is within 0.5 m: empty lists (zeros and no presence flag)
    few = VecSwarmEnv(1, n_agents=2, k_neighbors=4, seed=0)  # fewer other agents than slots: padded
    assert few._obs().shape[-1] == few.obs_dim and np.all(few._obs()[0, 0, 21 + 7 :] == 0.0)


def test_episodes_are_team_events_that_auto_reset_and_reproduce_from_the_seed():
    def run(seed):
        env = VecSwarmEnv(2, task="formation", n_agents=3, max_episode_s=1.0, seed=seed)
        obs = env.reset(seed=seed)
        log = []
        for _ in range(45):
            obs, r, te, tr, info = env.step(np.full((2, 3, 4), 0.2))
            log.append((obs.copy(), r.copy(), info["done"].copy()))
        return log

    a, b = run(7), run(7)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    dones = np.array([x[2] for x in a])
    assert dones.any() and dones.sum(axis=0).min() >= 1  # both teams hit the 1 s limit and restarted
    c = run(8)
    assert not np.array_equal(a[0][0], c[0][0])


def test_a_collision_ends_the_episode_for_the_whole_team_and_is_penalised():
    env = VecSwarmEnv(1, task="rendezvous", n_agents=2, collision_radius=0.4, separation=1.0, seed=3)
    env.reset()
    pol = SwarmScriptedPolicy(env, repulsion=0.0, speed=3.0)  # no avoidance: they fly into each other
    for _ in range(200):
        obs, r, te, tr, info = env.step(pol(env._obs()))
        if info["done"][0]:
            break
    assert te[0] and info["collided"][0] and info["min_separation"][0] < 0.4 and np.all(r < -5.0)


def test_rewards_follow_the_task_and_the_coverage_reward_is_the_cost_decrease():
    env = VecSwarmEnv(1, task="formation", n_agents=4, seed=4)
    env.reset()
    st = env._state()
    # put the vehicles exactly on their slots: the error part of the reward is maximal
    env.slots[0] = st["pos"][0]
    _, r_on, *_ = env.step(np.zeros((1, 4, 4)))
    env.slots[0] = st["pos"][0] + 3.0
    _, r_off, *_ = env.step(np.zeros((1, 4, 4)))
    assert r_on.mean() > r_off.mean() + 0.5
    cov = VecSwarmEnv(1, task="coverage", n_agents=3, seed=5)
    cov.reset()
    c0 = cov.prev_cost[0]
    _, r, _, _, info = cov.step(np.zeros((1, 3, 4)))
    assert np.allclose(r, r[0, 0], atol=0.2) and info["coverage_cost"][0] == pytest.approx(cov.prev_cost[0])
    assert c0 > 0


# ------------------------------------------------------------------------------------------------ scripted experts
@pytest.mark.parametrize("task,n_agents,k", [("formation", 4, 3), ("formation", 8, 4), ("rendezvous", 4, 3), ("coverage", 6, 5)])
def test_the_scripted_decentralised_policy_solves_each_task_without_crashes(task, n_agents, k):
    env = VecSwarmEnv(6, task=task, n_agents=n_agents, k_neighbors=k, seed=0)
    succ, crash, ret = episodes(env, SwarmScriptedPolicy(env))
    assert len(succ) >= 6 and succ.mean() == 1.0 and crash.mean() == 0.0


def test_a_random_policy_does_much_worse_than_the_expert():
    env = VecSwarmEnv(6, task="formation", n_agents=4, seed=0)
    rng = np.random.default_rng(0)
    rand = episodes(env, lambda o: rng.uniform(-1, 1, o.shape[:2] + (4,)))
    exp = episodes(VecSwarmEnv(6, task="formation", n_agents=4, seed=0), SwarmScriptedPolicy(env))
    assert rand[0].mean() < 0.2 and rand[2].mean() < exp[2].mean()


def test_with_too_few_visible_neighbours_gathering_can_split_into_clusters():
    full = VecSwarmEnv(8, task="rendezvous", n_agents=8, k_neighbors=7, seed=0)
    assert (
        episodes(full, SwarmScriptedPolicy(full))[0].mean() >= 0.85
    )  # with everybody visible it gathers (an occasional 8-agent team ends a step outside the radius)
    sparse = VecSwarmEnv(8, task="rendezvous", n_agents=8, k_neighbors=2, comm_range=3.0, seed=0)
    assert (
        episodes(sparse, SwarmScriptedPolicy(sparse))[0].mean() <= 0.1
    )  # the decentralised view matters: limited communication can fragment the team


# ------------------------------------------------------------------------------------------------ backends
def test_torch_and_numpy_dynamics_give_the_same_observations_and_rewards():
    a = VecSwarmEnv(2, task="formation", n_agents=4, seed=3, backend="numpy", dtype="float64")
    b = VecSwarmEnv(2, task="formation", n_agents=4, seed=3, backend="torch", dtype="float64")
    assert np.array_equal(a.reset(seed=5), b.reset(seed=5)) or np.abs(a.reset(seed=5) - b.reset(seed=5)).max() < 1e-12
    act = np.random.default_rng(0).uniform(-1, 1, (2, 4, 4))
    for _ in range(25):
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
    assert np.abs(oa - ob).max() < 1e-9 and np.abs(ra - rb).max() < 1e-9


# ------------------------------------------------------------------------------------------------ PettingZoo-style API
def test_parallel_env_follows_the_pettingzoo_contract():
    env = parallel_env(task="formation", n_agents=3, max_episode_s=1.0, seed=0)
    assert isinstance(env, ParallelSwarmEnv) and env.possible_agents == ["agent_0", "agent_1", "agent_2"] and env.agents == []
    obs, infos = env.reset(seed=1)
    assert env.agents == env.possible_agents and set(obs) == set(infos) == set(env.agents)
    for a in env.agents:
        assert env.observation_space(a).contains(obs[a]) and env.action_space(a).shape == (4,)
    assert env.state().shape == (3 * env.venv.obs_dim,)
    steps = 0
    while env.agents:
        acts = {a: env.action_space(a).sample() * 0.0 for a in env.agents}
        obs, rew, term, trunc, infos = env.step(acts)
        assert set(obs) == set(rew) == set(term) == set(trunc) == set(infos)
        assert all(isinstance(v, float) for v in rew.values())
        steps += 1
        assert steps < 200
    assert all(trunc.values()) or all(term.values())  # the episode ended for the whole team
    with pytest.raises(RuntimeError, match="reset"):
        env.step({a: np.zeros(4) for a in env.possible_agents})
    env.reset()
    with pytest.raises(KeyError):
        env.step({"agent_0": np.zeros(4)})
    assert env.render() is None
    env.close()


# ------------------------------------------------------------------------------------------------ configuration and SB3 view
def test_configuration_is_validated():
    with pytest.raises(ValueError, match="task"):
        VecSwarmEnv(1, task="herding")
    with pytest.raises(ValueError, match="two agents"):
        VecSwarmEnv(1, n_agents=1)
    with pytest.raises(ValueError):
        VecSwarmEnv(1, k_neighbors=0)
    with pytest.raises(TypeError, match="unknown"):
        VecSwarmEnv(1, colour="red")
    assert SwarmEnvConfig().task == "formation"


def test_the_parameter_sharing_view_exposes_every_agent_as_an_environment():
    pytest.importorskip("stable_baselines3")
    venv = make_swarm_sb3_vec_env(2, task="rendezvous", n_agents=3, max_episode_s=0.5, seed=0)
    assert venv.num_envs == 6
    obs = venv.reset()
    assert obs.shape == (6, venv.venv.obs_dim) and obs.dtype == np.float32
    finished = False
    for _ in range(30):
        obs, r, done, infos = venv.step(np.zeros((6, 4), np.float32))
        if done.any():
            finished = True
            i = int(np.nonzero(done)[0][0])
            assert (
                "terminal_observation" in infos[i] and "episode" in infos[i] and done[i // 3 * 3 : i // 3 * 3 + 3].all()
            )  # the whole team finishes together
            break
    assert finished


def test_a_few_ppo_updates_run_on_the_shared_policy_view():
    sb3 = pytest.importorskip("stable_baselines3")
    venv = make_swarm_sb3_vec_env(2, task="rendezvous", n_agents=3, max_episode_s=2.0, seed=0)
    model = sb3.PPO("MlpPolicy", venv, n_steps=16, batch_size=32, n_epochs=1, device="cpu", verbose=0, seed=0)
    model.learn(total_timesteps=6 * 16 * 2)
    obs = venv.reset()
    act, _ = model.predict(obs, deterministic=True)
    assert act.shape == (6, 4) and np.all(np.isfinite(act))
