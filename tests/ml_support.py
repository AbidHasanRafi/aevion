"""Importable (top-level) factories for the tests that send work to worker processes."""

from aevion.envs import VecQuadEnv
from aevion.envs.experts import make_expert


def make_env(seed):
    return VecQuadEnv(4, task="landing", action_mode="velocity", seed=seed, max_episode_s=4.0, control_dt=0.05, physics_dt=0.01)


def expert_policy(env):
    return make_expert(env)


def zero_policy(env):
    return lambda obs: env.B.zeros(env.n, 4)
