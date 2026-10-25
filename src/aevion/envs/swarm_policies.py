"""Scripted decentralised policies for :class:`~aevion.envs.swarm_env.VecSwarmEnv`: expert baselines and demonstrators for imitation learning.

Each policy is a function of one agent's observation only (its own state, the goal vector, and the neighbour list), exactly what a learned policy
would receive, so it can serve as an expert for behaviour cloning and as a performance reference in experiments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..multiagent.coverage import voronoi_assign
from .swarm_env import VecSwarmEnv

__all__ = ["SwarmScriptedPolicy"]


class SwarmScriptedPolicy:
    """Task-specific proportional policies: slot tracking (formation), gathering (rendezvous), Lloyd on the visible neighbours (coverage).

    All of them add a repulsion from any neighbour closer than the environment's ``separation``.
    """

    def __init__(self, env: VecSwarmEnv, gain: float = 1.0, repulsion: float = 3.0, speed: float = 1.5, range_factor: float = 2.2) -> None:
        self.env, self.gain, self.rep, self.speed, self.rf = env, gain, repulsion, speed, range_factor

    def __call__(self, obs: Any) -> np.ndarray:
        env, cfg = self.env, self.env.cfg
        o = np.asarray(obs, float)
        E, A, k = o.shape[0], o.shape[1], cfg.k_neighbors
        pos = o[..., 0:3] * cfg.arena
        goal = o[..., 18:21] * cfg.arena
        nb = o[..., 21:].reshape(E, A, k, env.obs_per_neighbor)
        rel = nb[..., 0:3] * cfg.arena
        present = nb[..., 6] > 0.5
        if cfg.task == "formation":
            want = goal
        elif cfg.task == "rendezvous":
            cnt = np.maximum(present.sum(-1, keepdims=True), 1)
            want = (rel * present[..., None]).sum(2) / cnt
        else:
            want = np.zeros((E, A, 3))
            dom = env.domain
            for e in range(E):
                for i in range(A):
                    sites = np.vstack([pos[e, i, :2], pos[e, i, :2] + rel[e, i, present[e, i], :2]])
                    cell = voronoi_assign(sites, dom.points) == 0
                    m = dom.weights[cell].sum()
                    if m > 0:
                        want[e, i, :2] = (dom.points[cell] * dom.weights[cell, None]).sum(0) / m - pos[e, i, :2]
        d = np.linalg.norm(rel, axis=-1)
        reach = cfg.separation * self.rf
        close = present & (d < reach)
        w = np.where(close, (reach - d) / np.maximum(d, 1e-6) / reach, 0.0)
        push = -(w[..., None] * rel).sum(2)
        v = self.gain * want
        vn = np.linalg.norm(v, axis=-1, keepdims=True)
        v = v * np.minimum(1.0, self.speed / np.maximum(vn, 1e-9))  # approach at a bounded speed so the repulsion has time to act
        v = v + self.rep * push
        a = np.zeros((E, A, 4))
        a[..., :3] = np.clip(v / cfg.max_speed, -1.0, 1.0)
        return a
