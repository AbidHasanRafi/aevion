"""Swarm coordination: lossy/delayed inter-agent communication and decentralised collision avoidance.

Each agent only knows the *last received* position of its neighbours (stale when packets are lost or
delayed). A short-range repulsive potential shifts each agent's reference away from neighbours that are
closer than ``avoid_radius``; with imperfect communication this degrades, which is exactly the effect
swarm-robotics studies want to quantify.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CommModel:
    delay_s: float = 0.0
    loss: float = 0.0  # independent packet-loss probability per link per broadcast
    range_m: float = np.inf
    rate_hz: float = 20.0


class SwarmCoordinator:
    def __init__(
        self, n: int, dt: float, avoid_radius: float = 1.0, gain: float = 0.8, max_push: float = 1.5, comm: Optional[CommModel] = None, seed: int = 0
    ):
        self.n, self.dt = n, dt
        self.R, self.gain, self.max_push = avoid_radius, gain, max_push
        self.comm = comm or CommModel()
        self.rng = np.random.default_rng(seed)
        self.known: Optional[np.ndarray] = None  # (N, N, 3): agent i's belief about agent j's position
        self.age = np.zeros((n, n))
        self._queue: deque = deque()
        self._next_tx = 0.0
        self.sent = self.delivered = 0
        self.laws: list = []  # cooperative laws (see :mod:`aevion.multiagent.laws`) applied to the references each step

    @classmethod
    def from_dict(cls, d: dict, n: int, dt: float, seed: int = 0):
        d = dict(d)
        comm = CommModel(**d.pop("comm", {})) if "comm" in d else CommModel()
        return cls(n, dt, comm=comm, seed=seed, **d)

    def _broadcast(self, t, pos):
        c = self.comm
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        ok = (self.rng.random((self.n, self.n)) >= c.loss) & (dist <= c.range_m)
        np.fill_diagonal(ok, True)
        self.sent += self.n * (self.n - 1)
        self.delivered += int(ok.sum()) - self.n
        self._queue.append((t + c.delay_s, pos.copy(), ok))

    def update(self, t: float, pos: np.ndarray):
        """Advance the communication layer to time ``t`` with true positions ``pos`` (N,3)."""
        if self.known is None:
            self.known = np.repeat(pos[None], self.n, 0).copy()
        if t + 1e-9 >= self._next_tx:
            self._next_tx += 1.0 / self.comm.rate_hz
            self._broadcast(t, pos)
        self.age += self.dt
        while self._queue and self._queue[0][0] <= t + 1e-9:
            _, p, ok = self._queue.popleft()
            known = self.known
            assert known is not None
            for i in range(self.n):
                m = ok[i]  # agent i receives from j where ok[i, j]... link j->i
                known[i, m] = p[m]
                self.age[i, m] = 0.0
        self.known[np.arange(self.n), np.arange(self.n)] = pos  # self-knowledge is exact

    def repulsion(self, pos: np.ndarray) -> np.ndarray:
        """Reference offset (N,3) pushing each agent away from believed-close neighbours."""
        if self.known is None:
            return np.zeros_like(pos)
        d = pos[:, None, :] - self.known  # from believed neighbour position to me
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, np.inf)
        w = np.clip(self.R - dist, 0.0, None) / np.maximum(dist, 1e-6)
        push = self.gain * (w[..., None] * d).sum(1)
        n = np.linalg.norm(push, axis=-1, keepdims=True)
        return push * np.minimum(1.0, self.max_push / np.maximum(n, 1e-9))

    def delivery_ratio(self) -> float:
        return self.delivered / max(self.sent, 1)


def min_separation(pos: np.ndarray) -> np.ndarray:
    """Per-step minimum pairwise distance for positions ``(T, N, 3)``."""
    T, N, _ = pos.shape
    if N < 2:
        return np.full(T, np.inf)
    d = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1)
    d[:, np.arange(N), np.arange(N)] = np.inf
    return d.reshape(T, -1).min(1)
