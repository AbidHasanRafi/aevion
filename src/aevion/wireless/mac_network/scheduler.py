"""Multi-user schedulers for a shared cell: round robin, max-C/I and proportional fair.

Each scheduler picks, per slot, the user to serve in a cell from the users' *instantaneous achievable rates*. The
:func:`simulate_cell` helper plays ``n_slots`` slots with independent Rayleigh fast fading per user and returns time shares and
mean throughputs, which exposes the **multi-user diversity gain** (max-C/I > PF > RR in sum rate; RR is perfectly fair in time,
PF is fair in the proportional sense).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["CellScheduler", "MaxCI", "ProportionalFair", "RoundRobin", "Scheduler", "SCHEDULERS", "make_scheduler", "simulate_cell"]


class Scheduler(ABC):
    name = "scheduler"

    @abstractmethod
    def pick(self, inst_rate: np.ndarray, avg_rate: np.ndarray, slot: int) -> int:
        """Index of the user to serve (``inst_rate`` and ``avg_rate`` are ``(K,)``)."""

    def reset(self) -> None:  # noqa: B027 - optional hook for stateful schedulers
        """Clear any per-run state."""


class RoundRobin(Scheduler):
    name = "round_robin"

    def pick(self, inst_rate: np.ndarray, avg_rate: np.ndarray, slot: int) -> int:
        return slot % len(inst_rate)


class MaxCI(Scheduler):
    """Serve the user with the best instantaneous rate (maximises sum rate, starves cell-edge users)."""

    name = "max_ci"

    def pick(self, inst_rate: np.ndarray, avg_rate: np.ndarray, slot: int) -> int:
        return int(np.argmax(inst_rate))


class ProportionalFair(Scheduler):
    """Serve ``argmax r_inst / r_avg^beta``; with ``beta = 1`` this maximises the sum of log-throughputs."""

    name = "proportional_fair"

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def pick(self, inst_rate: np.ndarray, avg_rate: np.ndarray, slot: int) -> int:
        return int(np.argmax(inst_rate / np.maximum(avg_rate, 1e-9) ** self.beta))


SCHEDULERS: Dict[str, type] = {"round_robin": RoundRobin, "max_ci": MaxCI, "proportional_fair": ProportionalFair}


def make_scheduler(name: str, **kw: Any) -> Scheduler:
    key = name.strip().lower().replace("-", "_")
    aliases = {"rr": "round_robin", "pf": "proportional_fair", "maxci": "max_ci", "max_c_i": "max_ci"}
    key = aliases.get(key, key)
    if key not in SCHEDULERS:
        raise KeyError(f"unknown scheduler {name!r}; choose from {sorted(SCHEDULERS)}")
    return SCHEDULERS[key](**kw)


def simulate_cell(
    mean_sinr_linear: Any,
    bandwidth_hz: float,
    scheduler: Scheduler,
    n_slots: int = 2000,
    rng: Optional[np.random.Generator] = None,
    pf_window: float = 100.0,
    rate_loss_db: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play ``n_slots`` of a cell with users of given mean SINR and Rayleigh fast fading.

    Returns ``(time_share (K,), mean_throughput_bps (K,))``. ``pf_window`` is the PF averaging window in slots.
    """
    rng = rng if rng is not None else np.random.default_rng()
    sinr = np.atleast_1d(np.asarray(mean_sinr_linear, float))
    k = len(sinr)
    if k == 1:
        r = bandwidth_hz * np.log2(1.0 + sinr / 10 ** (rate_loss_db / 10)) * np.ones(1)
        return np.ones(1), r * 1.0
    fade = rng.exponential(1.0, size=(n_slots, k))
    inst = bandwidth_hz * np.log2(1.0 + sinr[None, :] * fade / 10 ** (rate_loss_db / 10))
    avg = np.maximum(inst[0].copy(), 1e-9)
    served = np.zeros(k)
    delivered = np.zeros(k)
    scheduler.reset()
    for s in range(n_slots):
        u = scheduler.pick(inst[s], avg, s)
        served[u] += 1
        delivered[u] += inst[s, u]
        upd = np.zeros(k)
        upd[u] = inst[s, u]
        avg = (1.0 - 1.0 / pf_window) * avg + upd / pf_window
    return served / n_slots, delivered / n_slots


class CellScheduler:
    """Stateful per-cell scheduling of UAVs that share each cell with background (ground) users.

    ``shares(serving, sinr_linear)`` returns the fraction of slots each UAV obtains in its serving cell. Background users have
    fixed mean SINRs drawn once at reset (``bg_sinr_db = (mean, std)``). Proportional-fair averages persist across calls.
    ``policy="none"`` gives every UAV the whole cell (no contention).
    """

    def __init__(
        self,
        n_cells: int,
        policy: str = "proportional_fair",
        background_users: int = 0,
        bg_sinr_db: Tuple[float, float] = (8.0, 6.0),
        n_slots: int = 200,
        pf_window: float = 100.0,
        rng: Optional[np.random.Generator] = None,
    ):
        if policy != "none":
            make_scheduler(policy)  # validate the name
        self.n_cells, self.policy, self.k_bg = n_cells, policy, int(background_users)
        self.bg_sinr_db, self.n_slots, self.pf_window = bg_sinr_db, n_slots, pf_window
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset(0)

    def reset(self, n_uav: int, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        self.bg_lin = 10.0 ** (self.rng.normal(self.bg_sinr_db[0], self.bg_sinr_db[1], size=(self.n_cells, self.k_bg)) / 10.0)
        self.avg_uav = np.zeros(n_uav)
        self.avg_bg = np.zeros((self.n_cells, self.k_bg))
        self._last_cell = np.full(n_uav, -1)

    def shares(self, serving: Any, sinr_linear: Any, bandwidth_hz: float = 1.0) -> np.ndarray:
        serving = np.asarray(serving, int)
        sinr = np.asarray(sinr_linear, float)
        n = len(serving)
        if len(self.avg_uav) != n:
            self.reset(n)
        if self.policy == "none" or (self.k_bg == 0 and n == 0):
            return np.ones(n)
        out = np.ones(n)
        changed = serving != self._last_cell
        self.avg_uav[changed] = 0.0
        self._last_cell = serving.copy()
        for c in np.unique(serving):
            members = np.nonzero(serving == c)[0]
            k_u = len(members)
            users = np.concatenate([sinr[members], self.bg_lin[c]])
            n_users = len(users)
            if n_users == 1:
                continue
            if self.policy == "round_robin":
                out[members] = 1.0 / n_users
                continue
            fade = self.rng.exponential(1.0, size=(self.n_slots, n_users))
            inst = bandwidth_hz * np.log2(1.0 + users[None, :] * fade)
            if self.policy == "max_ci":
                winners = inst.argmax(axis=1)
            else:
                avg = np.concatenate([self.avg_uav[members], self.avg_bg[c]])
                avg = np.where(avg <= 0, inst[0], avg)
                winners = np.empty(self.n_slots, int)
                w = 1.0 / self.pf_window
                for s in range(self.n_slots):
                    u = int(np.argmax(inst[s] / np.maximum(avg, 1e-9)))
                    winners[s] = u
                    avg *= 1.0 - w
                    avg[u] += w * inst[s, u]
                self.avg_uav[members] = avg[:k_u]
                self.avg_bg[c] = avg[k_u:]
            counts = np.bincount(winners, minlength=n_users)[:k_u]
            out[members] = counts / self.n_slots
        return out
