"""Handover management: layer-3 filtering, event A3 with hysteresis and time-to-trigger, interruption, failures and ping-pong.

Event A3 (TS 38.331 5.5.4.4): a neighbour is better than the serving cell by an offset, ``M_n + O_cn - Hys > M_s + O_cs + Off``,
for ``time_to_trigger_s``; the UAV then hands over, with service interrupted for ``interruption_s``. Measurements ``M`` are RSRP [dBm]
smoothed by the L3 filter ``F_n = (1 - a) F_{n-1} + a M_n``, ``a = 1 / 2^(k / 4)`` (TS 38.331 5.5.3.2).

Mobility failures (simplified, after 3GPP TR 36.839):

* **handover failure**: the serving SINR at the handover command was below ``failure_sinr_db`` (the command cannot be received, the
  "too late handover" symptom of a fast UAV); the UAV is out of service for ``failure_recovery_s``.
* **ping-pong**: handing back to the previous cell within ``pingpong_window_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

__all__ = ["HandoverEvent", "HandoverManager", "HandoverStats"]


@dataclass
class HandoverEvent:
    t: float
    link: int
    source: int
    target: int
    failed: bool = False
    pingpong: bool = False


@dataclass
class HandoverStats:
    n_handovers: np.ndarray
    n_failures: np.ndarray
    n_pingpong: np.ndarray

    def rate_per_min(self, duration_s: float) -> np.ndarray:
        return self.n_handovers / max(duration_s, 1e-9) * 60.0


class HandoverManager:
    """A3-based handover controller for ``n_links`` terminals and ``n_cells`` cells."""

    def __init__(
        self,
        n_links: int,
        n_cells: int,
        hysteresis_db: float = 3.0,
        offset_db: float = 0.0,
        time_to_trigger_s: float = 0.16,
        l3_filter_k: int = 4,
        interruption_s: float = 0.03,
        failure_sinr_db: float = -8.0,
        failure_recovery_s: float = 0.5,
        pingpong_window_s: float = 1.0,
    ):
        if n_cells < 1:
            raise ValueError("need at least one cell")
        self.n, self.nb = n_links, n_cells
        self.hys, self.off, self.ttt = hysteresis_db, offset_db, time_to_trigger_s
        self.alpha = 1.0 / 2.0 ** (l3_filter_k / 4.0)
        self.interruption, self.fail_sinr, self.fail_recovery = interruption_s, failure_sinr_db, failure_recovery_s
        self.pp_window = pingpong_window_s
        self.events: List[HandoverEvent] = []
        self.reset()

    def reset(self) -> None:
        self.serving: Optional[np.ndarray] = None
        self._filt: Optional[np.ndarray] = None
        self._ttt_target = np.full(self.n, -1)
        self._ttt_elapsed = np.zeros(self.n)
        self._blocked_until = np.zeros(self.n)
        self._prev_cell = np.full(self.n, -1)
        self._last_ho_t = np.full(self.n, -1e9)
        self.n_ho = np.zeros(self.n, int)
        self.n_fail = np.zeros(self.n, int)
        self.n_pp = np.zeros(self.n, int)
        self.events = []

    @property
    def stats(self) -> HandoverStats:
        return HandoverStats(self.n_ho.copy(), self.n_fail.copy(), self.n_pp.copy())

    def step(self, t: float, rsrp_dbm: Any, dt: float, serving_sinr_db: Optional[Any] = None) -> tuple:
        """Process one measurement report. ``rsrp_dbm`` is ``(N, B)``. Returns ``(serving (N,), interrupted (N,) bool)``."""
        m = np.asarray(rsrp_dbm, float)
        if self._filt is None:
            self._filt = m.copy()
            self.serving = m.argmax(axis=1)  # initial attach: strongest cell
        else:
            self._filt = (1.0 - self.alpha) * self._filt + self.alpha * m
        assert self.serving is not None
        idx = np.arange(self.n)
        f = self._filt
        ms = f[idx, self.serving]
        masked = f.copy()
        masked[idx, self.serving] = -np.inf
        best = masked.argmax(axis=1)
        mn = masked[idx, best]
        a3 = (mn - self.hys > ms + self.off) & (self._blocked_until <= t)
        same = a3 & (best == self._ttt_target)
        self._ttt_elapsed = np.where(same, self._ttt_elapsed + dt, np.where(a3, dt, 0.0))
        self._ttt_target = np.where(a3, best, -1)
        trigger = a3 & (self._ttt_elapsed >= self.ttt - 1e-12)
        for i in np.nonzero(trigger)[0]:
            src, dst = int(self.serving[i]), int(best[i])
            sinr_now = None if serving_sinr_db is None else float(np.broadcast_to(np.asarray(serving_sinr_db, float), (self.n,))[i])
            failed = sinr_now is not None and sinr_now < self.fail_sinr
            pingpong = (dst == self._prev_cell[i]) and (t - self._last_ho_t[i] <= self.pp_window)
            self.events.append(HandoverEvent(t, int(i), src, dst, failed, pingpong))
            self.n_ho[i] += 1
            self.n_pp[i] += int(pingpong)
            self.n_fail[i] += int(failed)
            self._prev_cell[i], self._last_ho_t[i] = src, t
            self.serving[i] = dst
            self._blocked_until[i] = t + (self.fail_recovery if failed else self.interruption)
            self._ttt_target[i], self._ttt_elapsed[i] = -1, 0.0
        return self.serving.copy(), self._blocked_until > t

    def force_reselect(self, links: Any, rsrp_dbm: Any) -> None:
        """After an RLF the terminal re-establishes on the strongest cell (no A3 needed)."""
        links = np.atleast_1d(np.asarray(links, int))
        if self.serving is None or len(links) == 0:
            return
        m = np.asarray(rsrp_dbm, float)
        self.serving[links] = m[links].argmax(axis=1)
        self._ttt_target[links], self._ttt_elapsed[links] = -1, 0.0
        if self._filt is not None:
            self._filt[links] = m[links]
