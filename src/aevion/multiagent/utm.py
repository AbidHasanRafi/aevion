"""UTM-style deconfliction: 4-D trajectory conflicts, strategic ground delays, and prioritised space-time planning.

* A :class:`Trajectory4D` is a piecewise-linear path in space and time. :func:`detect_conflicts` finds the time intervals in which two of
  them are closer than a separation distance, exactly: relative motion is linear between breakpoints, so the squared distance is a
  quadratic in time with closed-form roots (no time-stepping, no missed brief encounters).
* :func:`schedule_departures` is strategic deconfliction by ground delay: routes are accepted in priority order, each with the smallest
  departure delay (on a grid of ``step`` seconds) that keeps it conflict-free against those already accepted.
* :class:`SpaceTimePlanner` is prioritised planning (Silver 2005) with space-time A*: each agent plans on the voxel grid and in time (it
  may wait) against the reserved trajectories of higher-priority agents, so a returned plan is conflict-free for the stated
  separation (the tests re-verify it with :func:`detect_conflicts`).
* :func:`predict_conflicts` is the tactical layer: constant-velocity closest-point-of-approach prediction for a snapshot of positions
  and velocities. Resolving them is the job of the ORCA and barrier filters in :mod:`aevion.planning`.

This is a planning component, not an implementation of any U-space or UTM standard.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Trajectory4D",
    "Conflict",
    "detect_conflicts",
    "closest_approach",
    "predict_conflicts",
    "schedule_departures",
    "SpaceTimePlanner",
    "UTMPlan",
]


class Trajectory4D:
    """Piecewise-linear trajectory ``p(t)`` through ``points`` at strictly increasing ``times``.

    The vehicle exists on ``[t_0, t_end + hold_after]``; outside that window it is absent (no conflicts are reported), and it hovers at its last point
    for ``hold_after`` seconds after arriving (a landed or parked vehicle still occupies space).
    """

    def __init__(self, times: Any, points: Any, hold_after: float = 0.0) -> None:
        t = np.asarray(times, float).reshape(-1)
        p = np.asarray(points, float).reshape(len(t), 3)
        if len(t) < 1 or np.any(np.diff(t) <= 0) or not (np.isfinite(t).all() and np.isfinite(p).all()):
            raise ValueError("times must be finite and strictly increasing, with finite points")
        self.t, self.p, self.hold_after = t, p, float(hold_after)

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1] + self.hold_after)

    def position(self, t: Any) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, float))
        out = np.stack([np.interp(tt, self.t, self.p[:, k]) for k in range(3)], axis=-1)
        return out[0] if np.ndim(t) == 0 else out

    def shifted(self, delay: float) -> "Trajectory4D":
        return Trajectory4D(self.t + delay, self.p, self.hold_after)

    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.p, axis=0), axis=1).sum())

    def max_speed(self) -> float:
        return float((np.linalg.norm(np.diff(self.p, axis=0), axis=1) / np.diff(self.t)).max()) if len(self.t) > 1 else 0.0

    def _breaks(self) -> np.ndarray:
        return self.t if self.hold_after <= 0 else np.append(self.t, self.end)

    def _pos_at(self, tt: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(tt, self.t, self.p[:, k]) for k in range(3)], axis=-1)


@dataclass
class Conflict:
    i: int
    j: int
    t_start: float
    t_end: float
    d_min: float
    t_min: float


def closest_approach(a: Trajectory4D, b: Trajectory4D) -> Tuple[float, float]:
    """Smallest distance between two trajectories over the time both exist, and when (``(inf, nan)`` if they never coexist)."""
    lo, hi = max(a.start, b.start), min(a.end, b.end)
    if hi < lo:
        return float("inf"), float("nan")
    ts = np.unique(np.concatenate([a._breaks(), b._breaks(), [lo, hi]]))
    ts = ts[(ts >= lo) & (ts <= hi)]
    if len(ts) == 1:
        return float(np.linalg.norm(a._pos_at(ts)[0] - b._pos_at(ts)[0])), float(ts[0])
    ra = a._pos_at(ts) - b._pos_at(ts)
    best, tb = np.inf, float("nan")
    for k in range(len(ts) - 1):
        r0, r1, dt = ra[k], ra[k + 1], ts[k + 1] - ts[k]
        dr = r1 - r0
        den = float(dr @ dr)
        s = 0.0 if den < 1e-300 else float(np.clip(-(r0 @ dr) / den, 0.0, 1.0))
        d = float(np.linalg.norm(r0 + s * dr))
        if d < best:
            best, tb = d, float(ts[k] + s * dt)
    return best, tb


def detect_conflicts(trajs: Sequence[Trajectory4D], separation: float) -> List[Conflict]:
    """Every maximal interval in which a pair is closer than ``separation`` (exact roots of the relative-distance quadratic)."""
    out: List[Conflict] = []
    for i, j in itertools.combinations(range(len(trajs)), 2):
        a, b = trajs[i], trajs[j]
        lo, hi = max(a.start, b.start), min(a.end, b.end)
        if hi < lo:
            continue
        ts = np.unique(np.concatenate([a._breaks(), b._breaks(), [lo, hi]]))
        ts = ts[(ts >= lo) & (ts <= hi)]
        ra = a._pos_at(ts) - b._pos_at(ts)
        if len(ts) == 1:
            if np.linalg.norm(ra[0]) < separation:
                out.append(Conflict(i, j, float(ts[0]), float(ts[0]), float(np.linalg.norm(ra[0])), float(ts[0])))
            continue
        cur: Optional[List[float]] = None  # [t_start, t_end, d_min, t_min]
        for k in range(len(ts) - 1):
            t0, t1 = float(ts[k]), float(ts[k + 1])
            r0, dr = ra[k], (ra[k + 1] - ra[k]) / (t1 - t0)  # relative velocity
            A, Bq, Cq = float(dr @ dr), 2.0 * float(r0 @ dr), float(r0 @ r0) - separation**2
            tin: Optional[float]
            tout: Optional[float]
            if A < 1e-300:
                tin, tout = (0.0, t1 - t0) if Cq < 0 else (None, None)
            else:
                disc = Bq * Bq - 4 * A * Cq
                if disc <= 0:
                    tin = tout = None
                else:
                    sq = np.sqrt(disc)
                    r1, r2 = (-Bq - sq) / (2 * A), (-Bq + sq) / (2 * A)
                    tin, tout = max(r1, 0.0), min(r2, t1 - t0)
                    if tin >= tout:
                        tin = tout = None
            if tin is None or tout is None:
                if cur is not None:
                    out.append(Conflict(i, j, cur[0], cur[1], cur[2], cur[3]))
                    cur = None
                continue
            s = float(np.clip(-Bq / (2 * A), tin, tout)) if A >= 1e-300 else tin
            dmin = float(np.linalg.norm(r0 + dr * s))
            if cur is not None and abs(cur[1] - (t0 + tin)) < 1e-9:
                cur[1] = t0 + tout
                if dmin < cur[2]:
                    cur[2], cur[3] = dmin, t0 + s
            else:
                if cur is not None:
                    out.append(Conflict(i, j, cur[0], cur[1], cur[2], cur[3]))
                cur = [t0 + tin, t0 + tout, dmin, t0 + s]
        if cur is not None:
            out.append(Conflict(i, j, cur[0], cur[1], cur[2], cur[3]))
    return out


def predict_conflicts(pos: Any, vel: Any, separation: float, horizon: float = 30.0) -> List[Dict[str, float]]:
    """Pairs that would violate ``separation`` within ``horizon`` seconds if everybody kept their velocity (closest-point-of-approach geometry)."""
    p, v = np.asarray(pos, float), np.asarray(vel, float)
    out = []
    for i, j in itertools.combinations(range(len(p)), 2):
        r, w = p[i] - p[j], v[i] - v[j]
        den = float(w @ w)
        tc = 0.0 if den < 1e-12 else float(np.clip(-(r @ w) / den, 0.0, horizon))
        d = float(np.linalg.norm(r + w * tc))
        if d < separation:
            if den < 1e-12:
                t_in = 0.0
            else:
                b = float(r @ w)
                disc = b * b - den * (float(r @ r) - separation**2)
                t_in = max(0.0, (-b - np.sqrt(max(disc, 0.0))) / den)
            out.append(dict(i=i, j=j, t_cpa=tc, d_cpa=d, t_violation=float(t_in)))
    out.sort(key=lambda c: c["t_violation"])
    return out


def schedule_departures(
    routes: Sequence[Trajectory4D], separation: float, max_delay: float = 120.0, step: float = 1.0, order: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, bool]:
    """Smallest departure delays (multiples of ``step``) that make the routes mutually conflict-free, accepting routes in priority ``order``.

    Returns ``(delays, success)``; an agent that cannot be fitted within ``max_delay`` gets ``nan`` and ``success`` is ``False``.
    """
    n = len(routes)
    order = list(range(n)) if order is None else list(order)
    delays = np.full(n, np.nan)
    accepted: List[Trajectory4D] = []
    ok = True
    for i in order:
        d = 0.0
        while d <= max_delay + 1e-9:
            cand = routes[i].shifted(d)
            if not any(detect_conflicts([cand, a], separation) for a in accepted):
                delays[i] = d
                accepted.append(cand)
                break
            d += step
        else:
            ok = False
    return delays, ok


# --------------------------------------------------------------------------------------------- space-time planner
@dataclass
class UTMPlan:
    trajectories: List[Optional[Trajectory4D]]
    success: bool
    order: List[int]
    reasons: Dict[int, str] = field(default_factory=dict)
    expansions: int = 0

    def makespan(self) -> float:
        return max((t.t[-1] for t in self.trajectories if t is not None), default=0.0)

    def conflicts(self, separation: float) -> List[Conflict]:
        ts = [t for t in self.trajectories if t is not None]
        return detect_conflicts(ts, separation)


class SpaceTimePlanner:
    """Prioritised space-time A* on a :class:`~aevion.planning.space.PlanningSpace`.

    Every step lasts ``dt`` seconds and moves a vehicle to one of its neighbouring cells (or keeps it where it is); ``connectivity`` is 6, 18 or 26.
    A move is accepted only if the cell is free, the straight segment is free of obstacles, and the *continuous* relative motion keeps at least
    ``separation`` metres from every higher-priority agent during the whole step (exact minimum of the linear relative motion). An agent that has
    arrived stays at its goal, so it keeps blocking the cell (and later agents are checked against that).
    """

    def __init__(
        self, space: Any, dt: float = 1.0, separation: float = 1.0, connectivity: int = 26, horizon_steps: int = 200, max_expansions: int = 400_000
    ) -> None:
        from ..planning.graph import neighbour_offsets

        if dt <= 0 or separation <= 0:
            raise ValueError("dt and separation must be positive")
        self.space, self.dt, self.sep = space, float(dt), float(separation)
        self.offsets = neighbour_offsets(connectivity)
        self.moves = np.vstack([np.zeros((1, 3), np.int64), self.offsets])
        self.step_len = np.linalg.norm(self.moves, axis=1) * space.res
        self.horizon, self.max_exp = int(horizon_steps), int(max_expansions)

    @staticmethod
    def _seg_min_dist(a: np.ndarray, b: np.ndarray, pk: np.ndarray, pk1: np.ndarray) -> float:
        """Smallest distance over one step between a mover ``a -> b`` and reserved movers ``pk -> pk1`` (arrays of shape (M, 3))."""
        r0 = a - pk
        dr = (b - a) - (pk1 - pk)
        den = (dr * dr).sum(axis=1)
        s = np.where(den < 1e-300, 0.0, np.clip(-(r0 * dr).sum(axis=1) / np.maximum(den, 1e-300), 0.0, 1.0))
        return float(np.linalg.norm(r0 + s[:, None] * dr, axis=1).min())

    def _plan_one(self, start: np.ndarray, goal: np.ndarray, reserved: List[np.ndarray]) -> Tuple[Optional[np.ndarray], int, str]:
        sp = self.space
        si, gi = sp.index(start), sp.index(goal)
        gs = (int(si[0]), int(si[1]), int(si[2]))
        gg = (int(gi[0]), int(gi[1]), int(gi[2]))
        for name, c in (("start", gs), ("goal", gg)):
            if sp.occ[c]:
                return None, 0, f"the {name} cell is blocked"
        M = len(reserved)
        T_res = max((len(r) for r in reserved), default=1)

        def res_at(k: int) -> np.ndarray:
            return np.array([r[min(k, len(r) - 1)] for r in reserved]) if M else np.zeros((0, 3))

        def move_ok(ca, cb, k) -> bool:
            if M == 0:
                return True
            pa, pb = sp.center(ca), sp.center(cb)
            return self._seg_min_dist(pa, pb, res_at(k), res_at(k + 1)) >= self.sep - 1e-9

        def goal_clear_from(k: int) -> bool:
            if M == 0:
                return True
            pg = sp.center(gg)
            return all(np.linalg.norm(res_at(kk) - pg, axis=1).min() >= self.sep - 1e-9 for kk in range(k, T_res + 1))

        if M and np.linalg.norm(res_at(0) - sp.center(gs), axis=1).min() < self.sep - 1e-9:
            return None, 0, "the start is within the separation distance of a reserved vehicle"
        h_scale = max(self.step_len.max(), 1e-9)
        pg = sp.center(gg)

        def h(c) -> float:
            return float(np.linalg.norm(sp.center(c) - pg)) / h_scale

        cnt = itertools.count()
        open_: List[Tuple[float, int, int, Tuple[int, int, int]]] = [(h(gs), next(cnt), 0, gs)]
        came: Dict[Tuple[Tuple[int, int, int], int], Tuple[Tuple[int, int, int], int]] = {}
        best_g: Dict[Tuple[Tuple[int, int, int], int], int] = {(gs, 0): 0}
        exp = 0
        shape = np.asarray(sp.shape)
        while open_ and exp < self.max_exp:
            f, _, k, c = heapq.heappop(open_)
            if best_g.get((c, k), 1 << 30) < k:
                continue
            exp += 1
            if c == gg and goal_clear_from(k):
                path = [(c, k)]
                while path[-1] in came:
                    path.append(came[path[-1]])
                path.reverse()
                return np.array([sp.center(cc) for cc, _ in path]), exp, ""
            if k >= self.horizon:
                continue
            for mv in self.moves:
                nc = (c[0] + int(mv[0]), c[1] + int(mv[1]), c[2] + int(mv[2]))
                if not (0 <= nc[0] < shape[0] and 0 <= nc[1] < shape[1] and 0 <= nc[2] < shape[2]) or sp.occ[nc]:
                    continue
                if nc != c and not sp.segment_free(sp.center(c), sp.center(nc)):
                    continue
                key = (nc, k + 1)
                if key in best_g:
                    continue
                if not move_ok(c, nc, k):
                    continue
                best_g[key] = k + 1
                came[key] = (c, k)
                heapq.heappush(open_, (k + 1 + h(nc), next(cnt), k + 1, nc))
        return None, exp, "no conflict-free path within the horizon" if exp < self.max_exp else "search budget exhausted"

    def plan(self, starts: Any, goals: Any, priority: Optional[Sequence[int]] = None, retries: int = 3) -> UTMPlan:
        """Plan every agent in priority order; if one fails it is moved to the front and the whole plan is retried (up to ``retries`` times)."""
        S, G = np.asarray(starts, float).reshape(-1, 3), np.asarray(goals, float).reshape(-1, 3)
        if len(S) != len(G):
            raise ValueError("starts and goals must have the same length")
        order = list(range(len(S))) if priority is None else list(priority)
        total_exp = 0
        last: Optional[UTMPlan] = None
        for _ in range(retries + 1):
            paths: Dict[int, Optional[np.ndarray]] = {}
            reasons: Dict[int, str] = {}
            reserved: List[np.ndarray] = []
            failed: Optional[int] = None
            for i in order:
                p, e, why = self._plan_one(S[i], G[i], reserved)
                total_exp += e
                paths[i] = p
                if p is None:
                    reasons[i] = why
                    failed = i
                    break
                reserved.append(p)
            trajs: List[Optional[Trajectory4D]] = [None] * len(S)
            longest = max((len(p) for p in paths.values() if p is not None), default=1)
            for i, p in paths.items():
                if p is not None:  # a vehicle that has arrived stays at its goal until the last one is done
                    hold = (longest - len(p)) * self.dt
                    trajs[i] = (
                        Trajectory4D(np.arange(len(p)) * self.dt, p, hold_after=hold) if len(p) > 1 else Trajectory4D([0.0], p, hold_after=hold)
                    )
            last = UTMPlan(trajs, failed is None, list(order), reasons, total_exp)
            if failed is None:
                return last
            order = [failed] + [i for i in order if i != failed]
        assert last is not None
        return last
