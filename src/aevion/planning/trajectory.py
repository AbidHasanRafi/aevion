"""From a collision-free path to a flyable trajectory.

* :func:`prune_path` and :func:`shortcut_path` remove unnecessary vertices (greedy line-of-sight pruning, random shortcutting).
* :class:`ArcCurve` is a smooth arc-length parametrised curve through a path. :func:`time_optimal_profile` gives the time-optimal speed
  profile along it for a point mass with bounded speed and bounded total acceleration ``sqrt(a_t² + a_n²) <= a_max``, ``a_n = κ v²``
  (forward-backward integration of the phase plane; Bobrow, Dubowsky & Gibson 1985; Pham 2014).
* :class:`MinSnap` solves the minimum-snap problem (Mellinger & Kumar 2011; Richter, Bry & Roy 2013) for piecewise 7th-order polynomials
  through waypoints: minimise ``∫ |p⁗|² dt`` subject to the waypoint positions and continuity of velocity, acceleration and jerk, as an
  equality-constrained QP solved exactly (one KKT system per axis).
* :func:`plan_trajectory` combines them: prune the path, allocate segment times, solve minimum snap, scale time so the velocity and
  acceleration limits are met exactly (scaling time by ``k`` scales velocity by ``1/k`` and acceleration by ``1/k²`` and keeps the
  polynomial optimal), then repair the corridor: where the trajectory leaves free space or the clearance margin, the segment is split
  at a point of the collision-free path and re-solved (Richter et al.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..control.reference import Trajectory
from .graph import PlanResult, astar, path_length, theta_star
from .space import PlanningSpace

__all__ = [
    "prune_path",
    "shortcut_path",
    "resample_path",
    "ArcCurve",
    "TimeProfile",
    "time_optimal_profile",
    "MinSnap",
    "PlannedTrajectory",
    "plan_path",
    "plan_trajectory",
    "trajectory_limits",
]


# ======================================================================================================== path clean-up
def prune_path(space: PlanningSpace, path: Any) -> np.ndarray:
    """Greedy line-of-sight pruning: from each kept vertex jump to the farthest vertex that is directly visible."""
    P = np.asarray(path, float)
    if len(P) <= 2:
        return P.copy()
    out, i = [P[0]], 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not space.segment_free(P[i], P[j]):
            j -= 1
        out.append(P[j])
        i = j
    return np.array(out)


def shortcut_path(space: PlanningSpace, path: Any, n_iter: int = 200, seed: int = 0) -> np.ndarray:
    """Random shortcutting: pick two points on the path and replace what lies between by a straight segment if it is free."""
    P = np.asarray(path, float).copy()
    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        if len(P) < 3:
            break
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        a, b = np.sort(rng.uniform(0.0, s[-1], 2))
        if b - a < 1e-6:
            continue
        ia, ib = int(np.searchsorted(s, a, "right") - 1), int(np.searchsorted(s, b, "right") - 1)
        if ib <= ia:
            continue
        pa = P[ia] + (P[ia + 1] - P[ia]) * (a - s[ia]) / max(seg[ia], 1e-12)
        pb = P[ib] + (P[min(ib + 1, len(P) - 1)] - P[ib]) * (b - s[ib]) / max(seg[ib], 1e-12)
        if space.segment_free(pa, pb):
            P = np.vstack([P[: ia + 1], pa, pb, P[ib + 1 :]])
            keep = np.ones(len(P), bool)
            keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9
            P = P[keep]
    return prune_path(space, P)


def resample_path(path: Any, spacing: float) -> np.ndarray:
    """Points every ``spacing`` metres along the polyline (the end point is kept)."""
    P = np.asarray(path, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] <= 0:
        return P[:1].copy()
    q = np.unique(np.concatenate([np.arange(0.0, s[-1], spacing), [s[-1]]]))
    return np.stack([np.interp(q, s, P[:, k]) for k in range(3)], axis=1)


# ======================================================================================================== smooth curve + TOPP
class ArcCurve:
    """A C² cubic-spline curve through ``points`` with arc-length parametrisation.

    ``sample(s)`` returns position, unit tangent and the curvature vector ``p''(s)`` (so ``κ = |p''|``).
    """

    def __init__(self, points: Any, n_fine: int = 4000) -> None:
        from scipy.interpolate import CubicSpline

        P = np.asarray(points, float)
        if len(P) < 2:
            raise ValueError("need at least two points")
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        if chord[-1] <= 0:
            raise ValueError("the points coincide")
        self._u = chord
        bc = "natural" if len(P) > 2 else "not-a-knot"
        self._spl = CubicSpline(chord, P, bc_type=bc)
        uu = np.linspace(0.0, chord[-1], n_fine)
        d = np.linalg.norm(self._spl(uu, 1), axis=1)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(uu))])
        self._uu, self._s = uu, s
        self.length = float(s[-1])
        self.points = P

    def u_of_s(self, s: Any) -> np.ndarray:
        return np.interp(np.clip(s, 0.0, self.length), self._s, self._uu)

    def sample(self, s: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = self.u_of_s(np.atleast_1d(s))
        p, pu, puu = self._spl(u), self._spl(u, 1), self._spl(u, 2)
        sp = np.linalg.norm(pu, axis=1, keepdims=True)
        t = pu / np.maximum(sp, 1e-12)
        pss = (puu - np.sum(puu * t, axis=1, keepdims=True) * t) / np.maximum(sp, 1e-12) ** 2  # the normal part of d²p/ds²
        return p, t, pss


@dataclass
class TimeProfile:
    """Speed along an arc-length parametrised curve: ``s``, ``v`` and the time ``t`` at which each ``s`` is reached."""

    s: np.ndarray
    v: np.ndarray
    t: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def s_of_t(self, t: Any) -> np.ndarray:
        return np.interp(t, self.t, self.s)

    def v_of_t(self, t: Any) -> np.ndarray:
        return np.interp(t, self.t, self.v)


def time_optimal_profile(
    curve: ArcCurve, v_max: float, a_max: float, v_start: float = 0.0, v_end: float = 0.0, ds: Optional[float] = None
) -> TimeProfile:
    """Time-optimal speed along ``curve`` with ``v <= v_max`` and ``sqrt(a_t² + (κ v²)²) <= a_max``."""
    if v_max <= 0 or a_max <= 0:
        raise ValueError("v_max and a_max must be positive")
    L = curve.length
    n = max(math.ceil(L / (ds or min(0.05, L / 200.0))), 2) + 1
    s = np.linspace(0.0, L, n)
    h = s[1] - s[0]
    _, _, pss = curve.sample(s)
    kappa = np.linalg.norm(pss, axis=1)
    v_lim = np.minimum(v_max, np.sqrt(a_max / np.maximum(kappa, 1e-9)))
    v2 = np.empty(n)
    v2[0] = min(v_start, v_lim[0]) ** 2

    def a_t(i: int, vv2: float) -> float:
        return math.sqrt(max(a_max**2 - (kappa[i] * vv2) ** 2, 0.0))

    for i in range(n - 1):  # forward: accelerate as hard as allowed
        v2[i + 1] = min(v_lim[i + 1] ** 2, v2[i] + 2.0 * a_t(i, v2[i]) * h)
    v2[-1] = min(v2[-1], v_end**2) if v_end >= 0 else v2[-1]
    for i in range(n - 2, -1, -1):  # backward: be able to stop at the end
        v2[i] = min(v2[i], v2[i + 1] + 2.0 * a_t(i + 1, v2[i + 1]) * h)
    v = np.sqrt(np.maximum(v2, 0.0))
    dt = 2.0 * h / np.maximum(v[:-1] + v[1:], 1e-9)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return TimeProfile(s, v, t)


# ======================================================================================================== minimum snap
def _basis(tau: float, order: int, n: int = 8) -> np.ndarray:
    row = np.zeros(n)
    for i in range(order, n):
        row[i] = math.factorial(i) / math.factorial(i - order) * tau ** (i - order)
    return row


def _snap_cost(T: float) -> np.ndarray:
    """``∫₀^T (d⁴/dt⁴ Σ c_i (t/T)^i)² dt`` as a quadratic form in the (normalised-time) coefficients."""
    Q = np.zeros((8, 8))
    for i in range(4, 8):
        for j in range(4, 8):
            Q[i, j] = (math.factorial(i) / math.factorial(i - 4)) * (math.factorial(j) / math.factorial(j - 4)) / (i + j - 7) / T**7
    return Q


class MinSnap:
    """Minimum-snap piecewise polynomial through ``points`` with segment durations ``times`` (7th order, C³ at interior knots).

    The start and end derivatives (velocity, acceleration, jerk) default to rest. The polynomial of segment ``s`` is
    ``Σ_i c[s, i, :] (t / T_s)^i`` for ``t`` in ``[0, T_s]``.
    """

    def __init__(
        self,
        points: Any,
        times: Any,
        start_derivs: Optional[Sequence[Sequence[float]]] = None,
        end_derivs: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        P = np.asarray(points, float)
        T = np.asarray(times, float)
        if len(P) < 2 or len(T) != len(P) - 1 or np.any(T <= 0):
            raise ValueError("need m+1 points and m positive segment times")
        self.P, self.T = P, T
        self.knots = np.concatenate([[0.0], np.cumsum(T)])
        m = len(T)
        sd = np.zeros((3, 3)) if start_derivs is None else np.asarray(start_derivs, float)  # rows: vel, acc, jerk
        ed = np.zeros((3, 3)) if end_derivs is None else np.asarray(end_derivs, float)
        nvar = 8 * m
        Q = np.zeros((nvar, nvar))
        for s in range(m):
            Q[8 * s : 8 * s + 8, 8 * s : 8 * s + 8] = _snap_cost(T[s])
        rows: List[np.ndarray] = []
        rhs: List[np.ndarray] = []

        def add(entries: List[Tuple[int, np.ndarray]], b: np.ndarray) -> None:
            r = np.zeros(nvar)
            for s_, vec in entries:
                r[8 * s_ : 8 * s_ + 8] += vec
            rows.append(r)
            rhs.append(b)

        for s in range(m):
            add([(s, _basis(0.0, 0))], P[s])
            add([(s, _basis(1.0, 0))], P[s + 1])
        for o in (1, 2, 3):  # derivative d^o/dt^o = (1/T^o) d^o/dτ^o
            add([(0, _basis(0.0, o) / T[0] ** o)], sd[o - 1])
            add([(m - 1, _basis(1.0, o) / T[-1] ** o)], ed[o - 1])
        for s in range(m - 1):
            for o in (1, 2, 3):
                add([(s, _basis(1.0, o) / T[s] ** o), (s + 1, -_basis(0.0, o) / T[s + 1] ** o)], np.zeros(3))
        A = np.array(rows)
        b = np.array(rhs)  # (n_con, 3)
        K = np.block([[Q + 1e-12 * np.eye(nvar), A.T], [A, np.zeros((len(A), len(A)))]])
        sol = np.linalg.solve(K, np.vstack([np.zeros((nvar, 3)), b]))
        self.c = sol[:nvar].reshape(m, 8, 3)
        self.cost = float(sum(self.c[:, :, k].reshape(-1) @ Q @ self.c[:, :, k].reshape(-1) for k in range(3)))

    @property
    def duration(self) -> float:
        return float(self.knots[-1])

    def sample(self, t: float, order: int = 0) -> np.ndarray:
        t = min(max(float(t), 0.0), self.duration)
        s = min(int(np.searchsorted(self.knots, t, side="right") - 1), len(self.T) - 1)
        tau = (t - self.knots[s]) / self.T[s]
        return (_basis(tau, order) @ self.c[s]) / self.T[s] ** order

    def sample_many(self, ts: Any, order: int = 0) -> np.ndarray:
        return np.stack([self.sample(t, order) for t in np.asarray(ts, float)])


# ======================================================================================================== planned trajectory
def trajectory_limits(traj: MinSnap, dt: float = 0.02) -> Tuple[float, float]:
    """Peak speed and peak acceleration magnitude along a minimum-snap trajectory."""
    ts = np.arange(0.0, traj.duration + dt, dt)
    v = np.linalg.norm(traj.sample_many(ts, 1), axis=1)
    a = np.linalg.norm(traj.sample_many(ts, 2), axis=1)
    return float(v.max()), float(a.max())


class PlannedTrajectory(Trajectory):
    """A smooth trajectory (a :class:`~aevion.control.reference.Trajectory`) produced by :func:`plan_trajectory`.

    ``sample(t)`` follows the controller convention ``(pos, vel, acc, jerk, yaw, yaw_rate)``; after the end the vehicle holds
    the goal at rest. Provenance (the raw path, the waypoints, planner info) is kept in :attr:`info`.
    """

    def __init__(self, snap: MinSnap, yaw: float = 0.0, yaw_follow: bool = False, hold: float = 0.0, info: Optional[Dict[str, Any]] = None) -> None:
        self.snap = snap
        self.yaw, self.yaw_follow = float(yaw), bool(yaw_follow)
        self.duration = snap.duration + float(hold)
        self.info: Dict[str, Any] = info or {}
        self._last_yaw = float(yaw)

    @property
    def waypoints(self) -> np.ndarray:
        return self.snap.P

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        tt = min(max(float(t), 0.0), self.snap.duration)
        p = self.snap.sample(tt, 0)
        v = self.snap.sample(tt, 1)
        a = self.snap.sample(tt, 2)
        j = self.snap.sample(tt, 3)
        yaw, yr = self.yaw, 0.0
        if self.yaw_follow and float(np.linalg.norm(v[:2])) > 0.3:
            yaw = math.atan2(v[1], v[0])
            self._last_yaw = yaw
        elif self.yaw_follow:
            yaw = self._last_yaw
        return p, v, a, j, yaw, yr

    def path(self, dt: float = 0.05) -> np.ndarray:
        return self.snap.sample_many(np.arange(0.0, self.snap.duration + dt, dt))


# ======================================================================================================== planning front end
_PLANNERS = {"astar": "A*", "theta_star": "Theta*", "rrt_star": "RRT*", "prm": "PRM"}


def plan_path(space: PlanningSpace, start: Any, goal: Any, method: str = "theta_star", seed: int = 0, **kw: Any) -> PlanResult:
    """One entry point for the planners: ``astar``, ``theta_star``, ``rrt_star`` or ``prm`` (extra arguments go to the planner)."""
    from .sampling import PRM, rrt_star

    m = method.lower().replace("*", "_star").replace("-", "_")
    if m == "astar" or m == "a_star":
        return astar(space, start, goal, **kw)
    if m == "theta_star":
        return theta_star(space, start, goal, **kw)
    if m == "rrt_star":
        return rrt_star(space, start, goal, seed=seed, **kw)
    if m == "prm":
        n, k = kw.pop("n_samples", 1000), kw.pop("k", 12)
        return PRM(space, seed=seed).build(n, k).query(start, goal, **kw)
    raise ValueError(f"unknown planner {method!r}; choose from {sorted(_PLANNERS)}")


def _segment_times(P: np.ndarray, v_max: float, a_max: float) -> np.ndarray:
    """Initial segment durations: a rest-to-rest trapezoid per segment, stretched for the corners they share."""
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    T = np.where(2 * d_acc <= L, 2 * t_acc + (L - 2 * d_acc) / v_max, 2.0 * np.sqrt(L / a_max))
    return np.maximum(0.6 * T, 0.2)  # neighbouring segments share their speed, so this is cheaper than rest-to-rest (time is rescaled later)


def plan_trajectory(
    space: Optional[PlanningSpace],
    path: Any,
    v_max: float = 3.0,
    a_max: float = 3.0,
    margin: float = 0.0,
    max_repairs: int = 30,
    prune: bool = True,
    yaw: float = 0.0,
    yaw_follow: bool = False,
    check_dt: float = 0.05,
    tighten: bool = True,
    start_derivs: Any = None,
    end_derivs: Any = None,
) -> PlannedTrajectory:
    """Minimum-snap trajectory along ``path`` that respects ``v_max``, ``a_max`` and stays in the free space of ``space``.

    ``margin`` additionally requires the clearance field (when the space has one) to be at least that large along the
    trajectory. ``tighten`` rescales time so the binding limit is met exactly (otherwise the trajectory may be slower than
    necessary). ``start_derivs`` / ``end_derivs`` (rows: velocity, acceleration, jerk) default to rest; give the current motion to
    continue smoothly from a moving vehicle. If the corridor cannot be repaired within ``max_repairs`` splits, a ``RuntimeError`` is raised.
    """
    P0 = np.asarray(path, float)
    if len(P0) < 2:
        raise ValueError("a path needs at least two points")
    P = prune_path(space, P0) if (prune and space is not None) else P0.copy()
    if len(P) < 2:
        P = P0[[0, -1]]
    T = _segment_times(P, v_max, a_max)
    repairs = 0
    snap = MinSnap(P, T, start_derivs, end_derivs)
    for _ in range(max_repairs + 1):
        for _ in range(8):  # with rest boundaries one rescale is exact; with a moving start the shape changes, so iterate
            vp, ap = trajectory_limits(snap)
            k = max(vp / v_max, math.sqrt(ap / a_max))
            if (k <= 1.0 + 1e-3 and not tighten) or abs(k - 1.0) <= 1e-3:
                break
            if k <= 1.0 and not tighten:
                break
            T = T * (k if tighten or k > 1.0 else 1.0)
            snap = MinSnap(P, T, start_derivs, end_derivs)
        if space is None:
            break
        ts = np.arange(0.0, snap.duration + check_dt, check_dt)
        pts = snap.sample_many(ts)
        bad = ~space.is_free(pts)
        if margin > 0.0 and space.clearance is not None:
            idx = space.index(pts)
            bad |= space.clearance[idx[:, 0], idx[:, 1], idx[:, 2]] < margin
        if not bad.any():
            break
        if repairs >= max_repairs:
            raise RuntimeError("could not keep the smooth trajectory inside free space; use a larger inflation or a denser path")
        tb = float(ts[int(np.argmax(bad))])
        s = min(int(np.searchsorted(snap.knots, tb, side="right") - 1), len(T) - 1)
        frac = float(np.clip((tb - snap.knots[s]) / T[s], 0.1, 0.9))
        P = np.vstack([P[: s + 1], P[s] + frac * (P[s + 1] - P[s]), P[s + 1 :]])  # a point on the (visible) straight segment
        T = np.concatenate([T[:s], [T[s] * frac, T[s] * (1.0 - frac)], T[s + 1 :]])
        snap = MinSnap(P, T, start_derivs, end_derivs)
        repairs += 1
    vp, ap = trajectory_limits(snap)
    info = dict(
        raw_path=P0, repairs=repairs, peak_speed=vp, peak_acceleration=ap, v_max=v_max, a_max=a_max, path_length=path_length(P), snap_cost=snap.cost
    )
    return PlannedTrajectory(snap, yaw=yaw, yaw_follow=yaw_follow, info=info)
