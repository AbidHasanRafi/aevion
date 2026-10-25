"""Collision avoidance: ORCA in three dimensions and control-barrier-function safety filters.

**ORCA** (Optimal Reciprocal Collision Avoidance; van den Berg, Guy, Lin & Manocha 2011, in 3-D as in RVO2-3D) gives each agent a
half-space of velocities that are collision-free for the next ``τ`` seconds, assuming the neighbour takes its half of the avoidance.
The agent then picks the velocity in the intersection of its half-spaces closest to its preferred velocity. Because each pair splits
the effort, agents running the same rule neither collide nor oscillate. Static obstacles in a signed-distance field add one
half-space per agent.

**Control barrier functions** (Ames, Xu, Grizzle & Tabuada 2017) filter a nominal command so a safe set ``{h(x) >= 0}`` stays
forward-invariant: the filtered ``u`` minimises ``|u - u_nom|²`` subject to ``ḣ(x, u) >= -α h(x)``. With ``h`` the distance to an
obstacle minus the vehicle radius, this is a half-space constraint built from the distance field's gradient.
:meth:`CBFSafetyFilter.filter_acceleration` is the second-order version (high-order CBF, Xiao & Belta 2019) for a double integrator,
``ḧ + (α₁+α₂) ḣ + α₁α₂ h >= 0``.

Both reduce to the small projection problems in :mod:`aevion.planning.qp`.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from .qp import project

__all__ = ["orca_half_space", "ORCA3D", "CBFSafetyFilter"]


def orca_half_space(p: np.ndarray, v: np.ndarray, radius: float, tau: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """The ORCA geometry for one neighbour.

    ``p`` is the neighbour's position relative to the agent, ``v`` the agent's velocity relative to the neighbour, ``radius`` the
    sum of the two radii, ``tau`` the time horizon and ``dt`` the control period (used when the agents already overlap).
    Returns ``(u, n)``: ``u`` is the smallest change of the relative velocity that reaches the boundary of the velocity obstacle
    and ``n`` is the outward unit normal there.
    """
    dist2 = float(p @ p)
    R = float(radius)
    if dist2 > R * R:
        w = v - p / tau
        wl2 = float(w @ w)
        dot1 = float(w @ p)
        if dot1 < 0.0 and dot1 * dot1 > R * R * wl2:  # closest boundary point is on the truncating sphere
            wl = math.sqrt(wl2)
            n = w / wl
            return (R / tau - wl) * n, n
        dist = math.sqrt(dist2)  # closest boundary point is on the cone: solve in the plane of the axis and v
        a = p / dist
        v_par = float(v @ a)
        perp = v - v_par * a
        v_perp = float(np.linalg.norm(perp))
        if v_perp > 1e-12:
            e = perp / v_perp
        else:  # v is on the axis: every perpendicular direction is equivalent
            t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e = np.cross(a, t)
            e /= np.linalg.norm(e)
        cos_t, sin_t = math.sqrt(dist2 - R * R) / dist, R / dist
        tt = v_par * cos_t + v_perp * sin_t
        u_par, u_perp = tt * cos_t - v_par, tt * sin_t - v_perp
        n = -sin_t * a + cos_t * e
        return u_par * a + u_perp * e, n
    w = v - p / dt  # already touching: separate within one control period
    wl = float(np.linalg.norm(w))
    n = w / wl if wl > 1e-12 else (-p / math.sqrt(max(dist2, 1e-24)) if dist2 > 1e-24 else np.array([1.0, 0.0, 0.0]))
    return (R / dt - wl) * n, n


class ORCA3D:
    """Reciprocal collision avoidance for ``N`` spherical agents, plus an optional static signed-distance obstacle field.

    ``radius`` is the agent radius (scalar or per agent); ``time_horizon`` is how far ahead collisions are avoided;
    ``max_speed`` bounds the chosen velocity; ``dt`` is the control period; ``neighbor_dist`` limits which neighbours matter
    (default ``max_speed * time_horizon * 2 + 2 radius``); ``margin`` inflates the radii for a safety buffer.
    """

    def __init__(
        self,
        radius: Any = 0.4,
        time_horizon: float = 3.0,
        max_speed: float = 3.0,
        dt: float = 0.05,
        neighbor_dist: Optional[float] = None,
        max_neighbors: int = 12,
        margin: float = 0.05,
        sdf: Any = None,
        obstacle_horizon: Optional[float] = None,
    ) -> None:
        if time_horizon <= 0 or max_speed <= 0 or dt <= 0:
            raise ValueError("time_horizon, max_speed and dt must be positive")
        self.radius, self.tau, self.v_max, self.dt = radius, float(time_horizon), float(max_speed), float(dt)
        self.margin = float(margin)
        self.neighbor_dist, self.max_neighbors = neighbor_dist, int(max_neighbors)
        self.sdf = sdf
        self.obstacle_tau = float(obstacle_horizon if obstacle_horizon is not None else time_horizon)
        self.infeasible_count = 0

    def _radii(self, n: int) -> np.ndarray:
        r = np.asarray(self.radius, float)
        return np.full(n, float(r)) if r.ndim == 0 else r.reshape(n)

    def compute(self, pos: Any, vel: Any, pref_vel: Any, other_pos: Any = None, other_vel: Any = None, other_radius: Any = None) -> np.ndarray:
        """New velocities for all agents. ``other_*`` describe non-cooperative obstacles (e.g. an unresponsive vehicle)."""
        P, V, W = (np.atleast_2d(np.asarray(x, float)) for x in (pos, vel, pref_vel))
        n = len(P)
        r = self._radii(n) + 0.5 * self.margin
        nd = self.neighbor_dist if self.neighbor_dist is not None else 2.0 * self.v_max * self.tau + 2.0 * float(r.max())
        if other_pos is not None:
            OP = np.atleast_2d(np.asarray(other_pos, float))
            OV = np.zeros_like(OP) if other_vel is None else np.atleast_2d(np.asarray(other_vel, float))
            OR = np.full(len(OP), 0.5) if other_radius is None else np.broadcast_to(np.asarray(other_radius, float), (len(OP),))
        out = np.empty_like(W)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        for i in range(n):
            cons: list = []
            order = np.argsort(D[i])
            taken = 0
            for j in order:
                if j == i or D[i, j] > nd:
                    continue
                cons.append((P[j] - P[i], V[i] - V[j], r[i] + r[j], 0.5))
                taken += 1
                if taken >= self.max_neighbors:
                    break
            if other_pos is not None:
                for k in range(len(OP)):
                    if np.linalg.norm(OP[k] - P[i]) <= nd:
                        cons.append((OP[k] - P[i], V[i] - OV[k], r[i] + OR[k] + 0.5 * self.margin, 1.0))
            out[i] = self._solve_one(P[i], V[i], W[i], cons, r[i])
        return out

    def _solve_one(self, p: np.ndarray, v: np.ndarray, pref: np.ndarray, cons: list, radius: float) -> np.ndarray:
        for shrink in (1.0, 0.5, 0.25, 0.1):
            A, b = [], []
            for rel_p, rel_v, R, share in cons:
                u, nrm = orca_half_space(rel_p, rel_v, R, self.tau * shrink, self.dt)
                A.append(nrm)
                b.append(float(nrm @ (v + share * u)))
            if self.sdf is not None:
                d = float(self.sdf.distance(p[None, :])[0])
                g = np.asarray(self.sdf.gradient(p[None, :]))[0]
                gn = np.linalg.norm(g)
                if gn > 1e-9 and d - radius < self.v_max * self.obstacle_tau * 2:
                    A.append(g / gn)
                    b.append(-(d - radius) / (self.obstacle_tau * shrink))  # the closing speed leaves at least tau to contact
            res = project(pref, np.array(A) if A else None, np.array(b) if b else None, np.zeros(3), self.v_max)
            if res.feasible:
                return res.x
        self.infeasible_count += 1
        return res.x


class CBFSafetyFilter:
    """Minimum-intervention safety filter from control barrier functions.

    Parameters: ``sdf`` (static obstacles as a :class:`~aevion.environments.sdf.SignedDistanceField`), ``radius`` (vehicle
    radius) and ``margin`` define the barrier ``h = d(x) - radius - margin``; ``alpha`` is the class-K gain of the velocity
    filter (larger = the vehicle may approach faster); ``v_max`` / ``a_max`` bound the output.
    """

    def __init__(
        self,
        sdf: Any = None,
        radius: float = 0.4,
        margin: float = 0.2,
        alpha: float = 1.5,
        v_max: Optional[float] = None,
        a_max: Optional[float] = None,
        activation_distance: float = 10.0,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.sdf, self.radius, self.margin, self.alpha = sdf, float(radius), float(margin), float(alpha)
        self.v_max, self.a_max = v_max, a_max
        self.act = float(activation_distance)
        self.active_count = 0

    # ---- barrier helpers
    def _static(self, p: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
        if self.sdf is None:
            return None
        d = float(self.sdf.distance(p[None, :])[0])
        if d > self.act:
            return None
        g = np.asarray(self.sdf.gradient(p[None, :]))[0]
        gn = np.linalg.norm(g)
        return (d - self.radius - self.margin, g / gn) if gn > 1e-9 else None

    def _agents(self, i: int, P: np.ndarray, r_other: float) -> list:
        out = []
        for j in range(len(P)):
            if j == i:
                continue
            d = P[i] - P[j]
            dn = float(np.linalg.norm(d))
            if dn < 1e-9 or dn > self.act:
                continue
            out.append((j, dn - (self.radius + r_other + self.margin), d / dn))
        return out

    def filter_velocity(
        self, pos: Any, v_des: Any, vel: Any = None, other_pos: Any = None, other_vel: Any = None, other_radius: float = 0.4
    ) -> np.ndarray:
        """Safe velocity commands for single-integrator vehicles: ``∇h · (u - v_other) >= -α h`` for every barrier.

        ``vel`` are the vehicles' current velocities (default: their desired ones); other vehicles, in the same batch or given by
        ``other_pos`` / ``other_vel``, are assumed to keep their current velocity, i.e. they are treated as *unresponsive*, which
        is the safe assumption (for cooperating agents use :class:`ORCA3D`, which splits the effort).
        """
        P, U = np.atleast_2d(np.asarray(pos, float)), np.atleast_2d(np.asarray(v_des, float))
        Vc = U if vel is None else np.atleast_2d(np.asarray(vel, float))
        out = np.empty_like(U)
        allp = P if other_pos is None else np.vstack([P, np.atleast_2d(np.asarray(other_pos, float))])
        allv = (
            Vc
            if other_pos is None
            else np.vstack([Vc, np.zeros((len(allp) - len(P), 3)) if other_vel is None else np.atleast_2d(np.asarray(other_vel, float))])
        )
        for i in range(len(P)):
            A, b = [], []
            s = self._static(P[i])
            if s is not None:
                h, g = s
                A.append(g)
                b.append(-self.alpha * h)
            for j, h, g in self._agents(i, allp, other_radius):
                A.append(g)
                b.append(float(g @ allv[j]) - self.alpha * h)
            if A:
                self.active_count += 1
            res = project(U[i], np.array(A) if A else None, np.array(b) if b else None, np.zeros(3) if self.v_max else None, self.v_max)
            out[i] = res.x
        return out

    def filter_acceleration(self, pos: Any, vel: Any, a_des: Any, alpha1: float = 2.0, alpha2: float = 2.0) -> np.ndarray:
        """Safe accelerations for double-integrator vehicles against the static field (high-order CBF).

        ``∇d · a >= -(α₁+α₂) ∇d · v - α₁ α₂ h``: dropping the (non-negative, for convex obstacles) curvature term of ``ḧ`` only
        makes the constraint more conservative, so the guarantee holds.
        """
        P, V, A_des = (np.atleast_2d(np.asarray(x, float)) for x in (pos, vel, a_des))
        out = np.empty_like(A_des)
        for i in range(len(P)):
            A, b = [], []
            s = self._static(P[i])
            if s is not None:
                h, g = s
                A.append(g)
                b.append(-(alpha1 + alpha2) * float(g @ V[i]) - alpha1 * alpha2 * h)
            res = project(A_des[i], np.array(A) if A else None, np.array(b) if b else None, np.zeros(3) if self.a_max else None, self.a_max)
            out[i] = res.x
        return out
