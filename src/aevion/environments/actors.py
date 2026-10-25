"""Dynamic actors: obstacles that move along a prescribed or stochastic path (birds, other aircraft, vehicles, people, balloons).

An :class:`Actor` has a shape (``sphere``, ``box`` or vertical ``cylinder``) and a pure function ``position(t)`` / ``velocity(t)``.
Actors are added to a world with :meth:`~aevion.environments.world.World.add_actor`: each owns one row of the world's primitive arrays, which
:meth:`~aevion.environments.world.World.advance` rewrites at the start of every physics step, so ray casting, collision, distance
queries and the fused dynamics kernel all see the moving obstacle without any special case. Radar uses :meth:`World.primitive_velocity`
to give the Doppler of a hit.

Implementations (``actors`` registry): ``linear``, ``circular``, ``waypoints`` (looping or ping-pong), ``bouncing`` (reflects inside a box)
and ``random_walk`` (smooth Ornstein-Uhlenbeck velocity, a deterministic function of its seed). Everything is deterministic and
independent of how often it is queried.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = ["Actor", "LinearActor", "CircularActor", "WaypointActor", "BouncingActor", "RandomWalkActor", "ACTORS", "actor_from_dict"]

SHAPES = ("sphere", "box", "cylinder")


def _v3(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be three finite numbers")
    return a


class Actor:
    """Base class. Subclasses implement :meth:`position`; the default :meth:`velocity` is a central difference of it."""

    kind = "actor"

    def __init__(
        self,
        shape: str = "sphere",
        radius: float = 0.5,
        size: Sequence[float] = (1.0, 1.0, 1.0),
        height: float = 1.0,
        color: Optional[Sequence[float]] = None,
        name: str = "",
    ):
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        if radius <= 0 or height <= 0 or np.any(np.asarray(size, float) <= 0):
            raise ValueError("actor dimensions must be positive")
        self.shape, self.radius, self.height = shape, float(radius), float(height)
        self.size = _v3(size, "size")
        self.color = None if color is None else tuple(float(c) for c in color)
        self.name = name or self.kind

    def position(self, t: float) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def velocity(self, t: float) -> np.ndarray:
        h = 1e-4
        return (self.position(t + h) - self.position(max(t - h, 0.0))) / (h + min(h, t))

    def primitive_row(self, t: float) -> np.ndarray:
        """The row written into the world's ``spheres`` (x, y, z, r), ``boxes`` (lo, hi) or ``cylinders`` (x, y, z0, z1, r) array.

        For a cylinder the position is the centre of its axis; a box is centred on the position.
        """
        p = self.position(t)
        if self.shape == "sphere":
            return np.array([p[0], p[1], p[2], self.radius])
        if self.shape == "box":
            return np.concatenate([p - self.size / 2.0, p + self.size / 2.0])
        return np.array([p[0], p[1], p[2] - self.height / 2.0, p[2] + self.height / 2.0, self.radius])

    def _common(self) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(type=self.kind, shape=self.shape, radius=self.radius, size=self.size.tolist(), height=self.height, name=self.name)
        if self.color is not None:
            d["color"] = list(self.color)
        return d

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class LinearActor(Actor):
    """Constant velocity from ``start``: ``p(t) = start + velocity * t``."""

    kind = "linear"

    def __init__(self, start: Sequence[float] = (0.0, 0.0, 0.0), velocity: Sequence[float] = (1.0, 0.0, 0.0), **kw: Any):
        super().__init__(**kw)
        self.start, self._vel = _v3(start, "start"), _v3(velocity, "velocity")

    def position(self, t: float) -> np.ndarray:
        return self.start + self._vel * t

    def velocity(self, t: float) -> np.ndarray:
        return self._vel.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {**self._common(), "start": self.start.tolist(), "velocity": self._vel.tolist()}


class CircularActor(Actor):
    """Horizontal circle of ``radius_path`` about ``center`` at angular rate ``omega`` [rad/s] (counter-clockwise for positive ``omega``)."""

    kind = "circular"

    def __init__(self, center: Sequence[float] = (0.0, 0.0, 5.0), path_radius: float = 10.0, omega: float = 0.2, phase: float = 0.0, **kw: Any):
        super().__init__(**kw)
        if path_radius <= 0:
            raise ValueError("path_radius must be positive")
        self.center, self.path_radius, self.omega, self.phase = _v3(center, "center"), float(path_radius), float(omega), float(phase)

    def position(self, t: float) -> np.ndarray:
        a = self.phase + self.omega * t
        return self.center + np.array([self.path_radius * math.cos(a), self.path_radius * math.sin(a), 0.0])

    def velocity(self, t: float) -> np.ndarray:
        a = self.phase + self.omega * t
        return np.array([-self.path_radius * self.omega * math.sin(a), self.path_radius * self.omega * math.cos(a), 0.0])

    def to_dict(self) -> Dict[str, Any]:
        return {**self._common(), "center": self.center.tolist(), "path_radius": self.path_radius, "omega": self.omega, "phase": self.phase}


class WaypointActor(Actor):
    """Constant-speed travel along straight segments through ``points``; ``mode`` ``"loop"`` closes the path, ``"pingpong"`` reverses at the ends,
    ``"stop"`` halts at the last point."""

    kind = "waypoints"

    def __init__(self, points: Sequence[Sequence[float]] = ((0, 0, 3), (10, 0, 3)), speed: float = 1.0, mode: str = "loop", **kw: Any):
        super().__init__(**kw)
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.all(np.isfinite(p)):
            raise ValueError("points must be an (n, 3) list with at least two waypoints")
        if speed <= 0 or mode not in ("loop", "pingpong", "stop"):
            raise ValueError("speed must be positive and mode one of loop, pingpong, stop")
        self.points, self.speed, self.mode = p, float(speed), mode
        pts = np.vstack([p, p[:1]]) if mode == "loop" else p
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if np.any(seg <= 0):
            raise ValueError("consecutive waypoints must differ")
        self._pts, self._cum = pts, np.concatenate([[0.0], np.cumsum(seg)])
        self._len = float(self._cum[-1])

    def _s(self, t: float) -> float:
        s = self.speed * max(t, 0.0)
        if self.mode == "loop":
            return s % self._len
        if self.mode == "pingpong":
            m = s % (2.0 * self._len)
            return m if m <= self._len else 2.0 * self._len - m
        return min(s, self._len)

    def _segment(self, s: float) -> int:
        return int(min(np.searchsorted(self._cum, s, side="right") - 1, len(self._pts) - 2))

    def position(self, t: float) -> np.ndarray:
        s = self._s(t)
        k = self._segment(s)
        a = (s - self._cum[k]) / (self._cum[k + 1] - self._cum[k])
        return self._pts[k] + a * (self._pts[k + 1] - self._pts[k])

    def velocity(self, t: float) -> np.ndarray:
        s = self.speed * max(t, 0.0)
        if self.mode == "stop" and s >= self._len:
            return np.zeros(3)
        k = self._segment(self._s(t))
        d = (self._pts[k + 1] - self._pts[k]) / (self._cum[k + 1] - self._cum[k])
        if self.mode == "pingpong" and (s % (2.0 * self._len)) > self._len:
            d = -d
        return d * self.speed

    def to_dict(self) -> Dict[str, Any]:
        return {**self._common(), "points": self.points.tolist(), "speed": self.speed, "mode": self.mode}


class BouncingActor(Actor):
    """Moves at constant velocity and reflects off the faces of the axis-aligned box ``bounds = [x0, y0, z0, x1, y1, z1]`` (like a ball)."""

    kind = "bouncing"

    def __init__(
        self,
        start: Sequence[float] = (0.0, 0.0, 3.0),
        velocity: Sequence[float] = (2.0, 1.3, 0.4),
        bounds: Sequence[float] = (-20, -20, 1, 20, 20, 15),
        **kw: Any,
    ):
        super().__init__(**kw)
        b = np.asarray(bounds, float)
        if b.shape != (6,) or np.any(b[3:] <= b[:3]):
            raise ValueError("bounds must be [x0, y0, z0, x1, y1, z1] with x1 > x0 etc.")
        self.start, self._vel, self.bounds = _v3(start, "start"), _v3(velocity, "velocity"), b
        if np.any(self.start < b[:3]) or np.any(self.start > b[3:]):
            raise ValueError("start must lie inside bounds")

    def position(self, t: float) -> np.ndarray:
        lo, hi = self.bounds[:3], self.bounds[3:]
        w = hi - lo
        u = (self.start - lo + self._vel * max(t, 0.0)) % (2.0 * w)  # unfold the reflections onto a triangle wave
        return lo + np.where(u <= w, u, 2.0 * w - u)

    def velocity(self, t: float) -> np.ndarray:
        lo, hi = self.bounds[:3], self.bounds[3:]
        w = hi - lo
        u = (self.start - lo + self._vel * max(t, 0.0)) % (2.0 * w)
        return np.where(u <= w, self._vel, -self._vel)

    def to_dict(self) -> Dict[str, Any]:
        return {**self._common(), "start": self.start.tolist(), "velocity": self._vel.tolist(), "bounds": self.bounds.tolist()}


class RandomWalkActor(Actor):
    """Smooth random motion: the velocity is an Ornstein-Uhlenbeck process (mean 0, std ``speed``, correlation time ``tau``).

    The path is generated once on a fixed ``dt`` grid from ``seed`` and interpolated, so ``position(t)`` does not depend on the order of
    queries. A restoring pull keeps the actor within ``bounds`` (``[x0, y0, z0, x1, y1, z1]``) by reflecting the velocity.
    """

    kind = "random_walk"

    def __init__(
        self,
        start: Sequence[float] = (0.0, 0.0, 3.0),
        speed: float = 1.5,
        tau: float = 3.0,
        seed: int = 0,
        bounds: Sequence[float] = (-30, -30, 1, 30, 30, 20),
        dt: float = 0.05,
        **kw: Any,
    ):
        super().__init__(**kw)
        b = np.asarray(bounds, float)
        if b.shape != (6,) or np.any(b[3:] <= b[:3]) or speed <= 0 or tau <= 0 or dt <= 0:
            raise ValueError("invalid bounds, speed, tau or dt")
        self.start, self.speed, self.tau, self.seed, self.bounds, self.dt = _v3(start, "start"), float(speed), float(tau), int(seed), b, float(dt)
        self._rng = np.random.default_rng(self.seed)
        self._p, self._v = [self.start.copy()], [self._rng.normal(0.0, self.speed, 3) * np.array([1.0, 1.0, 0.3])]
        self._k = math.exp(-dt / tau)

    def _extend(self, n: int) -> None:
        lo, hi = self.bounds[:3], self.bounds[3:]
        while len(self._p) <= n:
            v = self._v[-1] * self._k + self._rng.normal(0.0, self.speed * math.sqrt(1.0 - self._k**2), 3) * np.array([1.0, 1.0, 0.3])
            p = self._p[-1] + v * self.dt
            over, under = p > hi, p < lo
            v = np.where(over | under, -v, v)
            p = np.clip(p, lo, hi)
            self._p.append(p)
            self._v.append(v)

    def position(self, t: float) -> np.ndarray:
        s = max(t, 0.0) / self.dt
        k = int(s)
        self._extend(k + 1)
        a = s - k
        return (1 - a) * self._p[k] + a * self._p[k + 1]

    def velocity(self, t: float) -> np.ndarray:
        s = max(t, 0.0) / self.dt
        k = int(s)
        self._extend(k + 1)
        a = s - k
        return (1 - a) * self._v[k] + a * self._v[k + 1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self._common(),
            "start": self.start.tolist(),
            "speed": self.speed,
            "tau": self.tau,
            "seed": self.seed,
            "bounds": self.bounds.tolist(),
            "dt": self.dt,
        }


ACTORS = {c.kind: c for c in (LinearActor, CircularActor, WaypointActor, BouncingActor, RandomWalkActor)}


def actor_from_dict(d: Dict[str, Any]) -> Actor:
    d = dict(d)
    kind = d.pop("type")
    if kind not in ACTORS:
        raise ValueError(f"unknown actor type {kind!r}; known: {sorted(ACTORS)}")
    return ACTORS[kind](**d)
