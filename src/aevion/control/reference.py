"""Reference (setpoint) containers and trajectory generators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class Reference:
    """Batched tracking reference. All fields are backend tensors with leading dim N."""

    pos: Any
    vel: Any = None
    acc: Any = None
    yaw: Any = None  # (N,)
    yaw_rate: Any = None  # (N,)
    jerk: Any = None

    @staticmethod
    def hover(B, n, pos=(0.0, 0.0, 1.0), yaw=0.0):
        p = B.ones(n, 3) * B.asarray(pos)
        z = B.zeros(n, 3)
        return Reference(pos=p, vel=z, acc=z, yaw=B.zeros(n) + yaw, yaw_rate=B.zeros(n), jerk=z)

    def filled(self, B):
        """Return a copy with every optional field present."""
        z = self.pos * 0.0
        n = self.pos.shape[0]
        return Reference(
            pos=self.pos,
            vel=self.vel if self.vel is not None else z,
            acc=self.acc if self.acc is not None else z,
            yaw=self.yaw if self.yaw is not None else B.zeros(n),
            yaw_rate=self.yaw_rate if self.yaw_rate is not None else B.zeros(n),
            jerk=self.jerk if self.jerk is not None else z,
        )


class Trajectory:
    """Base class: ``sample(t)`` returns NumPy arrays; :meth:`reference` makes a batched Reference."""

    duration: float = math.inf

    def sample(self, t: float):
        raise NotImplementedError

    def reference(self, B, t: float, n: int = 1, offsets=None) -> Reference:
        p, v, a, j, yaw, yr = self.sample(t)
        f = B.asarray
        off = 0.0 if offsets is None else f(offsets)
        return Reference(
            pos=B.ones(n, 3) * f(p) + off,
            vel=B.ones(n, 3) * f(v),
            acc=B.ones(n, 3) * f(a),
            jerk=B.ones(n, 3) * f(j),
            yaw=B.ones(n) * float(yaw),
            yaw_rate=B.ones(n) * float(yr),
        )

    def sample_many(self, ts):
        out = [np.asarray(self.sample(float(t))[0]) for t in ts]
        return np.stack(out)


class Hover(Trajectory):
    def __init__(self, pos=(0, 0, 1.0), yaw=0.0):
        self.pos = np.asarray(pos, float)
        self.yaw = yaw

    def sample(self, t):
        z = np.zeros(3)
        return self.pos, z, z, z, self.yaw, 0.0


class Circle(Trajectory):
    """Horizontal circle of radius ``r`` at angular rate ``omega`` [rad/s] (ramped in over ``ramp`` s)."""

    def __init__(self, center=(0, 0, 1.5), radius=1.0, omega=1.0, phase=0.0, yaw_follow=False, ramp=2.0):
        self.c = np.asarray(center, float)
        self.r, self.w, self.ph = radius, omega, phase
        self.yaw_follow, self.ramp = yaw_follow, ramp

    def _phase(self, t):
        # smooth start: theta = w * (t - ramp/2 + ramp/pi * ...) avoided; use smoothstep-integrated rate
        T = self.ramp
        if T <= 0 or t >= T:
            th = self.w * (t - T / 2 if T > 0 else t)
            thd, thdd = self.w, 0.0
        else:
            s = t / T
            sm = 3 * s * s - 2 * s**3
            dsm = (6 * s - 6 * s * s) / T
            (6 - 12 * s) / (T * T)
            th = self.w * T * (s**3 - s**4 / 2)
            thd = self.w * sm
            thdd = self.w * dsm
        return th + self.ph, thd, thdd

    def sample(self, t):
        th, thd, thdd = self._phase(t)
        r = self.r
        c, s = math.cos(th), math.sin(th)
        p = self.c + r * np.array([c, s, 0.0])
        v = r * thd * np.array([-s, c, 0.0])
        a = r * (thdd * np.array([-s, c, 0.0]) - thd**2 * np.array([c, s, 0.0]))
        yaw = th + math.pi / 2 if self.yaw_follow else 0.0
        return p, v, a, np.zeros(3), yaw, thd if self.yaw_follow else 0.0


class Lissajous(Trajectory):
    """Smooth 3-D Lissajous curve; ``figure8()`` is the 1:2 case in the horizontal plane."""

    def __init__(self, center=(0, 0, 1.5), amp=(1.5, 1.0, 0.0), freq=(0.6, 1.2, 0.0), phase=(0, 0, 0), ramp=2.0):
        self.c = np.asarray(center, float)
        self.A, self.f, self.ph = np.asarray(amp, float), np.asarray(freq, float), np.asarray(phase, float)
        self.ramp = ramp

    @classmethod
    def figure8(cls, center=(0, 0, 1.5), size=1.5, period=10.0):
        w = 2 * math.pi / period
        return cls(center, amp=(size, size / 2, 0.0), freq=(w, 2 * w, 0.0), phase=(0, 0, 0))

    def sample(self, t):
        # phase is shifted so the curve starts at its centre-crossing with zero ramp artefacts
        w = self.f
        arg = w * t + self.ph
        s, c = np.sin(arg), np.cos(arg)
        p = self.c + self.A * s
        v = self.A * w * c
        a = -self.A * w**2 * s
        j = -self.A * w**3 * c
        return p, v, a, j, 0.0, 0.0


class Waypoints(Trajectory):
    """Minimum-snap-style polynomial through waypoints (7th-order segments, C^6 joins).

    Start and end are at rest. ``times`` are segment durations (or give ``speed`` to derive them
    from segment lengths).
    """

    def __init__(
        self, points: Sequence[Sequence[float]], times: Optional[Sequence[float]] = None, speed: float = 1.5, yaw=0.0, yaw_follow=False, hold=0.0
    ):
        P = np.asarray(points, float)
        if len(P) < 2:
            raise ValueError("need at least two waypoints")
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        if times is None:
            # trapezoid-ish allocation: time = length/speed with a floor so accelerations stay finite
            times = np.maximum(seg / speed * 1.6, 0.8)
        self.P, self.T = P, np.asarray(times, float)
        self.knots = np.concatenate([[0.0], np.cumsum(self.T)])
        self.duration = float(self.knots[-1]) + hold
        self.yaw, self.yaw_follow = yaw, yaw_follow
        self.coef = np.stack([self._solve(P[:, k]) for k in range(3)], axis=-1)  # (m, 8, 3)

    @staticmethod
    def _basis(t, order):
        """Row of the d^order/dt^order monomial basis for a 7th-order polynomial."""
        row = np.zeros(8)
        for i in range(order, 8):
            row[i] = math.factorial(i) / math.factorial(i - order) * t ** (i - order)
        return row

    def _solve(self, w):
        m = len(w) - 1
        A = np.zeros((8 * m, 8 * m))
        b = np.zeros(8 * m)
        r = 0
        for s in range(m):
            A[r, 8 * s : 8 * s + 8] = self._basis(0.0, 0)
            b[r] = w[s]
            r += 1
            A[r, 8 * s : 8 * s + 8] = self._basis(self.T[s], 0)
            b[r] = w[s + 1]
            r += 1
        for o in (1, 2, 3):
            A[r, 0:8] = self._basis(0.0, o)
            r += 1
        for o in (1, 2, 3):
            A[r, 8 * (m - 1) : 8 * m] = self._basis(self.T[-1], o)
            r += 1
        for s in range(m - 1):
            for o in range(1, 7):
                A[r, 8 * s : 8 * s + 8] = self._basis(self.T[s], o)
                A[r, 8 * (s + 1) : 8 * (s + 1) + 8] = -self._basis(0.0, o)
                r += 1
        return np.linalg.solve(A, b).reshape(m, 8)

    def sample(self, t):
        t = min(max(t, 0.0), self.knots[-1])
        s = min(int(np.searchsorted(self.knots, t, side="right") - 1), len(self.T) - 1)
        tau = t - self.knots[s]
        c = self.coef[s]  # (8,3)
        p, v, a, j = (self._basis(tau, o) @ c for o in range(4))
        yaw, yr = self.yaw, 0.0
        if self.yaw_follow and np.linalg.norm(v[:2]) > 0.3:
            yaw = math.atan2(v[1], v[0])
        return p, v, a, j, yaw, yr


class Polyline(Trajectory):
    """Constant-speed straight segments (discontinuous acceleration): handy for navigation tasks."""

    def __init__(self, points, speed=1.0):
        self.P = np.asarray(points, float)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.speed = speed
        self.duration = float(self.s[-1] / speed)

    def sample(self, t):
        d = min(t * self.speed, self.s[-1])
        i = min(int(np.searchsorted(self.s, d, side="right") - 1), len(self.P) - 2)
        f = (d - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-9)
        p = self.P[i] + f * (self.P[i + 1] - self.P[i])
        dirv = (self.P[i + 1] - self.P[i]) / max(self.s[i + 1] - self.s[i], 1e-9)
        v = dirv * self.speed if d < self.s[-1] else np.zeros(3)
        return p, v, np.zeros(3), np.zeros(3), 0.0, 0.0
