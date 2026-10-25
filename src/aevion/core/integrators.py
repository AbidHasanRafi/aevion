"""Explicit ODE integrators operating on tuples of arrays.

The state of a vehicle is a tuple ``y = (pos, vel, quat, omega, rotor, ...)``.
``f(y)`` returns the tuple of time derivatives. Quaternions are renormalised by
the caller-supplied ``project`` hook after every full step.
"""

from __future__ import annotations

from typing import Callable, Sequence

State = Sequence


def _axpy(a: float, x: State, y: State):
    return tuple(yi + a * xi for xi, yi in zip(x, y))


def euler(f: Callable, y: State, dt: float, project: Callable | None = None):
    k1 = f(y)
    out = _axpy(dt, k1, y)
    return project(out) if project else out


def semi_implicit_euler(f: Callable, y: State, dt: float, project: Callable | None = None):
    """Symplectic-style update for (pos, vel, ...) states: velocity first, then position.

    Assumes ``y[0]`` is position and ``y[1]`` is velocity.
    """
    k = f(y)
    vel = y[1] + dt * k[1]
    pos = y[0] + dt * vel
    rest = tuple(yi + dt * ki for yi, ki in zip(y[2:], k[2:]))
    out = (pos, vel) + rest
    return project(out) if project else out


def midpoint(f: Callable, y: State, dt: float, project: Callable | None = None):
    k1 = f(y)
    ym = _axpy(0.5 * dt, k1, y)
    ym = project(ym) if project else ym
    k2 = f(ym)
    out = _axpy(dt, k2, y)
    return project(out) if project else out


def rk4(f: Callable, y: State, dt: float, project: Callable | None = None):
    k1 = f(y)
    y2 = _axpy(0.5 * dt, k1, y)
    y2 = project(y2) if project else y2
    k2 = f(y2)
    y3 = _axpy(0.5 * dt, k2, y)
    y3 = project(y3) if project else y3
    k3 = f(y3)
    y4 = _axpy(dt, k3, y)
    y4 = project(y4) if project else y4
    k4 = f(y4)
    out = tuple(yi + (dt / 6.0) * (a + 2 * b + 2 * c + d) for yi, a, b, c, d in zip(y, k1, k2, k3, k4))
    return project(out) if project else out


METHODS = {
    "euler": euler,
    "semi_implicit": semi_implicit_euler,
    "midpoint": midpoint,
    "rk4": rk4,
}


def get_integrator(name: str):
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(f"unknown integrator {name!r}; choose from {sorted(METHODS)}") from None
