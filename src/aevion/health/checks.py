"""Individual health checks. A check inspects the simulator state and returns zero or more violations.

*Hard* violations mean the simulation is no longer meaningful (non-finite values, runaway speeds);
*soft* ones are warnings worth recording (actuator saturation, battery depleted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Set

import numpy as np


@dataclass
class Violation:
    check: str
    message: str
    vehicle: Optional[int] = None
    quantity: Optional[str] = None
    value: Optional[float] = None
    hard: bool = True

    def __str__(self) -> str:
        v = f" vehicle {self.vehicle}" if self.vehicle is not None else ""
        return f"[{self.check}]{v}: {self.message}"


class HealthCheck:
    """Base class. Subclasses implement :meth:`check` and set ``hard``."""

    name = "check"
    hard = True
    stride = 1  # run every ``stride``-th monitor pass (cheap checks every pass, slower-moving ones less often)

    def check(self, state: Any, ctx: Any) -> List[Violation]:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass


def _np(B: Any, x: Any) -> Any:
    return B.to_numpy(x) if B is not None else np.asarray(x)


class FiniteCheck(HealthCheck):
    """Every state array is finite. Uses a single summed probe first (cheap), details only on failure."""

    name = "finite"
    FIELDS = ("pos", "vel", "quat", "omega", "rotor", "soc", "energy")

    def check(self, state: Any, ctx: Any) -> Any:
        B = ctx.B
        arrays = {f: _np(B, getattr(state, f)) for f in self.FIELDS}
        probe = sum(float(a.sum()) for a in arrays.values())
        if np.isfinite(probe):
            return []
        out = []
        for f, a in arrays.items():
            bad = ~np.isfinite(a)
            if bad.any():
                veh = int(np.nonzero(bad.reshape(bad.shape[0], -1).any(1))[0][0])
                out.append(Violation(self.name, f"non-finite values in '{f}'", veh, f))
        return out or [Violation(self.name, "non-finite state (overflow)")]


class QuaternionNormCheck(HealthCheck):
    name = "quaternion_norm"
    stride = 5

    def __init__(self, tol: float = 1e-06) -> None:
        self.tol = tol

    def check(self, state: Any, ctx: Any) -> List[Any]:
        n = np.linalg.norm(_np(ctx.B, state.quat), axis=-1)
        err = np.abs(n - 1.0)
        if float(err.max()) > self.tol:
            i = int(err.argmax())
            return [Violation(self.name, f"|q| = {n[i]:.9f} deviates from 1 by {err[i]:.2e} (> {self.tol:g})", i, "quat", float(err[i]))]
        return []


class SpeedCheck(HealthCheck):
    name = "speed"
    stride = 5

    def __init__(self, max_speed: float = 200.0) -> None:
        self.max_speed = max_speed

    def check(self, state: Any, ctx: Any) -> List[Any]:
        v = np.linalg.norm(_np(ctx.B, state.vel), axis=-1)
        if float(v.max()) > self.max_speed:
            i = int(v.argmax())
            return [Violation(self.name, f"speed {v[i]:.1f} m/s exceeds {self.max_speed:g} m/s", i, "vel", float(v[i]))]
        return []


class RateCheck(HealthCheck):
    name = "body_rate"
    stride = 5

    def __init__(self, max_rate: float = 500.0) -> None:
        self.max_rate = max_rate

    def check(self, state: Any, ctx: Any) -> List[Any]:
        w = np.linalg.norm(_np(ctx.B, state.omega), axis=-1)
        if float(w.max()) > self.max_rate:
            i = int(w.argmax())
            return [Violation(self.name, f"angular rate {w[i]:.1f} rad/s exceeds {self.max_rate:g} rad/s", i, "omega", float(w[i]))]
        return []


class AltitudeCheck(HealthCheck):
    name = "altitude"
    stride = 5

    def __init__(self, max_altitude: float = 20000.0) -> None:
        self.max_altitude = max_altitude

    def check(self, state: Any, ctx: Any) -> List[Any]:
        z = _np(ctx.B, state.pos)[:, 2]
        if float(z.max()) > self.max_altitude or float(z.min()) < -1e3:
            i = int(np.abs(z).argmax())
            return [Violation(self.name, f"altitude {z[i]:.1f} m outside the valid range", i, "pos.z", float(z[i]))]
        return []


class RotorSpeedCheck(HealthCheck):
    """Rotor speeds stay non-negative and below 1.25 x the nominal maximum."""

    name = "rotor_speed"
    stride = 5

    def check(self, state: Any, ctx: Any) -> List[Any]:
        r = _np(ctx.B, state.rotor)
        wmax = ctx.spec.w_max if ctx.spec is not None else np.inf
        if float(r.min()) < -1e-6 or float(r.max()) > 1.25 * wmax:
            i = int(np.nonzero((r < -1e-6).any(1) | (r > 1.25 * wmax).any(1))[0][0])
            return [Violation(self.name, f"rotor speed {float(r[i].max()):.0f} rad/s outside [0, 1.25 w_max]", i, "rotor", float(r[i].max()))]
        return []


class BatteryCheck(HealthCheck):
    """Soft: warns once a battery is depleted."""

    name = "battery"
    hard = False
    stride = 10

    def __init__(self) -> None:
        self._warned: Set[int] = set()

    def reset(self) -> None:
        self._warned.clear()

    def check(self, state: Any, ctx: Any) -> Any:
        soc = _np(ctx.B, state.soc)
        out = []
        for i in np.nonzero(soc <= 0.0)[0]:
            if int(i) not in self._warned:
                self._warned.add(int(i))
                out.append(Violation(self.name, "battery depleted (state of charge reached 0)", int(i), "soc", 0.0, hard=False))
        return out


class SaturationCheck(HealthCheck):
    """Soft: warns when any rotor is pinned at its limit for more than ``window_s`` seconds."""

    name = "saturation"
    hard = False
    stride = 10

    def __init__(self, window_s: float = 1.0, level: float = 0.98) -> None:
        self.window_s, self.level = window_s, level
        self._run: Optional[np.ndarray] = None
        self._warned: Set[int] = set()

    def reset(self) -> None:
        self._run, self._warned = None, set()

    def check(self, state: Any, ctx: Any) -> Any:
        if ctx.spec is None:
            return []
        sat = (_np(ctx.B, state.rotor) > self.level * ctx.spec.w_max).any(1)
        if self._run is None:
            self._run = np.zeros(len(sat))
        self._run = np.where(sat, self._run + ctx.dt_check * self.stride, 0.0)
        out = []
        for i in np.nonzero(self._run > self.window_s)[0]:
            if int(i) not in self._warned:
                self._warned.add(int(i))
                out.append(
                    Violation(self.name, f"a rotor has been saturated for {self._run[i]:.1f} s", int(i), "rotor", float(self._run[i]), hard=False)
                )
        return out


DEFAULT_CHECKS = (FiniteCheck, QuaternionNormCheck, SpeedCheck, RateCheck, AltitudeCheck, RotorSpeedCheck, BatteryCheck, SaturationCheck)
