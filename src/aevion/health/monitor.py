"""The health monitor: runs checks, keeps the last good state, applies a policy on violations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..errors import SimulationDiverged
from .checks import (
    AltitudeCheck,
    BatteryCheck,
    FiniteCheck,
    HealthCheck,
    QuaternionNormCheck,
    RateCheck,
    RotorSpeedCheck,
    SaturationCheck,
    SpeedCheck,
    Violation,
)


@dataclass
class HealthContext:
    """What checks may look at besides the state."""

    B: object
    spec: object = None
    dt_check: float = 0.005  # simulated time between consecutive checks
    emit: object = None  # callable(name, vehicle=None, **data): lets a check publish engine events (e.g. geofence.breach)


class HealthMonitor:
    """Run :class:`HealthCheck` objects every ``every`` steps.

    ``policy``: ``"raise"`` (hard violation raises :class:`SimulationDiverged` carrying the last good state),
    ``"stop"`` (sets :attr:`should_stop`; the engine ends the run cleanly) or ``"warn"``. Soft violations never
    raise; they are recorded and (once each) emitted as warnings/events.
    """

    def __init__(
        self,
        checks: Optional[Sequence[HealthCheck]] = None,
        policy: str = "raise",
        every: int = 1,
        on_violation: Optional[Callable] = None,
        snapshot_every: int = 10,
    ) -> None:
        if policy not in ("raise", "stop", "warn"):
            raise ValueError("policy must be 'raise', 'stop' or 'warn'")
        self.checks: List[HealthCheck] = (
            list(checks)
            if checks is not None
            else [
                FiniteCheck(),
                QuaternionNormCheck(),
                SpeedCheck(),
                RateCheck(),
                AltitudeCheck(),
                RotorSpeedCheck(),
                BatteryCheck(),
                SaturationCheck(),
            ]
        )
        self.policy, self.every = policy, max(1, int(every))
        self.snapshot_every = max(1, int(snapshot_every))
        self.on_violation = on_violation
        self.violations: List[Violation] = []
        self.last_good = None
        self.should_stop = False
        self.n_checks = 0
        self._warned: Set[Tuple[Any, ...]] = set()

    def reset(self) -> None:
        self.violations.clear()
        self.last_good, self.should_stop, self.n_checks = None, False, 0
        self._warned.clear()
        for c in self.checks:
            c.reset()

    @staticmethod
    def _snapshot(state: Any) -> Any:
        try:
            return state.clone().detach()
        except Exception:
            return state.clone()

    def check(self, state: Any, ctx: HealthContext, step: int, t: float) -> List[Violation]:
        if step % self.every:
            return []
        self.n_checks += 1
        found: List[Violation] = []
        for c in self.checks:
            if self.n_checks % c.stride == 1 % c.stride or c.stride == 1:  # first pass, then every `stride`-th
                found.extend(c.check(state, ctx))
        hard = [v for v in found if v.hard]
        if not hard and (self.last_good is None or self.n_checks % self.snapshot_every == 0):
            self.last_good = self._snapshot(state)  # cloning every step is wasteful; a recent snapshot is enough for debugging
        for v in found:
            self.violations.append(v)
            if self.on_violation is not None:
                self.on_violation(v, step, t)
            if not v.hard:
                key = (v.check, v.vehicle)
                if key not in self._warned:
                    self._warned.add(key)
                    warnings.warn(f"health: {v}", stacklevel=3)
        if hard:
            v = hard[0]
            msg = f"simulation diverged: {v.message}" + (f" (+{len(hard) - 1} more violation(s))" if len(hard) > 1 else "")
            if self.policy == "raise":
                raise SimulationDiverged(msg, step, t, v.vehicle, v.quantity, self.last_good)
            if self.policy == "stop":
                self.should_stop = True
            else:
                hard_key: Tuple[Any, ...] = (v.check, v.vehicle, "hard")
                if hard_key not in self._warned:
                    self._warned.add(hard_key)
                    warnings.warn(msg, stacklevel=3)
        return found

    def summary(self) -> dict:
        by: Dict[str, int] = {}
        for v in self.violations:
            by[v.check] = by.get(v.check, 0) + 1
        return dict(checks=[c.name for c in self.checks], n_checks=self.n_checks, violations=by, stopped=self.should_stop)
