"""Deterministic multi-rate task scheduler.

Time advances in physics ticks. A task runs every ``every_ticks`` ticks (or at ``rate_hz``, converted to the
nearest whole number of ticks), at a ``phase`` offset, in ``priority`` order within a tick. Integer-tick
scheduling makes runs exactly reproducible and free of floating-point drift.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..errors import SchedulerError


@dataclass
class Task:
    name: str
    fn: Callable[[], None]
    every_ticks: int = 1
    phase: int = 0
    priority: int = 50
    enabled: bool = True
    requested_hz: Optional[float] = None
    # statistics
    calls: int = 0
    wall_ns: int = 0
    max_ns: int = 0

    def due(self, tick: int) -> bool:
        return self.enabled and tick >= self.phase and (tick - self.phase) % self.every_ticks == 0


class Scheduler:
    def __init__(self, dt: float, profile: bool = False) -> None:
        if dt <= 0:
            raise SchedulerError("dt must be positive")
        self.dt = dt
        self.tick = 0
        self.profile = profile
        self._tasks: List[Task] = []
        self._order: List[Task] = []

    # ------------------------------------------------------------------- tasks
    def ticks_for_rate(self, rate_hz: float, name: str = "task") -> int:
        if rate_hz <= 0:
            raise SchedulerError(f"task '{name}': rate must be positive, got {rate_hz}")
        exact = 1.0 / (rate_hz * self.dt)
        if exact < 1.0 - 1e-6:
            raise SchedulerError(f"task '{name}': {rate_hz:g} Hz exceeds the physics rate {1.0 / self.dt:g} Hz")
        n = max(1, round(exact))
        if abs(n - exact) / exact > 0.01:
            warnings.warn(
                f"task '{name}': {rate_hz:g} Hz is not a whole multiple of the {1.0 / self.dt:g} Hz physics rate; running at {1.0 / (n * self.dt):g} Hz",
                stacklevel=3,
            )
        return n

    def add(
        self,
        name: str,
        fn: Callable[[], None],
        *,
        rate_hz: Optional[float] = None,
        every_ticks: Optional[int] = None,
        phase: int = 0,
        priority: int = 50,
        enabled: bool = True,
    ) -> Task:
        if any(t.name == name for t in self._tasks):
            raise SchedulerError(f"duplicate task name '{name}'")
        if rate_hz is not None and every_ticks is not None:
            raise SchedulerError(f"task '{name}': give rate_hz or every_ticks, not both")
        if rate_hz is not None:
            every_ticks = self.ticks_for_rate(rate_hz, name)
        every_ticks = 1 if every_ticks is None else int(every_ticks)
        if every_ticks < 1:
            raise SchedulerError(f"task '{name}': every_ticks must be >= 1")
        t = Task(name, fn, every_ticks, int(phase), int(priority), enabled, rate_hz)
        self._tasks.append(t)
        self._order = sorted(self._tasks, key=lambda x: (x.priority, self._tasks.index(x)))
        return t

    def remove(self, name: str) -> None:
        self._tasks = [t for t in self._tasks if t.name != name]
        self._order = sorted(self._tasks, key=lambda x: (x.priority, self._tasks.index(x)))

    def get(self, name: str) -> Task:
        for t in self._tasks:
            if t.name == name:
                return t
        raise SchedulerError(f"unknown task '{name}'")

    @property
    def tasks(self) -> List[Task]:
        return list(self._order)

    def effective_rate(self, name: str) -> float:
        return 1.0 / (self.get(name).every_ticks * self.dt)

    # --------------------------------------------------------------------- run
    def run_tick(self) -> None:
        """Run every due task for the current tick (the caller advances physics and then :meth:`advance`)."""
        k = self.tick
        if self.profile:
            for t in self._order:
                if t.due(k):
                    t0 = time.perf_counter_ns()
                    t.fn()
                    d = time.perf_counter_ns() - t0
                    t.calls += 1
                    t.wall_ns += d
                    t.max_ns = max(t.max_ns, d)
        else:
            for t in self._order:
                if t.due(k):
                    t.fn()
                    t.calls += 1

    def advance(self) -> None:
        self.tick += 1

    def reset(self) -> None:
        self.tick = 0
        for t in self._tasks:
            t.calls = t.wall_ns = t.max_ns = 0

    def report(self) -> str:
        total = sum(t.wall_ns for t in self._tasks) or 1
        lines = [f"{'task':<16}{'rate Hz':>9}{'calls':>9}{'ms total':>10}{'us/call':>9}{'share':>7}"]
        for t in self._order:
            per = t.wall_ns / max(t.calls, 1) / 1e3
            lines.append(
                f"{t.name:<16}{1.0 / (t.every_ticks * self.dt):>9.1f}{t.calls:>9}{t.wall_ns / 1e6:>10.1f}{per:>9.1f}{100 * t.wall_ns / total:>6.0f}%"
            )
        return "\n".join(lines)
