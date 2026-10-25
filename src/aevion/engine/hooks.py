"""Simulation lifecycle hooks (callbacks).

Subclass :class:`Hook` and override any of the lifecycle methods, or wrap functions with
:class:`CallbackHook`. Hooks run in ``priority`` order; an exception inside a hook is re-raised with the hook's
name and the simulation step attached, unless the manager was created with ``isolate=True`` (then the hook is
disabled and the failure recorded).
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from ..datasets.archive import RunArchive
    from .simulation import Simulation


class Hook:
    """Base class: override what you need. ``sim`` is the :class:`~aevion.engine.simulation.Simulation`."""

    name = "hook"
    priority = 50

    def on_start(self, sim: "Simulation") -> None:  # before the first step (after reset)
        pass

    def on_reset(self, sim: "Simulation") -> None:
        pass

    def pre_step(self, sim: "Simulation") -> None:  # after control, before physics; may modify ``sim.ctx.cmd``
        pass

    def post_step(self, sim: "Simulation") -> None:  # after physics and health checks
        pass

    def on_event(self, sim: "Simulation", event: Any) -> None:
        pass

    def on_end(self, sim: "Simulation", archive: Optional["RunArchive"] = None) -> None:
        pass


class CallbackHook(Hook):
    """Wrap plain functions: ``CallbackHook(pre_step=lambda sim: ...)``."""

    def __init__(
        self,
        name: str = "callback",
        on_start: Any = None,
        pre_step: Any = None,
        post_step: Any = None,
        on_event: Any = None,
        on_end: Any = None,
        priority: int = 50,
    ) -> None:
        self.name, self.priority = name, priority
        self._f = dict(on_start=on_start, pre_step=pre_step, post_step=post_step, on_event=on_event, on_end=on_end)

    def on_start(self, sim: "Simulation") -> None:
        if self._f["on_start"]:
            self._f["on_start"](sim)

    def pre_step(self, sim: "Simulation") -> None:
        if self._f["pre_step"]:
            self._f["pre_step"](sim)

    def post_step(self, sim: "Simulation") -> None:
        if self._f["post_step"]:
            self._f["post_step"](sim)

    def on_event(self, sim: "Simulation", event: Any) -> None:
        if self._f["on_event"]:
            self._f["on_event"](sim, event)

    def on_end(self, sim: "Simulation", archive: Optional["RunArchive"] = None) -> None:
        if self._f["on_end"]:
            self._f["on_end"](sim, archive)


class HookError(RuntimeError):
    pass


class HookManager:
    def __init__(self, hooks: Any = (), isolate: bool = False) -> None:
        self.hooks: List[Hook] = []
        self.isolate = isolate
        self.failures: List[tuple] = []
        for h in hooks:
            self.add(h)

    def add(self, hook: Hook) -> Any:
        self.hooks.append(hook)
        self.hooks.sort(key=lambda h: getattr(h, "priority", 50))
        return hook

    def remove(self, hook: Hook) -> None:
        if hook in self.hooks:
            self.hooks.remove(hook)

    def call(self, method: str, sim: "Simulation", *args: Any) -> None:
        for h in list(self.hooks):
            fn = getattr(h, method, None)
            if fn is None:
                continue
            try:
                fn(sim, *args)
            except Exception as ex:
                if self.isolate:
                    self.failures.append((h.name, method, repr(ex)))
                    self.hooks.remove(h)
                    warnings.warn(f"hook '{h.name}' failed in {method} and was disabled: {ex!r}", stacklevel=2)
                else:
                    raise HookError(f"hook '{h.name}' raised in {method} at step {getattr(sim, 'k', '?')}: {ex!r}") from ex


# ------------------------------------------------------------------------------ built-in hooks
class ProgressHook(Hook):
    """Print a one-line progress indicator (wall-clock throttled)."""

    name = "progress"
    priority = 90

    def __init__(self, every_s: float = 0.5, stream: Any = None) -> None:
        self.every_s, self.stream = every_s, stream or sys.stderr
        self._last = 0.0

    def on_start(self, sim: "Simulation") -> None:
        self._t0 = time.perf_counter()
        self._last = 0.0

    def post_step(self, sim: "Simulation") -> None:
        now = time.perf_counter()
        if now - self._last >= self.every_s:
            self._last = now
            frac = sim.t / max(sim.cfg.duration, 1e-9)
            self.stream.write(f"\r  t={sim.t:7.2f}/{sim.cfg.duration:.1f}s  [{'#' * int(20 * frac):<20}] {100 * frac:3.0f}%")
            self.stream.flush()

    def on_end(self, sim: "Simulation", archive: Optional["RunArchive"] = None) -> None:
        self.stream.write("\r" + " " * 60 + "\r")
        self.stream.flush()


class EventLogHook(Hook):
    """Print events as they happen."""

    name = "event_log"

    def __init__(self, pattern: str = "*", stream: Any = None) -> None:
        self.pattern, self.stream = pattern, stream or sys.stderr

    def on_event(self, sim: "Simulation", event: Any) -> None:
        import fnmatch

        if fnmatch.fnmatchcase(event.name, self.pattern):
            v = f" vehicle {event.vehicle}" if event.vehicle is not None else ""
            self.stream.write(f"[t={event.t:8.3f}s] {event.name}{v} {event.data or ''}\n")


class StopOnEventHook(Hook):
    """Stop the run cleanly when an event matching ``pattern`` occurs (e.g. ``"crash"``)."""

    name = "stop_on_event"

    def __init__(self, pattern: str = "crash") -> None:
        self.pattern = pattern

    def on_event(self, sim: "Simulation", event: Any) -> None:
        import fnmatch

        if fnmatch.fnmatchcase(event.name, self.pattern):
            sim.request_stop(f"stopped by event '{event.name}'")


class ProfilerHook(Hook):
    """Collect per-task wall time (enables the scheduler profiler) and print a table at the end."""

    name = "profiler"

    def __init__(self, print_report: bool = True, stream: Any = None) -> None:
        self.print_report, self.stream = print_report, stream or sys.stderr

    def on_start(self, sim: "Simulation") -> None:
        sim.scheduler.profile = True

    def on_end(self, sim: "Simulation", archive: Optional["RunArchive"] = None) -> None:
        if self.print_report:
            self.stream.write(sim.scheduler.report() + "\n")
