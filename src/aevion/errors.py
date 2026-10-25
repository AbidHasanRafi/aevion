"""Exception hierarchy. Every error raised deliberately by Aevion derives from :class:`AevionError`."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


class AevionError(Exception):
    """Base class for all Aevion errors."""


class RegistryError(AevionError, KeyError):
    """Unknown or duplicate registry entry."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return Exception.__str__(self)


class PluginError(AevionError):
    """A plugin failed to load or register."""


class ConfigIssue:
    """One validation problem: where it is, what is wrong, and (optionally) how to fix it."""

    __slots__ = ("hint", "message", "path")

    def __init__(self, path: str, message: str, hint: Optional[str] = None) -> None:
        self.path, self.message, self.hint = path or "<root>", message, hint

    def __str__(self) -> str:
        s = f"{self.path}: {self.message}"
        return s + (f"  (hint: {self.hint})" if self.hint else "")

    def __repr__(self) -> str:
        return f"ConfigIssue({str(self)!r})"


class ConfigError(AevionError, ValueError):
    """Configuration is invalid. ``issues`` lists *every* problem found, not just the first."""

    def __init__(self, issues: Sequence[ConfigIssue], source: Optional[str] = None) -> None:
        self.issues: List[ConfigIssue] = list(issues)
        self.source = source
        head = f"invalid configuration{f' in {source}' if source else ''}: {len(self.issues)} problem(s)"
        super().__init__(head + "\n" + "\n".join(f"  - {i}" for i in self.issues))


class SimulationDiverged(AevionError, FloatingPointError):
    """A health check failed (non-finite state, runaway speed, …). Carries the last good state for debugging."""

    def __init__(
        self, message: str, step: int, t: float, vehicle: Optional[int] = None, quantity: Optional[str] = None, last_good: Any = None
    ) -> None:
        self.step, self.t, self.vehicle, self.quantity, self.last_good = step, t, vehicle, quantity, last_good
        where = f"step {step} (t = {t:.4f} s)" + (f", vehicle {vehicle}" if vehicle is not None else "")
        super().__init__(f"{message} at {where}" + (f" [{quantity}]" if quantity else ""))


class ArchiveError(AevionError):
    """Corrupt, incompatible or incomplete run archive."""


class SchedulerError(AevionError):
    """Invalid task or rate configuration in the simulation engine."""
