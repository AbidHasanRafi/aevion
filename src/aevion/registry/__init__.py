"""Component registries and plugin discovery.

>>> from aevion.registry import registries
>>> registries.controllers.names()
['cascaded', 'geometric', 'lqr']
"""

from __future__ import annotations

from typing import Any, Dict

from ..errors import RegistryError
from .core import Entry, Registry
from .plugins import PluginRecord, discover, load_plugin_function, records


class Registries:
    """The set of all component registries."""

    KINDS = (
        "vehicles",
        "controllers",
        "trajectories",
        "sensors",
        "estimators",
        "winds",
        "worlds",
        "actors",
        "scan_patterns",
        "channels",
        "propagation",
        "antennas",
        "schedulers",
        "networks",
        "tasks",
        "planners",
        "formations",
        "swarm_laws",
        "faults",
        "hooks",
        "health_checks",
        "metrics",
    )

    vehicles: Registry
    controllers: Registry
    trajectories: Registry
    sensors: Registry
    estimators: Registry
    winds: Registry
    worlds: Registry
    actors: Registry
    scan_patterns: Registry
    channels: Registry
    propagation: Registry
    antennas: Registry
    schedulers: Registry
    networks: Registry
    tasks: Registry
    planners: Registry
    formations: Registry
    swarm_laws: Registry
    faults: Registry
    hooks: Registry
    health_checks: Registry
    metrics: Registry

    def __init__(self) -> None:
        for k in self.KINDS:
            setattr(self, k, Registry(k))

    def all(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.KINDS}

    def describe(self) -> str:
        return "\n\n".join(r.describe() for r in self.all().values() if len(r))

    def summary(self) -> dict:
        return {k: r.names() for k, r in self.all().items()}


registries = Registries()


def load_builtins() -> None:
    from . import builtin  # noqa: F401  (registers on import)


load_builtins()
discover(registries)

__all__ = ["Entry", "Registry", "Registries", "registries", "PluginRecord", "discover", "records", "load_plugin_function", "RegistryError"]
