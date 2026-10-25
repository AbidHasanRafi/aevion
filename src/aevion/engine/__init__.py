"""Simulation engine: scheduler, events, hooks, faults, RNG streams and the :class:`Simulation` itself.

``Simulation`` is imported lazily (it pulls in the config and registry layers) so that lightweight pieces such as
:mod:`aevion.engine.faults` can be imported without a circular dependency.
"""

from typing import Any

from .events import Event, EventBus
from .faults import (
    ActuatorStuck,
    BatteryDrop,
    CommBlackout,
    ComShift,
    Fault,
    FaultError,
    FaultTimeline,
    GpsOutage,
    PayloadChange,
    RotorFailure,
    SensorBiasStep,
    SensorDelay,
    SensorDropout,
    SensorNoiseScale,
    SensorStuck,
    ThrustLoss,
    WindDisturbance,
    WindGust,
)
from .hooks import CallbackHook, EventLogHook, Hook, HookManager, ProfilerHook, ProgressHook, StopOnEventHook
from .rng import RngStreams
from .scheduler import Scheduler, Task

_LAZY = {"Simulation": "simulation", "StepContext": "simulation", "build_parts": "builder", "Parts": "builder", "legacy_scenario_view": "simulation"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module 'aevion.engine' has no attribute {name!r}")


__all__ = [
    "Event",
    "EventBus",
    "Fault",
    "FaultError",
    "FaultTimeline",
    "RotorFailure",
    "ThrustLoss",
    "PayloadChange",
    "ComShift",
    "ActuatorStuck",
    "BatteryDrop",
    "SensorDropout",
    "GpsOutage",
    "SensorStuck",
    "SensorBiasStep",
    "SensorNoiseScale",
    "SensorDelay",
    "WindDisturbance",
    "WindGust",
    "CommBlackout",
    "Hook",
    "HookManager",
    "CallbackHook",
    "ProgressHook",
    "EventLogHook",
    "StopOnEventHook",
    "ProfilerHook",
    "RngStreams",
    "Scheduler",
    "Task",
    "Simulation",
    "StepContext",
    "build_parts",
    "Parts",
    "legacy_scenario_view",
]
