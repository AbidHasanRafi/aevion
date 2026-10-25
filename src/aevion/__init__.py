"""Aevion: research-grade UAV simulation with CPU-first, GPU-ready execution."""

from __future__ import annotations

__version__ = "0.1.0"

from .core.backend import cuda_available, get_backend, torch_available
from .core.constants import Atmosphere, GravityModel
from .dynamics.multirotor import Multirotor, State
from .environments.world import World
from .vehicles.spec import BatteryModel, VehicleSpec, get_vehicle

__all__ = [
    "__version__",
    "get_backend",
    "torch_available",
    "cuda_available",
    "Atmosphere",
    "GravityModel",
    "Multirotor",
    "State",
    "World",
    "BatteryModel",
    "VehicleSpec",
    "get_vehicle",
]
