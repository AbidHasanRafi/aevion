"""Reduced-order aircraft: fixed-wing and vertical-take-off-and-landing models, trim, linearisation and autopilots."""

from .autopilot import Autopilot, PathFollower
from .fixedwing import AEROSONDE, AircraftParams, AircraftState, FixedWing
from .quadplane import QuadPlane, QuadPlaneController, QuadPlaneParams, TiltRotorController, TiltRotorPlane
from .trim import TrimResult, linearize, modes, state_from_trim, trim

__all__ = [
    "AEROSONDE",
    "AircraftParams",
    "AircraftState",
    "FixedWing",
    "Autopilot",
    "PathFollower",
    "QuadPlane",
    "QuadPlaneController",
    "QuadPlaneParams",
    "TiltRotorPlane",
    "TiltRotorController",
    "TrimResult",
    "trim",
    "linearize",
    "modes",
    "state_from_trim",
]
