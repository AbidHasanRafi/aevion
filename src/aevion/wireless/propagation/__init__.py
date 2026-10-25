"""Large-scale propagation models behind one interface (:class:`PathLossModel`).

>>> from aevion.wireless.propagation import make_model
>>> m = make_model("tr38901", freq_hz=3.5e9, scenario="uma")
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .al_hourani import AlHourani
from .base import LinkGeometry, PathLossModel, breakpoint_distance, fspl_db, realise
from .free_space import FreeSpace, LogDistance, TwoRayGround
from .knife_edge import bullington_loss_db, fresnel_parameter, knife_edge_loss_db, path_profile, world_diffraction_loss_db
from .ray_tracing import MATERIALS, PathSet, complex_permittivity, fresnel_reflection, trace_paths
from .tr38901 import SCENARIOS, TR36777, TR38901

MODELS: Dict[str, Callable[..., PathLossModel]] = {
    "free_space": FreeSpace,
    "two_ray": TwoRayGround,
    "log_distance": LogDistance,
    "al_hourani": AlHourani,
    "tr38901": TR38901,
    "tr36777": TR36777,
}


def make_model(name: str, **kw: Any) -> PathLossModel:
    """Instantiate a propagation model by name (``free_space``, ``two_ray``, ``log_distance``, ``al_hourani``, ``tr38901``, ``tr36777``)."""
    key = name.strip().lower().replace("-", "_")
    if key not in MODELS:
        raise KeyError(f"unknown propagation model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[key](**kw)


__all__ = [
    "MATERIALS",
    "MODELS",
    "SCENARIOS",
    "AlHourani",
    "FreeSpace",
    "LinkGeometry",
    "LogDistance",
    "PathSet",
    "PathLossModel",
    "TR36777",
    "TR38901",
    "TwoRayGround",
    "breakpoint_distance",
    "complex_permittivity",
    "bullington_loss_db",
    "fresnel_parameter",
    "fresnel_reflection",
    "fspl_db",
    "knife_edge_loss_db",
    "make_model",
    "path_profile",
    "realise",
    "trace_paths",
    "world_diffraction_loss_db",
]
