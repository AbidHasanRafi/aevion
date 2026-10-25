"""Schema migrations: convert older configuration dicts to the current schema.

Each migration is a function ``dict -> dict`` registered for the *source* version. legacy scenarios have no
``schema_version`` key and are treated as version 1.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict

from .blocks import SCHEMA_VERSION

MIGRATIONS: Dict[int, Callable[[dict], dict]] = {}


def migration(from_version: int) -> Any:
    def deco(fn: Any) -> Any:
        MIGRATIONS[from_version] = fn
        return fn

    return deco


# every key a legacy Scenario dict can contain
V1_KEYS = {
    "name",
    "vehicle",
    "world",
    "wind",
    "controller",
    "controller_kwargs",
    "trajectory",
    "duration",
    "dt",
    "integrator",
    "n",
    "formation",
    "seed",
    "sensors",
    "use_estimator",
    "estimator_handover_s",
    "record_every",
    "record_sensors",
    "initial",
    "perturbation",
    "wireless",
    "swarm",
    "notes",
}


def detect_version(d: dict) -> int:
    """Explicit ``schema_version`` wins. Without one, a mapping that only uses legacy keys is a legacy scenario;
    anything else is assumed to be current-schema (so typos are *reported*, never silently dropped)."""
    v = d.get("schema_version")
    if v is not None:
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError(f"schema_version must be an integer, got {type(v).__name__}")
        try:
            return int(v)
        except ValueError:
            raise ValueError(f"schema_version must be an integer, got {v!r}") from None
    return 1 if set(d) <= V1_KEYS and not _has_v2_shape(d) else SCHEMA_VERSION


# legacy and v2 share most key names; these are the places where the same key holds a different *shape* in v2
_V2_VEHICLE_KEYS = {"model", "count", "formation", "overrides", "initial"}


def _has_v2_shape(d: dict) -> bool:
    """Evidence that an unversioned document is written for the current schema rather than as a legacy scenario."""
    if isinstance(d.get("controller"), dict):
        return True
    veh = d.get("vehicle")
    if isinstance(veh, dict) and set(veh) & _V2_VEHICLE_KEYS:
        return True
    traj = d.get("trajectory")  # legacy trajectories are {"kind": ...}; world/wind dicts reuse "type", so only "params" is conclusive
    if isinstance(traj, dict) and ("type" in traj or "params" in traj):
        return True
    for key in ("world", "wind"):
        sec = d.get(key)
        if isinstance(sec, dict) and "params" in sec:
            return True
    sensors = d.get("sensors")
    return isinstance(sensors, list) and any(isinstance(s, dict) for s in sensors)


@migration(1)
def _v1_to_v2(d: dict) -> dict:
    """Legacy ``Scenario.to_dict()`` to a :class:`SimulationConfig` dict."""
    d = copy.deepcopy(d)
    out: dict = {
        "schema_version": 2,
        "name": d.get("name", "scenario"),
        "description": d.get("notes", ""),
        "seed": d.get("seed", 0),
        "duration": d.get("duration", 10.0),
    }
    out["engine"] = {"dt": d.get("dt", 0.005), "integrator": d.get("integrator", "rk4")}
    w = d.get("world")
    out["world"] = "empty" if w is None else w
    out["wind"] = d["wind"] if d.get("wind") else "none"
    veh = {"model": d.get("vehicle", "generic_1kg"), "count": d.get("n", 1)}
    if d.get("formation"):
        veh["formation"] = d["formation"]
    if d.get("initial"):
        veh["initial"] = {k: v for k, v in d["initial"].items() if k in ("pos", "vel", "euler", "omega", "soc")}
    out["vehicle"] = veh
    ctrl = {"type": d.get("controller", "geometric")}
    ctrl.update(d.get("controller_kwargs") or {})
    if isinstance(d.get("controller"), dict):
        ctrl = dict(d["controller"])
    out["controller"] = ctrl
    out["trajectory"] = d.get("trajectory") or {"kind": "hover", "pos": [0, 0, 2.0]}
    rec = set(d.get("record_sensors") or [])
    out["sensors"] = [({"type": s, "record": True} if s in rec else s) for s in (d.get("sensors") or [])]
    if d.get("use_estimator"):
        out["estimator"] = {"type": "mahony_kf", "handover_s": d.get("estimator_handover_s", 6.0)}
    if d.get("wireless"):
        out["network"] = {"type": "cellular", **d["wireless"]}
    if d.get("swarm"):
        out["swarm"] = d["swarm"]
    p = d.get("perturbation")
    if p:
        if p.get("kind") != "rotor_failure":
            raise ValueError(f"cannot migrate unknown legacy perturbation kind {p.get('kind')!r}")
        out["faults"] = [
            {
                "type": "rotor_failure",
                "t_start": p["t"],
                "vehicle": p.get("vehicle", 0),
                "rotor": p.get("rotor", 0),
                "severity": p.get("severity", 1.0),
            }
        ]
    out["logging"] = {"record_every": d.get("record_every", 4), "sensors": list(d.get("record_sensors") or [])}
    return out


def migrate(d: dict) -> dict:
    """Apply migrations until ``d`` is at :data:`SCHEMA_VERSION`."""
    d = dict(d)
    v = detect_version(d)
    if v > SCHEMA_VERSION:
        raise ValueError(f"configuration schema version {v} is newer than this Aevion supports ({SCHEMA_VERSION}); upgrade Aevion")
    while v < SCHEMA_VERSION:
        if v not in MIGRATIONS:
            raise ValueError(f"no migration from schema version {v}")
        d = MIGRATIONS[v](d)
        v = detect_version(d)
    return d
