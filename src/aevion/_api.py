"""Public-API snapshotting (used by ``tests/test_public_api.py`` and ``tools/api_snapshot.py``).

The snapshot records, for every public module, each public name with its kind and call signature (and, for classes,
the public method names). Comparing the live API with the stored snapshot separates **breaking** changes (a name or
parameter removed/reordered, a required parameter added) from **additions**. Breaking changes need a deprecation
cycle (see the versioning policy in ``DOCUMENTATION.md``); additions require a conscious snapshot update so nothing becomes public by accident.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, List, Tuple

PUBLIC_MODULES = [
    "aevion",
    "aevion.errors",
    "aevion.deprecation",
    "aevion.config",
    "aevion.registry",
    "aevion.engine",
    "aevion.health",
    "aevion.frames",
    "aevion.datasets",
    "aevion.dynamics",
    "aevion.core.backend",
    "aevion.environments.world",
    "aevion.environments.wind",
    "aevion.environments.geodesy",
    "aevion.environments.geomagnetic",
    "aevion.environments.solar",
    "aevion.environments.weather",
    "aevion.environments.wind_fields",
    "aevion.environments.mesh",
    "aevion.environments.mesh_index",
    "aevion.environments.terrain",
    "aevion.environments.osm",
    "aevion.environments.sdf",
    "aevion.environments.scene",
    "aevion.environments.actors",
    "aevion.environments.geofence",
    "aevion.sensors.allan",
    "aevion.sensors.gnss",
    "aevion.sensors.airdata",
    "aevion.sensors.proximity",
    "aevion.sensors.radar",
    "aevion.sensors.scan_patterns",
    "aevion.sensors.camera_models",
    "aevion.sensors.event_camera",
    "aevion.vehicles.spec",
    "aevion.control",
    "aevion.sensors",
    "aevion.estimation",
    "aevion.estimation.mapping",
    "aevion.planning",
    "aevion.control.advanced",
    "aevion.vehicles.propeller",
    "aevion.vehicles.motor",
    "aevion.vehicles.battery_pack",
    "aevion.engine.effects",
    "aevion.aircraft",
    "aevion.identification",
    "aevion.bridges.mavlink",
    "aevion.bridges.telemetry",
    "aevion.bridges.ulog",
    "aevion.bridges.transport",
    "aevion.bridges.hil",
    "aevion.bridges.ros2",
    "aevion.wireless",
    "aevion.wireless.propagation",
    "aevion.wireless.fading",
    "aevion.wireless.antennas",
    "aevion.wireless.channel",
    "aevion.wireless.phy",
    "aevion.wireless.mac_network",
    "aevion.wireless.ris",
    "aevion.wireless.ntn",
    "aevion.wireless.maps",
    "aevion.wireless.security",
    "aevion.wireless.metrics",
    "aevion.wireless.interop",
    "aevion.multiagent",
    "aevion.multiagent.graphs",
    "aevion.multiagent.consensus",
    "aevion.multiagent.allocation",
    "aevion.multiagent.coverage",
    "aevion.multiagent.utm",
    "aevion.multiagent.laws",
    "aevion.ml",
    "aevion.ml.dataset",
    "aevion.ml.policies",
    "aevion.ml.imitation",
    "aevion.ml.curriculum",
    "aevion.ml.tracking",
    "aevion.ml.tuning",
    "aevion.ml.rollouts",
    "aevion.envs.swarm_env",
    "aevion.envs.tasks_ext",
    "aevion.envs.experts",
    "aevion.datasets.vision",
    "aevion.experiments.design",
    "aevion.experiments.sensitivity",
    "aevion.experiments.bayesopt",
    "aevion.experiments.cache",
    "aevion.experiments.runners",
    "aevion.experiments.database",
    "aevion.experiments.config_experiment",
    "aevion.experiments.suites",
    "aevion.analysis.diagnostics",
    "aevion.analysis.figures_diag",
    "aevion.analysis.templates",
    "aevion.analysis.appendix",
    "aevion.envs",
    "aevion.experiments",
    "aevion.scenario",
    "aevion.metrics",
    "aevion.analysis.style",
    "aevion.analysis.figures",
    "aevion.analysis.figures_stats",
    "aevion.analysis.stats",
    "aevion.analysis.report",
    "aevion.viewer",
    "aevion.safety",
]

_KINDS: Dict[Callable[[Any], bool], str] = {
    inspect.isclass: "class",
    inspect.isfunction: "function",
    inspect.ismodule: "module",
    inspect.isbuiltin: "function",
}


def public_names(mod: Any) -> List[str]:
    if hasattr(mod, "__all__"):
        return sorted(set(mod.__all__))
    out = []
    for n, o in vars(mod).items():
        if n.startswith("_"):
            continue
        if (inspect.isclass(o) or inspect.isfunction(o)) and getattr(o, "__module__", None) == mod.__name__:
            out.append(n)
    return sorted(out)


def _sig(obj: Any) -> List[Dict[str, Any]]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    params = []
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        params.append(
            {
                "name": p.name,
                "kind": p.kind.name,
                "required": p.default is inspect.Parameter.empty and p.kind.name not in ("VAR_POSITIONAL", "VAR_KEYWORD"),
            }
        )
    return params


def _describe(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {"kind": "optional-missing"}
    kind = next((k for test, k in _KINDS.items() if test(obj)), None)
    if kind is None:
        return {"kind": "constant", "type": type(obj).__name__}
    d: Dict[str, Any] = {"kind": kind}
    if kind in ("function", "class"):
        d["signature"] = _sig(obj)
    if kind == "class":
        d["methods"] = sorted(
            n for n, m in inspect.getmembers(obj, callable) if not n.startswith("_") and (inspect.isfunction(m) or inspect.ismethod(m))
        )
    return d


def snapshot() -> Dict[str, Dict[str, Any]]:
    snap: Dict[str, Dict[str, Any]] = {}
    for name in PUBLIC_MODULES:
        mod = importlib.import_module(name)
        snap[name] = {n: _describe(getattr(mod, n, None)) for n in public_names(mod) if hasattr(mod, n)}
    return snap


def _compatible(old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """New signature accepts every call the old one did."""
    new_by = {p["name"]: p for p in new}
    new_has_kwargs = any(p["kind"] == "VAR_KEYWORD" for p in new)
    new_has_varargs = any(p["kind"] == "VAR_POSITIONAL" for p in new)
    old_pos = [p["name"] for p in old if p["kind"] in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")]
    new_pos = [p["name"] for p in new if p["kind"] in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")]
    for p in old:
        if p["kind"] in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            if not any(q["kind"] == p["kind"] for q in new):
                return False, f"lost {'*args' if p['kind'] == 'VAR_POSITIONAL' else '**kwargs'}"
            continue
        if p["name"] not in new_by:
            if p["kind"] == "KEYWORD_ONLY" and new_has_kwargs:
                continue
            if p["kind"] != "KEYWORD_ONLY" and new_has_varargs and new_has_kwargs:
                continue
            return False, f"parameter '{p['name']}' removed"
    # relative order of retained positional parameters must be preserved
    kept = [n for n in old_pos if n in new_pos]
    if [n for n in new_pos if n in kept] != kept:
        return False, "positional parameters reordered"
    for p in new:
        if p["name"] not in {o["name"] for o in old} and p["required"]:
            return False, f"new required parameter '{p['name']}'"
    return True, ""


def compare(old: Dict[str, Any], new: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return ``(breaking, additions)`` descriptions."""
    breaking: List[str] = []
    additions: List[str] = []
    for mod, names in old.items():
        if mod not in new:
            breaking.append(f"module {mod} removed")
            continue
        for n, d in names.items():
            if d.get("kind") == "optional-missing":
                continue
            cur = new[mod].get(n)
            if cur is None or cur.get("kind") == "optional-missing":
                breaking.append(f"{mod}.{n} removed")
                continue
            if cur["kind"] != d["kind"]:
                breaking.append(f"{mod}.{n} changed kind {d['kind']} -> {cur['kind']}")
                continue
            if "signature" in d:
                ok, why = _compatible(d["signature"], cur.get("signature", []))
                if not ok:
                    breaking.append(f"{mod}.{n}: incompatible signature ({why})")
            for m in set(d.get("methods", [])) - set(cur.get("methods", [])):
                breaking.append(f"{mod}.{n}.{m}() removed")
            for m in set(cur.get("methods", [])) - set(d.get("methods", [])):
                additions.append(f"{mod}.{n}.{m}()")
        for n in new[mod]:
            if n not in names:
                additions.append(f"{mod}.{n}")
    for mod in new:
        if mod not in old:
            additions.append(f"module {mod}")
    return sorted(breaking), sorted(additions)
