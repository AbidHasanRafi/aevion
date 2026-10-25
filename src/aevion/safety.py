"""Resource limits and checks for running untrusted configurations (shared services, web front ends, grading).

A configuration is data that decides how much CPU, memory and disk a run uses and which code it can reach. :class:`ResourceLimits`
bounds this before anything is built:

* file size, nesting depth, node count and YAML anchor/alias count (parser bombs);
* simulated duration, steps, vehicles and vehicle-steps (CPU);
* world objects, sensor ray and pixel throughput, and estimated archive size (memory, disk);
* an optional wall-clock limit enforced by a hook while running;
* a confined output directory;
* built-in components only (no plugin code, no custom hooks).

``ResourceLimits.untrusted()`` is a conservative preset. The limits bound resource use; they are not a sandbox for arbitrary Python.
Never load untrusted plugins or pickles: plugins remain trusted code.
"""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ._types import PathLike
from .engine.hooks import Hook
from .errors import AevionError, ConfigError, ConfigIssue

if TYPE_CHECKING:
    from .config import SimulationConfig
    from .datasets.archive import RunArchive


__all__ = ["ResourceLimitError", "ResourceLimits", "WallClockLimit", "check_config", "estimate_cost", "load_config_checked", "run_limited"]


class ResourceLimitError(AevionError):
    """A configuration or run exceeds the configured resource limits. ``violations`` lists every problem."""

    def __init__(self, violations: List[str]):
        self.violations = list(violations)
        super().__init__("resource limits exceeded:\n" + "\n".join(f"  - {v}" for v in self.violations))


@dataclass(frozen=True)
class ResourceLimits:
    max_config_bytes: int = 16 * 1024 * 1024
    max_config_depth: int = 64
    max_config_nodes: int = 1_000_000
    max_yaml_aliases: int = 100
    max_duration_s: float = 24 * 3600.0
    max_steps: float = 5e8
    max_vehicles: int = 100_000
    max_vehicle_steps: float = 1e11
    max_rotors: int = 64
    max_world_objects: int = 1_000_000
    max_rays_per_second: float = 1e9
    max_pixels_per_frame: int = 4_000_000
    max_archive_bytes: float = 64 * 1024**3
    max_wall_s: Optional[float] = None
    output_root: Optional[Path] = None
    builtin_components_only: bool = False
    allow_input_files: bool = True  # meshes, terrain, map data, gridded wind: any component that reads a file named in the configuration

    @classmethod
    def untrusted(cls, output_root: Optional[PathLike] = None) -> "ResourceLimits":
        """Conservative preset for configurations from unknown sources."""
        return cls(
            max_config_bytes=256 * 1024,
            max_config_depth=24,
            max_config_nodes=20_000,
            max_yaml_aliases=0,
            max_duration_s=600.0,
            max_steps=2e5,
            max_vehicles=64,
            max_vehicle_steps=5e6,
            max_rotors=12,
            max_world_objects=5_000,
            max_rays_per_second=2e6,
            max_pixels_per_frame=320 * 240,
            max_archive_bytes=256 * 1024**2,
            max_wall_s=300.0,
            output_root=Path(output_root) if output_root else None,
            builtin_components_only=True,
            allow_input_files=False,
        )

    def replace(self, **changes: Any) -> "ResourceLimits":
        return replace(self, **changes)


# ------------------------------------------------------------------------------------ parsing guards
def measure_tree(data: Any, max_nodes: int, max_depth: int) -> None:
    """Raise :class:`ResourceLimitError` if ``data`` (nested dicts/lists) is too big or deep.

    Shared references (YAML aliases) are walked every time they are met, so an alias expansion attack is
    counted at its *expanded* size and stopped early.
    """
    count = 0
    stack: List[Any] = [(data, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > max_nodes:
            raise ResourceLimitError([f"configuration has more than {max_nodes:,} nodes (after alias expansion)"])
        if depth > max_depth:
            raise ResourceLimitError([f"configuration is nested deeper than {max_depth} levels"])
        if isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend((v, depth + 1) for v in node)


def read_config_text(path: Path, limits: ResourceLimits) -> str:
    size = path.stat().st_size
    if size > limits.max_config_bytes:
        raise ResourceLimitError([f"configuration file is {size:,} bytes (limit {limits.max_config_bytes:,})"])
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        aliases = len(re.findall(r"(?m)(?:^|[\s\[\{,:])\*[A-Za-z0-9_-]+", text))
        if aliases > limits.max_yaml_aliases:
            raise ResourceLimitError(
                [f"YAML uses {aliases} alias references (limit {limits.max_yaml_aliases}); anchors/aliases can encode expansion bombs"]
            )
    return text


def load_config_checked(source: Union[PathLike, dict], limits: ResourceLimits) -> "SimulationConfig":
    """Load a configuration with every parser guard applied, then enforce the resource limits on the parsed result."""
    import json

    from .config import load_config

    if isinstance(source, dict):
        measure_tree(source, limits.max_config_nodes, limits.max_config_depth)
        cfg = load_config(source)
    else:
        path = Path(source)
        text = read_config_text(path, limits)
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(text)
            else:
                from .config.io import yaml_loads

                data = yaml_loads(text)
        except RecursionError as ex:
            raise ResourceLimitError(["configuration is nested too deeply to parse"]) from ex
        except Exception as ex:
            raise ConfigError([ConfigIssue("<file>", f"cannot parse: {ex}")], str(path)) from ex
        measure_tree(data, limits.max_config_nodes, limits.max_config_depth)
        if not isinstance(data, dict):
            raise ConfigError([ConfigIssue("<root>", "top level must be a mapping")], str(path))
        cfg = load_config(data)
    check_config(cfg, limits)
    return cfg


# ------------------------------------------------------------------------------------------ estimation
def _default(entry: Any, params: Dict[str, Any], name: str, fallback: float) -> float:
    if name in params:
        return float(params[name])
    d = entry.meta.get("defaults", {})
    if name in d:
        return float(d[name])
    try:
        p = inspect.signature(entry.factory).parameters.get(name)
        if p is not None and isinstance(p.default, (int, float)):
            return float(p.default)
    except (TypeError, ValueError):
        pass
    return fallback


def _world_objects(world: Any) -> int:
    p, t = world.params, world.type
    if t == "forest":
        return int(p.get("n_trees", 40))
    if t == "city":
        b = p.get("blocks", (4, 4))
        return int(b[0]) * int(b[1])
    if t == "corridor":
        return int(p.get("n_pillars", 6)) + 3
    if t == "hills":
        he, res = float(p.get("half_extent", 50.0)), float(p.get("resolution", 1.0))
        return int((2 * he / max(res, 1e-6) + 1) ** 2 // 64)  # a grid cell is far cheaper than an obstacle
    if t == "from_dict":
        d = p.get("data", {})
        hm = d.get("heightmap")
        cells = 0 if not hm else sum(len(r) for r in hm.get("heights", [])) // 64
        return len(d.get("boxes", [])) + len(d.get("spheres", [])) + len(d.get("cylinders", [])) + cells
    return 0


def estimate_cost(cfg: "SimulationConfig") -> Dict[str, float]:
    """Rough, conservative cost model of a configuration (computed without building anything)."""
    from .registry import registries

    steps = float(cfg.n_steps)
    n = float(cfg.vehicle.count)
    rows = steps / max(cfg.logging.record_every if not cfg.logging.rate_hz else (1.0 / (cfg.logging.rate_hz * cfg.engine.dt)), 1.0)
    rays = pixels = stream_bytes = 0.0
    for s in cfg.sensors:
        e = registries.sensors.entry(s.type)
        rate = _default(e, s.params, "rate", 10.0)
        if s.type == "lidar":
            r = _default(e, s.params, "channels", 16) * _default(e, s.params, "samples", 360)
            rays += r * rate * n
            frame = r * 4 * n
        elif s.type == "camera":
            px = _default(e, s.params, "width", 64) * _default(e, s.params, "height", 48)
            pixels = max(pixels, px)
            rays += px * rate * n * 2  # primary + shadow rays
            frame = px * 4 * n
        else:
            rays += rate * n if s.type == "rangefinder" else 0
            frame = 6 * 8 * n
        if s.record or s.key in cfg.logging.sensors:
            stream_bytes += frame * rate * cfg.duration
    archive = rows * n * 56 * 8 + stream_bytes
    return dict(
        steps=steps,
        vehicle_steps=steps * n,
        rays_per_second=rays,
        pixels_per_frame=pixels,
        archive_bytes=archive,
        world_objects=float(_world_objects(cfg.world)),
    )


def _components(cfg: "SimulationConfig") -> List[tuple]:
    out = [("controllers", cfg.controller), ("trajectories", cfg.trajectory), ("worlds", cfg.world), ("winds", cfg.wind)]
    out += [("sensors", s) for s in cfg.sensors]
    out += [("faults", f) for f in cfg.faults]
    out += [("hooks", h) for h in cfg.hooks]
    out += [("health_checks", c) for c in cfg.health.checks]
    if cfg.estimator is not None:
        out.append(("estimators", cfg.estimator))
    if cfg.network is not None:
        out.append(("networks", cfg.network))
    if cfg.vehicle.formation is not None:
        out.append(("formations", cfg.vehicle.formation))
    return out


_FILE_KEYS = {"file", "path", "files", "source"}


def _file_references(node: Any, where: str) -> List[Any]:
    """(location, key) of every ``file`` / ``path`` entry in a nested configuration fragment."""
    found: List[Any] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _FILE_KEYS and v:
                found.append((where, k))
            found += _file_references(v, f"{where}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            found += _file_references(v, f"{where}[{i}]")
    return found


def check_config(cfg: "SimulationConfig", limits: ResourceLimits) -> None:
    """Raise :class:`ResourceLimitError` listing every limit the configuration violates."""
    from .registry import registries

    v: List[str] = []
    L = limits
    if cfg.duration > L.max_duration_s:
        v.append(f"duration {cfg.duration:g} s exceeds the limit of {L.max_duration_s:g} s")
    if cfg.n_steps > L.max_steps:
        v.append(f"{cfg.n_steps:,} physics steps exceed the limit of {L.max_steps:,.0f} (increase engine.dt or shorten the run)")
    if cfg.vehicle.count > L.max_vehicles:
        v.append(f"{cfg.vehicle.count:,} vehicles exceed the limit of {L.max_vehicles:,}")
    if isinstance(cfg.vehicle.model, dict):
        rp = cfg.vehicle.model.get("rotor_positions", [])
        if len(rp) > L.max_rotors:
            v.append(f"vehicle has {len(rp)} rotors (limit {L.max_rotors})")
    est = estimate_cost(cfg)
    checks = [
        ("vehicle_steps", L.max_vehicle_steps, "vehicle-steps"),
        ("rays_per_second", L.max_rays_per_second, "ray casts per simulated second"),
        ("world_objects", L.max_world_objects, "world objects"),
        ("archive_bytes", L.max_archive_bytes, "bytes of archive data"),
        ("pixels_per_frame", L.max_pixels_per_frame, "camera pixels per frame"),
    ]
    for key, limit, label in checks:
        if est[key] > limit:
            v.append(f"estimated {est[key]:,.0f} {label} exceeds the limit of {limit:,.0f}")
    if L.output_root is not None and cfg.logging.path:
        root = Path(L.output_root).resolve()
        target = Path(cfg.logging.path).resolve()
        if target != root and root not in target.parents:
            v.append(f"logging.path {cfg.logging.path!r} is outside the permitted output directory")
    if not L.allow_input_files:
        for where, key in (
            _file_references(cfg.world.to_dict(), "world")
            + _file_references(cfg.wind.to_dict(), "wind")
            + _file_references([c.to_dict() for c in cfg.sensors], "sensors")
        ):
            v.append(f"{where}: '{key}' names a file; untrusted configurations may not read input files")
    if L.builtin_components_only:
        for kind, comp in _components(cfg):
            reg = getattr(registries, kind)
            if comp.type in reg and reg.entry(comp.type).source != "builtin":
                v.append(f"{kind[:-1]} '{comp.type}' comes from plugin '{reg.entry(comp.type).source}'; only built-in components are allowed")
        if cfg.hooks:
            v.append("custom hooks are not allowed in untrusted configurations")
    if v:
        raise ResourceLimitError(v)


# ------------------------------------------------------------------------------------------ run-time
class WallClockLimit(Hook):
    """Abort the run with :class:`ResourceLimitError` once its wall-clock time exceeds ``seconds``."""

    name = "wall_clock_limit"
    priority = 1

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self._t0 = 0.0

    def on_start(self, sim: Any) -> None:
        self._t0 = time.monotonic()

    def post_step(self, sim: Any) -> None:
        if sim.k % 50 == 0 and time.monotonic() - self._t0 > self.seconds:
            raise ResourceLimitError([f"wall-clock limit of {self.seconds:g} s exceeded at simulated t = {sim.t:.2f} s"])


def run_limited(source: Union[PathLike, dict, "SimulationConfig"], limits: Optional[ResourceLimits] = None, progress: bool = False) -> "RunArchive":
    """Validate against ``limits`` and run. Accepts a path, a dict or a built config."""
    from .config import SimulationConfig
    from .engine import Simulation

    limits = limits or ResourceLimits.untrusted()
    if isinstance(source, SimulationConfig):
        check_config(source, limits)
        cfg = source
    else:
        cfg = load_config_checked(source, limits)
    hooks = [WallClockLimit(limits.max_wall_s)] if limits.max_wall_s else []
    # the engine's own health monitor stays on; `hooks` errors surface as HookError wrapping ResourceLimitError
    try:
        return Simulation(cfg, hooks=hooks).run(progress=progress)
    except Exception as ex:
        cause = ex.__cause__
        if isinstance(cause, ResourceLimitError):
            raise cause from None
        raise
