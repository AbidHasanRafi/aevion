"""Configuration sections for a simulation (schema version 2).

Components that come from a registry (controllers, sensors, winds, …) are written either as a bare name
(``"geometric"``) or as a mapping with a ``type`` (alias ``kind``) key plus parameters::

    controller: {type: geometric, ki_pos: 3.0}
    sensors:
      - imu
      - {type: lidar, name: front_lidar, channels: 16, rate: 10}
"""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from ..errors import ConfigIssue
from ..registry.introspect import accepted_params
from .schema import Block, _check_meta, _hints, _join, _parse, _suggest, spec

SCHEMA_VERSION = 2

# context positional arguments each registry's factories receive before the user parameters
_CONTEXT_ARGS = {"controllers": 1, "estimators": 1, "formations": 1, "swarm_laws": 1}


def _singular(kind: str) -> str:
    return kind[:-3] + "y" if kind.endswith("ies") else kind[:-1] if kind.endswith("s") else kind


@dataclass
class ComponentConfig(Block):
    """A named registry component plus its parameters."""

    type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    registry_kind: ClassVar[str] = ""
    default_type: ClassVar[Optional[str]] = None

    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        return {"type": value} if isinstance(value, str) else None

    @classmethod
    def _from_value(cls, value: Any, path: str, issues: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = {"type": value}
        if not isinstance(value, dict):
            issues.append(ConfigIssue(path, f"expected a name or a mapping with a 'type' key, got {type(value).__name__}"))
            return None
        d = dict(value)
        own = {f.name for f in fields(cls)} - {"type", "params"}
        raw_params = d.pop("params", None)
        if raw_params is not None and not isinstance(raw_params, dict):
            issues.append(ConfigIssue(_join(path, "params"), f"expected a mapping of parameters, got {type(raw_params).__name__}"))
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        t = d.pop("type", None)
        k = d.pop("kind", None)
        t = t if t is not None else k
        if t is None:
            t = cls.default_type
        if t is None:
            issues.append(
                ConfigIssue(_join(path, "type"), "required field is missing", f"one of the registered {cls.registry_kind or 'component'} names")
            )
            t = ""
        elif not isinstance(t, str):
            issues.append(ConfigIssue(_join(path, "type"), f"expected a string, got {t!r}"))
            t = str(t)
        hints, flds = _hints(cls), {f.name: f for f in fields(cls)}
        kwargs = {}
        for name in own:
            if name in d:
                v = _parse(hints[name], d.pop(name), _join(path, name), issues)
                _check_meta(v, flds[name].metadata, _join(path, name), issues)
                kwargs[name] = v
        params.update(d)
        obj = cls(type=t, params=params, **kwargs)
        obj._validate(path, issues)
        return obj

    def _validate(self, path: str, issues: Any) -> None:
        kind = self.registry_kind
        if not kind or not self.type:
            return
        from ..registry import registries

        reg = getattr(registries, kind)
        if self.type not in reg:
            issues.append(
                ConfigIssue(
                    _join(path, "type"),
                    f"unknown {_singular(kind)} '{self.type}'",
                    _suggest(self.type, reg.names()) or f"available: {', '.join(reg.names())}",
                )
            )
            return
        e = reg.entry(self.type)
        acc = accepted_params(e.factory, _CONTEXT_ARGS.get(kind, 0))
        if acc is not None:
            for k in self.params:
                if k not in acc:
                    issues.append(
                        ConfigIssue(
                            _join(path, k),
                            f"unknown parameter for {_singular(kind)} '{self.type}'",
                            _suggest(k, acc) or f"accepted: {', '.join(sorted(acc))}",
                        )
                    )

    def to_dict(self, skip_defaults: bool = False) -> dict:
        out = {"type": self.type}
        for f in fields(self):
            if f.name in ("type", "params"):
                continue
            v = getattr(self, f.name)
            if skip_defaults and f.default is not MISSING and v == f.default:
                continue
            out[f.name] = v
        out.update(self.params)
        return out


# ------------------------------------------------------------------------------ component kinds
@dataclass
class ControllerConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "controllers"
    default_type: ClassVar[Optional[str]] = "geometric"
    type: str = "geometric"


@dataclass
class TrajectoryConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "trajectories"
    default_type: ClassVar[Optional[str]] = "hover"
    type: str = "hover"


@dataclass
class WindConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "winds"
    default_type: ClassVar[Optional[str]] = "none"
    type: str = "none"


@dataclass
class WorldConfig(ComponentConfig):
    """A procedural world (``forest``, ``city``, …) or a full serialised world (``World.to_dict()``)."""

    registry_kind: ClassVar[str] = "worlds"
    default_type: ClassVar[Optional[str]] = "empty"
    type: str = "empty"

    @classmethod
    def _from_value(cls, value: Any, path: str, issues: Any) -> Any:
        if isinstance(value, dict) and "boxes" in value and "type" not in value and "kind" not in value:
            value = {"type": "from_dict", "data": value}
        return super()._from_value(value, path, issues)


@dataclass
class FormationConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "formations"
    type: str = "line"


@dataclass
class SensorConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "sensors"
    name: Optional[str] = None  # unique instance name (defaults to the type)
    record: bool = False  # also store this sensor's readings in the archive

    @property
    def key(self) -> str:
        return self.name or self.type


@dataclass
class EstimatorConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "estimators"
    default_type: ClassVar[Optional[str]] = "mahony_kf"
    type: str = "mahony_kf"
    handover_s: float = spec(6.0, ge=0.0, doc="fly the controller on truth until the filter has converged", unit="s")


@dataclass
class NetworkConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "networks"
    default_type: ClassVar[Optional[str]] = "cellular"
    type: str = "cellular"
    rate_hz: Optional[float] = spec(None, gt=0, doc="evaluation rate; default = recording rate", unit="Hz")


@dataclass
class HookConfig(ComponentConfig):
    registry_kind: ClassVar[str] = "hooks"


@dataclass
class FaultConfig(ComponentConfig):
    """A fault active during ``[t_start, t_end)`` (``t_end`` omitted = permanent)."""

    registry_kind: ClassVar[str] = "faults"
    t_start: float = spec(0.0, ge=0.0, unit="s")
    t_end: Optional[float] = spec(None, ge=0.0, unit="s")
    vehicle: Optional[int] = spec(None, ge=0, doc="vehicle index (omit = all)")

    def _validate(self, path: str, issues: Any) -> None:
        super()._validate(path, issues)
        if self.t_end is not None and self.t_end <= self.t_start:
            issues.append(ConfigIssue(_join(path, "t_end"), f"must be greater than t_start ({self.t_start})"))


# --------------------------------------------------------------------------------- plain blocks
@dataclass
class InitialConfig(Block):
    pos: Optional[Tuple[float, float, float]] = None
    vel: Optional[Tuple[float, float, float]] = None
    euler: Optional[Tuple[float, float, float]] = spec(None, doc="roll, pitch, yaw", unit="rad")
    omega: Optional[Tuple[float, float, float]] = None
    soc: float = spec(1.0, ge=0.0, le=1.0, doc="initial battery state of charge")


@dataclass
class VehicleConfig(Block):
    model: Union[str, Dict[str, Any]] = "generic_1kg"  # preset name or a VehicleSpec dict
    count: int = spec(1, ge=1)
    formation: Optional[FormationConfig] = None
    overrides: Dict[str, Any] = field(default_factory=dict)  # VehicleSpec field overrides
    initial: InitialConfig = field(default_factory=InitialConfig)

    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        return {"model": value} if isinstance(value, str) else None

    def _validate(self, path: str, issues: Any) -> None:
        if isinstance(self.model, str):
            from ..registry import registries

            if self.model not in registries.vehicles:
                issues.append(
                    ConfigIssue(_join(path, "model"), f"unknown vehicle preset '{self.model}'", _suggest(self.model, registries.vehicles.names()))
                )
        if self.overrides:
            from ..vehicles.spec import VehicleSpec

            valid = set(VehicleSpec.__dataclass_fields__)
            for k in self.overrides:
                if k not in valid:
                    issues.append(ConfigIssue(_join(_join(path, "overrides"), k), "not a VehicleSpec field", _suggest(k, valid)))


@dataclass
class CommModelConfig(Block):
    delay_s: float = spec(0.0, ge=0.0, unit="s")
    loss: float = spec(0.0, ge=0.0, le=1.0, doc="packet-loss probability per link")
    range_m: float = spec(math.inf, gt=0.0, unit="m")
    rate_hz: float = spec(20.0, gt=0.0, unit="Hz")


@dataclass
class SwarmLawConfig(ComponentConfig):
    """A cooperative law (registry: swarm_laws) acting on the references through the lossy inter-agent link."""

    registry_kind: ClassVar[str] = "swarm_laws"


@dataclass
class SwarmConfig(Block):
    avoid_radius: float = spec(1.0, gt=0.0, unit="m")
    gain: float = spec(0.8, ge=0.0)
    max_push: float = spec(1.5, gt=0.0, unit="m")
    comm: CommModelConfig = field(default_factory=CommModelConfig)
    rate_hz: Optional[float] = spec(None, gt=0.0, doc="coordination rate; default = every physics step", unit="Hz")
    laws: List[SwarmLawConfig] = field(default_factory=list)


@dataclass
class HealthConfig(Block):
    enabled: bool = True
    on_diverge: Literal["raise", "stop", "warn"] = "raise"
    check_every: int = spec(1, ge=1, doc="run checks every N physics steps")
    max_speed: float = spec(200.0, gt=0.0, unit="m/s")
    max_rate: float = spec(500.0, gt=0.0, doc="maximum body rate", unit="rad/s")
    max_altitude: float = spec(20000.0, gt=0.0, unit="m")
    quat_tol: float = spec(1e-6, gt=0.0)
    checks: List[ComponentConfig] = field(default_factory=list)  # extra registered checks (registry: health_checks)


@dataclass
class OriginConfig(Block):
    """Where the world's (0, 0, 0) is on Earth: anchors the ENU frame for GNSS, the magnetic field, the sun and map data."""

    lat: float = spec(47.3769, ge=-90.0, le=90.0, unit="deg")
    lon: float = spec(8.5417, ge=-180.0, le=180.0, unit="deg")
    alt: float = spec(0.0, unit="m", doc="ellipsoidal height of the world's z = 0")


@dataclass
class ActorConfig(ComponentConfig):
    """A moving obstacle (registry: actors): ``linear``, ``circular``, ``waypoints``, ``bouncing`` or ``random_walk``."""

    registry_kind: ClassVar[str] = "actors"
    default_type: ClassVar[Optional[str]] = "linear"
    type: str = "linear"


@dataclass
class GeofenceConfig(Block):
    """A geofence: ``zone`` is ``{type: cylinder|box|sphere|polygon|altitude, ...}`` (see ``aevion.environments.geofence``)."""

    name: str = ""
    kind: Literal["inclusion", "exclusion"] = "exclusion"
    margin: float = spec(0.0, ge=0.0, unit="m")
    zone: Dict[str, Any] = field(default_factory=dict)

    def _validate(self, path: str, issues: Any) -> None:
        from ..environments.geofence import ZONES

        t = self.zone.get("type") if isinstance(self.zone, dict) else None
        if t not in ZONES:
            issues.append(ConfigIssue(_join(path, "zone.type"), f"unknown zone type {t!r}", f"choose from {sorted(ZONES)}"))


@dataclass
class EnvironmentConfig(Block):
    """Earth, weather, moving obstacles and geofences, applied on top of whatever ``world`` builds."""

    origin: Optional[OriginConfig] = None
    utc: Optional[str] = spec(None, doc="ISO-8601 UTC time of t = 0 (sun position, GNSS satellite geometry)")
    weather: Optional[Dict[str, Any]] = spec(None, doc="{preset: clear|overcast|rain|heavy_rain|fog|hot|cold|storm} and/or Weather fields")  # noqa: RUF009
    actors: List[ActorConfig] = field(default_factory=list)
    geofences: List[GeofenceConfig] = field(default_factory=list)
    geofence_stop: bool = spec(False, doc="end the run at the first geofence breach (needs health.on_diverge: stop)")
    gravity: Optional[Literal["constant", "altitude", "wgs84"]] = spec(
        None, doc="gravity model; wgs84 uses the latitude and height of the origin (Somigliana + free-air gradient)"
    )

    def _validate(self, path: str, issues: Any) -> None:
        if self.utc is not None:
            try:
                from ..environments.solar import julian_day

                julian_day(self.utc)
            except Exception:
                issues.append(ConfigIssue(_join(path, "utc"), f"cannot read {self.utc!r} as an ISO-8601 time", "e.g. 2026-06-21T10:00:00Z"))
        if self.weather is not None:
            try:
                from ..environments.weather import Weather

                Weather.from_dict(self.weather)
            except (ValueError, KeyError, TypeError) as e:
                issues.append(ConfigIssue(_join(path, "weather"), str(e)))
        names = [g.name for g in self.geofences if g.name]
        for n in set(names):
            if names.count(n) > 1:
                issues.append(ConfigIssue(_join(path, "geofences"), f"duplicate geofence name '{n}'"))


@dataclass
class LoggingConfig(Block):
    record_every: int = spec(4, ge=1, doc="record every N physics steps")
    rate_hz: Optional[float] = spec(None, gt=0.0, doc="record rate; overrides record_every", unit="Hz")
    mode: Literal["memory", "stream"] = "memory"
    path: Optional[str] = None  # archive directory for mode='stream'
    chunk_steps: int = spec(512, ge=16)
    sensors: List[str] = field(default_factory=list)  # names of sensors whose readings are stored
    channels: Optional[List[str]] = None  # keep only these channels (None = all)
    vehicles: Optional[List[int]] = None  # record only these vehicles
    provenance: bool = True


@dataclass
class EngineConfig(Block):
    dt: float = spec(0.005, gt=0.0, le=0.05, doc="physics step", unit="s")
    integrator: Literal["euler", "semi_implicit", "midpoint", "rk4"] = "rk4"
    backend: str = "numpy"
    dtype: Optional[Literal["float32", "float64"]] = None
    kernel: Literal["auto", "reference", "numba"] = "auto"
    verify_kernel: bool = True
    controller_rate_hz: Optional[float] = spec(None, gt=0.0, doc="controller rate; default = physics rate", unit="Hz")

    def _validate(self, path: str, issues: Any) -> None:
        b = self.backend.lower()
        if not (b == "numpy" or b.startswith(("torch", "cuda")) or b == "cpu"):
            issues.append(
                ConfigIssue(_join(path, "backend"), f"unknown backend {self.backend!r}", "use 'numpy', 'torch', 'torch:cpu', 'cuda' or 'cuda:0'")
            )


@dataclass
class SimulationConfig(Block):
    """Everything needed to reproduce a simulation."""

    schema_version: int = SCHEMA_VERSION  # write 'schema_version: 2' at the top of hand-written files
    name: str = "simulation"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    seed: int = spec(0, ge=0)
    duration: float = spec(10.0, gt=0.0, unit="s")
    engine: EngineConfig = field(default_factory=EngineConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    sensors: List[SensorConfig] = field(default_factory=list)
    estimator: Optional[EstimatorConfig] = None
    network: Optional[NetworkConfig] = None
    swarm: Optional[SwarmConfig] = None
    faults: List[FaultConfig] = field(default_factory=list)
    hooks: List[HookConfig] = field(default_factory=list)
    health: HealthConfig = field(default_factory=HealthConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def _validate(self, path: str, issues: Any) -> None:
        names = [s.key for s in self.sensors]
        for i, n in enumerate(names):
            if names.count(n) > 1:
                issues.append(ConfigIssue(_join(_join(path, "sensors"), i), f"duplicate sensor name '{n}'", "give each instance a unique 'name'"))
        if self.estimator is not None:
            types = {s.type for s in self.sensors}
            if "imu" not in types:
                issues.append(
                    ConfigIssue(
                        _join(path, "estimator"), "an estimator needs an 'imu' sensor", "add 'imu' (and ideally 'gps', 'baro', 'mag') to sensors"
                    )
                )
        for n in self.logging.sensors:
            if n not in names:
                issues.append(ConfigIssue(_join(path, "logging.sensors"), f"'{n}' is not a configured sensor", _suggest(n, names)))
        if self.logging.mode == "stream" and not self.logging.path:
            issues.append(ConfigIssue(_join(path, "logging.path"), "stream mode needs a directory path"))
        if self.engine.controller_rate_hz is not None and self.engine.controller_rate_hz > 1.0 / self.engine.dt + 1e-9:
            issues.append(ConfigIssue(_join(path, "engine.controller_rate_hz"), f"cannot exceed the physics rate {1.0 / self.engine.dt:g} Hz"))
        n_steps = self.duration / self.engine.dt
        if n_steps > 5e8:
            issues.append(ConfigIssue(_join(path, "duration"), f"{n_steps:.3g} physics steps is unreasonably large"))
        if self.vehicle.count > 1 and self.vehicle.formation is None and self.swarm is not None:
            issues.append(
                ConfigIssue(
                    _join(path, "vehicle.formation"),
                    "a swarm of coincident vehicles would start inside each other",
                    "set vehicle.formation (ring/line/grid/vee)",
                )
            )
        for i, f in enumerate(self.faults):
            if f.vehicle is not None and f.vehicle >= self.vehicle.count:
                issues.append(
                    ConfigIssue(
                        _join(_join(path, "faults"), i) + ".vehicle", f"vehicle index {f.vehicle} out of range (count = {self.vehicle.count})"
                    )
                )
            if f.t_start > self.duration:
                issues.append(ConfigIssue(_join(_join(path, "faults"), i) + ".t_start", f"starts after the simulation ends ({self.duration} s)"))
        if self.environment.geofence_stop and self.health.on_diverge == "raise":
            issues.append(
                ConfigIssue(
                    _join(path, "environment.geofence_stop"),
                    "a geofence stop is a hard violation; with health.on_diverge: raise it would abort the run with an exception",
                    "set health.on_diverge: stop",
                )
            )
        if self.schema_version != SCHEMA_VERSION:
            issues.append(
                ConfigIssue(
                    _join(path, "schema_version"),
                    f"unsupported schema version {self.schema_version}",
                    "use aevion.config.load_config(), which migrates older files",
                )
            )

    # ------------------------------------------------------------------ conveniences
    def sensor_by_key(self, key: str) -> Optional[SensorConfig]:
        return next((s for s in self.sensors if s.key == key), None)

    @property
    def n_steps(self) -> int:
        return round(self.duration / self.engine.dt)
