"""Construct simulation components from a validated :class:`~aevion.config.SimulationConfig` via the registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..config import SimulationConfig
    from ..dynamics.multirotor import Multirotor


@dataclass
class Parts:
    """Everything a :class:`Simulation` needs, built but not yet wired together."""

    spec: Any
    world: Any
    dyn: Any
    ctrl: Any
    traj: Any
    offsets: Optional[np.ndarray]
    suite: Any
    est: Any
    net: Any
    swarm: Any
    faults: Any
    hooks: list
    health: Any
    recorded_sensors: List[str] = field(default_factory=list)


def _reg() -> Any:
    from ..registry import registries

    return registries


def build_spec(cfg: "SimulationConfig") -> Any:
    from ..vehicles.spec import VehicleSpec

    reg = _reg()
    v = cfg.vehicle
    spec = VehicleSpec.from_dict(v.model) if isinstance(v.model, dict) else reg.vehicles.create(v.model)
    if v.overrides:
        spec = spec.copy(**v.overrides)
    return spec


def build_world(cfg: "SimulationConfig") -> Any:
    reg = _reg()
    d = cfg.world.to_dict()
    typ = d.pop("type")
    if typ == "scene" and cfg.environment.origin is not None:  # map data and terrain are placed with the environment's anchor
        d.setdefault("origin", cfg.environment.origin.to_dict())
    source_dir = getattr(cfg, "_source_dir", None)
    if typ == "scene" and source_dir:  # relative map, terrain and mesh files are found next to the configuration file
        d.setdefault("base_dir", source_dir)
    world = reg.worlds.create(typ, **{"seed": cfg.seed, **d})
    if cfg.wind.type != "none":
        world.wind = reg.winds.build(cfg.wind.to_dict())
    apply_environment(cfg, world)
    return world


def apply_environment(cfg: "SimulationConfig", world: Any) -> None:
    """Apply ``cfg.environment`` (origin, UTC time, weather, actors) to ``world``."""
    env = cfg.environment
    reg = _reg()
    if env.origin is not None:
        from ..environments.geodesy import LocalFrame

        world.origin = LocalFrame(env.origin.lat, env.origin.lon, env.origin.alt)
    if env.gravity is not None:
        from ..core.constants import GravityModel

        if env.gravity == "wgs84":  # latitude-dependent gravity follows the anchor
            lat, alt = (env.origin.lat, env.origin.alt) if env.origin is not None else (45.0, 0.0)
            world.gravity = GravityModel("wgs84", latitude_deg=lat, altitude_ref=alt)
        else:
            world.gravity = GravityModel(env.gravity, g=world.gravity.g)
    if env.utc is not None:
        world.utc = env.utc
    if env.weather is not None:
        from ..environments.weather import Weather

        Weather.from_dict(env.weather).apply(world)
    for a in env.actors:
        world.add_actor(reg.actors.create(a.type, **a.params))


def build_sensors(cfg: "SimulationConfig", sim: "Multirotor") -> Any:
    from ..sensors import SensorSuite

    reg = _reg()
    if not cfg.sensors:
        return None, []
    sensors, recorded = [], []
    for sc in cfg.sensors:
        s = reg.sensors.create(sc.type, **sc.params)
        if sc.name:
            s.name = sc.name
        sensors.append(s)
        if sc.record:
            recorded.append(s.name)
    for n in cfg.logging.sensors:
        if n not in recorded:
            recorded.append(n)
    suite = SensorSuite(*sensors)
    if len(suite.sensors) != len(sensors):
        raise ValueError("duplicate sensor instance names after construction")
    suite.bind(sim, cfg.seed)
    return suite, recorded


def build_swarm(cfg: "SimulationConfig", offsets: Any = None) -> Any:
    from ..multiagent import CommModel, SwarmCoordinator

    if cfg.swarm is None or cfg.vehicle.count < 2:
        return None
    s = cfg.swarm
    comm = CommModel(delay_s=s.comm.delay_s, loss=s.comm.loss, range_m=s.comm.range_m, rate_hz=s.comm.rate_hz)
    swarm = SwarmCoordinator(
        cfg.vehicle.count, cfg.engine.dt, avoid_radius=s.avoid_radius, gain=s.gain, max_push=s.max_push, comm=comm, seed=cfg.seed
    )
    for lc in s.laws:
        law = _reg().swarm_laws.create(lc.type, cfg.vehicle.count, **lc.params)
        law.bind(offsets, cfg.engine.dt)
        swarm.laws.append(law)
    return swarm


def build_faults(cfg: "SimulationConfig") -> Any:
    from .faults import FaultTimeline

    reg = _reg()
    tl = FaultTimeline()
    for f in cfg.faults:
        tl.add(reg.faults.create(f.type, **f.params), f.t_start, f.t_end, f.vehicle)
    return tl


def build_health(cfg: "SimulationConfig", backend_name: str) -> Any:
    from ..health import HealthMonitor
    from ..health import checks as HC

    h = cfg.health
    if not h.enabled:
        return None
    reg = _reg()
    every = h.check_every if backend_name == "numpy" else max(h.check_every, 10)  # avoid a device sync every tick on GPU
    checks = [
        HC.FiniteCheck(),
        HC.QuaternionNormCheck(h.quat_tol),
        HC.SpeedCheck(h.max_speed),
        HC.RateCheck(h.max_rate),
        HC.AltitudeCheck(h.max_altitude),
        HC.RotorSpeedCheck(),
        HC.BatteryCheck(),
        HC.SaturationCheck(),
    ]
    for c in h.checks:
        checks.append(reg.health_checks.create(c.type, **c.params))
    if cfg.environment.geofences:
        checks.append(
            reg.health_checks.create("geofence", fences=[g.to_dict() for g in cfg.environment.geofences], stop=cfg.environment.geofence_stop)
        )
    return HealthMonitor(checks, policy=h.on_diverge, every=every)


def build_trajectory(cfg: "SimulationConfig", world: Any) -> Any:
    """Build the reference trajectory; factories that declare a ``world`` parameter (planned, mission) are given the world."""
    import inspect

    reg = _reg()
    spec = cfg.trajectory.to_dict()
    try:
        takes_world = "world" in inspect.signature(reg.trajectories.get(spec.get("type") or spec.get("kind") or "")).parameters
    except (TypeError, ValueError):
        takes_world = False
    return reg.trajectories.build(spec, world=world) if takes_world else reg.trajectories.build(spec)


def build_parts(cfg: "SimulationConfig") -> Parts:
    """Build all components for ``cfg`` (no simulation state is created yet)."""
    from ..dynamics.multirotor import Multirotor
    from ..estimation import Estimator  # noqa: F401  (registered via registry)

    reg = _reg()
    spec = build_spec(cfg)
    world = build_world(cfg)
    e = cfg.engine
    sim = Multirotor(
        spec,
        n=cfg.vehicle.count,
        backend=e.backend,
        dtype=e.dtype,
        dt=e.dt,
        integrator=e.integrator,
        world=world,
        seed=cfg.seed,
        kernel=e.kernel,
        verify_kernel=e.verify_kernel,
    )
    ctrl = reg.controllers.build(cfg.controller.to_dict(), sim)
    if hasattr(ctrl, "reset"):
        ctrl.reset(sim.n)
    traj = build_trajectory(cfg, world)
    n = cfg.vehicle.count
    offsets: Optional[np.ndarray]
    if cfg.vehicle.formation is not None:
        fd = cfg.vehicle.formation.to_dict()
        offsets = np.asarray(reg.formations.create(fd.pop("type"), n, **fd), float)
    else:
        offsets = np.zeros((n, 3)) if n > 1 else None
    if hasattr(ctrl, "attach_trajectory"):  # predictive controllers preview the reference exactly
        ctrl.attach_trajectory(traj, offsets)
    suite, recorded = build_sensors(cfg, sim)
    est = None
    if cfg.estimator is not None:
        est = reg.estimators.create(cfg.estimator.type, sim, **cfg.estimator.params)
        est.reset(None)
    net = None
    if cfg.network is not None:
        net = reg.networks.create(cfg.network.type, world=world, seed=cfg.seed, **cfg.network.params)
    hooks = [reg.hooks.create(h.type, **h.params) for h in cfg.hooks]
    return Parts(
        spec,
        world,
        sim,
        ctrl,
        traj,
        offsets,
        suite,
        est,
        net,
        build_swarm(cfg, offsets),
        build_faults(cfg),
        hooks,
        build_health(cfg, sim.B.name),
        recorded,
    )
