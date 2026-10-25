"""Declarative, serialisable simulation scenarios.

A :class:`Scenario` captures everything needed to reproduce a run (vehicle, world, wind,
controller, trajectory, sensors, seed). ``Scenario.run()`` produces a :class:`RunArchive`; the
scenario is stored in the archive manifest so ``verify()`` can re-simulate and compare.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .control import controllers as C
from .control import reference as R
from .datasets.archive import RunArchive
from .environments.wind import wind_from_dict
from .environments.world import World
from .multiagent import make_formation  # re-exported: this function lived here originally
from .sensors import GPS, IMU, Barometer, Camera, Lidar, Magnetometer, RangeFinder
from .vehicles.spec import VehicleSpec, get_vehicle

__all__ = ["CONTROLLERS", "SENSOR_FACTORIES", "Scenario", "make_formation", "make_trajectory", "verify"]

# ----------------------------------------------------------------------- registries
CONTROLLERS: Dict[str, Callable] = {
    "geometric": lambda sim, kw: C.GeometricController(sim, C.Gains(**kw) if kw else None),
    "cascaded": lambda sim, kw: C.CascadedPID(sim, **kw),
    "lqr": lambda sim, kw: C.LQRHover(sim, **kw),
}


def make_trajectory(spec: dict) -> R.Trajectory:
    kind = spec.get("kind", "hover")
    kw = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "hover":
        return R.Hover(**kw)
    if kind == "circle":
        return R.Circle(**kw)
    if kind == "figure8":
        return R.Lissajous.figure8(**kw)
    if kind == "lissajous":
        return R.Lissajous(**kw)
    if kind == "waypoints":
        return R.Waypoints(**kw)
    if kind == "polyline":
        return R.Polyline(**kw)
    raise ValueError(f"unknown trajectory kind {kind!r}")


SENSOR_FACTORIES = {
    "imu": IMU,
    "gps": GPS,
    "baro": Barometer,
    "mag": Magnetometer,
    "rangefinder": RangeFinder,
    "lidar": lambda: Lidar(channels=8, samples=180, return_points=False),
    "camera": lambda: Camera(width=64, height=48, modes=("depth",)),
}


@dataclass
class Scenario:
    name: str = "scenario"
    vehicle: Any = "generic_1kg"  # preset name or VehicleSpec.to_dict()
    world: Any = None  # None | "forest"/"city"/... | World.to_dict() | {"kind": "forest", ...}
    wind: Optional[dict] = None  # wind model dict (see environments.wind)
    controller: str = "geometric"
    controller_kwargs: dict = field(default_factory=dict)
    trajectory: dict = field(default_factory=lambda: dict(kind="hover", pos=[0, 0, 2.0]))
    duration: float = 10.0
    dt: float = 0.005
    integrator: str = "rk4"
    n: int = 1
    formation: Optional[dict] = None  # e.g. {"kind": "ring", "spacing": 1.5}
    seed: int = 0
    sensors: List[str] = field(default_factory=list)
    use_estimator: bool = False
    estimator_handover_s: float = 6.0  # fly on truth until the filter has converged
    record_every: int = 4
    record_sensors: List[str] = field(default_factory=list)
    initial: Optional[dict] = None  # {"pos": [...], "vel": [...], "euler": [...]}
    swarm: Optional[dict] = None  # {"avoid_radius":1.0, "gain":0.8, "comm":{"delay_s":0.05,"loss":0.1}}
    wireless: Optional[dict] = None  # CellularNetwork.to_dict(): base stations + channel; records link_* channels
    perturbation: Optional[dict] = None  # {"kind": "rotor_failure", "t": 5.0, "vehicle": 0, "rotor": 1, "severity": 1.0}
    notes: str = ""

    # -------------------------------------------------------------------- build
    def build_world(self) -> World:
        w = self.world
        if w is None:
            world = World()
        elif isinstance(w, str):
            world = World.from_spec(w, seed=self.seed)
        elif "kind" in w:
            kw = {k: v for k, v in w.items() if k != "kind"}
            world = World.from_spec(w["kind"], **{"seed": self.seed, **kw})
        else:
            world = World.from_dict(w)
        if self.wind is not None:
            world.wind = wind_from_dict(self.wind)
        return world

    def build_spec(self) -> VehicleSpec:
        return (
            self.vehicle
            if isinstance(self.vehicle, VehicleSpec)
            else (VehicleSpec.from_dict(self.vehicle) if isinstance(self.vehicle, dict) else get_vehicle(self.vehicle))
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(self.vehicle, VehicleSpec):
            d["vehicle"] = self.vehicle.to_dict()
        if isinstance(self.world, World):
            d["world"] = self.world.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(**d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    # ---------------------------------------------------------------------- run
    def to_config(self):
        """This scenario as a schema-v2 :class:`~aevion.config.SimulationConfig` (migrated from the legacy form)."""
        from .config import SimulationConfig
        from .config.migrate import migrate

        return SimulationConfig.from_dict(migrate(self.to_dict()), source=f"Scenario '{self.name}'")

    def run(self, backend: str = "numpy", dtype=None, progress: bool = False, kernel: str = "auto") -> RunArchive:
        """Run through the simulation engine (see :mod:`aevion.engine`) and return the archive.

        ``kernel`` is ``"auto"``, ``"reference"`` or ``"numba"``.
        """
        from .engine import Simulation

        cfg = self.to_config()
        cfg.engine.backend = backend
        cfg.engine.dtype = dtype
        cfg.engine.kernel = kernel
        arch = Simulation(cfg).run(progress=progress)
        arch.meta["scenario"] = self.to_dict()  # keep the full legacy form for older tools
        return arch


def verify(archive: RunArchive, tol: float = 1e-9, backend: Optional[str] = None) -> dict:
    """Re-simulate an archive from its recorded configuration and compare channel by channel."""
    cfg = archive.meta.get("config")
    if cfg:
        from .config import load_config
        from .engine import Simulation

        c = load_config(cfg)
        c.engine.backend = backend or archive.meta.get("backend", c.engine.backend)
        again = Simulation(c).run()
    else:  # archives written before schema v2
        sc = Scenario.from_dict(archive.meta["scenario"])
        again = sc.run(backend=backend or archive.meta.get("backend", "numpy"))
    diffs = {}
    for k in ("pos", "vel", "quat", "omega", "rotor", "soc"):
        if k in archive and k in again:
            n = min(len(archive[k]), len(again[k]))
            diffs[k] = float(np.max(np.abs(archive[k][:n] - again[k][:n])))
    worst = max(diffs.values()) if diffs else float("nan")
    return dict(ok=bool(worst <= tol), max_abs_diff=worst, per_channel=diffs, tol=tol)
