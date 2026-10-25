"""Fault injection: a timeline of events that degrade the vehicle, its sensors, the wind or the network.

A :class:`Fault` has ``on_start`` / ``on_end`` (revert) and may modify the rotor command each active tick.
:class:`FaultTimeline` activates and deactivates faults at their scheduled times and emits
``fault.start`` / ``fault.end`` events. Every fault is reversible, so transient faults (a GPS outage from
10 s to 20 s) behave correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..errors import AevionError

if TYPE_CHECKING:
    from .simulation import Simulation


class FaultError(AevionError):
    """A fault refers to something that does not exist in the simulation (unknown sensor, rotor index, …)."""


class Fault:
    """Base class. Subclasses implement :meth:`on_start` and (for reversible faults) :meth:`on_end`."""

    name = "fault"

    def __init__(self) -> None:
        self.vehicle: Optional[int] = None  # set by the timeline (None = all vehicles)
        self.t_start: float = 0.0

    # ---------------------------------------------------------------- helpers
    def targets(self, sim: "Simulation") -> List[int]:
        n = sim.dyn.n
        if self.vehicle is None:
            return list(range(n))
        if not 0 <= self.vehicle < n:
            raise FaultError(f"fault '{self.name}': vehicle {self.vehicle} out of range (n = {n})")
        return [self.vehicle]

    def _sensor(self, sim: "Simulation", name: str) -> Any:
        suite = sim.suite
        if suite is None or name not in suite:
            known = [] if suite is None else list(suite.sensors)
            raise FaultError(f"fault '{self.name}': no sensor named '{name}' (configured: {known})")
        return suite[name]

    # -------------------------------------------------------------- lifecycle
    def on_start(self, sim: "Simulation") -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_end(self, sim: "Simulation") -> None:
        """Revert. Default: nothing (permanent faults)."""

    def modify_command(self, sim: "Simulation", cmd: Any) -> Any:
        """Optionally alter the rotor command (N, M) while active."""
        return cmd

    def details(self) -> Dict[str, Any]:
        return {}


# ------------------------------------------------------------------------------ vehicle faults
class RotorFailure(Fault):
    """Rotor ``rotor`` loses ``severity`` of its thrust (1.0 = dead; the commanded speed is also forced to zero)."""

    name = "rotor_failure"

    def __init__(self, rotor: int = 0, severity: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= severity <= 1.0:
            raise FaultError("severity must be in [0, 1]")
        self.rotor, self.severity = int(rotor), float(severity)
        self._prev: Dict[int, Any] = {}

    def on_start(self, sim: "Simulation") -> None:
        P = sim.dyn.params
        if not 0 <= self.rotor < sim.dyn.n_rotors:
            raise FaultError(f"rotor_failure: rotor {self.rotor} out of range (vehicle has {sim.dyn.n_rotors})")
        for v in self.targets(sim):
            self._prev[v] = float(P.rotor_health[v, self.rotor])
            P.rotor_health[v, self.rotor] = 1.0 - self.severity

    def on_end(self, sim: "Simulation") -> None:
        for v, h in self._prev.items():
            sim.dyn.params.rotor_health[v, self.rotor] = h
        self._prev.clear()

    def details(self) -> Any:
        return dict(rotor=self.rotor, severity=self.severity)


class ThrustLoss(Fault):
    """Scale the thrust coefficient of all rotors of the target vehicle(s) (e.g. damaged propellers)."""

    name = "thrust_loss"

    def __init__(self, factor: float = 0.8) -> None:
        super().__init__()
        if factor <= 0:
            raise FaultError("factor must be positive")
        self.factor = float(factor)
        self._prev: Dict[int, Any] = {}

    def on_start(self, sim: "Simulation") -> None:
        P = sim.dyn.params
        for v in self.targets(sim):
            self._prev[v] = P.kT[v].clone() if hasattr(P.kT[v], "clone") else P.kT[v].copy()
            P.kT[v] = P.kT[v] * self.factor

    def on_end(self, sim: "Simulation") -> None:
        for v, k in self._prev.items():
            sim.dyn.params.kT[v] = k
        self._prev.clear()

    def details(self) -> Any:
        return dict(factor=self.factor)


class PayloadChange(Fault):
    """Add (or release, if negative) a payload mass [kg]."""

    name = "payload_change"

    def __init__(self, mass: float = 0.2) -> None:
        super().__init__()
        self.mass = float(mass)

    def on_start(self, sim: "Simulation") -> None:
        P = sim.dyn.params
        for v in self.targets(sim):
            P.payload_mass[v] = P.payload_mass[v] + self.mass

    def on_end(self, sim: "Simulation") -> None:
        P = sim.dyn.params
        for v in self.targets(sim):
            P.payload_mass[v] = P.payload_mass[v] - self.mass

    def details(self) -> Any:
        return dict(mass=self.mass)


class ComShift(Fault):
    """Shift the centre of mass by ``offset`` [m] (body frame)."""

    name = "com_shift"

    def __init__(self, offset: Sequence[float] = (0.01, 0.0, 0.0)) -> None:
        super().__init__()
        self.offset = np.asarray(offset, float).reshape(3)

    def on_start(self, sim: "Simulation") -> None:
        P, B = sim.dyn.params, sim.B
        o = B.asarray(self.offset)
        for v in self.targets(sim):
            P.com_offset[v] = P.com_offset[v] + o

    def on_end(self, sim: "Simulation") -> None:
        P, B = sim.dyn.params, sim.B
        o = B.asarray(self.offset)
        for v in self.targets(sim):
            P.com_offset[v] = P.com_offset[v] - o

    def details(self) -> Any:
        return dict(offset=self.offset.tolist())


class ActuatorStuck(Fault):
    """Rotor ``rotor`` ignores the controller and stays at ``speed`` rad/s (``None`` = its speed when the fault begins)."""

    name = "actuator_stuck"

    def __init__(self, rotor: int = 0, speed: Optional[float] = None) -> None:
        super().__init__()
        self.rotor, self.speed = int(rotor), speed
        self._held: Dict[int, float] = {}

    def on_start(self, sim: "Simulation") -> None:
        if not 0 <= self.rotor < sim.dyn.n_rotors:
            raise FaultError(f"actuator_stuck: rotor {self.rotor} out of range")
        for v in self.targets(sim):
            self._held[v] = float(self.speed) if self.speed is not None else float(sim.B.to_numpy(sim.state.rotor)[v, self.rotor])

    def on_end(self, sim: "Simulation") -> None:
        self._held.clear()

    def modify_command(self, sim: "Simulation", cmd: Any) -> Any:
        for v, w in self._held.items():
            cmd[v, self.rotor] = w
        return cmd

    def details(self) -> Any:
        return dict(rotor=self.rotor, speed=self.speed)


class BatteryDrop(Fault):
    """Instantly reduce the state of charge to at most ``soc`` (a cell failure / sudden brown-out)."""

    name = "battery_drop"

    def __init__(self, soc: float = 0.15) -> None:
        super().__init__()
        self.soc = float(soc)

    def on_start(self, sim: "Simulation") -> None:
        s = sim.state.soc
        for v in self.targets(sim):
            s[v] = min(float(s[v]), self.soc)

    def details(self) -> Any:
        return dict(soc=self.soc)


# ------------------------------------------------------------------------------- sensor faults
class _SensorFault(Fault):
    def __init__(self, sensor: str) -> None:
        super().__init__()
        self.sensor = sensor

    def details(self) -> Any:
        return dict(sensor=self.sensor)


class SensorDropout(_SensorFault):
    """The sensor loses each sample with ``probability`` (1.0 = complete outage)."""

    name = "sensor_dropout"

    def __init__(self, sensor: str = "gps", probability: float = 1.0) -> None:
        super().__init__(sensor)
        self.probability, self._prev = float(probability), None

    def on_start(self, sim: "Simulation") -> None:
        s = self._sensor(sim, self.sensor)
        self._prev = s.dropout
        s.dropout = self.probability

    def on_end(self, sim: "Simulation") -> None:
        self._sensor(sim, self.sensor).dropout = self._prev

    def details(self) -> Any:
        return dict(sensor=self.sensor, probability=self.probability)


class GpsOutage(SensorDropout):
    name = "gps_outage"

    def __init__(self, sensor: str = "gps") -> None:
        super().__init__(sensor, 1.0)


class SensorStuck(_SensorFault):
    """The sensor keeps republishing its last reading."""

    name = "sensor_stuck"

    def on_start(self, sim: "Simulation") -> None:
        self._sensor(sim, self.sensor).frozen = True

    def on_end(self, sim: "Simulation") -> None:
        self._sensor(sim, self.sensor).frozen = False


class SensorBiasStep(_SensorFault):
    """Add a constant offset to one field of the sensor's reading (e.g. a gyro bias jump)."""

    name = "sensor_bias_step"

    def __init__(self, sensor: str = "imu", field: str = "gyro", bias: Sequence[float] = (0.0, 0.0, 0.05)) -> None:
        super().__init__(sensor)
        self.field, self.bias = field, np.asarray(bias, float)
        self._filter: Optional[Callable[[Any], Any]] = None

    def on_start(self, sim: "Simulation") -> None:
        s = self._sensor(sim, self.sensor)
        rows = self.targets(sim)
        B, field, bias = sim.B, self.field, self.bias

        def add_bias(reading: Any) -> Any:
            if field not in reading:
                raise FaultError(f"sensor_bias_step: sensor '{self.sensor}' has no field '{field}' (fields: {list(reading)})")
            r = dict(reading)
            x = B.clone(r[field])
            b = B.asarray(bias)
            for v in rows:
                x[v] = x[v] + b
            r[field] = x
            return r

        self._filter = add_bias
        s.filters.append(add_bias)

    def on_end(self, sim: "Simulation") -> None:
        s = self._sensor(sim, self.sensor)
        if self._filter in s.filters:
            s.filters.remove(self._filter)

    def details(self) -> Any:
        return dict(sensor=self.sensor, field=self.field, bias=self.bias.tolist())


class SensorNoiseScale(_SensorFault):
    """Multiply the sensor's noise by ``factor`` (degradation, vibration, interference)."""

    name = "sensor_noise_scale"

    def __init__(self, sensor: str = "imu", factor: float = 5.0) -> None:
        super().__init__(sensor)
        self.factor, self._prev = float(factor), 1.0

    def on_start(self, sim: "Simulation") -> None:
        s = self._sensor(sim, self.sensor)
        self._prev = s.noise_scale
        s.noise_scale = self._prev * self.factor

    def on_end(self, sim: "Simulation") -> None:
        self._sensor(sim, self.sensor).noise_scale = self._prev

    def details(self) -> Any:
        return dict(sensor=self.sensor, factor=self.factor)


class SensorDelay(_SensorFault):
    """Add ``delay_s`` of latency (a congested bus, a processing stall)."""

    name = "sensor_delay"

    def __init__(self, sensor: str = "gps", delay_s: float = 0.5) -> None:
        super().__init__(sensor)
        self.delay_s, self._prev = float(delay_s), 0.0

    def on_start(self, sim: "Simulation") -> None:
        s = self._sensor(sim, self.sensor)
        self._prev = s.latency
        s.latency = self._prev + self.delay_s

    def on_end(self, sim: "Simulation") -> None:
        self._sensor(sim, self.sensor).latency = self._prev

    def details(self) -> Any:
        return dict(sensor=self.sensor, delay_s=self.delay_s)


# -------------------------------------------------------------------------- environment faults
class _AddedWind:
    """Wraps the existing wind model and adds an extra component (used by wind faults)."""

    def __init__(self, base: Any, extra: Any) -> None:
        self.base, self.extra = base, extra

    def __getattr__(self, k: str) -> Any:  # delegate resample(), reset(), to_dict(), ... to the wrapped model
        return getattr(self.base, k)

    def sample(self, B: Any, pos: Any, vel: Any, t: float, dt: float) -> Any:
        return self.base.sample(B, pos, vel, t, dt) + self.extra.sample(B, pos, vel, t, dt)


class WindDisturbance(Fault):
    """Add a constant wind ``velocity`` [m/s] (ENU) on top of the existing wind for the fault's duration."""

    name = "wind_disturbance"

    def __init__(self, velocity: Sequence[float] = (5.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.velocity = np.asarray(velocity, float)
        self._orig = None

    def _extra(self, sim: "Simulation") -> Any:
        from ..environments.wind import ConstantWind

        w = ConstantWind(self.velocity)
        w.reset(sim.B, sim.dyn.n, None)
        return w

    def on_start(self, sim: "Simulation") -> None:
        self._orig = sim.world.wind
        sim.world.wind = _AddedWind(self._orig, self._extra(sim))

    def on_end(self, sim: "Simulation") -> None:
        sim.world.wind = self._orig

    def details(self) -> Any:
        return dict(velocity=self.velocity.tolist())


class WindGust(WindDisturbance):
    """A 1-cos gust of ``amplitude`` [m/s] lasting ``duration`` s, starting when the fault starts."""

    name = "wind_gust"

    def __init__(self, amplitude: Sequence[float] = (6.0, 0.0, 0.0), duration: float = 2.0) -> None:
        super().__init__(amplitude)
        self.duration = float(duration)

    def _extra(self, sim: "Simulation") -> Any:
        from ..environments.wind import Gust

        g = Gust(self.velocity, self.t_start, self.duration)
        g.reset(sim.B, sim.dyn.n, None)
        return g

    def details(self) -> Any:
        return dict(amplitude=self.velocity.tolist(), duration=self.duration)


class CommBlackout(Fault):
    """Total loss of the inter-agent link (``swarm``), the cellular link (``network``) or both."""

    name = "comm_blackout"

    def __init__(self, target: str = "both") -> None:
        super().__init__()
        if target not in ("swarm", "network", "both"):
            raise FaultError("target must be 'swarm', 'network' or 'both'")
        self.target = target
        self._prev: Dict[str, Any] = {}

    def on_start(self, sim: "Simulation") -> None:
        if self.target in ("swarm", "both") and sim.swarm is not None:
            self._prev["swarm"] = sim.swarm.comm.loss
            sim.swarm.comm.loss = 1.0
        if self.target in ("network", "both") and sim.net is not None:
            self._prev["net"] = getattr(sim.net, "blackout", False)
            sim.net.blackout = True
        if not self._prev:
            raise FaultError("comm_blackout: the simulation has no swarm/network to black out")

    def on_end(self, sim: "Simulation") -> None:
        if "swarm" in self._prev:
            sim.swarm.comm.loss = self._prev["swarm"]
        if "net" in self._prev:
            sim.net.blackout = self._prev["net"]
        self._prev.clear()

    def details(self) -> Any:
        return dict(target=self.target)


# --------------------------------------------------------------------------------- timeline
class FaultTimeline:
    """Schedules faults. ``add(fault, t_start, t_end=None, vehicle=None)``."""

    def __init__(self) -> None:
        self._items: List[dict] = []

    def add(self, fault: Fault, t_start: float, t_end: Optional[float] = None, vehicle: Optional[int] = None) -> Fault:
        fault.vehicle, fault.t_start = vehicle, float(t_start)
        self._items.append(dict(fault=fault, t0=float(t_start), t1=None if t_end is None else float(t_end), active=False, done=False))
        self._items.sort(key=lambda d: d["t0"])
        return fault

    def __len__(self) -> int:
        return len(self._items)

    @property
    def active(self) -> List[Fault]:
        return [d["fault"] for d in self._items if d["active"]]

    def reset(self) -> None:
        for d in self._items:
            d["active"] = d["done"] = False

    def update(self, sim: "Simulation", t: float, step: int) -> None:
        """Start/stop faults due at time ``t`` (tolerance: half a physics step)."""
        tol = 0.5 * sim.dt
        for d in self._items:
            f = d["fault"]
            if not d["active"] and not d["done"] and t >= d["t0"] - tol:
                f.on_start(sim)
                d["active"] = True
                sim.emit("fault.start", vehicle=f.vehicle, fault=f.name, **f.details())
            if d["active"] and d["t1"] is not None and t >= d["t1"] - tol:
                f.on_end(sim)
                d["active"], d["done"] = False, True
                sim.emit("fault.end", vehicle=f.vehicle, fault=f.name)

    def modify_command(self, sim: "Simulation", cmd: Any) -> Any:
        for d in self._items:
            if d["active"]:
                cmd = d["fault"].modify_command(sim, cmd)
        return cmd

    def describe(self) -> List[dict]:
        return [dict(fault=d["fault"].name, t_start=d["t0"], t_end=d["t1"], vehicle=d["fault"].vehicle, **d["fault"].details()) for d in self._items]
