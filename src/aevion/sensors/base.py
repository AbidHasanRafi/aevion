"""Sensor base class with sampling rate, latency and dropout, plus a sensor suite."""

from __future__ import annotations

import zlib
from collections import deque
from typing import Any, Dict, Optional


class Sensor:
    """A sensor produces readings at ``rate`` Hz, delivered ``latency`` seconds later.

    Subclasses implement :meth:`measure` which returns a dict of arrays with leading dim N.
    ``dropout`` is the probability that a given sample is lost entirely.
    """

    name = "sensor"

    def __init__(self, rate: float = 100.0, latency: float = 0.0, dropout: float = 0.0, name: Optional[str] = None, clock=None, extrinsics=None):
        from ..frames import ClockModel, Transform

        self.rate, self.latency, self.dropout = float(rate), float(latency), float(dropout)
        if name:
            self.name = name
        # timestamp error model and sensor pose in the body frame (T_body_sensor)
        self.clock = clock if (clock is None or isinstance(clock, ClockModel)) else ClockModel(**clock)
        self.extrinsics = extrinsics if (extrinsics is None or isinstance(extrinsics, Transform)) else Transform.from_dict(extrinsics)
        # fault-injection hooks (see aevion.engine.faults)
        self.noise_scale = 1.0
        self.frozen = False
        self.filters: list = []
        self.latency_extra: Optional[Any] = None  # callable() -> extra delay [s] for the next sample (latency-spike faults)
        self.stamp_reported = -1.0
        self.sim: Any = None
        self.B: Any = None
        self._next = 0.0
        self._queue: deque = deque()
        self.latest: Optional[dict] = None
        self.fresh = False
        self.stamp = -1.0

    # -- lifecycle -----------------------------------------------------------
    def bind(self, sim, seed: int = 0):
        self.sim, self.B, self.n = sim, sim.B, sim.n
        dt = getattr(sim, "dt", None)
        if dt and self.rate > 1.0 / dt * 1.001:
            import warnings

            warnings.warn(
                f"sensor '{self.name}' is set to {self.rate:g} Hz but the physics step is {dt:g} s ({1.0 / dt:g} Hz): it can deliver at most one reading per step",
                stacklevel=3,
            )
        self.gen = self.B.make_rng(int(seed) + zlib.crc32(self.name.encode()) % 100000)
        self._next = 0.0
        self._queue.clear()
        self.latest, self.fresh, self.stamp = None, False, -1.0
        self.on_bind()

    def on_bind(self):  # hook
        pass

    def measure(self, state) -> Optional[dict]:
        raise NotImplementedError

    def reset_state(self, state):
        """Re-initialise internal error states (biases, filters) at episode start."""
        self.on_bind()
        self._next = state.t
        self._queue.clear()
        self.latest, self.fresh = None, False

    # -- per-step update -------------------------------------------------------
    def update(self, state) -> bool:
        """Advance with the simulator; returns True if a new reading became available."""
        t = state.t
        period = 1.0 / self.rate
        self.fresh = False
        while t + 1e-9 >= self._next:
            self._next += period
            if self.dropout > 0.0 and float(self.B.to_numpy(self.B.rand((1,), self.gen))[0]) < self.dropout:
                continue
            if self.frozen and self.latest is not None:
                reading = self.latest  # stuck sensor: keeps publishing its last value
            else:
                measured = self.measure(state)
                if measured is None:  # the sensor had nothing to publish (e.g. a GNSS receiver without a fix): a gap, not a sample
                    continue
                reading = measured
                for f in self.filters:
                    reading = f(reading)
            self._queue.append((t + self.latency + (self.latency_extra() if self.latency_extra is not None else 0.0), t, reading))
        while self._queue and self._queue[0][0] <= t + 1e-9:
            _, ts, reading = self._queue.popleft()
            self.latest, self.stamp, self.fresh = reading, ts, True
            self.stamp_reported = self.clock.stamp(ts) if self.clock is not None else ts
        return self.fresh

    @property
    def dt(self):
        return 1.0 / self.rate

    def noise(self, shape, std):
        return self.B.randn(shape, self.gen) * (std * self.noise_scale)

    def _apply_extrinsics(self):
        """Hook for subclasses that cast rays: rebuild body-frame geometry from ``self.extrinsics``."""

    def to_dict(self):
        d = dict(type=type(self).__name__, rate=self.rate, latency=self.latency, dropout=self.dropout)
        if self.extrinsics is not None:
            d["extrinsics"] = self.extrinsics.to_dict()
        if self.clock is not None:
            d["clock"] = dict(offset_s=self.clock.offset_s, drift_ppm=self.clock.drift_ppm, jitter_std_s=self.clock.jitter_std_s)
        return d


class SensorSuite:
    """A named collection of sensors updated together."""

    def __init__(self, *sensors: Sensor):
        self.sensors: Dict[str, Sensor] = {s.name: s for s in sensors}

    def add(self, sensor: Sensor):
        self.sensors[sensor.name] = sensor
        return self

    def bind(self, sim, seed: int = 0):
        for s in self.sensors.values():
            s.bind(sim, seed)
        return self

    def reset(self, state):
        for s in self.sensors.values():
            s.reset_state(state)

    def update(self, state) -> Dict[str, Optional[Dict[str, Any]]]:
        out: Dict[str, Optional[Dict[str, Any]]] = {}
        for k, s in self.sensors.items():
            s.update(state)
            out[k] = s.latest
        return out

    def fresh(self) -> Dict[str, bool]:
        return {k: s.fresh for k, s in self.sensors.items()}

    def __getitem__(self, k):
        return self.sensors[k]

    def __contains__(self, k):
        return k in self.sensors

    def to_dict(self):
        return {k: s.to_dict() for k, s in self.sensors.items()}
