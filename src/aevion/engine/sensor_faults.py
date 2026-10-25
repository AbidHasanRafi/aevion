"""More sensor faults: scale error, saturation, spikes, drift, invalid (NaN) output and latency spikes.

Together with ``sensor_dropout`` / ``gps_outage`` (lost samples), ``sensor_stuck`` (frozen output), ``sensor_bias_step``, ``sensor_noise_scale`` and
``sensor_delay`` (constant extra latency) of :mod:`aevion.engine.faults` this is the fault-injection library for estimator and controller
robustness studies. Every fault acts on one *field* of the sensor's reading (``field="gyro"``, ``"accel"``, ``"pos"``, ``"range"`` ...) for
the vehicles it targets, starts at ``t_start`` and ends at ``t_end`` (set in the configuration's ``faults:`` entry).

All randomness comes from a generator seeded by the fault's ``seed``, so runs stay reproducible.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .faults import FaultError, _SensorFault

__all__ = ["SensorScaleError", "SensorSaturation", "SensorSpikes", "SensorDrift", "SensorNaN", "SensorDelaySpikes"]


class _FieldFault(_SensorFault):
    """A filter on one field of the sensor reading, applied to the targeted vehicles."""

    def __init__(self, sensor: str, field: str, seed: int = 0) -> None:
        super().__init__(sensor)
        self.field, self.seed = field, int(seed)
        self._filter: Optional[Callable[[Any], Any]] = None

    def transform(self, x: np.ndarray, rows: np.ndarray, t: float) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def on_start(self, sim: Any) -> None:
        s = self._sensor(sim, self.sensor)
        rows = np.asarray(self.targets(sim), int)
        B = sim.B
        self._rng = np.random.default_rng(self.seed + 7919 * (1 + (self.vehicle or 0)))

        def apply(reading: Any) -> Any:
            if self.field not in reading:
                raise FaultError(f"{self.name}: sensor '{self.sensor}' has no field '{self.field}' (fields: {list(reading)})")
            r = dict(reading)
            x = np.array(B.to_numpy(r[self.field]), dtype=float, copy=True)
            x[rows] = self.transform(x[rows], rows, float(sim.t))
            r[self.field] = B.asarray(x)
            return r

        self._filter = apply
        s.filters.append(apply)

    def on_end(self, sim: Any) -> None:
        s = self._sensor(sim, self.sensor)
        if self._filter in s.filters:
            s.filters.remove(self._filter)

    def details(self) -> Dict[str, Any]:
        return dict(sensor=self.sensor, field=self.field)


class SensorScaleError(_FieldFault):
    """Multiply a field by ``factor`` (e.g. 1.05: a 5 % scale-factor error from temperature or a damaged sensor)."""

    name = "sensor_scale"

    def __init__(self, sensor: str = "imu", field: str = "gyro", factor: float = 1.05) -> None:
        super().__init__(sensor, field)
        self.factor = float(factor)

    def transform(self, x: np.ndarray, rows: np.ndarray, t: float) -> np.ndarray:
        return x * self.factor

    def details(self) -> Dict[str, Any]:
        return dict(super().details(), factor=self.factor)


class SensorSaturation(_FieldFault):
    """Clip a field to ``[-limit, limit]`` (a range that is too small, a railed ADC)."""

    name = "sensor_saturation"

    def __init__(self, sensor: str = "imu", field: str = "gyro", limit: float = 1.0) -> None:
        super().__init__(sensor, field)
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = float(limit)

    def transform(self, x: np.ndarray, rows: np.ndarray, t: float) -> np.ndarray:
        return np.clip(x, -self.limit, self.limit)

    def details(self) -> Dict[str, Any]:
        return dict(super().details(), limit=self.limit)


class SensorSpikes(_FieldFault):
    """Each sample of a vehicle is corrupted with probability ``probability`` by an outlier: Gaussian with standard deviation ``magnitude``
    (``mode="gaussian"``) or ``+-magnitude`` with random sign (``mode="sign"``), added to every component."""

    name = "sensor_spikes"

    def __init__(
        self, sensor: str = "gps", field: str = "pos", probability: float = 0.05, magnitude: float = 20.0, mode: str = "gaussian", seed: int = 0
    ) -> None:
        super().__init__(sensor, field, seed)
        if not 0.0 <= probability <= 1.0 or magnitude < 0 or mode not in ("gaussian", "sign"):
            raise ValueError("need 0 <= probability <= 1, magnitude >= 0 and mode 'gaussian' or 'sign'")
        self.probability, self.magnitude, self.mode = float(probability), float(magnitude), mode

    def transform(self, x: np.ndarray, rows: np.ndarray, t: float) -> np.ndarray:
        hit = self._rng.random(len(x)) < self.probability
        noise = self._rng.normal(0.0, self.magnitude, x.shape) if self.mode == "gaussian" else self.magnitude * self._rng.choice([-1.0, 1.0], x.shape)
        shape = (len(x),) + (1,) * (x.ndim - 1)
        return x + noise * hit.reshape(shape)

    def details(self) -> Dict[str, Any]:
        return dict(super().details(), probability=self.probability, magnitude=self.magnitude)


class SensorDrift(_FieldFault):
    """A bias that grows linearly from the start of the fault: ``bias(t) = rate * (t - t_start)`` (per component, or a 3-vector ``rate``)."""

    name = "sensor_drift"

    def __init__(self, sensor: str = "imu", field: str = "gyro", rate: Any = 0.002) -> None:
        super().__init__(sensor, field)
        self.rate = np.asarray(rate, float)

    def transform(self, x: np.ndarray, rows: np.ndarray, t: float) -> np.ndarray:
        return x + self.rate * max(t - self.t_start, 0.0)

    def details(self) -> Dict[str, Any]:
        return dict(super().details(), rate=self.rate.tolist())


class SensorNaN(_FieldFault):
    """With probability ``probability`` the field arrives as NaN (a corrupted frame, a bus error). Estimators must reject it."""

    name = "sensor_nan"

    def __init__(self, sensor: str = "gps", field: str = "pos", probability: float = 0.1, seed: int = 0) -> None:
        super().__init__(sensor, field, seed)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.probability = float(probability)

    def transform(self, x: np.ndarray, rows: np.ndarray, t: float) -> np.ndarray:
        bad = self._rng.random(len(x)) < self.probability
        out = x.copy()
        out[bad] = np.nan
        return out

    def details(self) -> Dict[str, Any]:
        return dict(super().details(), probability=self.probability)


class SensorDelaySpikes(_SensorFault):
    """Random extra latency: each sample is delayed by ``delay_s`` with probability ``probability``. Delivery stays in order (a late sample holds
    back the ones behind it), as on a serial link."""

    name = "sensor_delay_spikes"

    def __init__(self, sensor: str = "gps", probability: float = 0.1, delay_s: float = 0.3, seed: int = 0) -> None:
        super().__init__(sensor)
        if not 0.0 <= probability <= 1.0 or delay_s < 0:
            raise ValueError("need 0 <= probability <= 1 and delay_s >= 0")
        self.probability, self.delay_s, self.seed = float(probability), float(delay_s), int(seed)
        self._prev: Optional[Callable[[], float]] = None

    def on_start(self, sim: Any) -> None:
        s = self._sensor(sim, self.sensor)
        rng = np.random.default_rng(self.seed + 104729)
        self._prev = getattr(s, "latency_extra", None)
        s.latency_extra = lambda: self.delay_s if rng.random() < self.probability else 0.0

    def on_end(self, sim: Any) -> None:
        self._sensor(sim, self.sensor).latency_extra = self._prev

    def details(self) -> Dict[str, Any]:
        return dict(super().details(), probability=self.probability, delay_s=self.delay_s)
