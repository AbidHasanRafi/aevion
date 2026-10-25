"""RF-environment faults for resilience studies: GNSS interference, a spoofing threat model and communication interference.

They plug into the engine's fault timeline (``faults: [{type: gnss_jamming, ...}]``) and act through the existing sensor and
network hooks, so every effect is reversible and recorded as ``fault.start`` / ``fault.end`` events.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ...engine.faults import Fault, FaultError
from .jamming import GNSS_L1_HZ, GNSS_NOMINAL_POWER_DBM, Jammer, gnss_cn0_effective_dbhz, gnss_jam_to_signal_db
from .spoofing import capture_probability, pull_offset, spoofer_power_advantage_db

__all__ = ["GnssJamming", "GnssSpoofing", "NetworkJamming"]


class GnssJamming(Fault):
    """Interference source near the vehicle degrades the GNSS receiver.

    The effective ``C/N0`` follows Betz' formula from the free-space jammer-to-signal ratio at the vehicle. Position noise grows as
    ``10^((C/N0_nominal - C/N0_eff)/20)``; below ``lock_threshold_dbhz`` the receiver loses lock and its output **freezes** at the
    last good fix (per vehicle) until the interference subsides.
    """

    name = "gnss_jamming"

    def __init__(
        self,
        pos: Sequence[float] = (0.0, 0.0, 2.0),
        power_dbm: float = 30.0,
        sensor: str = "gps",
        cn0_nominal_dbhz: float = 43.5,
        lock_threshold_dbhz: float = 28.0,
        gain_dbi: float = 0.0,
        freq_hz: float = GNSS_L1_HZ,
        noise_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.jammer = Jammer(pos=np.asarray(pos, float), power_dbm=power_dbm, gain_dbi=gain_dbi, freq_hz=freq_hz)
        self.sensor, self.cn0_nom, self.lock = sensor, cn0_nominal_dbhz, lock_threshold_dbhz
        self.noise_gain = noise_gain
        self._filter: Optional[Callable[[Any], Any]] = None
        self._last: Optional[Dict[str, np.ndarray]] = None
        self.min_cn0_dbhz = math.inf

    def on_start(self, sim: Any) -> None:
        s = self._sensor(sim, self.sensor)
        rows = self.targets(sim)
        B = sim.B
        self._last = None
        if hasattr(s, "interference_fn"):  # a pseudorange receiver: lower C/N0 per vehicle and let it lose satellites and lock itself
            mask = np.zeros(sim.n, bool)
            mask[rows] = True

            def penalty(pos: Any) -> Any:
                cn0 = gnss_cn0_effective_dbhz(self.cn0_nom, gnss_jam_to_signal_db(self.jammer.received_dbm(np.asarray(pos, float))))
                self.min_cn0_dbhz = min(self.min_cn0_dbhz, float(np.min(cn0[mask])) if mask.any() else math.inf)
                return np.where(mask, np.clip(self.cn0_nom - cn0, 0.0, 80.0), 0.0)

            self._filter = penalty
            s.interference_fn = penalty
            return

        def degrade(reading: Any) -> Any:
            pos_true = B.to_numpy(sim.state.pos)
            j_dbm = self.jammer.received_dbm(pos_true)
            cn0 = gnss_cn0_effective_dbhz(self.cn0_nom, gnss_jam_to_signal_db(j_dbm))
            self.min_cn0_dbhz = min(self.min_cn0_dbhz, float(np.min(cn0[rows])))
            out = dict(reading)
            pos, vel = B.to_numpy(reading["pos"]).copy(), B.to_numpy(reading["vel"]).copy()
            lost_lock = cn0 < self.lock
            scale = np.where(lost_lock, 1.0, 10.0 ** (np.clip(self.cn0_nom - cn0, 0.0, 60.0) / 20.0))  # noise grows until lock is lost
            extra = B.to_numpy(s.noise(pos.shape, 1.0)) * (self.noise_gain * (scale - 1.0))[:, None]  # the sensor's own seeded stream
            mask = np.zeros(len(pos), bool)
            mask[rows] = True
            pos = pos + np.where(mask[:, None], extra, 0.0)
            lost = mask & lost_lock
            if self._last is None:
                self._last = dict(pos=pos.copy(), vel=vel.copy())
            self._last["pos"] = np.where(lost[:, None], self._last["pos"], pos)
            self._last["vel"] = np.where(lost[:, None], self._last["vel"], vel)
            out["pos"], out["vel"] = B.asarray(self._last["pos"]), B.asarray(self._last["vel"])
            return out

        self._filter = degrade
        s.filters.append(degrade)

    def on_end(self, sim: Any) -> None:
        s = self._sensor(sim, self.sensor)
        if self._filter in s.filters:
            s.filters.remove(self._filter)
        if getattr(s, "interference_fn", None) is self._filter:
            s.interference_fn = None

    def details(self) -> Any:
        return dict(sensor=self.sensor, jammer_pos=self.jammer.pos.tolist(), power_dbm=self.jammer.power_dbm, lock_threshold_dbhz=self.lock)


class GnssSpoofing(Fault):
    """Threat model: a counterfeit GNSS signal drags the reported position away from the truth.

    If ``spoofer_pos`` is given, capture succeeds only when the counterfeit signal beats the authentic one at the vehicle (see
    :func:`~aevion.wireless.security.spoofing.capture_probability`; decided once at the start with a fixed seed); otherwise the
    spoofer is assumed to have captured the receiver. The induced offset grows at ``rate_mps`` along ``direction`` up to ``max_offset_m``.
    """

    name = "gnss_spoofing"

    def __init__(
        self,
        direction: Sequence[float] = (1.0, 0.0, 0.0),
        rate_mps: float = 0.5,
        max_offset_m: float = 200.0,
        sensor: str = "gps",
        spoofer_pos: Optional[Sequence[float]] = None,
        spoofer_power_dbm: float = 10.0,
        capture_threshold_db: float = 2.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        d = np.asarray(direction, float)
        if np.linalg.norm(d) == 0:
            raise FaultError("gnss_spoofing: direction must be non-zero")
        self.direction = d / np.linalg.norm(d)
        self.rate, self.max_offset, self.sensor = float(rate_mps), float(max_offset_m), sensor
        self.spoofer_pos = None if spoofer_pos is None else np.asarray(spoofer_pos, float)
        self.spoofer_power_dbm, self.capture_threshold, self.seed = spoofer_power_dbm, capture_threshold_db, int(seed)
        self._filter: Optional[Callable[[Any], Any]] = None
        self.captured = True

    def on_start(self, sim: Any) -> None:
        s = self._sensor(sim, self.sensor)
        rows = self.targets(sim)
        B = sim.B
        t0 = float(sim.t)
        n = sim.dyn.n
        if self.spoofer_pos is not None:
            adv = spoofer_power_advantage_db(
                self.spoofer_pos, B.to_numpy(sim.state.pos), self.spoofer_power_dbm, authentic_dbm=GNSS_NOMINAL_POWER_DBM
            )
            p = capture_probability(adv, self.capture_threshold)
            captured = np.random.default_rng(self.seed).random(n) < p
        else:
            captured = np.ones(n, bool)
        mask = np.zeros(n, bool)
        mask[rows] = True
        mask &= captured
        self.captured = bool(mask.any())
        direction, rate, cap = self.direction, self.rate, self.max_offset

        def spoof(reading: Any) -> Any:
            t = float(reading.get("t", sim.t))
            off = float(pull_offset(t, t0, rate, cap))
            out = dict(reading)
            pos, vel = B.to_numpy(reading["pos"]).copy(), B.to_numpy(reading["vel"]).copy()
            pos = pos + np.where(mask[:, None], off * direction[None, :], 0.0)
            moving = off < cap
            vel = vel + np.where(mask[:, None], (rate if moving else 0.0) * direction[None, :], 0.0)
            out["pos"], out["vel"] = B.asarray(pos), B.asarray(vel)
            return out

        self._filter = spoof
        s.filters.append(spoof)

    def on_end(self, sim: Any) -> None:
        s = self._sensor(sim, self.sensor)
        if self._filter in s.filters:
            s.filters.remove(self._filter)

    def details(self) -> Any:
        return dict(sensor=self.sensor, rate_mps=self.rate, max_offset_m=self.max_offset, direction=self.direction.tolist(), captured=self.captured)


class NetworkJamming(Fault):
    """An interference source degrades the cellular link: its power adds to the interference at the UAV (needs a network with ``jammers``)."""

    name = "network_jamming"

    def __init__(
        self,
        pos: Sequence[float] = (0.0, 0.0, 2.0),
        power_dbm: float = 30.0,
        gain_dbi: float = 0.0,
        bandwidth_hz: Optional[float] = None,
        freq_hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.spec: Dict[str, Any] = dict(pos=list(map(float, pos)), power_dbm=float(power_dbm), gain_dbi=float(gain_dbi))
        if bandwidth_hz is not None:
            self.spec["bandwidth_hz"] = float(bandwidth_hz)
        if freq_hz is not None:
            self.spec["freq_hz"] = float(freq_hz)

    def on_start(self, sim: Any) -> None:
        if sim.net is None or not hasattr(sim.net, "jammers"):
            raise FaultError("network_jamming: the simulation needs a network that supports jammers (type: nr_cellular)")
        sim.net.jammers.append(self.spec)

    def on_end(self, sim: Any) -> None:
        if sim.net is not None and self.spec in sim.net.jammers:
            sim.net.jammers.remove(self.spec)

    def details(self) -> Any:
        return dict(self.spec)
