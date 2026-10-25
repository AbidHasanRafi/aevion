"""Inertial / navigation sensors: IMU, GPS, barometer, magnetometer.

Noise models follow the standard MEMS error model: additive white noise with a given
spectral density, bias that does a random walk, constant scale-factor error, bandwidth
limiting and saturation/quantisation. Defaults approximate a consumer-grade MEMS IMU.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import rotation as rot
from ..core.constants import altitude_from_pressure
from .allan import gauss_markov_avar
from .base import Sensor

DEG = math.pi / 180.0
G0 = 9.80665

#: representative orders of magnitude per grade (not a specific product): gyro ARW [deg/sqrt(h)], gyro bias instability [deg/h],
#: gyro rate random walk [deg/h/sqrt(h)], accel VRW [m/s/sqrt(h)], accel bias instability [micro-g], accel RRW [m/s^2/sqrt(h)]
IMU_GRADES = {
    "consumer": dict(gyro_arw=0.60, gyro_bi=30.0, gyro_rrw=20.0, accel_vrw=0.12, accel_bi=100.0, accel_rrw=1e-3),
    "industrial": dict(gyro_arw=0.30, gyro_bi=8.0, gyro_rrw=4.0, accel_vrw=0.05, accel_bi=40.0, accel_rrw=4e-4),
    "tactical": dict(gyro_arw=0.08, gyro_bi=1.5, gyro_rrw=0.5, accel_vrw=0.02, accel_bi=10.0, accel_rrw=1e-4),
    "navigation": dict(gyro_arw=0.004, gyro_bi=0.01, gyro_rrw=0.002, accel_vrw=0.002, accel_bi=2.0, accel_rrw=1e-5),
}


class _Flicker:
    """1/f ("flicker") bias noise as a sum of first-order Gauss-Markov processes with log-spaced correlation times.

    The amplitude is calibrated analytically so that the Allan-deviation *floor* of the sum equals ``floor`` (what datasheets quote as the
    in-run bias instability). Vectorised over ``(n, 3)`` axes.
    """

    def __init__(self, B, n: int, dt: float, floor: float, tau_range=(2.0, 3600.0), per_decade: float = 1.5, gen=None):
        n_proc = max(2, round(per_decade * math.log10(tau_range[1] / tau_range[0])) + 1)
        self.tau_c = np.logspace(math.log10(tau_range[0]), math.log10(tau_range[1]), n_proc)
        if tau_range[1] < 16.0 * tau_range[0]:
            raise ValueError("bias_tau_range must span at least a factor of 16 for the 1/f model")
        grid = np.logspace(
            math.log10(2.0 * tau_range[0]), math.log10(tau_range[1] / 2.0), 200
        )  # the flat plateau inside the range (the ends roll off)
        unit = np.sum([gauss_markov_avar(grid, 1.0, tc) for tc in self.tau_c], axis=0)
        self.var = floor**2 / float(unit.min())  # per-process stationary variance
        self.phi = np.exp(-dt / self.tau_c)  # (K,)
        self.sd_drive = math.sqrt(self.var) * np.sqrt(1.0 - self.phi**2)
        self.sd0 = math.sqrt(self.var)
        self.B, self.gen = B, gen
        self.state = B.randn((n, 3, n_proc), gen) * self.sd0
        self._phi, self._drive = B.asarray(self.phi), B.asarray(self.sd_drive)

    def step(self, scale: float = 1.0):
        w = self.B.randn(self.state.shape, self.gen)
        self.state = self.state * self._phi + w * self._drive * scale
        return self.state.sum(-1)


class IMU(Sensor):
    """6-axis IMU. Output: ``accel`` (specific force, body, m/s²) and ``gyro`` (rad/s).

    Error model (all terms optional beyond the original white noise + bias random walk + turn-on bias + scale factor):

    * white noise from the noise densities (angle / velocity random walk), bias **random walk** (rate random walk) and a constant
      **turn-on bias**;
    * 1/f **bias instability** (``accel_bias_instability`` / ``gyro_bias_instability``, the Allan-deviation floor in m/s² / rad/s);
    * rotor-synchronous **vibration**: each rotor adds a sinusoid at its rotation frequency (sampled, so it aliases like on hardware)
      whose amplitude scales with the square of its speed relative to hover; ``accel_vibration`` / ``gyro_vibration`` are the amplitudes at
      hover per axis;
    * **thermal drift** of the bias (``*_bias_tempco`` per kelvin) driven by ``temperature_c`` and an exponential warm-up;
    * gyro **g-sensitivity**, random axis **misalignment**, and ADC **quantisation**.

    :meth:`from_allan` and :meth:`from_grade` build a sensor from datasheet-style Allan-variance numbers.
    """

    name = "imu"

    def __init__(
        self,
        rate=200.0,
        accel_noise_density=2.0e-3,  # m/s²/√Hz  (≈200 µg/√Hz)
        gyro_noise_density=2.4e-4,  # rad/s/√Hz (≈0.014 °/s/√Hz)
        accel_bias_walk=3.0e-4,  # m/s²/√s
        gyro_bias_walk=2.0e-5,  # rad/s/√s
        accel_bias_init=0.05,  # 1σ turn-on bias [m/s²]
        gyro_bias_init=0.002,  # 1σ turn-on bias [rad/s]
        scale_factor_std=0.002,
        accel_range=16 * 9.80665,
        gyro_range=34.9,  # ≈2000 °/s
        bandwidth=None,  # Hz, first-order low-pass (None = off)
        ideal=False,
        accel_bias_instability=0.0,  # Allan floor [m/s²] of the 1/f bias term
        gyro_bias_instability=0.0,  # Allan floor [rad/s]
        bias_tau_range=(2.0, 3600.0),  # correlation times spanned by the 1/f model [s]
        accel_vibration=(0.0, 0.0, 0.0),  # m/s² amplitude per axis at hover rotor speed
        gyro_vibration=(0.0, 0.0, 0.0),  # rad/s amplitude per axis at hover rotor speed
        vibration_harmonics=1,  # also add the 2nd, 3rd ... multiple of the rotor frequency (amplitude 1/h)
        temperature_c=25.0,  # ambient temperature
        warmup_c=0.0,  # self-heating rise at steady state
        warmup_tau=120.0,  # warm-up time constant [s]
        accel_bias_tempco=0.0,  # m/s² per K away from 25 C
        gyro_bias_tempco=0.0,  # rad/s per K away from 25 C
        gyro_g_sensitivity=0.0,  # rad/s per m/s² of specific force (isotropic)
        misalignment_std_deg=0.0,  # 1σ random axis misalignment of the accelerometer and gyro triads
        accel_resolution=0.0,  # quantisation step [m/s²] (0 = off)
        gyro_resolution=0.0,  # quantisation step [rad/s]
        **kw,
    ):
        super().__init__(rate=rate, **kw)
        self.cfg = dict(
            an=accel_noise_density,
            gn=gyro_noise_density,
            abw=accel_bias_walk,
            gbw=gyro_bias_walk,
            ab0=accel_bias_init,
            gb0=gyro_bias_init,
            sf=scale_factor_std,
            ar=accel_range,
            gr=gyro_range,
        )
        self.bandwidth, self.ideal = bandwidth, ideal
        self.bi = (float(accel_bias_instability), float(gyro_bias_instability))
        self.bias_tau_range = tuple(float(x) for x in bias_tau_range)
        self.vib = (np.asarray(accel_vibration, float).reshape(3), np.asarray(gyro_vibration, float).reshape(3))
        self.vib_harmonics = max(1, int(vibration_harmonics))
        self.thermal = dict(
            ambient=float(temperature_c), warmup=float(warmup_c), tau=float(warmup_tau), atc=float(accel_bias_tempco), gtc=float(gyro_bias_tempco)
        )
        self.g_sens, self.misalign_deg = float(gyro_g_sensitivity), float(misalignment_std_deg)
        self.res = (float(accel_resolution), float(gyro_resolution))
        for name, v in (
            ("accel_bias_instability", self.bi[0]),
            ("gyro_bias_instability", self.bi[1]),
            ("warmup_tau", warmup_tau),
            ("misalignment_std_deg", misalignment_std_deg),
        ):
            if v < 0 or (name == "warmup_tau" and v == 0):
                raise ValueError(f"{name} must be {'positive' if name == 'warmup_tau' else 'non-negative'}")

    # ------------------------------------------------------------------------------------------ builders
    @classmethod
    def from_allan(cls, gyro_arw=0.3, gyro_bias_instability=8.0, gyro_rrw=0.0, accel_vrw=0.05, accel_bias_instability=40.0, accel_rrw=0.0, **kw):
        """Build an IMU from datasheet-style Allan-variance figures.

        ``gyro_arw`` [deg/√h] angle random walk, ``gyro_bias_instability`` [deg/h] (Allan floor), ``gyro_rrw`` [deg/h/√h] rate random walk;
        ``accel_vrw`` [m/s/√h] velocity random walk, ``accel_bias_instability`` [µg] (Allan floor), ``accel_rrw`` [m/s²/√h].
        """
        return cls(
            gyro_noise_density=gyro_arw * DEG / 60.0,
            gyro_bias_instability=gyro_bias_instability * DEG / 3600.0,
            gyro_bias_walk=gyro_rrw * DEG / 3600.0 / 60.0,
            accel_noise_density=accel_vrw / 60.0,
            accel_bias_instability=accel_bias_instability * 1e-6 * G0,
            accel_bias_walk=accel_rrw / 60.0,
            **kw,
        )

    @classmethod
    def from_grade(cls, grade: str = "industrial", **kw):
        """``consumer``, ``industrial``, ``tactical`` or ``navigation``: representative Allan figures of that class of sensor."""
        if grade not in IMU_GRADES:
            raise KeyError(f"unknown IMU grade {grade!r}; choose from {sorted(IMU_GRADES)}")
        g = IMU_GRADES[grade]
        return cls.from_allan(g["gyro_arw"], g["gyro_bi"], g["gyro_rrw"], g["accel_vrw"], g["accel_bi"], g["accel_rrw"], **kw)

    # ------------------------------------------------------------------------------------------ simulation
    def on_bind(self):
        _B, n, c = self.B, self.n, self.cfg
        z = 0.0 if self.ideal else 1.0
        self.ab = self.noise((n, 3), c["ab0"] * z)
        self.gb = self.noise((n, 3), c["gb0"] * z)
        self.asf = 1.0 + self.noise((n, 3), c["sf"] * z)
        self.gsf = 1.0 + self.noise((n, 3), c["sf"] * z)
        self._lp = None
        self._flk = None
        if not self.ideal and (self.bi[0] > 0 or self.bi[1] > 0):
            self._flk = (
                _Flicker(_B, n, self.dt, self.bi[0], self.bias_tau_range, gen=self.gen) if self.bi[0] > 0 else None,
                _Flicker(_B, n, self.dt, self.bi[1], self.bias_tau_range, gen=self.gen) if self.bi[1] > 0 else None,
            )
        self._vib_phase = None
        self._vib_on = not self.ideal and bool(np.any(self.vib[0]) or np.any(self.vib[1]))
        self._rot = None
        self._misalign = None
        if not self.ideal and self.misalign_deg > 0:

            def triad():
                a = _B.to_numpy(self.noise((n, 3), self.misalign_deg * math.pi / 180.0))
                m = np.tile(np.eye(3), (n, 1, 1))
                m[:, 0, 1], m[:, 0, 2], m[:, 1, 0], m[:, 1, 2], m[:, 2, 0], m[:, 2, 1] = -a[:, 2], a[:, 1], a[:, 2], -a[:, 0], -a[:, 1], a[:, 0]
                return _B.asarray(m)

            self._misalign = (triad(), triad())
        self._t0 = None

    def _vibration(self, state):
        """Rotor-synchronous accel and gyro vibration at the sampling instant (n, 3) each."""
        B, n = self.B, self.n
        w = state.rotor  # (n, M) rad/s
        M = w.shape[-1]
        if self._vib_phase is None:
            self._vib_phase = B.rand((n, M), self.gen) * (2.0 * math.pi)
            self._hover = float(self.sim.hover_speed()) if hasattr(self.sim, "hover_speed") else float(B.to_numpy(w).mean())
            self._axis_phase = B.rand((n, M, 6), self.gen) * (2.0 * math.pi)
        self._vib_phase = (self._vib_phase + w * self.dt) % (2.0 * math.pi)
        rel2 = (w / max(self._hover, 1e-9)) ** 2  # (n, M) amplitude factor ~ thrust ~ w^2
        acc = B.zeros(n, 3)
        gyr = B.zeros(n, 3)
        scale_m = 1.0 / math.sqrt(M)  # M rotors add incoherently: keep the total amplitude near the specified one
        for h in range(1, self.vib_harmonics + 1):
            wave = B.sin(h * self._vib_phase[..., None] + self._axis_phase)  # (n, M, 6)
            term = (rel2[..., None] * wave).sum(1) * (scale_m / h)  # (n, 6)
            acc = acc + term[:, :3] * B.asarray(self.vib[0])
            gyr = gyr + term[:, 3:] * B.asarray(self.vib[1])
        return acc, gyr

    def measure(self, state):
        B, c, dt = self.B, self.cfg, self.dt
        if self.ideal:
            return dict(accel=state.f_body, gyro=state.omega, t=state.t)
        sq = math.sqrt(dt)
        self.ab = self.ab + self.noise((self.n, 3), c["abw"] * sq)
        self.gb = self.gb + self.noise((self.n, 3), c["gbw"] * sq)
        ab, gb = self.ab, self.gb
        if self._flk is not None:
            if self._flk[0] is not None:
                ab = ab + self._flk[0].step(self.noise_scale)
            if self._flk[1] is not None:
                gb = gb + self._flk[1].step(self.noise_scale)
        th = self.thermal
        if th["atc"] or th["gtc"]:
            if self._t0 is None:
                self._t0 = state.t
            temp = th["ambient"] + th["warmup"] * (1.0 - math.exp(-(state.t - self._t0) / th["tau"])) - 25.0
            ab = ab + th["atc"] * temp
            gb = gb + th["gtc"] * temp
        if self.g_sens:
            gb = gb + self.g_sens * state.f_body
        f, w = state.f_body, state.omega
        if self._misalign is not None:
            f = B.einsum("nij,nj->ni", self._misalign[0], f)
            w = B.einsum("nij,nj->ni", self._misalign[1], w)
        acc = f * self.asf + ab + self.noise((self.n, 3), c["an"] / sq)
        gyr = w * self.gsf + gb + self.noise((self.n, 3), c["gn"] / sq)
        if self._vib_on:
            va, vg = self._vibration(state)
            acc, gyr = acc + va, gyr + vg
        if self.bandwidth:
            a = 1.0 - math.exp(-2 * math.pi * self.bandwidth * dt)
            if self._lp is None:
                self._lp = (acc, gyr)
            else:
                self._lp = (self._lp[0] + a * (acc - self._lp[0]), self._lp[1] + a * (gyr - self._lp[1]))
            acc, gyr = self._lp
        acc = B.clip(acc, -c["ar"], c["ar"])
        gyr = B.clip(gyr, -c["gr"], c["gr"])
        if self.res[0] > 0:
            acc = B.floor(acc / self.res[0] + 0.5) * self.res[0]
        if self.res[1] > 0:
            gyr = B.floor(gyr / self.res[1] + 0.5) * self.res[1]
        return dict(accel=acc, gyro=gyr, t=state.t)

    def to_dict(self):
        d = {**super().to_dict(), **self.cfg, "bandwidth": self.bandwidth, "ideal": self.ideal}
        if any(self.bi):
            d.update(accel_bias_instability=self.bi[0], gyro_bias_instability=self.bi[1])
        if self._vib_on or np.any(self.vib[0]) or np.any(self.vib[1]):
            d.update(accel_vibration=self.vib[0].tolist(), gyro_vibration=self.vib[1].tolist())
        if self.thermal["atc"] or self.thermal["gtc"]:
            d.update(
                temperature_c=self.thermal["ambient"],
                warmup_c=self.thermal["warmup"],
                accel_bias_tempco=self.thermal["atc"],
                gyro_bias_tempco=self.thermal["gtc"],
            )
        return d


class GPS(Sensor):
    """GNSS receiver in a local ENU frame. Position error is a first-order Gauss-Markov process
    (slowly wandering, as real GNSS errors do) plus white noise."""

    name = "gps"

    def __init__(self, rate=10.0, latency=0.1, sigma_h=0.8, sigma_v=1.5, tau=60.0, white_h=0.15, white_v=0.3, vel_sigma=0.08, ideal=False, **kw):
        super().__init__(rate=rate, latency=latency, **kw)
        self.sig = np.array([sigma_h, sigma_h, sigma_v])
        self.white = np.array([white_h, white_h, white_v])
        self.tau, self.vel_sigma, self.ideal = tau, vel_sigma, ideal

    def on_bind(self):
        z = 0.0 if self.ideal else 1.0
        self.err = self.noise((self.n, 3), 1.0) * self.B.asarray(self.sig * z)

    def measure(self, state):
        B = self.B
        if self.ideal:
            return dict(pos=state.pos, vel=state.vel, t=state.t)
        k = math.exp(-self.dt / self.tau)
        self.err = self.err * k + self.noise((self.n, 3), 1.0) * B.asarray(self.sig * math.sqrt(1 - k * k))
        pos = state.pos + self.err + self.noise((self.n, 3), 1.0) * B.asarray(self.white)
        vel = state.vel + self.noise((self.n, 3), self.vel_sigma)
        return dict(pos=pos, vel=vel, t=state.t)


class Barometer(Sensor):
    """Static-pressure altimeter with noise and random-walk drift. Output pressure [Pa] and altitude [m]."""

    name = "baro"

    def __init__(self, rate=25.0, noise_pa=6.0, drift_pa=0.4, ideal=False, **kw):
        super().__init__(rate=rate, **kw)
        self.noise_pa, self.drift_pa, self.ideal = noise_pa, drift_pa, ideal

    def on_bind(self):
        self.drift = self.B.zeros(self.n)

    def measure(self, state):
        B = self.B
        atm = self.sim.world.atmosphere
        p = atm.pressure(B, state.pos[..., 2])
        if not self.ideal:
            self.drift = self.drift + self.noise((self.n,), self.drift_pa * math.sqrt(self.dt))
            p = p + self.drift + self.noise((self.n,), self.noise_pa)
        alt = altitude_from_pressure(p) - atm.base_altitude
        return dict(pressure=p, altitude=alt, t=state.t)


class Magnetometer(Sensor):
    """3-axis magnetometer. Earth field of 50 µT with configurable inclination/declination."""

    name = "mag"

    def __init__(
        self, rate=50.0, field_ut=50.0, inclination_deg=60.0, declination_deg=0.0, noise_ut=0.4, hard_iron_ut=2.0, ideal=False, model=None, **kw
    ):
        super().__init__(rate=rate, **kw)
        self.model_spec = model  # None: the uniform field above; "dipole" or a MagneticModel: the field at the vehicle's geodetic position
        inc, dec = math.radians(inclination_deg), math.radians(declination_deg)
        # ENU: north = +y rotated by declination toward east; down component negative z
        h = math.cos(inc)
        self.field_world = field_ut * np.array([h * math.sin(dec), h * math.cos(dec), -math.sin(inc)])
        self.noise_ut, self.hard_iron_ut, self.ideal = noise_ut, hard_iron_ut, ideal

    def on_bind(self):
        self.bias = self.noise((self.n, 3), 0.0 if self.ideal else self.hard_iron_ut)
        self._model = None
        if self.model_spec is not None:
            from ..environments.geomagnetic import FixedField, MagneticModel, TiltedDipole

            spec = self.model_spec
            if isinstance(spec, MagneticModel):
                self._model = spec
            elif spec == "dipole" or (isinstance(spec, dict) and spec.get("type") == "dipole"):
                self._model = TiltedDipole(**({k: v for k, v in spec.items() if k != "type"} if isinstance(spec, dict) else {}))
            elif spec == "fixed" or (isinstance(spec, dict) and spec.get("type") == "fixed"):
                self._model = FixedField(**({k: v for k, v in spec.items() if k != "type"} if isinstance(spec, dict) else {}))
            else:
                raise ValueError(f"unknown magnetic model {spec!r}; use 'dipole', 'fixed' or a MagneticModel")
            w = self.sim.world
            from ..environments.geodesy import LocalFrame

            self._frame = w.origin or LocalFrame(45.0, 0.0, 0.0)
            from ..environments.solar import _to_unix

            self._year = 1970.0 + (float(_to_unix(w.utc)) / 31_556_952.0 if w.utc else 56.5)

    def measure(self, state):
        B = self.B
        if self._model is not None:
            lla = self._frame.to_lla(B.to_numpy(state.pos))
            ned_field = self._model.field_enu(lla[:, 0], lla[:, 1], lla[:, 2], self._year)
            m = rot.rotate_inv(B, state.quat, B.asarray(ned_field))
            if not self.ideal:
                m = m + self.bias + self.noise((self.n, 3), self.noise_ut)
            return dict(field=m, t=state.t)
        m = rot.rotate_inv(B, state.quat, B.ones(self.n, 3) * B.asarray(self.field_world))
        if not self.ideal:
            m = m + self.bias + self.noise((self.n, 3), self.noise_ut)
        return dict(field=m, t=state.t)
