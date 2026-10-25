"""Air-data and flow sensors: pitot tube and downward optical flow.

* :class:`PitotTube` measures the differential pressure ``dp = 1/2 rho V^2 C(alpha)`` along its axis from the relative air velocity
  (vehicle velocity minus the wind at the vehicle), with a cosine-squared loss for off-axis flow, a total loss beyond the acceptance
  cone, a first-order pneumatic lag, noise, drift and an optional icing/blockage fault. It reports indicated (sea-level density) and
  true airspeed (local density, which follows weather and altitude).
* :class:`OpticalFlow` is a downward-looking flow sensor (PX4Flow-style). Its raw output is the angular rate of the line of sight to
  the ground, ``-(n x v)/range - omega_perp``; the gyro-compensated output isolates the translational part, so ``flow_comp * range``
  is the horizontal velocity. Quality drops with height, speed and darkness, and invalid readings are reported rather than guessed.

Fidelity: the pitot model is algebraic plus a first-order lag (no compressibility correction; error below about 1 % under 100 m/s). The
flow model assumes a textured, locally flat ground patch and a perfectly stabilised mount.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from ..core import rotation as rot
from .base import Sensor

__all__ = ["PitotTube", "OpticalFlow", "RHO_SEA_LEVEL"]

RHO_SEA_LEVEL = 1.225


class PitotTube(Sensor):
    """Differential-pressure airspeed sensor. Output ``diff_pressure`` [Pa], ``ias`` and ``tas`` [m/s] (signed along the tube axis)."""

    name = "pitot"

    def __init__(
        self,
        rate: float = 50.0,
        axis: Sequence[float] = (1.0, 0.0, 0.0),
        noise_pa: float = 1.5,
        drift_pa: float = 0.05,
        lag_s: float = 0.05,
        acceptance_deg: float = 20.0,
        cutoff_deg: float = 45.0,
        ideal: bool = False,
        **kw: Any,
    ):
        super().__init__(rate=rate, **kw)
        a = np.asarray(axis, float)
        if a.shape != (3,) or np.linalg.norm(a) == 0:
            raise ValueError("axis must be a non-zero 3-vector")
        if not 0.0 < acceptance_deg < cutoff_deg <= 90.0 or lag_s < 0 or noise_pa < 0:
            raise ValueError("need 0 < acceptance_deg < cutoff_deg <= 90, lag_s >= 0 and noise_pa >= 0")
        self.axis = a / np.linalg.norm(a)
        if self.extrinsics is not None:
            self.axis = self.extrinsics.rotate(self.axis)
        self.noise_pa, self.drift_pa, self.lag_s, self.ideal = float(noise_pa), float(drift_pa), float(lag_s), bool(ideal)
        self.accept, self.cutoff = math.radians(acceptance_deg), math.radians(cutoff_deg)
        #: set to 1.0 by a blockage fault: the tube is iced over and the reading decays towards zero (per-vehicle array allowed)
        self.blockage = 0.0

    def on_bind(self) -> None:
        B = self.B
        self.drift = B.zeros(self.n)
        self._dp = None

    def acceptance(self, angle: Any) -> Any:
        """Pressure recovery ``C(alpha)``: ``cos^2`` inside the acceptance cone, falling linearly to zero at the cut-off angle."""
        B = self.B
        c2 = B.cos(angle) ** 2
        ramp = B.clip((self.cutoff - B.abs(angle)) / (self.cutoff - self.accept), 0.0, 1.0)
        return B.where(B.abs(angle) <= self.accept, c2, c2 * ramp)

    def measure(self, state: Any) -> dict:
        B = self.B
        v_air = rot.rotate_inv(B, state.quat, state.vel - state.wind)  # air velocity relative to the vehicle, body frame (sign: vehicle moves)
        ax = B.asarray(self.axis)
        along = B.sum(v_air * ax, axis=-1)
        speed = B.sqrt(B.sum(v_air * v_air, axis=-1) + 1e-12)
        perp = B.sqrt(B.maximum(speed * speed - along * along, speed * 0.0))
        angle = B.atan2(perp, B.abs(along))  # off-axis angle in [0, 90 deg]: flow from behind the tube is treated like flow from the front
        rho = self.sim.world.atmosphere.density(B, state.pos[..., 2])
        dp = 0.5 * rho * speed * speed * self.acceptance(angle) * B.sign(along)
        if self.ideal:
            dpm = dp
        else:
            if self._dp is None:
                self._dp = dp
            a = 1.0 - math.exp(-self.dt / self.lag_s) if self.lag_s > 0 else 1.0
            self._dp = self._dp + a * (dp - self._dp)
            self.drift = self.drift + self.noise((self.n,), self.drift_pa * math.sqrt(self.dt))
            dpm = self._dp * (1.0 - B.asarray(self.blockage)) + self.drift + self.noise((self.n,), self.noise_pa)
        sign = B.sign(dpm)
        mag = B.abs(dpm)
        return dict(diff_pressure=dpm, ias=sign * B.sqrt(2.0 * mag / RHO_SEA_LEVEL), tas=sign * B.sqrt(2.0 * mag / rho), t=state.t)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "axis": self.axis.tolist(),
            "noise_pa": self.noise_pa,
            "lag_s": self.lag_s,
            "acceptance_deg": math.degrees(self.accept),
        }


class OpticalFlow(Sensor):
    """Downward optical-flow sensor. Output: ``flow`` (raw line-of-sight rate, rad/s, body x/y), ``flow_comp`` (gyro-compensated),
    ``range`` [m], ``quality`` (0-255) and ``valid``."""

    name = "flow"

    def __init__(
        self,
        rate: float = 50.0,
        min_range: float = 0.2,
        max_range: float = 15.0,
        max_flow: float = 4.0,
        noise: float = 0.01,
        noise_rel: float = 0.01,
        gyro_noise: float = 0.002,
        min_sun_elevation_deg: float = -3.0,
        illuminated: bool = False,
        **kw: Any,
    ):
        super().__init__(rate=rate, **kw)
        if min_range <= 0 or max_range <= min_range or max_flow <= 0:
            raise ValueError("need 0 < min_range < max_range and max_flow > 0")
        self.min_range, self.max_range, self.max_flow = float(min_range), float(max_range), float(max_flow)
        self.noise_abs, self.noise_rel, self.gyro_noise = float(noise), float(noise_rel), float(gyro_noise)
        self.min_sun, self.illuminated = float(min_sun_elevation_deg), bool(illuminated)
        self._n_body = np.array([0.0, 0.0, -1.0])
        if self.extrinsics is not None:
            self._n_body = self.extrinsics.rotate(self._n_body)

    def _dark(self, t: float) -> bool:
        """Is it too dark for the sensor? Uses the world's UTC time and origin when set; otherwise assumes daylight."""
        w = self.sim.world
        if self.illuminated or w.utc is None or w.origin is None:
            return False
        from ..environments.solar import sun_position

        return float(sun_position(w.origin.lat, w.origin.lon, _add_seconds(w.utc, t))["elevation_deg"]) < self.min_sun

    def measure(self, state: Any) -> dict:
        B, n = self.B, self.n
        nb = B.asarray(self._n_body)
        n_world = rot.rotate(B, state.quat, B.ones(n, 3) * nb)
        from .raycast import cast_rays

        o = B.to_numpy(state.pos)
        d = B.to_numpy(n_world)
        hit = cast_rays(self.sim.world, o, d, self.max_range * 1.5)
        rng = B.asarray(np.asarray(hit.t, float))
        v_b = rot.rotate_inv(B, state.quat, state.vel)
        w = state.omega
        # line-of-sight rate of the ground point under the sensor: -(n x v)/rho - omega_perp
        cross = B.stack(
            [nb[1] * v_b[..., 2] - nb[2] * v_b[..., 1], nb[2] * v_b[..., 0] - nb[0] * v_b[..., 2], nb[0] * v_b[..., 1] - nb[1] * v_b[..., 0]], -1
        )
        w_perp = w - B.sum(w * nb, axis=-1, keepdims=True) * nb
        flow3 = -cross / B.maximum(rng, rng * 0.0 + 1e-3)[..., None] - w_perp
        flow = flow3[..., :2]
        w_meas = w + self.noise((n, 3), self.gyro_noise)
        comp3 = flow3 + (w_meas - B.sum(w_meas * nb, axis=-1, keepdims=True) * nb)
        flow_n = flow + self.noise((n, 2), 1.0) * (self.noise_abs + self.noise_rel * B.abs(flow))
        comp_n = comp3[..., :2] + self.noise((n, 2), 1.0) * (self.noise_abs + self.noise_rel * B.abs(flow))
        mag = B.sqrt(B.sum(flow * flow, axis=-1))
        got = np.asarray(hit.prim != -1)
        ok = (
            B.asarray(got.astype(float))
            * (rng >= self.min_range)
            * (rng <= self.max_range)
            * (mag <= self.max_flow)
            * (0.0 if self._dark(float(state.t)) else 1.0)
        )
        quality = 255.0 * ok * B.clip(1.0 - mag / self.max_flow, 0.1, 1.0) * B.clip(1.0 - (rng / self.max_range) ** 2, 0.1, 1.0)
        valid = ok > 0.5
        z = flow_n * 0.0
        return dict(
            flow=B.where(valid[..., None], flow_n, z),
            flow_comp=B.where(valid[..., None], comp_n, z),
            range=rng,
            quality=quality,
            valid=valid,
            t=state.t,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "min_range": self.min_range, "max_range": self.max_range, "max_flow": self.max_flow}


def _add_seconds(utc: str, t: float) -> Any:
    from ..environments.solar import _to_unix

    return float(_to_unix(utc)) + float(t)
