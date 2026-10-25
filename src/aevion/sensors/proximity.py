"""Short-range sensors: ultrasonic rangefinder and UWB two-way ranging, plus a multilateration helper.

* :class:`Ultrasonic` has a wide beam: it returns the nearest echo inside a cone (sampled by a ring of rays), loses echoes from surfaces
  that are too oblique, and converts time of flight with an assumed speed of sound, so temperature biases it unless
  ``temperature_compensated``.
* :class:`UwbRanging` measures distance to fixed anchors with centimetre noise. Anchors without line of sight read long (exponentially
  distributed excess delay), a small fraction of readings are outliers, and each anchor has a fixed antenna-delay offset.
* :func:`multilaterate` solves a position from anchor ranges by (weighted) Gauss-Newton.

Fidelity: ultrasonic returns are geometric (no sidelobes or cross-talk); UWB uses a statistical NLoS model, not a ray-traced channel
impulse response.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core import rotation as rot
from .base import Sensor
from .raycast import cast_rays

__all__ = ["Ultrasonic", "UwbRanging", "multilaterate", "SPEED_OF_SOUND_ASSUMED"]

SPEED_OF_SOUND_ASSUMED = 343.0  # m/s at 20 C: what a typical module's firmware assumes


class Ultrasonic(Sensor):
    """Cone-beam ultrasonic rangefinder (default: pointing along body +x). Output ``range`` [m] and ``valid``."""

    name = "ultrasonic"

    def __init__(
        self,
        rate: float = 20.0,
        direction: Sequence[float] = (1.0, 0.0, 0.0),
        half_angle_deg: float = 15.0,
        min_range: float = 0.2,
        max_range: float = 5.0,
        max_incidence_deg: float = 40.0,
        noise_std: float = 0.002,
        noise_rel: float = 0.001,
        resolution: float = 0.001,
        temperature_compensated: bool = False,
        assumed_speed: float = SPEED_OF_SOUND_ASSUMED,
        rays: int = 8,
        **kw: Any,
    ):
        super().__init__(rate=rate, **kw)
        d = np.asarray(direction, float)
        if d.shape != (3,) or np.linalg.norm(d) == 0:
            raise ValueError("direction must be a non-zero 3-vector")
        if not 0 < half_angle_deg < 90 or min_range <= 0 or max_range <= min_range or rays < 3:
            raise ValueError("need 0 < half_angle_deg < 90, 0 < min_range < max_range and rays >= 3")
        axis = d / np.linalg.norm(d)
        if self.extrinsics is not None:
            axis = self.extrinsics.rotate(axis)
        self.axis, self.half = axis, math.radians(half_angle_deg)
        self.min_range, self.max_range = float(min_range), float(max_range)
        self.max_incidence = math.radians(max_incidence_deg)
        self.noise_std, self.noise_rel, self.resolution = float(noise_std), float(noise_rel), float(resolution)
        self.compensated, self.assumed = bool(temperature_compensated), float(assumed_speed)
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ang = np.linspace(0.0, 2.0 * np.pi, rays, endpoint=False)
        ring = axis * math.cos(self.half) + math.sin(self.half) * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
        self.dirs_body = np.vstack([axis[None, :], ring])  # centre ray + a ring on the cone's edge

    def measure(self, state: Any) -> dict:
        B, n = self.B, self.n
        R = B.to_numpy(rot.to_matrix(B, state.quat))
        pos = B.to_numpy(state.pos)
        dw = np.einsum("nij,kj->nki", R, self.dirs_body)  # (n, K, 3)
        K = dw.shape[1]
        o = np.repeat(pos[:, None, :], K, axis=1).reshape(-1, 3)
        h = cast_rays(self.sim.world, o, dw.reshape(-1, 3), self.max_range)
        t = np.asarray(h.t).reshape(n, K)
        cosinc = np.clip(-np.sum(np.asarray(h.normal).reshape(n, K, 3) * dw, axis=-1), 0.0, 1.0)
        echo = (np.asarray(h.prim).reshape(n, K) != -1) & (t >= self.min_range) & (np.arccos(cosinc) <= self.max_incidence)
        rng = np.where(echo, t, np.inf).min(axis=1)  # the nearest strong echo wins
        valid = np.isfinite(rng)
        scale: Any
        if self.compensated:
            scale = 1.0
        else:
            c_true = np.asarray(B.to_numpy(self.sim.world.atmosphere.speed_of_sound(B, state.pos[..., 2])))
            scale = self.assumed / c_true  # time of flight is set by the true c; the firmware divides by the assumed one
        noisy = rng * scale + B.to_numpy(self.noise((n,), 1.0)) * (self.noise_std + self.noise_rel * np.where(valid, rng, 0.0))
        if self.resolution > 0:
            noisy = np.round(noisy / self.resolution) * self.resolution
        out = np.where(valid, noisy, self.max_range)
        return dict(range=B.asarray(out), valid=B.asarray(valid), t=state.t)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "half_angle_deg": math.degrees(self.half),
            "max_range": self.max_range,
            "temperature_compensated": self.compensated,
        }


class UwbRanging(Sensor):
    """Two-way ranging to fixed anchors. Output ``ranges`` ``(N, A)`` [m], ``valid`` ``(N, A)`` and ``nlos`` ``(N, A)`` (ground truth, for analysis)."""

    name = "uwb"

    def __init__(
        self,
        anchors: Sequence[Sequence[float]] = ((0.0, 0.0, 2.0), (20.0, 0.0, 2.0), (20.0, 20.0, 2.0), (0.0, 20.0, 2.0)),
        rate: float = 10.0,
        noise_std: float = 0.04,
        max_range: float = 60.0,
        nlos_excess_mean: float = 0.6,
        nlos_noise_std: float = 0.15,
        outlier_probability: float = 0.01,
        outlier_max: float = 4.0,
        antenna_delay_std: float = 0.03,
        use_world_los: bool = True,
        **kw: Any,
    ):
        super().__init__(rate=rate, **kw)
        a = np.asarray(anchors, float)
        if a.ndim != 2 or a.shape[1] != 3 or len(a) < 1:
            raise ValueError("anchors must be a non-empty (A, 3) list")
        if noise_std < 0 or not 0.0 <= outlier_probability <= 1.0 or max_range <= 0:
            raise ValueError("invalid noise_std, outlier_probability or max_range")
        self.anchors, self.noise_std, self.max_range = a, float(noise_std), float(max_range)
        self.nlos_mean, self.nlos_noise = float(nlos_excess_mean), float(nlos_noise_std)
        self.p_out, self.out_max, self.delay_std, self.use_los = (
            float(outlier_probability),
            float(outlier_max),
            float(antenna_delay_std),
            bool(use_world_los),
        )

    def on_bind(self) -> None:
        self.delay = self.B.to_numpy(self.noise((1, len(self.anchors)), self.delay_std))  # fixed per-anchor calibration error

    def measure(self, state: Any) -> dict:
        B, n = self.B, self.n
        pos = B.to_numpy(state.pos)
        A = len(self.anchors)
        d = np.linalg.norm(pos[:, None, :] - self.anchors[None, :, :], axis=-1)
        nlos = np.zeros((n, A), bool)
        world = self.sim.world
        if self.use_los and world.has_obstacles:
            los = world.line_of_sight(np.repeat(pos, A, axis=0), np.tile(self.anchors, (n, 1)))
            nlos = ~np.asarray(los).reshape(n, A)
        g = B.to_numpy(self.noise((n, A), 1.0))
        excess = nlos * (
            -self.nlos_mean * np.log(np.clip(B.to_numpy(B.rand((n, A), self.gen)), 1e-12, 1.0))
            + self.nlos_noise * B.to_numpy(self.noise((n, A), 1.0))
        )
        out = (B.to_numpy(B.rand((n, A), self.gen)) < self.p_out) * B.to_numpy(B.rand((n, A), self.gen)) * self.out_max
        r = d + self.delay + self.noise_std * g + np.abs(excess) + out
        valid = d <= self.max_range
        return dict(ranges=B.asarray(np.where(valid, r, 0.0)), valid=B.asarray(valid), nlos=nlos, t=state.t)

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "anchors": self.anchors.tolist(), "noise_std": self.noise_std, "max_range": self.max_range}


def multilaterate(
    ranges: np.ndarray,
    anchors: np.ndarray,
    valid: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    iterations: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Position from anchor ranges by Gauss-Newton. ``ranges`` ``(N, A)``, ``anchors`` ``(A, 3)``; returns ``(positions (N, 3), n_used (N,))``
    (``nan`` rows where fewer than three usable anchors remain). With coplanar anchors the height is ambiguous in sign: pass ``x0`` on the
    correct side."""
    r = np.asarray(ranges, float)
    a = np.asarray(anchors, float)
    n, A = r.shape
    v = np.ones((n, A), bool) if valid is None else np.asarray(valid, bool)
    w = np.ones((n, A)) if weights is None else np.asarray(weights, float)
    x = np.tile(a.mean(0) + np.array([0.0, 0.0, 1.0]), (n, 1)) if x0 is None else np.array(np.broadcast_to(x0, (n, 3)), float)
    for _ in range(iterations):
        diff = x[:, None, :] - a[None]
        d = np.linalg.norm(diff, axis=-1)
        J = diff / np.maximum(d, 1e-9)[..., None]
        res = (r - d) * (w * v)
        Jw = J * (w * v)[..., None]
        H = np.einsum("nai,naj->nij", Jw, J) + 1e-9 * np.eye(3)
        dx = np.linalg.solve(H, np.einsum("nai,na->ni", Jw, res)[..., None])[..., 0]
        x = x + dx
        if np.max(np.abs(dx)) < 1e-9:
            break
    used = v.sum(1)
    x[used < 3] = np.nan
    return x, used
