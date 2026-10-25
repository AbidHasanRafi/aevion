"""UAV attitude-dependent antenna gain.

A UAV pitches and rolls to fly, which rotates its antennas relative to the ground. The pattern is evaluated in the
**antenna frame**: world direction -> body frame (``R_wb^T``, body is FLU, quaternions are ``[w, x, y, z]`` body-to-world) ->
antenna frame (``R_ba^T``, the fixed mounting rotation).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .patterns import AntennaPattern, HalfWaveDipole, Patch3GPP

__all__ = ["UavAntenna", "direction_in_antenna_frame", "quat_to_matrices", "rpy_to_matrix"]


def quat_to_matrices(q: Any) -> np.ndarray:
    """Batch rotation matrices ``(..., 3, 3)`` for quaternions ``(..., 4)`` ``[w, x, y, z]`` (normalised internally)."""
    q = np.asarray(q, float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0], r[..., 0, 1], r[..., 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)
    r[..., 1, 0], r[..., 1, 1], r[..., 1, 2] = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
    r[..., 2, 0], r[..., 2, 1], r[..., 2, 2] = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    return r


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """``R = Rz(yaw) Ry(pitch) Rx(roll)`` (the aerospace ZYX convention used throughout Aevion), radians."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def direction_in_antenna_frame(
    quat: Any, direction_world: Any, mount_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """``(az, el)`` in radians of world-frame direction(s) as seen by an antenna mounted on a UAV with attitude ``quat``.

    ``mount_rpy`` is the rotation of the antenna frame relative to the body frame (roll, pitch, yaw, radians). With the
    default the antenna x axis points forward, z up. ``quat`` is ``(..., 4)`` and ``direction_world`` ``(..., 3)``
    (need not be normalised); they broadcast against each other.
    """
    d = np.asarray(direction_world, float)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    r_wb = quat_to_matrices(quat)
    d_body = np.einsum("...ji,...j->...i", r_wb, np.broadcast_to(d, np.broadcast_shapes(d.shape, r_wb.shape[:-1])))  # R^T d
    r_ba = rpy_to_matrix(*mount_rpy)
    d_ant = d_body @ r_ba  # = (R_ba^T d_body)^T
    az = np.arctan2(d_ant[..., 1], d_ant[..., 0])
    el = np.arctan2(d_ant[..., 2], np.hypot(d_ant[..., 0], d_ant[..., 1]))
    return az, el


class UavAntenna:
    """An antenna pattern mounted on a UAV."""

    PRESETS: Dict[str, Tuple[Any, Tuple[float, float, float], Dict[str, Any]]] = {
        "dipole": (HalfWaveDipole, (0.0, 0.0, 0.0), {}),  # vertical whip: omnidirectional in the horizontal plane
        "patch_down": (Patch3GPP, (0.0, math.pi / 2.0, 0.0), {}),  # boresight towards the ground (+90 deg pitch maps x to -z)
        "patch_forward": (Patch3GPP, (0.0, 0.0, 0.0), {}),
    }

    def __init__(self, pattern: Optional[AntennaPattern] = None, mount_rpy: Sequence[float] = (0.0, 0.0, 0.0)):
        self.pattern = pattern or HalfWaveDipole()
        r, p, y = (float(v) for v in mount_rpy)
        self.mount_rpy: Tuple[float, float, float] = (r, p, y)

    @classmethod
    def preset(cls, name: str, **kw: Any) -> "UavAntenna":
        if name not in cls.PRESETS:
            raise KeyError(f"unknown UAV antenna preset {name!r}; choose from {sorted(cls.PRESETS)}")
        pat, rpy, defaults = cls.PRESETS[name]
        return cls(pat(**{**defaults, **kw}), rpy)

    def gain_dbi(self, quat: Any, direction_world: Any) -> np.ndarray:
        """Gain towards ``direction_world`` (vector from the UAV to the other end) for UAV attitude ``quat``."""
        az, el = direction_in_antenna_frame(quat, direction_world, self.mount_rpy)
        return self.pattern.gain_dbi(az, el)
