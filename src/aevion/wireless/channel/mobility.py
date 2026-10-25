"""Mobility-induced channel dynamics: Doppler shift/spread, coherence time and handover-relevant kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fading.doppler import coherence_time
from ..units import C_LIGHT

__all__ = ["LinkKinematics", "link_kinematics"]


@dataclass
class LinkKinematics:
    """Per-link geometry and motion. Arrays are ``(..., )`` over links."""

    distance_m: np.ndarray
    radial_speed_mps: np.ndarray  # > 0: moving apart
    tangential_speed_mps: np.ndarray
    doppler_shift_hz: np.ndarray  # signed shift seen by the receiver
    max_doppler_hz: np.ndarray  # v f_c / c with v the relative speed
    coherence_time_s: np.ndarray  # correlation 0.5
    angular_rate_deg_s: np.ndarray  # rate of change of the bearing as seen from the base station


def link_kinematics(bs_pos: Any, ut_pos: Any, ut_vel: Any, freq_hz: float, bs_vel: Any = 0.0) -> LinkKinematics:
    """Kinematics of links from base stations at ``bs_pos`` to a moving terminal (broadcastable ``(..., 3)`` arrays)."""
    bs, ut = np.broadcast_arrays(np.asarray(bs_pos, float), np.asarray(ut_pos, float))
    v = np.asarray(ut_vel, float) - np.asarray(bs_vel, float)
    v = np.broadcast_to(v, bs.shape)
    d = ut - bs
    r = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
    u = d / r[..., None]  # unit vector base station -> terminal
    v_rad = np.sum(v * u, axis=-1)
    v_tan = np.linalg.norm(v - v_rad[..., None] * u, axis=-1)
    speed = np.linalg.norm(v, axis=-1)
    fd_max = speed * freq_hz / C_LIGHT
    return LinkKinematics(
        distance_m=r,
        radial_speed_mps=v_rad,
        tangential_speed_mps=v_tan,
        doppler_shift_hz=-v_rad * freq_hz / C_LIGHT,  # approaching (v_rad < 0) raises the received frequency
        max_doppler_hz=fd_max,
        coherence_time_s=coherence_time(np.maximum(fd_max, 1e-9)),
        angular_rate_deg_s=np.degrees(v_tan / r),
    )
