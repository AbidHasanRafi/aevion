"""Single-element antenna patterns.

Angles are in the **antenna frame**: x = boresight, y = left, z = up. ``az`` is the angle in the x-y plane from the boresight
(positive towards y), ``el`` the elevation above the x-y plane, both in radians. Gains are in dBi and are floored at
``-40 dBi`` so that pattern nulls never produce ``-inf``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from ..units import C_LIGHT

__all__ = [
    "GAIN_FLOOR_DBI",
    "AntennaPattern",
    "GaussianBeam",
    "HalfWaveDipole",
    "Isotropic",
    "Parabolic",
    "Patch3GPP",
    "SectorTR36814",
    "ShortDipole",
    "off_boresight_angle",
]

GAIN_FLOOR_DBI = -40.0


def off_boresight_angle(az: Any, el: Any) -> np.ndarray:
    """Angle between the direction ``(az, el)`` and the boresight (x axis), radians in ``[0, pi]``."""
    return np.arccos(np.clip(np.cos(np.asarray(el, float)) * np.cos(np.asarray(az, float)), -1.0, 1.0))


def _db(x: Any) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.maximum(np.asarray(x, float), 1e-12)), GAIN_FLOOR_DBI)


class AntennaPattern(ABC):
    name = "pattern"

    @abstractmethod
    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        """Gain towards ``(az, el)`` [dBi]."""

    @property
    def peak_dbi(self) -> float:
        return float(np.max(self.gain_dbi(np.linspace(-math.pi, math.pi, 361)[:, None], np.linspace(-math.pi / 2, math.pi / 2, 181)[None, :])))

    def describe(self) -> Dict[str, Any]:
        return dict(name=self.name, peak_dbi=round(self.peak_dbi, 2))


class Isotropic(AntennaPattern):
    name = "isotropic"

    def __init__(self, gain_dbi: float = 0.0):
        self._g = float(gain_dbi)

    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        return np.full(np.broadcast(np.asarray(az), np.asarray(el)).shape, self._g)


class ShortDipole(AntennaPattern):
    """Short (Hertzian) dipole along the antenna z axis: ``D = 1.5 cos^2(el)`` (1.76 dBi, omnidirectional in azimuth)."""

    name = "short_dipole"

    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        el = np.asarray(el, float) + 0.0 * np.asarray(az, float)
        return _db(1.5 * np.cos(el) ** 2)


class HalfWaveDipole(AntennaPattern):
    """Half-wave dipole along z: ``D = 1.64 [cos(pi/2 sin(el)) / cos(el)]^2`` (2.15 dBi). The usual UAV whip/dipole."""

    name = "half_wave_dipole"

    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        el = np.asarray(el, float) + 0.0 * np.asarray(az, float)
        c = np.cos(el)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.abs(c) < 1e-9, 0.0, np.cos(0.5 * math.pi * np.sin(el)) / np.where(np.abs(c) < 1e-9, 1.0, c))
        return _db(1.64 * f * f)


class Patch3GPP(AntennaPattern):
    """3GPP TR 38.901 Table 7.3-1 single-element pattern (also a good model of a directional patch/panel).

    ``G = G_max - min(-(A_V + A_H), A_max)`` with ``A_V = -min(12 (el/theta_3dB)^2, SLA_V)`` and
    ``A_H = -min(12 (az/phi_3dB)^2, A_max)``; defaults 65 degrees, 30 dB, 8 dBi.
    """

    name = "patch_3gpp"

    def __init__(
        self, g_max_dbi: float = 8.0, theta_3db_deg: float = 65.0, phi_3db_deg: float = 65.0, sla_v_db: float = 30.0, a_max_db: float = 30.0
    ):
        self.g_max, self.t3, self.p3, self.sla, self.am = g_max_dbi, theta_3db_deg, phi_3db_deg, sla_v_db, a_max_db

    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        az_d, el_d = np.degrees(np.asarray(az, float)), np.degrees(np.asarray(el, float))
        az_d = (az_d + 180.0) % 360.0 - 180.0
        a_v = -np.minimum(12.0 * (el_d / self.t3) ** 2, self.sla)
        a_h = -np.minimum(12.0 * (az_d / self.p3) ** 2, self.am)
        a = -np.minimum(-(a_v + a_h), self.am)
        return np.maximum(self.g_max + a, GAIN_FLOOR_DBI)


class SectorTR36814(AntennaPattern):
    """TR 36.814 macro-cell sector (vertical tilt pattern only, as used for cellular base stations serving UAVs).

    ``tilt_deg`` is the electrical down-tilt: the boresight points that many degrees below the horizon. The azimuth pattern
    is the 3GPP ``-min(12 (az/65)^2, 30)``.
    """

    name = "sector_36814"

    def __init__(self, g_max_dbi: float = 8.0, tilt_deg: float = 10.0, theta_3db_deg: float = 65.0, phi_3db_deg: float = 65.0, sla_db: float = 30.0):
        self.g_max, self.tilt, self.t3, self.p3, self.sla = g_max_dbi, tilt_deg, theta_3db_deg, phi_3db_deg, sla_db

    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        az_d = (np.degrees(np.asarray(az, float)) + 180.0) % 360.0 - 180.0
        below = -np.degrees(np.asarray(el, float))  # angle below the horizon
        a_v = np.minimum(12.0 * ((below - self.tilt) / self.t3) ** 2, self.sla)
        a_h = np.minimum(12.0 * (az_d / self.p3) ** 2, self.sla)
        return np.maximum(self.g_max - np.minimum(a_v + a_h, self.sla), GAIN_FLOOR_DBI)


class GaussianBeam(AntennaPattern):
    """Gaussian main lobe with independent azimuth/elevation half-power beamwidths and a sidelobe floor (relative to peak)."""

    name = "gaussian_beam"

    def __init__(self, g_max_dbi: float = 15.0, bw_az_deg: float = 30.0, bw_el_deg: float = 30.0, floor_db: float = 25.0):
        self.g_max, self.bw_az, self.bw_el, self.floor = g_max_dbi, bw_az_deg, bw_el_deg, floor_db

    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        az_d = (np.degrees(np.asarray(az, float)) + 180.0) % 360.0 - 180.0
        el_d = np.degrees(np.asarray(el, float))
        a = 12.0 * ((az_d / self.bw_az) ** 2 + (el_d / self.bw_el) ** 2)
        return np.maximum(self.g_max - np.minimum(a, self.floor), GAIN_FLOOR_DBI)


class Parabolic(AntennaPattern):
    """Aperture antenna: ``G = eta (pi D / lambda)^2``, half-power beamwidth ``70 lambda / D`` degrees, parabolic roll-off to a sidelobe floor."""

    name = "parabolic"

    def __init__(self, diameter_m: float = 0.3, freq_hz: float = 28e9, efficiency: float = 0.6, floor_db: float = 25.0):
        lam = C_LIGHT / freq_hz
        self.g_max = 10.0 * math.log10(efficiency * (math.pi * diameter_m / lam) ** 2)
        self.bw = 70.0 * lam / diameter_m
        self.floor = floor_db

    def gain_dbi(self, az: Any, el: Any) -> np.ndarray:
        th = np.degrees(off_boresight_angle(az, el))
        return np.maximum(self.g_max - np.minimum(12.0 * (th / self.bw) ** 2, self.floor), GAIN_FLOOR_DBI)
