"""Earth's magnetic field models for the magnetometer.

* :class:`FixedField`   one declination / inclination / intensity everywhere (the simple magnetometer behaviour; right for a few km).
* :class:`TiltedDipole` the degree-1 term of the IGRF-13 spherical-harmonic model (epoch 2020, with secular variation): a tilted
  centred dipole. It reproduces the global structure (inclination from -90 deg to +90 deg, a factor-2 equator/pole intensity ratio, the
  geomagnetic pole near 80.6 N, 72.7 W) but **not** the regional anomalies carried by degrees 2-13, so declination can be off by
  10-20 deg and intensity by 10-30 % at a given site. Use :class:`FixedField` with measured values (or a table from the NOAA/BGS
  calculators) when absolute accuracy matters.

Output is the field in the local **ENU** frame, in microtesla.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .geodesy import lla_to_ecef

__all__ = ["MagneticModel", "FixedField", "TiltedDipole", "field_angles"]

REF_RADIUS_M = 6371200.0  # IGRF reference sphere


def field_angles(enu_ut: np.ndarray) -> Dict[str, np.ndarray]:
    """Declination (east of true north), inclination (positive down) [deg] and total intensity [uT] of an ENU field."""
    e, n, u = enu_ut[..., 0], enu_ut[..., 1], enu_ut[..., 2]
    h = np.hypot(e, n)
    return dict(declination_deg=np.degrees(np.arctan2(e, n)), inclination_deg=np.degrees(np.arctan2(-u, h)), intensity_ut=np.sqrt(h * h + u * u))


class MagneticModel:
    """Interface: ``field_enu(lat_deg, lon_deg, alt_m, year) -> (..., 3)`` microtesla."""

    name = "magnetic"

    def field_enu(self, lat_deg: Any, lon_deg: Any, alt_m: Any = 0.0, year: float = 2020.0) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def angles(self, lat_deg: Any, lon_deg: Any, alt_m: Any = 0.0, year: float = 2020.0) -> Dict[str, np.ndarray]:
        return field_angles(self.field_enu(lat_deg, lon_deg, alt_m, year))

    def to_dict(self) -> dict:
        return dict(type=type(self).__name__)


class FixedField(MagneticModel):
    """A uniform field given by ``intensity_ut``, ``inclination_deg`` (positive down) and ``declination_deg`` (east of true north)."""

    name = "fixed"

    def __init__(self, intensity_ut: float = 50.0, inclination_deg: float = 60.0, declination_deg: float = 0.0):
        self.intensity_ut, self.inclination_deg, self.declination_deg = float(intensity_ut), float(inclination_deg), float(declination_deg)
        inc, dec = math.radians(inclination_deg), math.radians(declination_deg)
        h = math.cos(inc)
        self._v = intensity_ut * np.array([h * math.sin(dec), h * math.cos(dec), -math.sin(inc)])

    def field_enu(self, lat_deg: Any = 0.0, lon_deg: Any = 0.0, alt_m: Any = 0.0, year: float = 2020.0) -> np.ndarray:
        shape = np.broadcast(np.asarray(lat_deg), np.asarray(lon_deg), np.asarray(alt_m)).shape
        return np.broadcast_to(self._v, shape + (3,)).copy()

    def to_dict(self) -> dict:
        return dict(type="FixedField", intensity_ut=self.intensity_ut, inclination_deg=self.inclination_deg, declination_deg=self.declination_deg)


class TiltedDipole(MagneticModel):
    """IGRF-13 degree-1 field (Gauss coefficients in nT, secular variation in nT/yr, epoch 2020.0)."""

    name = "dipole"

    def __init__(
        self,
        g10: float = -29404.8,
        g11: float = -1450.9,
        h11: float = 4652.5,
        sv_g10: float = 5.7,
        sv_g11: float = 7.4,
        sv_h11: float = -25.9,
        epoch: float = 2020.0,
    ):
        self.g = (g10, g11, h11)
        self.sv = (sv_g10, sv_g11, sv_h11)
        self.epoch = float(epoch)

    def coefficients(self, year: float) -> tuple:
        dt = year - self.epoch
        return tuple(c + s * dt for c, s in zip(self.g, self.sv))

    @property
    def moment_nt(self) -> float:
        """Equatorial surface intensity B0 = sqrt(g10^2 + g11^2 + h11^2) [nT]."""
        return float(math.sqrt(sum(c * c for c in self.g)))

    def geomagnetic_pole(self, year: float = 2020.0) -> tuple:
        """(latitude, longitude) in degrees of the northern geomagnetic pole (where the dipole axis meets the surface)."""
        g10, g11, h11 = self.coefficients(year)
        b0 = math.sqrt(g10 * g10 + g11 * g11 + h11 * h11)
        lon = (math.degrees(math.atan2(h11, g11)) + 180.0 + 180.0) % 360.0 - 180.0
        return math.degrees(math.asin(-g10 / b0)), lon

    def field_enu(self, lat_deg: Any, lon_deg: Any, alt_m: Any = 0.0, year: float = 2020.0) -> np.ndarray:
        g10, g11, h11 = self.coefficients(year)
        lat, lon, alt = np.broadcast_arrays(np.asarray(lat_deg, float), np.asarray(lon_deg, float), np.asarray(alt_m, float))
        xyz = lla_to_ecef(lat, lon, alt)
        r = np.linalg.norm(xyz, axis=-1)
        theta = np.arccos(np.clip(xyz[..., 2] / r, -1.0, 1.0))  # geocentric colatitude
        phi = np.arctan2(xyz[..., 1], xyz[..., 0])
        a3 = (REF_RADIUS_M / r) ** 3
        st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        m = g11 * cp + h11 * sp
        b_r = 2.0 * a3 * (g10 * ct + m * st)
        b_t = a3 * (g10 * st - m * ct)
        b_p = a3 * (g11 * sp - h11 * cp)
        north_c, east, down_c = -b_t, b_p, -b_r  # geocentric NED
        psi = np.radians(lat) - (np.pi / 2.0 - theta)  # geodetic minus geocentric latitude
        north = north_c * np.cos(psi) + down_c * np.sin(psi)
        down = down_c * np.cos(psi) - north_c * np.sin(psi)
        return np.stack([east, north, -down], axis=-1) * 1e-3  # nT -> uT

    def to_dict(self) -> dict:
        return dict(type="TiltedDipole", g10=self.g[0], g11=self.g[1], h11=self.g[2], epoch=self.epoch)
