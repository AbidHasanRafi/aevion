"""Geodesy on the WGS-84 ellipsoid: geodetic (lat, lon, alt) <-> ECEF <-> local ENU, and geodesics.

Angles in this module's public API are **degrees**, lengths **metres**. Arrays broadcast; a "point" is the last axis of size 3
(``[lat, lon, alt]``, ``[x, y, z]`` or ``[e, n, u]``).

Fidelity: exact ellipsoidal formulae. ``ecef_to_lla`` uses the iterative Bowring/Newton scheme and is accurate to well below a
micrometre for altitudes from -10 km to +100 km; Vincenty's inverse/direct solutions agree with Geoscience Australia's published
test line to millimetres (see ``tests/test_geodesy.py``). Not modelled: the geoid (altitude is *ellipsoidal* height), plate motion,
datum shifts other than WGS-84.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np

__all__ = [
    "WGS84_A",
    "WGS84_F",
    "WGS84_B",
    "WGS84_E2",
    "lla_to_ecef",
    "ecef_to_lla",
    "LocalFrame",
    "haversine_distance",
    "vincenty_inverse",
    "vincenty_direct",
    "initial_bearing",
    "meridian_radius",
    "prime_vertical_radius",
]

WGS84_A = 6378137.0  # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared
MEAN_RADIUS = 6371008.8  # IUGG mean radius, used by the spherical formulae


def prime_vertical_radius(lat_deg: Union[float, np.ndarray]) -> np.ndarray:
    """Radius of curvature in the prime vertical, N(phi) [m]."""
    s = np.sin(np.radians(lat_deg))
    return WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)


def meridian_radius(lat_deg: Union[float, np.ndarray]) -> np.ndarray:
    """Radius of curvature in the meridian, M(phi) [m]."""
    s = np.sin(np.radians(lat_deg))
    return WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s * s) ** 1.5


def lla_to_ecef(lat_deg, lon_deg, alt_m=0.0) -> np.ndarray:
    """Geodetic coordinates to Earth-centred Earth-fixed ``(..., 3)``."""
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    h = np.asarray(alt_m, float)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * np.sin(lat)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_lla(xyz) -> np.ndarray:
    """ECEF ``(..., 3)`` to ``(..., 3)`` = ``[lat_deg, lon_deg, alt_m]`` (iterative; converges in <= 6 steps for any point off the centre)."""
    p3 = np.asarray(xyz, float)
    x, y, z = p3[..., 0], p3[..., 1], p3[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        s = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
        h = np.where(np.abs(np.cos(lat)) > 1e-6, p / np.cos(lat) - n, z / np.where(s == 0, 1.0, s) - n * (1.0 - WGS84_E2))
        new = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
        if np.all(np.abs(new - lat) < 1e-14):
            lat = new
            break
        lat = new
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    h = np.where(np.abs(np.cos(lat)) > 1e-6, p / np.cos(lat) - n, z / np.where(s == 0, 1.0, s) - n * (1.0 - WGS84_E2))
    return np.stack([np.degrees(lat), np.degrees(lon), h], axis=-1)


class LocalFrame:
    """A local East-North-Up frame anchored at a geodetic origin (the world frame of a simulation with an ``origin``).

    >>> f = LocalFrame(47.3769, 8.5417, 408.0)
    >>> enu = f.to_enu([47.3779, 8.5417, 408.0])   # about 111 m north of the origin
    """

    def __init__(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0):
        self.lat, self.lon, self.alt = float(lat_deg), float(lon_deg), float(alt_m)
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"latitude {lat_deg} outside [-90, 90]")
        self.origin_ecef = lla_to_ecef(self.lat, self.lon, self.alt)
        la, lo = math.radians(self.lat), math.radians(self.lon)
        sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
        #: rows are the east, north and up unit vectors in ECEF
        self.rotation = np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

    def ecef_to_enu(self, xyz) -> np.ndarray:
        return (np.asarray(xyz, float) - self.origin_ecef) @ self.rotation.T

    def enu_to_ecef(self, enu) -> np.ndarray:
        return np.asarray(enu, float) @ self.rotation + self.origin_ecef

    def to_enu(self, lla) -> np.ndarray:
        """``[lat, lon, alt]`` (degrees, metres) to local ``[e, n, u]``."""
        a = np.asarray(lla, float)
        return self.ecef_to_enu(lla_to_ecef(a[..., 0], a[..., 1], a[..., 2]))

    def to_lla(self, enu) -> np.ndarray:
        """Local ``[e, n, u]`` to ``[lat, lon, alt]``."""
        return ecef_to_lla(self.enu_to_ecef(enu))

    def to_dict(self) -> dict:
        return dict(lat=self.lat, lon=self.lon, alt=self.alt)

    @classmethod
    def from_dict(cls, d: dict) -> "LocalFrame":
        return cls(d["lat"], d["lon"], d.get("alt", 0.0))

    def __repr__(self) -> str:
        return f"LocalFrame(lat={self.lat:.6f}, lon={self.lon:.6f}, alt={self.alt:.1f})"


# ---------------------------------------------------------------------------------------------------------- geodesics
def haversine_distance(lat1, lon1, lat2, lon2, radius: float = MEAN_RADIUS) -> np.ndarray:
    """Great-circle distance on a sphere [m] (error up to ~0.5 % against the ellipsoid; use :func:`vincenty_inverse` for accuracy)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * radius * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def initial_bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north in [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def vincenty_inverse(lat1: float, lon1: float, lat2: float, lon2: float, tol: float = 1e-12, max_iter: int = 200) -> Tuple[float, float, float]:
    """Geodesic distance [m], the forward azimuth at point 1 and the back-azimuth (from point 2 towards point 1), both in degrees, between two points on the WGS-84 ellipsoid (Vincenty 1975).

    Returns ``(nan, nan, nan)`` if the iteration does not converge (nearly antipodal points).
    """
    a, b, f = WGS84_A, WGS84_B, WGS84_F
    u1, u2 = math.atan((1 - f) * math.tan(math.radians(lat1))), math.atan((1 - f) * math.tan(math.radians(lat2)))
    big_l = math.radians(lon2 - lon1)
    lam = big_l
    su1, cu1, su2, cu2 = math.sin(u1), math.cos(u1), math.sin(u2), math.cos(u2)
    for _ in range(max_iter):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.hypot(cu2 * sl, cu1 * su2 - su1 * cu2 * cl)
        if ss == 0.0:
            return 0.0, 0.0, 0.0  # coincident points
        cs = su1 * su2 + cu1 * cu2 * cl
        sigma = math.atan2(ss, cs)
        sa = cu1 * cu2 * sl / ss
        c2a = 1.0 - sa * sa
        c2sm = cs - 2.0 * su1 * su2 / c2a if c2a != 0.0 else 0.0  # equatorial line
        c = f / 16.0 * c2a * (4.0 + f * (4.0 - 3.0 * c2a))
        prev = lam
        lam = big_l + (1 - c) * f * sa * (sigma + c * ss * (c2sm + c * cs * (-1.0 + 2.0 * c2sm * c2sm)))
        if abs(lam - prev) < tol:
            break
    else:
        return math.nan, math.nan, math.nan
    u_sq = c2a * (a * a - b * b) / (b * b)
    big_a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)))
    big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))
    ds = big_b * ss * (c2sm + big_b / 4.0 * (cs * (-1.0 + 2.0 * c2sm**2) - big_b / 6.0 * c2sm * (-3.0 + 4.0 * ss**2) * (-3.0 + 4.0 * c2sm**2)))
    dist = b * big_a * (sigma - ds)
    az1 = math.degrees(math.atan2(cu2 * math.sin(lam), cu1 * su2 - su1 * cu2 * math.cos(lam))) % 360.0
    az2 = (math.degrees(math.atan2(cu1 * math.sin(lam), -su1 * cu2 + cu1 * su2 * math.cos(lam))) + 180.0) % 360.0  # back-azimuth
    return dist, az1, az2


def vincenty_direct(lat1: float, lon1: float, azimuth_deg: float, distance_m: float) -> Tuple[float, float, float]:
    """Point reached from ``(lat1, lon1)`` after ``distance_m`` along ``azimuth_deg``: ``(lat2, lon2, back-azimuth)`` in degrees."""
    a, b, f = WGS84_A, WGS84_B, WGS84_F
    al1 = math.radians(azimuth_deg)
    sa1, ca1 = math.sin(al1), math.cos(al1)
    tu1 = (1 - f) * math.tan(math.radians(lat1))
    cu1 = 1.0 / math.sqrt(1 + tu1 * tu1)
    su1 = tu1 * cu1
    sigma1 = math.atan2(tu1, ca1)
    sa = cu1 * sa1
    c2a = 1 - sa * sa
    u_sq = c2a * (a * a - b * b) / (b * b)
    big_a = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    big_b = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    sigma = distance_m / (b * big_a)
    for _ in range(200):
        c2sm = math.cos(2 * sigma1 + sigma)
        ss, cs = math.sin(sigma), math.cos(sigma)
        ds = big_b * ss * (c2sm + big_b / 4 * (cs * (-1 + 2 * c2sm**2) - big_b / 6 * c2sm * (-3 + 4 * ss**2) * (-3 + 4 * c2sm**2)))
        new = distance_m / (b * big_a) + ds
        if abs(new - sigma) < 1e-12:
            sigma = new
            break
        sigma = new
    ss, cs = math.sin(sigma), math.cos(sigma)
    c2sm = math.cos(2 * sigma1 + sigma)
    x = su1 * ss - cu1 * cs * ca1
    lat2 = math.atan2(su1 * cs + cu1 * ss * ca1, (1 - f) * math.hypot(sa, x))
    lam = math.atan2(ss * sa1, cu1 * cs - su1 * ss * ca1)
    c = f / 16 * c2a * (4 + f * (4 - 3 * c2a))
    big_l = lam - (1 - c) * f * sa * (sigma + c * ss * (c2sm + c * cs * (-1 + 2 * c2sm**2)))
    az2 = (math.degrees(math.atan2(sa, -x)) + 180.0) % 360.0  # back-azimuth
    return math.degrees(lat2), (lon1 + math.degrees(big_l) + 540.0) % 360.0 - 180.0, az2
