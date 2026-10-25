"""Solar position (NOAA / Meeus low-precision algorithm), sunrise/sunset and clear-sky irradiance.

Used for the sun direction in rendered cameras and the viewer's time of day, for shadow lengths, and for solar-powered vehicle studies.

Fidelity: the NOAA algorithm is accurate to about 0.01 deg in solar position for years 1800-2100 (before atmospheric refraction, which
is applied separately by :func:`refracted_elevation`). The clear-sky irradiance is the Haurwitz (1945) global-horizontal formula: a
sensible mean clear-day curve (typically within 10-15 % of measured clear-sky data), not a radiative-transfer model.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

__all__ = ["julian_day", "sun_position", "sun_direction_enu", "sunrise_sunset", "refracted_elevation", "clear_sky_ghi", "equation_of_time_min"]

TimeLike = Union[str, _dt.datetime, np.datetime64, float, np.ndarray]
_J2000_UNIX = 946728000.0  # 2000-01-01 12:00:00 UTC in Unix seconds


def _to_unix(t: TimeLike) -> np.ndarray:
    """Any supported time representation -> Unix seconds (UTC); plain numbers are taken as Unix seconds."""
    if isinstance(t, str):
        t = _dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
    if isinstance(t, _dt.datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=_dt.timezone.utc)
        return np.asarray(t.timestamp(), float)
    a = np.asarray(t)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[ns]").astype(np.int64) / 1e9
    return a.astype(float)  # plain numbers are Unix seconds


def julian_day(t: TimeLike) -> np.ndarray:
    """Julian day number (UT) of a datetime, ISO-8601 string or ``datetime64`` (arrays allowed)."""
    return _to_unix(t) / 86400.0 + 2440587.5


def equation_of_time_min(t: TimeLike) -> np.ndarray:
    """Equation of time [minutes]: apparent minus mean solar time."""
    return _solar_terms(julian_day(t))["eot_min"]


def _solar_terms(jd: np.ndarray) -> Dict[str, np.ndarray]:
    T = (np.asarray(jd, float) - 2451545.0) / 36525.0
    rad = np.radians
    l0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360.0
    m = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    c = np.sin(rad(m)) * (1.914602 - T * (0.004817 + 0.000014 * T)) + np.sin(rad(2 * m)) * (0.019993 - 0.000101 * T) + np.sin(rad(3 * m)) * 0.000289
    true_long = l0 + c
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * np.sin(rad(omega))
    eps0 = 23.0 + (26.0 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(rad(omega))
    decl = np.degrees(np.arcsin(np.sin(rad(eps)) * np.sin(rad(lam))))
    y = np.tan(rad(eps) / 2.0) ** 2
    eot = (
        y * np.sin(2 * rad(l0))
        - 2 * e * np.sin(rad(m))
        + 4 * e * y * np.sin(rad(m)) * np.cos(2 * rad(l0))
        - 0.5 * y * y * np.sin(4 * rad(l0))
        - 1.25 * e * e * np.sin(2 * rad(m))
    )
    return dict(declination_deg=decl, eot_min=4.0 * np.degrees(eot), distance_au=(1.000001018 * (1 - e * e)) / (1 + e * np.cos(rad(m + c))))


def sun_position(lat_deg: Any, lon_deg: Any, t: TimeLike) -> Dict[str, np.ndarray]:
    """Geometric solar position: ``azimuth_deg`` (clockwise from north), ``elevation_deg`` above the horizon, ``declination_deg``,
    ``hour_angle_deg`` and Earth-Sun ``distance_au``. Longitude is east-positive."""
    unix = _to_unix(t)
    jd = unix / 86400.0 + 2440587.5
    terms = _solar_terms(jd)
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.asarray(lon_deg, float)
    minutes = (unix % 86400.0) / 60.0
    tst = (minutes + terms["eot_min"] + 4.0 * lon) % 1440.0  # true solar time [min]
    ha = tst / 4.0 - 180.0
    ha = np.where(ha < -180.0, ha + 360.0, ha)
    decl = np.radians(terms["declination_deg"])
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(np.radians(ha))
    zen = np.arccos(np.clip(cos_z, -1.0, 1.0))
    # azimuth from north, clockwise
    az = np.degrees(np.arctan2(np.sin(np.radians(ha)), np.cos(np.radians(ha)) * np.sin(lat) - np.tan(decl) * np.cos(lat))) + 180.0
    return dict(
        azimuth_deg=az % 360.0,
        elevation_deg=90.0 - np.degrees(zen),
        declination_deg=terms["declination_deg"],
        hour_angle_deg=ha,
        distance_au=terms["distance_au"],
    )


def sun_direction_enu(azimuth_deg: Any, elevation_deg: Any) -> np.ndarray:
    """Unit vector towards the sun in the local ENU frame."""
    az, el = np.radians(np.asarray(azimuth_deg, float)), np.radians(np.asarray(elevation_deg, float))
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=-1)


def refracted_elevation(elevation_deg: Any, pressure_pa: float = 101325.0, temperature_c: float = 10.0) -> np.ndarray:
    """Apparent elevation including atmospheric refraction (Saemundsson's formula with pressure/temperature correction)."""
    h = np.asarray(elevation_deg, float)
    r = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) * (pressure_pa / 101000.0) * (283.0 / (273.0 + temperature_c)) / 60.0
    return np.where(h > -1.0, h + r, h)


def sunrise_sunset(lat_deg: float, lon_deg: float, date: Union[str, _dt.date, _dt.datetime]) -> Tuple[Optional[_dt.datetime], Optional[_dt.datetime]]:
    """UTC sunrise and sunset (centre of the disc 0.833 deg below the horizon) of the given calendar date; ``None`` for polar day/night."""
    if isinstance(date, str):
        date = _dt.date.fromisoformat(date[:10])
    if isinstance(date, _dt.datetime):
        date = date.date()
    noon = _dt.datetime(date.year, date.month, date.day, 12, tzinfo=_dt.timezone.utc)
    terms = _solar_terms(julian_day(noon))
    lat, decl = math.radians(lat_deg), math.radians(float(terms["declination_deg"]))
    cos_h = (math.cos(math.radians(90.833)) - math.sin(lat) * math.sin(decl)) / (math.cos(lat) * math.cos(decl))
    if cos_h > 1.0 or cos_h < -1.0:
        return None, None
    h = math.degrees(math.acos(cos_h))
    eot = float(terms["eot_min"])
    base = _dt.datetime(date.year, date.month, date.day, tzinfo=_dt.timezone.utc)
    rise = base + _dt.timedelta(minutes=720.0 - 4.0 * (lon_deg + h) - eot)
    sett = base + _dt.timedelta(minutes=720.0 - 4.0 * (lon_deg - h) - eot)
    return rise, sett


def clear_sky_ghi(elevation_deg: Any) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance [W/m^2]; zero when the sun is below the horizon."""
    el = np.asarray(elevation_deg, float)
    cz = np.sin(np.radians(np.maximum(el, 0.0)))
    return np.where(cz > 0.0, 1098.0 * cz * np.exp(-0.057 / np.maximum(cz, 1e-6)), 0.0)
