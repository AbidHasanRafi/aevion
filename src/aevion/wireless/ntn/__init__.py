"""Non-terrestrial network links: satellite geometry, link budget, Doppler and atmospheric loss."""

from .satellite_link import (
    EARTH_RADIUS_M,
    GM_EARTH,
    SatelliteLink,
    doppler_shift_hz,
    gaseous_attenuation_db,
    orbital_period_s,
    orbital_speed_mps,
    pass_geometry,
    rain_attenuation_db,
    slant_range_m,
)

__all__ = [
    "EARTH_RADIUS_M",
    "GM_EARTH",
    "SatelliteLink",
    "doppler_shift_hz",
    "gaseous_attenuation_db",
    "orbital_period_s",
    "orbital_speed_mps",
    "pass_geometry",
    "rain_attenuation_db",
    "slant_range_m",
]
