"""Physical constants and environment physics: gravity and atmosphere models."""

from __future__ import annotations

import math

# --- Earth / gravity -----------------------------------------------------
G0 = 9.80665  # standard gravity [m/s^2]
R_EARTH = 6_371_000.0  # mean radius [m]
WGS84_A = 6_378_137.0
WGS84_E2 = 6.694_379_990_14e-3
OMEGA_EARTH = 7.292_115e-5  # [rad/s]

# --- ISA atmosphere ------------------------------------------------------
T0 = 288.15  # sea-level temperature [K]
P0 = 101_325.0  # sea-level pressure [Pa]
RHO0 = 1.225  # sea-level density [kg/m^3]
LAPSE = 0.0065  # troposphere lapse rate [K/m]
R_AIR = 287.052_87  # specific gas constant [J/(kg K)]
GAMMA = 1.4
MU_AIR0 = 1.789e-5  # dynamic viscosity at 15C [Pa s]

C_LIGHT = 299_792_458.0
K_BOLTZMANN = 1.380_649e-23


class GravityModel:
    """Gravitational acceleration magnitude, pointing toward -z (world is z-up).

    ``kind``:

    * ``"constant"``: ``g = g0`` everywhere.
    * ``"altitude"``: inverse-square falloff, ``g(h) = g0 (Re / (Re + h))^2``.
    * ``"wgs84"``: Somigliana latitude model with free-air altitude correction.
    * ``"custom"``: user value, e.g. to simulate Mars (3.71) or Moon (1.62).
    """

    def __init__(self, kind: str = "constant", g: float = G0, latitude_deg: float = 45.0, altitude_ref: float = 0.0):
        if kind not in ("constant", "altitude", "wgs84", "custom"):
            raise ValueError(f"unknown gravity model {kind!r}")
        self.kind = kind
        self.g = float(g)
        self.latitude_deg = float(latitude_deg)
        self.altitude_ref = float(altitude_ref)

    def magnitude(self, B, z):
        """Gravity magnitude at height ``z`` above the reference level (array-valued)."""
        if self.kind in ("constant", "custom"):
            return self.g + 0.0 * z
        h = z + self.altitude_ref
        if self.kind == "altitude":
            return self.g * (R_EARTH / (R_EARTH + h)) ** 2
        # Somigliana latitude model with first-order free-air correction in h
        return self.latitude_g() - 3.086e-6 * h

    def latitude_g(self) -> float:
        """Somigliana normal gravity at sea level for this latitude [m/s^2]."""
        s2 = math.sin(math.radians(self.latitude_deg)) ** 2
        return 9.780_325_335_9 * (1 + 0.001_931_852_652_41 * s2) / math.sqrt(1 - WGS84_E2 * s2)

    @classmethod
    def mars(cls):
        return cls("custom", g=3.71)

    @classmethod
    def moon(cls):
        return cls("custom", g=1.62)

    def to_dict(self):
        return dict(kind=self.kind, g=self.g, latitude_deg=self.latitude_deg, altitude_ref=self.altitude_ref)


R_VAPOUR = 461.495  # specific gas constant of water vapour [J/(kg K)]


def saturation_vapour_pressure(temp_c: float) -> float:
    """Saturation vapour pressure over water [Pa] (Alduchov-Eskridge Magnus form, accurate to ~0.4 % from -40 to 50 C)."""
    return 610.94 * math.exp(17.625 * temp_c / (temp_c + 243.04))


class Atmosphere:
    """International Standard Atmosphere (troposphere, valid to ~11 km), optionally shifted by weather.

    ``altitude_msl = altitude + base_altitude``. ``temp_offset`` shifts temperature for hot/cold-day studies (ISA deviation);
    ``pressure_sl`` is the sea-level pressure (QNH; low/high-pressure days) and ``humidity`` the relative humidity in [0, 1], which
    lowers the density of moist air. With the defaults the model is exactly the ISA.
    """

    def __init__(self, base_altitude: float = 0.0, temp_offset: float = 0.0, enabled: bool = True, pressure_sl: float = P0, humidity: float = 0.0):
        self.base_altitude = float(base_altitude)
        self.temp_offset = float(temp_offset)
        self.enabled = enabled
        self.pressure_sl = float(pressure_sl)
        self.humidity = float(humidity)

    def temperature(self, B, z):
        h = z + self.base_altitude
        return T0 - LAPSE * h + self.temp_offset

    def pressure(self, B, z):
        h = z + self.base_altitude
        t_std = T0 - LAPSE * h
        p = P0 * (t_std / T0) ** (G0 / (R_AIR * LAPSE))
        return p if self.pressure_sl == P0 else p * (self.pressure_sl / P0)

    def density(self, B, z):
        if not self.enabled:
            return RHO0 + 0.0 * z
        p = self.pressure(B, z)
        t = self.temperature(B, z)
        if self.humidity <= 0.0:
            return p / (R_AIR * t)
        tc = t - 273.15
        e = B.minimum(self.humidity * 610.94 * B.exp(17.625 * tc / (tc + 243.04)), 0.99 * p)  # vapour pressure, never above the total
        return (p - e) / (R_AIR * t) + e / (R_VAPOUR * t)

    def speed_of_sound(self, B, z):
        return B.sqrt(GAMMA * R_AIR * self.temperature(B, z))

    def to_dict(self):
        d = dict(base_altitude=self.base_altitude, temp_offset=self.temp_offset, enabled=self.enabled)
        if self.pressure_sl != P0:
            d["pressure_sl"] = self.pressure_sl
        if self.humidity:
            d["humidity"] = self.humidity
        return d


def altitude_from_pressure(p):
    """ISA pressure altitude [m] from pressure [Pa] (NumPy-friendly)."""
    return (T0 / LAPSE) * (1.0 - (p / P0) ** (R_AIR * LAPSE / G0))
