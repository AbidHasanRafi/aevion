"""Weather: a small, explicit description of the air and what it does to vehicles and sensors.

A :class:`Weather` bundles temperature, sea-level pressure, humidity, rain rate, visibility and cloud cover. It

* configures the world's :class:`~aevion.core.constants.Atmosphere` (air density follows temperature, pressure and humidity, which
  changes thrust, drag and barometer readings),
* provides the optical **extinction coefficient** used by the camera (fog) and LiDAR (range loss, false returns) models, and
* offers presets (``clear``, ``overcast``, ``rain``, ``heavy_rain``, ``fog``, ``hot``, ``cold``, ``storm``).

Wind is a separate component (``winds`` registry) because it is a field, not a scalar; the ``storm`` preset only *suggests* one.

Fidelity: the physics here is standard textbook meteorology (Magnus vapour pressure, ideal-gas moist-air density, Koschmieder
visibility, Marshall-Palmer geometric-optics rain extinction). Weather is spatially uniform and constant in time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from ..core.constants import P0, R_AIR, R_VAPOUR, saturation_vapour_pressure

__all__ = ["Weather", "PRESETS", "moist_air_density", "rain_extinction", "fog_extinction", "visibility_from_extinction"]

KOSCHMIEDER = 3.912  # ln(1 / 0.02): the 2 % contrast threshold of the meteorological optical range


def moist_air_density(pressure_pa: float, temp_c: float, humidity: float) -> float:
    """Density of moist air [kg/m^3]: dry-air and water-vapour partial densities from the ideal-gas law."""
    e = min(humidity * saturation_vapour_pressure(temp_c), 0.99 * pressure_pa)
    t = temp_c + 273.15
    return (pressure_pa - e) / (R_AIR * t) + e / (R_VAPOUR * t)


def fog_extinction(visibility_m: float) -> float:
    """Extinction coefficient [1/m] of a medium with meteorological optical range ``visibility_m`` (Koschmieder: beta = 3.912 / V)."""
    return KOSCHMIEDER / max(visibility_m, 1.0)


def rain_extinction(rain_mm_h: float) -> float:
    """Extinction coefficient of rain [1/m] for visible/near-infrared light.

    Geometric optics (extinction efficiency 2) with the Marshall-Palmer drop-size distribution gives
    ``beta = 0.365 R^0.63`` per kilometre, R in mm/h.
    """
    return 0.365e-3 * rain_mm_h**0.63 if rain_mm_h > 0.0 else 0.0


def visibility_from_extinction(beta_per_m: float) -> float:
    return KOSCHMIEDER / max(beta_per_m, 1e-9)


@dataclass
class Weather:
    temperature_c: float = 15.0  # sea-level air temperature
    pressure_sl_pa: float = P0  # sea-level pressure (QNH)
    humidity: float = 0.5  # relative humidity in [0, 1]
    rain_mm_h: float = 0.0
    visibility_m: float = 50_000.0  # meteorological optical range without rain
    cloud_cover: float = 0.0  # fraction of the sky in [0, 1]

    def __post_init__(self) -> None:
        if not 0.0 <= self.humidity <= 1.0:
            raise ValueError("humidity must be in [0, 1]")
        if not 0.0 <= self.cloud_cover <= 1.0:
            raise ValueError("cloud_cover must be in [0, 1]")
        if self.rain_mm_h < 0.0 or self.visibility_m <= 0.0 or self.pressure_sl_pa <= 0.0:
            raise ValueError("rain_mm_h must be >= 0, visibility_m > 0 and pressure_sl_pa > 0")

    # ----------------------------------------------------------------------------------------------- air
    @property
    def temp_offset(self) -> float:
        """Deviation from the ISA sea-level temperature (15 C), the ``Atmosphere.temp_offset``."""
        return self.temperature_c - 15.0

    def air_density(self) -> float:
        return moist_air_density(self.pressure_sl_pa, self.temperature_c, self.humidity)

    def speed_of_sound(self) -> float:
        return math.sqrt(1.4 * R_AIR * (self.temperature_c + 273.15))

    def atmosphere_kwargs(self) -> Dict[str, float]:
        return dict(temp_offset=self.temp_offset, pressure_sl=self.pressure_sl_pa, humidity=self.humidity)

    def apply(self, world: Any) -> Any:
        """Configure ``world.atmosphere`` from this weather and remember it as ``world.weather``."""
        a = world.atmosphere
        a.temp_offset, a.pressure_sl, a.humidity = self.temp_offset, self.pressure_sl_pa, self.humidity
        world.weather = self
        return world

    # ---------------------------------------------------------------------------------------------- light
    def extinction(self) -> float:
        """Total optical extinction [1/m]: haze/fog (from the visibility) plus rain."""
        return fog_extinction(self.visibility_m) + rain_extinction(self.rain_mm_h)

    @property
    def effective_visibility_m(self) -> float:
        return visibility_from_extinction(self.extinction())

    def transmission(self, distance_m: float) -> float:
        """Fraction of light surviving a one-way path of ``distance_m`` (Beer-Lambert)."""
        return math.exp(-self.extinction() * distance_m)

    # ------------------------------------------------------------------------------------- construction
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Weather":
        d = dict(d)
        preset = d.pop("preset", None)
        base = dict(PRESETS[preset]) if preset is not None else {}
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown weather field(s) {sorted(unknown)}; known: {[f.name for f in fields(cls)]}")
        base.update(d)
        return cls(**base)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "Weather":
        if name not in PRESETS:
            raise KeyError(f"unknown weather preset {name!r}; choose from {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})


PRESETS: Dict[str, Dict[str, float]] = {
    "clear": dict(temperature_c=15.0, humidity=0.45, visibility_m=50_000.0, cloud_cover=0.1),
    "overcast": dict(temperature_c=12.0, humidity=0.8, visibility_m=20_000.0, cloud_cover=1.0),
    "rain": dict(temperature_c=10.0, pressure_sl_pa=100_200.0, humidity=0.95, rain_mm_h=4.0, visibility_m=10_000.0, cloud_cover=1.0),
    "heavy_rain": dict(temperature_c=12.0, pressure_sl_pa=99_500.0, humidity=1.0, rain_mm_h=25.0, visibility_m=4_000.0, cloud_cover=1.0),
    "fog": dict(temperature_c=6.0, humidity=1.0, visibility_m=200.0, cloud_cover=1.0),
    "hot": dict(temperature_c=38.0, humidity=0.3, visibility_m=30_000.0, cloud_cover=0.0),
    "cold": dict(temperature_c=-15.0, humidity=0.6, visibility_m=40_000.0, cloud_cover=0.2),
    "storm": dict(temperature_c=18.0, pressure_sl_pa=98_500.0, humidity=1.0, rain_mm_h=40.0, visibility_m=2_000.0, cloud_cover=1.0),
}
