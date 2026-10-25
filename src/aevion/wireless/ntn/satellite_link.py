"""Non-terrestrial network (NTN) links: UAV to LEO/MEO/GEO satellite.

Geometry (spherical Earth), free-space loss with slant range, gaseous and rain attenuation (ITU-R P.676/P.838 style, simplified),
Doppler along an overhead pass, and the resulting C/N0, SNR and rate. Atmospheric terms are *engineering estimates*: the rain
model uses ITU-R P.838 specific attenuation with an effective path length (P.618 style), valid for 4-40 GHz and elevations above
about 5 degrees; below 4 GHz rain loss is neglected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..propagation.base import fspl_db
from ..units import C_LIGHT, K_BOLTZMANN

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

EARTH_RADIUS_M = 6_371_000.0
GM_EARTH = 3.986004418e14

# ITU-R P.838-3 coefficients (horizontal polarisation): frequency GHz -> (k, alpha); approximate, log-log interpolated
_RAIN = np.array(
    [
        (4.0, 0.0001071, 1.6009),
        (6.0, 0.0007056, 1.5900),
        (8.0, 0.004115, 1.3905),
        (10.0, 0.01217, 1.2571),
        (12.0, 0.02386, 1.1825),
        (15.0, 0.04481, 1.1233),
        (20.0, 0.09164, 1.0568),
        (25.0, 0.1571, 0.9991),
        (30.0, 0.2403, 0.9485),
        (35.0, 0.3374, 0.9047),
        (40.0, 0.4431, 0.8673),
    ]
)


def slant_range_m(elevation_deg: Any, altitude_m: float, ut_height_m: float = 0.0) -> np.ndarray:
    """Distance from a terminal to a satellite at ``altitude_m`` seen at elevation ``e``: ``sqrt((Re+h)^2 - (Rc cos e)^2) - Rc sin e`` with ``Rc = Re + h_ut``."""
    e = np.radians(np.asarray(elevation_deg, float))
    rc = EARTH_RADIUS_M + ut_height_m
    rs = EARTH_RADIUS_M + altitude_m
    return np.sqrt(rs**2 - (rc * np.cos(e)) ** 2) - rc * np.sin(e)


def orbital_speed_mps(altitude_m: float) -> float:
    return math.sqrt(GM_EARTH / (EARTH_RADIUS_M + altitude_m))


def orbital_period_s(altitude_m: float) -> float:
    return 2.0 * math.pi * math.sqrt((EARTH_RADIUS_M + altitude_m) ** 3 / GM_EARTH)


def pass_geometry(
    altitude_m: float, max_elevation_deg: float = 90.0, dt: float = 1.0, min_elevation_deg: float = 10.0, ut_height_m: float = 0.0
) -> Dict[str, np.ndarray]:
    """Time series of a satellite pass over a terminal: elevation, slant range and range rate (positive = receding).

    The satellite follows a great-circle arc in the plane containing the terminal; ``max_elevation_deg`` fixes the cross-track
    offset (90 = overhead). The pass lasts while elevation >= ``min_elevation_deg``. Earth rotation is neglected (the terminal
    is the UAV, whose own velocity is negligible against 7.5 km/s).
    """
    rs = EARTH_RADIUS_M + altitude_m
    rc = EARTH_RADIUS_M + ut_height_m
    omega = orbital_speed_mps(altitude_m) / rs  # orbital angular rate
    # central angle at closest approach: elevation e <-> central angle psi, with cos(e + psi) = rc / rs * cos(e)
    e_max = math.radians(max_elevation_deg)
    psi_min = math.acos(min(rc / rs * math.cos(e_max), 1.0)) - e_max if max_elevation_deg < 90 else 0.0
    e_min = math.radians(min_elevation_deg)
    psi_visible = math.acos(min(rc / rs * math.cos(e_min), 1.0)) - e_min  # central angle at the elevation mask
    # along-track half-angle so that the total central angle reaches psi_visible
    half = math.acos(min(math.cos(psi_visible) / max(math.cos(psi_min), 1e-12), 1.0))
    t_half = half / omega
    t = np.arange(-t_half, t_half + dt / 2, dt)
    along = omega * t
    cos_psi = np.cos(psi_min) * np.cos(along)
    d = np.sqrt(rs**2 + rc**2 - 2.0 * rs * rc * cos_psi)
    sin_e = (rs * cos_psi - rc) / d
    elevation = np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))
    range_rate = np.gradient(d, t) if len(t) > 1 else np.zeros_like(d)
    return dict(t=t - t[0], elevation_deg=elevation, range_m=d, range_rate_mps=range_rate)


def doppler_shift_hz(range_rate_mps: Any, freq_hz: float) -> np.ndarray:
    """Received-frequency shift ``-(dR/dt) f / c`` (approaching, ``dR/dt < 0``, raises the frequency)."""
    return -np.asarray(range_rate_mps, float) * freq_hz / C_LIGHT


def gaseous_attenuation_db(elevation_deg: Any, freq_ghz: float = 2.0, zenith_db: Optional[float] = None) -> np.ndarray:
    """Gaseous (oxygen + water vapour) attenuation: a zenith value scaled by ``1 / sin(e)`` (flat-earth, above 5 degrees).

    Default zenith attenuation: 0.035 dB below 10 GHz, 0.1 dB at 12-20 GHz, 0.3 dB at 30 GHz (typical clear-sky mid-latitude values).
    """
    if zenith_db is None:
        zenith_db = 0.035 if freq_ghz < 10 else (0.1 if freq_ghz < 22 else 0.3)
    e = np.maximum(np.radians(np.asarray(elevation_deg, float)), math.radians(5.0))
    return zenith_db / np.sin(e)


def rain_attenuation_db(
    freq_ghz: float, rain_rate_mm_h: float, elevation_deg: Any, rain_height_km: float = 3.0, ut_height_km: float = 0.0
) -> np.ndarray:
    """Rain attenuation ``gamma_R L_eff``: specific attenuation ``k R^alpha`` times the slant path through the rain layer with a
    simple horizontal reduction factor ``1 / (1 + L_slant cos e / L_0)`` (``L_0 = 35 exp(-0.015 R)`` km). 0 below 4 GHz or in clear sky."""
    if rain_rate_mm_h <= 0 or freq_ghz < 3.0:
        return np.zeros_like(np.asarray(elevation_deg, float))
    if freq_ghz > _RAIN[-1, 0] + 1e-9:
        raise ValueError("rain model is tabulated up to 40 GHz")
    f = max(freq_ghz, _RAIN[0, 0])
    k = math.exp(np.interp(math.log(f), np.log(_RAIN[:, 0]), np.log(_RAIN[:, 1])))
    alpha = float(np.interp(math.log(f), np.log(_RAIN[:, 0]), _RAIN[:, 2]))
    gamma = k * rain_rate_mm_h**alpha  # dB/km
    e = np.maximum(np.radians(np.asarray(elevation_deg, float)), math.radians(5.0))
    l_slant = max(rain_height_km - ut_height_km, 0.0) / np.sin(e)
    l0 = 35.0 * math.exp(-0.015 * min(rain_rate_mm_h, 100.0))
    return gamma * l_slant / (1.0 + l_slant * np.cos(e) / l0)


@dataclass
class SatelliteLink:
    """Downlink budget from a satellite to a UAV terminal.

    ``eirp_dbw`` is the satellite EIRP towards the terminal, ``g_over_t_dbk`` the terminal figure of merit (a small UAV patch:
    about -20 dB/K at S-band, +10 dB/K for a VSAT dish at Ku).
    """

    altitude_m: float = 550_000.0
    freq_hz: float = 2.0e9
    bandwidth_hz: float = 5e6
    eirp_dbw: float = 34.0
    g_over_t_dbk: float = -12.0
    rain_rate_mm_h: float = 0.0
    implementation_loss_db: float = 2.0
    pointing_loss_db: float = 0.5
    se_max: float = 4.4

    def budget(self, elevation_deg: Any, ut_height_m: float = 0.0) -> Dict[str, np.ndarray]:
        e = np.asarray(elevation_deg, float)
        d = slant_range_m(e, self.altitude_m, ut_height_m)
        fspl = fspl_db(d, self.freq_hz)
        gas = gaseous_attenuation_db(e, self.freq_hz / 1e9)
        rain = rain_attenuation_db(self.freq_hz / 1e9, self.rain_rate_mm_h, e)
        loss = fspl + gas + rain + self.pointing_loss_db + self.implementation_loss_db
        cn0 = self.eirp_dbw - loss + self.g_over_t_dbk - 10.0 * math.log10(K_BOLTZMANN)  # dB-Hz
        snr = cn0 - 10.0 * math.log10(self.bandwidth_hz)
        from ..phy.bler_curves import attenuated_shannon_se

        rate = self.bandwidth_hz * attenuated_shannon_se(snr, se_max=self.se_max)
        return dict(range_m=d, fspl_db=fspl, gas_db=gas, rain_db=rain, cn0_dbhz=cn0, snr_db=snr, rate_bps=rate)

    def one_way_delay_s(self, elevation_deg: Any, ut_height_m: float = 0.0) -> np.ndarray:
        return slant_range_m(elevation_deg, self.altitude_m, ut_height_m) / C_LIGHT

    def max_doppler_hz(self) -> float:
        """Largest Doppler shift on a pass: orbital speed projected on the line of sight at the horizon, about ``v f / c`` times ``Re / (Re + h)``."""
        return orbital_speed_mps(self.altitude_m) * EARTH_RADIUS_M / (EARTH_RADIUS_M + self.altitude_m) * self.freq_hz / C_LIGHT

    def pass_profile(
        self, max_elevation_deg: float = 90.0, dt: float = 1.0, min_elevation_deg: float = 10.0, ut_height_m: float = 0.0
    ) -> Dict[str, Any]:
        g = pass_geometry(self.altitude_m, max_elevation_deg, dt, min_elevation_deg, ut_height_m)
        b = self.budget(g["elevation_deg"], ut_height_m)
        out: Dict[str, Any] = {**g, **b}
        out["doppler_hz"] = doppler_shift_hz(g["range_rate_mps"], self.freq_hz)
        out["doppler_rate_hz_s"] = np.gradient(out["doppler_hz"], g["t"]) if len(g["t"]) > 1 else np.zeros_like(g["t"])
        return out
