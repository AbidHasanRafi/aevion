"""Air-to-ground radio channel models.

References
----------
* A. Al-Hourani, S. Kandeepan, S. Lardner, "Optimal LAP altitude for maximum coverage",
  IEEE Wireless Commun. Lett., 2014 (probabilistic LoS + mean excess loss).
* 3GPP TR 36.777 (UAV channel measurements) and TR 36.814 (BS antenna pattern).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.constants import C_LIGHT, K_BOLTZMANN

# Al-Hourani (a, b, eta_LoS [dB], eta_NLoS [dB]) per environment class
ENVIRONMENTS = {
    "suburban": (4.88, 0.43, 0.1, 21.0),
    "urban": (9.61, 0.16, 1.0, 20.0),
    "dense_urban": (12.08, 0.11, 1.6, 23.0),
    "highrise": (27.23, 0.08, 2.3, 34.0),
}
SHADOWING_STD_DB = {"los": 4.0, "nlos": 6.0}


def fspl_db(distance_m, freq_hz):
    """Free-space path loss in dB: ``20 log10(4 π d f / c)``."""
    d = np.maximum(np.asarray(distance_m, float), 1e-3)
    return 20.0 * np.log10(4.0 * math.pi * d * freq_hz / C_LIGHT)


def elevation_deg(horizontal_m, height_m):
    return np.degrees(np.arctan2(height_m, np.maximum(horizontal_m, 1e-6)))


def los_probability(theta_deg, env: str = "urban"):
    """Al-Hourani LoS probability ``1 / (1 + a exp(-b (θ - a)))`` vs elevation angle θ [deg]."""
    a, b, *_ = ENVIRONMENTS[env]
    return 1.0 / (1.0 + a * np.exp(-b * (np.asarray(theta_deg, float) - a)))


def rician_k_db(theta_deg, k0_db: float = 5.0, k90_db: float = 15.0):
    """Elevation-dependent Rician K-factor for the LoS component (interpolated between horizon and zenith)."""
    return k0_db + (k90_db - k0_db) * np.asarray(theta_deg, float) / 90.0


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float = 7.0, temperature_k: float = 290.0) -> float:
    return 10.0 * math.log10(K_BOLTZMANN * temperature_k * bandwidth_hz / 1e-3) + noise_figure_db


def doppler_hz(vel_rx, tx_pos, rx_pos, freq_hz):
    """Doppler shift ``f_d = (v · u) f / c`` with ``u`` the unit vector from receiver to transmitter."""
    u = np.asarray(tx_pos, float) - np.asarray(rx_pos, float)
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
    return np.sum(np.asarray(vel_rx, float) * u, axis=-1) * freq_hz / C_LIGHT


@dataclass
class AirToGroundChannel:
    """Large- and small-scale air-to-ground channel.

    ``los_mode``:

    * ``"probabilistic"``: LoS drawn with the Al-Hourani probability (no geometry needed);
    * ``"geometric"``: LoS determined by ray casting against the :class:`World` obstacles
      (accounts for the actual buildings/trees between the UAV and the ground node);
    * ``"mean"``: deterministic expected path loss (probability-weighted), no randomness.
    """

    env: str = "urban"
    freq_hz: float = 2.0e9
    los_mode: str = "probabilistic"
    shadowing: bool = True
    fading: bool = False  # small-scale Rician/Rayleigh fading
    nlos_excess_db: Optional[float] = None  # override geometric-mode NLoS excess loss

    def __post_init__(self):
        if self.env not in ENVIRONMENTS:
            raise ValueError(f"env must be one of {sorted(ENVIRONMENTS)}")
        if self.los_mode not in ("probabilistic", "geometric", "mean"):
            raise ValueError("los_mode must be 'probabilistic', 'geometric' or 'mean'")

    def path_loss_db(self, tx, rx, rng: Optional[np.random.Generator] = None, world=None):
        """Path loss [dB] between positions ``tx`` and ``rx`` (``(..., 3)``). Returns ``(PL, los_mask, theta)``."""
        tx, rx = np.asarray(tx, float), np.asarray(rx, float)
        tx, rx = np.broadcast_arrays(tx, rx)
        d = np.linalg.norm(tx - rx, axis=-1)
        h = np.abs(tx[..., 2] - rx[..., 2])
        theta = elevation_deg(np.linalg.norm((tx - rx)[..., :2], axis=-1), h)
        _, _, eta_los, eta_nlos = ENVIRONMENTS[self.env]
        base = fspl_db(d, self.freq_hz)
        rng = rng or np.random.default_rng()
        if self.los_mode == "mean":
            p = los_probability(theta, self.env)
            return base + p * eta_los + (1 - p) * eta_nlos, p > 0.5, theta
        if self.los_mode == "probabilistic":
            los = rng.random(d.shape) < los_probability(theta, self.env)
        else:
            if world is None:
                raise ValueError("geometric LoS needs a world")
            flat_tx, flat_rx = tx.reshape(-1, 3), rx.reshape(-1, 3)
            los = world.line_of_sight(flat_tx, flat_rx).reshape(d.shape)
        nlos_loss = eta_nlos if self.nlos_excess_db is None else self.nlos_excess_db
        pl = base + np.where(los, eta_los, nlos_loss)
        if self.shadowing:
            sigma = np.where(los, SHADOWING_STD_DB["los"], SHADOWING_STD_DB["nlos"])
            pl = pl + rng.normal(size=d.shape) * sigma
        return pl, los, theta

    def fading_gain(self, los, theta_deg, rng: np.random.Generator):
        """Linear small-scale power gain (unit mean): Rician for LoS (elevation-dependent K), Rayleigh for NLoS."""
        if not self.fading:
            return np.ones_like(theta_deg, dtype=float)
        K = 10 ** (rician_k_db(theta_deg) / 10.0)
        K = np.where(los, K, 0.0)
        shape = np.shape(theta_deg)
        s = np.sqrt(K / (K + 1))
        sigma = np.sqrt(1.0 / (2 * (K + 1)))
        x = s + sigma * rng.normal(size=shape)
        y = sigma * rng.normal(size=shape)
        return x * x + y * y

    def to_dict(self):
        return dict(env=self.env, freq_hz=self.freq_hz, los_mode=self.los_mode, shadowing=self.shadowing, fading=self.fading)


# ---------------------------------------------------------------------- antennas
def sector_gain_dbi(elevation_below_horizon_deg, tilt_deg: float = 10.0, theta3db: float = 65.0, sla_db: float = 30.0, g_max: float = 8.0):
    """3GPP TR 36.814 vertical pattern of a down-tilted macro-cell sector antenna.

    ``elevation_below_horizon_deg`` is the angle of the target *below* the antenna horizon (negative for
    targets above it, e.g. UAVs): ``A(θ) = -min(12((θ - tilt)/θ3dB)², SLA)``.
    """
    th = np.asarray(elevation_below_horizon_deg, float)
    return g_max - np.minimum(12.0 * ((th - tilt_deg) / theta3db) ** 2, sla_db)
