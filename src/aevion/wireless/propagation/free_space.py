"""Simple analytic models: free space, two-ray ground reflection and log-distance."""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

from ..units import C_LIGHT
from .base import LinkGeometry, PathLossModel, _clip_d, _log10, fspl_db


class FreeSpace(PathLossModel):
    """Friis free-space loss: always line-of-sight, no shadowing. The upper bound on achievable link quality."""

    name = "free_space"
    d2d_range_m: Tuple[float, float] = (0.0, 1e6)

    def path_loss_db(self, g: LinkGeometry, los: Any = True) -> np.ndarray:
        return fspl_db(g.d3d, self.freq_hz)

    def los_probability(self, g: LinkGeometry) -> np.ndarray:
        return np.ones_like(g.d3d)

    def shadowing_std_db(self, g: LinkGeometry, los: Any = True) -> np.ndarray:
        return np.zeros_like(g.d3d)


class TwoRayGround(PathLossModel):
    """Two-ray ground-reflection model (perfectly reflecting flat ground, reflection coefficient -1).

    Exact two-path interference below the crossover distance would oscillate; this uses the standard asymptote:
    free space up to ``d_c = 4 pi h_t h_r / lambda`` and ``PL = 40 log10 d - 20 log10(h_t h_r)`` beyond it, which is
    the right *median* behaviour for terrestrial links and low-flying UAVs.
    """

    name = "two_ray"
    d2d_range_m: Tuple[float, float] = (1.0, 1e5)

    def crossover_m(self, h_bs: Any, h_ut: Any) -> np.ndarray:
        lam = C_LIGHT / self.freq_hz
        return 4.0 * math.pi * np.maximum(h_bs, 0.1) * np.maximum(h_ut, 0.1) / lam

    def path_loss_db(self, g: LinkGeometry, los: Any = True) -> np.ndarray:
        d = _clip_d(g.d3d, 1e-3)
        dc = self.crossover_m(g.h_bs, g.h_ut)
        far = 40.0 * _log10(d) - 20.0 * _log10(np.maximum(g.h_bs, 0.1) * np.maximum(g.h_ut, 0.1))
        return np.where(d <= dc, fspl_db(d, self.freq_hz), np.maximum(far, fspl_db(d, self.freq_hz)))

    def los_probability(self, g: LinkGeometry) -> np.ndarray:
        return np.ones_like(g.d3d)

    def shadowing_std_db(self, g: LinkGeometry, los: Any = True) -> np.ndarray:
        return np.zeros_like(g.d3d)


class LogDistance(PathLossModel):
    """``PL(d) = PL(d0) + 10 n log10(d / d0)`` with separate exponents for LoS and NLoS.

    ``PL(d0)`` is the free-space loss at the reference distance. Typical exponents: 2 (free space), 2.7-3.5 (urban),
    4-6 (obstructed indoor).
    """

    name = "log_distance"

    def __init__(
        self,
        freq_hz: float = 2.0e9,
        n_los: float = 2.0,
        n_nlos: float = 3.5,
        d0: float = 1.0,
        sigma_los_db: float = 4.0,
        sigma_nlos_db: float = 7.0,
        p_los_d0: float = 50.0,
    ):
        super().__init__(freq_hz)
        if min(n_los, n_nlos) <= 0 or d0 <= 0:
            raise ValueError("path-loss exponents and the reference distance must be positive")
        self.n_los, self.n_nlos, self.d0 = n_los, n_nlos, d0
        self.sigma_los, self.sigma_nlos = sigma_los_db, sigma_nlos_db
        self.p_los_d0 = p_los_d0  # exponential LoS-probability decay distance [m]

    def path_loss_db(self, g: LinkGeometry, los: Any = True) -> np.ndarray:
        n = np.where(np.asarray(los, bool), self.n_los, self.n_nlos)
        d = np.maximum(g.d3d, self.d0)
        return fspl_db(self.d0, self.freq_hz) + 10.0 * n * _log10(d / self.d0)

    def los_probability(self, g: LinkGeometry) -> np.ndarray:
        return np.exp(-np.maximum(g.d2d - self.d0, 0.0) / self.p_los_d0)

    def shadowing_std_db(self, g: LinkGeometry, los: Any = True) -> np.ndarray:
        return np.where(np.asarray(los, bool), self.sigma_los, self.sigma_nlos).astype(float) * np.ones_like(g.d3d)
