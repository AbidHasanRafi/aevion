"""Spatially correlated log-normal shadow fading (Gudmundson's exponential model).

``S_{k+1} = rho S_k + sqrt(1 - rho^2) sigma w``, ``rho = exp(-delta_d / d_corr)``: a UAV that moves little sees nearly the same
shadowing, one that moves far sees a fresh draw. Decorrelation distances follow 3GPP TR 38.901 Table 7.5-6.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["DECORRELATION_DISTANCE_M", "CorrelatedShadowing", "decorrelation_distance"]

#: (LoS, NLoS) decorrelation distance in metres per scenario (TR 38.901 Table 7.5-6, shadow fading row)
DECORRELATION_DISTANCE_M: Dict[str, Tuple[float, float]] = {
    "uma": (37.0, 50.0),
    "umi": (10.0, 13.0),
    "rma": (37.0, 120.0),
    "inh": (10.0, 6.0),
}


def decorrelation_distance(scenario: str, los: Any) -> np.ndarray:
    d_los, d_nlos = DECORRELATION_DISTANCE_M[scenario.lower()]
    return np.where(np.asarray(los, bool), d_los, d_nlos).astype(float)


class CorrelatedShadowing:
    """First-order autoregressive shadowing field for ``shape`` independent links (e.g. ``(n_uav, n_bs)``)."""

    def __init__(self, shape: Any, rng: Optional[np.random.Generator] = None):
        self.shape = tuple(np.atleast_1d(shape))
        self.rng = rng if rng is not None else np.random.default_rng()
        self._state: Optional[np.ndarray] = None  # unit-variance innovations

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        self._state = None

    def step(self, sigma_db: Any, moved_m: Any, d_corr_m: Any) -> np.ndarray:
        """New shadowing values [dB]. ``moved_m`` is the distance the terminal moved since the last call."""
        if self._state is None:
            self._state = self.rng.normal(size=self.shape)
        else:
            rho = np.exp(-np.maximum(np.asarray(moved_m, float), 0.0) / np.maximum(np.asarray(d_corr_m, float), 1e-9))
            self._state = rho * self._state + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * self.rng.normal(size=self.shape)
        return self._state * np.asarray(sigma_db, float)

    @staticmethod
    def autocorrelation(distance_m: Any, d_corr_m: float) -> np.ndarray:
        """Theoretical correlation ``exp(-d / d_corr)``."""
        return np.exp(-np.asarray(distance_m, float) / d_corr_m)
