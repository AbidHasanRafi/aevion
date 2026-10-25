"""Unit conversions and constants shared by the wireless models (all functions are vectorised NumPy)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.constants import C_LIGHT, K_BOLTZMANN

__all__ = [
    "C_LIGHT",
    "K_BOLTZMANN",
    "T0_K",
    "db2lin",
    "dbm2w",
    "lin2db",
    "noise_power_dbm",
    "w2dbm",
    "wavelength",
]

T0_K = 290.0  # IEEE standard noise reference temperature


def db2lin(x_db: Any) -> Any:
    """Decibel to linear power ratio."""
    return np.power(10.0, np.asarray(x_db, dtype=float) / 10.0)


def lin2db(x: Any, floor: float = 1e-30) -> Any:
    """Linear power ratio to decibel; values at or below ``floor`` map to ``10*log10(floor)`` instead of ``-inf``."""
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))


def dbm2w(p_dbm: Any) -> Any:
    return db2lin(p_dbm) * 1e-3


def w2dbm(p_w: Any) -> Any:
    return lin2db(np.asarray(p_w, dtype=float) / 1e-3)


def wavelength(freq_hz: Any) -> Any:
    return C_LIGHT / np.asarray(freq_hz, dtype=float)


def noise_power_dbm(bandwidth_hz: float, noise_figure_db: float = 7.0, temperature_k: float = T0_K) -> float:
    """Thermal noise power ``k T B`` plus the receiver noise figure, in dBm (-174 dBm/Hz at 290 K)."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth must be positive")
    return 10.0 * math.log10(K_BOLTZMANN * temperature_k * bandwidth_hz / 1e-3) + noise_figure_db
