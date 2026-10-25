"""OFDM link abstraction: map per-resource-block SINRs of a frequency-selective channel to one effective SINR (EESM).

Exponential effective SINR mapping: ``SINR_eff = -beta ln( mean_k exp(-SINR_k / beta) )`` (SINR linear). ``beta`` grows with the
modulation order. The values below are *indicative* (the literature spans roughly a factor of two per modulation and
depends on code rate); calibrate them against your own link-level curves when the absolute BLER matters.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from ..units import db2lin, lin2db

__all__ = ["EESM_BETA", "eesm_effective_sinr", "eesm_effective_sinr_db", "per_rb_sinr_db"]

EESM_BETA: Dict[int, float] = {2: 2.0, 4: 6.0, 6: 15.0, 8: 40.0}  # by modulation order Qm


def eesm_effective_sinr(sinr_linear: Any, beta: float, axis: int = -1) -> np.ndarray:
    """Effective SINR (linear) of resource blocks along ``axis``; equals the arithmetic mean as ``beta -> inf`` and the minimum as ``beta -> 0``."""
    x = np.asarray(sinr_linear, float)
    lo = x.min(axis=axis, keepdims=True)  # shift by the minimum: exp(-(x - lo) / beta) cannot underflow to all zeros
    m = np.mean(np.exp(-(x - lo) / beta), axis=axis)
    return np.squeeze(lo, axis=axis) - beta * np.log(m)


def eesm_effective_sinr_db(sinr_db: Any, qm: int = 4, axis: int = -1) -> np.ndarray:
    """Convenience wrapper in dB with the modulation-dependent ``beta``."""
    return lin2db(eesm_effective_sinr(db2lin(sinr_db), EESM_BETA[qm], axis))


def per_rb_sinr_db(mean_sinr_db: float, h_f: Any, n_rb: int) -> np.ndarray:
    """Per-RB SINR from a complex frequency response sampled on a uniform grid: ``mean_sinr * |H|^2`` averaged within each RB (unit-power ``H``)."""
    g = np.abs(np.asarray(h_f, complex)) ** 2
    chunks = np.array_split(g, n_rb)
    return mean_sinr_db + lin2db(np.array([c.mean() for c in chunks]))


def mutual_information_gap_db(sinr_db_flat: float, sinr_db_selective: Any, qm: int = 4) -> float:
    """Loss [dB] of a frequency-selective channel relative to a flat one with the same mean SINR (EESM): non-negative by Jensen."""
    eff = float(eesm_effective_sinr_db(sinr_db_selective, qm))
    return float(sinr_db_flat - eff) if math.isfinite(eff) else float("inf")
