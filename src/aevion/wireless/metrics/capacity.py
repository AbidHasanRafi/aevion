"""Capacity measures: Shannon, ergodic (analytic Rayleigh), finite blocklength (Polyanskiy et al., 2010) and fairness."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["ergodic_capacity_rayleigh", "finite_blocklength_rate", "jain_fairness", "shannon_capacity_bps", "shannon_capacity_bps_hz"]


def shannon_capacity_bps_hz(sinr_db: Any) -> np.ndarray:
    return np.log2(1.0 + 10.0 ** (np.asarray(sinr_db, float) / 10.0))


def shannon_capacity_bps(bandwidth_hz: float, sinr_db: Any) -> np.ndarray:
    return bandwidth_hz * shannon_capacity_bps_hz(sinr_db)


def ergodic_capacity_rayleigh(mean_snr_linear: Any) -> np.ndarray:
    """Ergodic capacity [bit/s/Hz] of a Rayleigh channel with mean SNR ``g``: ``e^(1/g) E1(1/g) / ln 2``."""
    from scipy.special import exp1

    g = np.asarray(mean_snr_linear, float)
    return np.exp(1.0 / g) * exp1(1.0 / g) / math.log(2.0)


def finite_blocklength_rate(snr_linear: Any, blocklength: int, error_prob: float = 1e-5) -> np.ndarray:
    """Maximal rate [bit per channel use] at blocklength ``n`` and error probability ``eps`` for the AWGN channel (normal approximation):
    ``R ~ C - sqrt(V / n) Q^-1(eps) + log2(n) / (2 n)``, ``V = (1 - (1 + snr)^-2) log2(e)^2``. The back-off from capacity is what
    ultra-reliable low-latency (URLLC) control links pay for short packets."""
    from scipy.special import ndtri

    g = np.asarray(snr_linear, float)
    c = np.log2(1.0 + g)
    v = (1.0 - 1.0 / (1.0 + g) ** 2) * math.log2(math.e) ** 2
    q_inv = -float(ndtri(error_prob))
    return np.maximum(c - np.sqrt(v / blocklength) * q_inv + math.log2(blocklength) / (2.0 * blocklength), 0.0)


def jain_fairness(x: Any) -> float:
    """Jain's index ``(sum x)^2 / (n sum x^2)``: 1 = perfectly equal, ``1/n`` = one user takes all."""
    v = np.asarray(x, float).ravel()
    if v.size == 0 or not np.any(v):
        return 1.0
    return float(v.sum() ** 2 / (v.size * np.sum(v * v)))
