"""Small-scale fading amplitude/power distributions (unit mean power) and their analytic properties."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = [
    "nakagami_gain",
    "rayleigh_gain",
    "rayleigh_outage_probability",
    "rician_gain",
    "rician_k_from_nakagami_m",
    "rician_outage_probability",
    "nakagami_m_from_rician_k",
]


def _rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def rayleigh_gain(shape: Any, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Power gain of Rayleigh fading: exponential with unit mean."""
    return _rng(rng).exponential(1.0, size=shape)


def rician_gain(k_factor: Any, shape: Any = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Power gain of Rician fading with linear K-factor (LoS-to-scattered power ratio); unit mean, ``K = 0`` is Rayleigh."""
    k = np.asarray(k_factor, float)
    shape = np.broadcast(k, np.empty(shape if shape is not None else ())).shape
    r = _rng(rng)
    s = np.sqrt(k / (k + 1.0))
    sigma = np.sqrt(1.0 / (2.0 * (k + 1.0)))
    x = s + sigma * r.normal(size=shape)
    y = sigma * r.normal(size=shape)
    return x * x + y * y


def nakagami_gain(m: Any, shape: Any = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Power gain of Nakagami-m fading: Gamma(m, 1/m), unit mean (``m = 1`` is Rayleigh, ``m -> inf`` no fading); ``m >= 0.5``."""
    m = np.asarray(m, float)
    if np.any(m < 0.5):
        raise ValueError("Nakagami m must be >= 0.5")
    shape = np.broadcast(m, np.empty(shape if shape is not None else ())).shape
    return _rng(rng).gamma(m, 1.0 / m, size=shape)


def nakagami_m_from_rician_k(k_factor: Any) -> np.ndarray:
    """Moment-matching Nakagami ``m = (K + 1)^2 / (2 K + 1)`` for a given Rician K (exact for the first two moments of the power)."""
    k = np.asarray(k_factor, float)
    return (k + 1.0) ** 2 / (2.0 * k + 1.0)


def rician_k_from_nakagami_m(m: Any) -> np.ndarray:
    """Inverse of :func:`nakagami_m_from_rician_k` for ``m >= 1``: ``K = sqrt(m^2 - m) / (m - sqrt(m^2 - m))``."""
    m = np.asarray(m, float)
    root = np.sqrt(np.maximum(m * m - m, 0.0))
    return root / np.maximum(m - root, 1e-12)


def rayleigh_outage_probability(threshold: Any) -> np.ndarray:
    """``P[g < threshold]`` for unit-mean exponential power: ``1 - exp(-threshold)``."""
    return 1.0 - np.exp(-np.asarray(threshold, float))


def rician_outage_probability(threshold: Any, k_factor: float) -> np.ndarray:
    """``P[g < threshold]`` for unit-mean Rician power: ``1 - Q_1(sqrt(2K), sqrt(2 (K + 1) threshold))`` (Marcum Q via the non-central chi-square)."""
    from scipy.stats import ncx2

    x = np.asarray(threshold, float)
    return ncx2.cdf(2.0 * (k_factor + 1.0) * x, 2, 2.0 * k_factor)
