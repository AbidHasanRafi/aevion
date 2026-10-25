"""Doppler spectrum, temporal correlation and level-crossing statistics of Rayleigh fading (Clarke/Jakes model)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..units import C_LIGHT

__all__ = [
    "average_fade_duration",
    "coherence_time",
    "jakes_autocorrelation",
    "jakes_spectrum",
    "level_crossing_rate",
    "max_doppler_hz",
]


def max_doppler_hz(speed_mps: Any, freq_hz: float) -> np.ndarray:
    """Maximum Doppler shift ``v f_c / c``."""
    return np.asarray(speed_mps, float) * freq_hz / C_LIGHT


def jakes_autocorrelation(tau_s: Any, f_d_hz: float) -> np.ndarray:
    """Normalised autocorrelation of the complex fading envelope, ``J0(2 pi f_d tau)``."""
    from scipy.special import j0  # imported lazily: scipy.special is a heavy import

    return j0(2.0 * math.pi * f_d_hz * np.asarray(tau_s, float))


def jakes_spectrum(f_hz: Any, f_d_hz: float) -> np.ndarray:
    """Normalised Doppler power spectrum ``1 / (pi f_d sqrt(1 - (f/f_d)^2))`` for ``|f| < f_d`` (0 outside; integrates to 1)."""
    f = np.asarray(f_hz, float)
    r = f / f_d_hz
    inside = np.abs(r) < 1.0
    return np.where(inside, 1.0 / (math.pi * f_d_hz * np.sqrt(np.where(inside, 1.0 - r * r, 1.0))), 0.0)


def coherence_time(f_d_hz: Any, correlation: float = 0.5) -> np.ndarray:
    """Time over which the envelope autocorrelation ``J0(2 pi f_d tau)`` first drops to ``correlation`` (exact root of J0).

    ``correlation = 0.5`` gives ``0.242 / f_d``; ``0.9`` gives ``0.0719 / f_d``. (Rules of thumb such as ``9 / (16 pi f_d)`` or
    ``0.423 / f_d`` correspond to other definitions; this one is unambiguous.)
    """
    if not 0.0 < correlation < 1.0:
        raise ValueError("correlation must be in (0, 1)")
    from scipy.optimize import brentq
    from scipy.special import j0

    x = brentq(lambda v: j0(v) - correlation, 0.0, 2.404825557695773)  # first zero of J0 bounds the root
    return x / (2.0 * math.pi * np.maximum(np.asarray(f_d_hz, float), 1e-12))


def level_crossing_rate(rho: Any, f_d_hz: float) -> np.ndarray:
    """Rayleigh LCR ``sqrt(2 pi) f_d rho exp(-rho^2)`` [1/s]; ``rho`` is the threshold relative to the RMS envelope."""
    rho = np.asarray(rho, float)
    return math.sqrt(2.0 * math.pi) * f_d_hz * rho * np.exp(-rho * rho)


def average_fade_duration(rho: Any, f_d_hz: float) -> np.ndarray:
    """Rayleigh AFD ``(exp(rho^2) - 1) / (rho f_d sqrt(2 pi))`` [s]."""
    rho = np.maximum(np.asarray(rho, float), 1e-9)
    return (np.exp(rho * rho) - 1.0) / (rho * f_d_hz * math.sqrt(2.0 * math.pi))
