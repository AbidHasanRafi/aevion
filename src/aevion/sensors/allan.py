"""Allan variance: characterising inertial-sensor noise, and the analytic curves used to parameterise the IMU model.

``allan_deviation`` computes the overlapping Allan deviation of a rate signal (gyro in rad/s, accelerometer in m/s^2). ``fit_noise``
decomposes a measured curve into the IEEE Std 952 terms (angle/velocity random walk ``N``, bias instability ``B``, rate random walk
``K``, quantisation ``Q``, rate ramp ``R``) by non-negative least squares, and ``model_curve`` evaluates a parameter set analytically.

IEEE 952 (rate sensors)::

    sigma^2(tau) = 3 Q^2 / tau^2  +  N^2 / tau  +  (2 ln2 / pi) B^2  +  K^2 tau / 3  +  R^2 tau^2 / 2

with ``N`` in rad/s/sqrt(Hz), ``B`` the 1/f coefficient (the Allan floor is ``sqrt(2 ln 2 / pi) B = 0.664 B``) and ``K`` in rad/s/sqrt(s).
Accelerometers use m/s^2 in place of rad/s.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import nnls

__all__ = ["allan_variance", "allan_deviation", "fit_noise", "model_curve", "gauss_markov_avar", "FLICKER_FLOOR", "log_taus"]

FLICKER_FLOOR = math.sqrt(2.0 * math.log(2.0) / math.pi)  # 0.664: Allan floor / B


def log_taus(n_samples: int, dt: float, per_decade: int = 8, min_clusters: int = 9) -> np.ndarray:
    """Log-spaced averaging times from ``dt`` up to the longest the record supports (at least ``min_clusters`` clusters)."""
    m_max = max(1, (n_samples - 1) // (2 * min_clusters // 2 + 1) // 2)
    m = np.unique(np.round(10.0 ** np.arange(0.0, math.log10(max(m_max, 1)) + 1e-9, 1.0 / per_decade)).astype(int))
    return m * dt


def allan_variance(rate: np.ndarray, dt: float, taus: Optional[Union[Sequence[float], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan variance of ``rate`` ``(T,)`` or ``(T, S)`` (S independent records: the variances are averaged, which
    tightens the estimate) sampled every ``dt`` seconds. Returns ``(taus, avar)``."""
    x = np.asarray(rate, float)
    if x.ndim == 1:
        x = x[:, None]
    T = x.shape[0]
    theta = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0) * dt])  # integrated signal (angle / velocity)
    if taus is None:
        taus = log_taus(T, dt)
    ms = np.unique(np.maximum(np.round(np.asarray(taus, float) / dt).astype(int), 1))
    ms = ms[2 * ms < T]
    av = np.empty(len(ms))
    for i, m in enumerate(ms):
        d = theta[2 * m :] - 2.0 * theta[m:-m] + theta[: -2 * m]  # (T + 1 - 2m, S)
        av[i] = np.mean(d * d) / (2.0 * (m * dt) ** 2)
    return ms * dt, av


def allan_deviation(rate: np.ndarray, dt: float, taus: Optional[Union[Sequence[float], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    t, av = allan_variance(rate, dt, taus)
    return t, np.sqrt(av)


def model_curve(taus: np.ndarray, N: float = 0.0, B: float = 0.0, K: float = 0.0, Q: float = 0.0, R: float = 0.0) -> np.ndarray:
    """Allan *deviation* of the IEEE 952 noise terms at ``taus``."""
    t = np.asarray(taus, float)
    return np.sqrt(3.0 * Q**2 / t**2 + N**2 / t + (2.0 * math.log(2.0) / math.pi) * B**2 + K**2 * t / 3.0 + R**2 * t**2 / 2.0)


def fit_noise(taus: np.ndarray, adev: np.ndarray, terms: Sequence[str] = ("N", "B", "K")) -> Dict[str, float]:
    """Non-negative least-squares decomposition of an Allan-deviation curve into the requested IEEE 952 terms.

    The fit is on the variance, weighted by ``1 / sigma^2`` so every decade counts equally. Returns the coefficients (``B`` is the 1/f
    coefficient; the bias-instability *floor* is ``0.664 B``).
    """
    t, a2 = np.asarray(taus, float), np.asarray(adev, float) ** 2
    basis = {
        "Q": 3.0 / t**2,
        "N": 1.0 / t,
        "B": np.full_like(t, 2.0 * math.log(2.0) / math.pi),
        "K": t / 3.0,
        "R": t**2 / 2.0,
    }
    unknown = set(terms) - set(basis)
    if unknown:
        raise ValueError(f"unknown term(s) {sorted(unknown)}; choose from {sorted(basis)}")
    A = np.stack([basis[k] for k in terms], axis=1)
    w = 1.0 / a2
    coef, _ = nnls(A * w[:, None], a2 * w)
    return {k: float(math.sqrt(max(c, 0.0))) for k, c in zip(terms, coef)}


def gauss_markov_avar(tau: np.ndarray, variance: float, tau_c: float) -> np.ndarray:
    """Allan variance of a first-order Gauss-Markov process with stationary ``variance`` and correlation time ``tau_c``."""
    t = np.asarray(tau, float)
    x = t / tau_c
    return 2.0 * variance / x**2 * (x + 2.0 * np.exp(-x) - 0.5 * np.exp(-2.0 * x) - 1.5)
