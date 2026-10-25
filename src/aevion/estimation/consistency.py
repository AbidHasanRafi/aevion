"""Filter consistency: NEES / NIS statistics and their chi-square acceptance intervals.

A filter is *consistent* when its reported covariance matches its actual error. For a state of dimension ``d`` the
normalised estimation error squared ``ε = eᵀ P⁻¹ e`` is chi-square with ``d`` degrees of freedom; averaged over ``N``
independent Monte Carlo runs, ``N ε̄`` is chi-square with ``N d`` degrees of freedom (Bar-Shalom, Li & Kirubarajan,
*Estimation with Applications to Tracking and Navigation*, 2001, §5.4). The same holds for the normalised innovation
squared (NIS) of a measurement of dimension ``m``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

__all__ = ["nees_interval", "average_nees", "fraction_inside", "consistency_report", "is_consistent"]


def nees_interval(dof: int, n_runs: int = 1, alpha: float = 0.05) -> Tuple[float, float]:
    """Two-sided ``1 - alpha`` acceptance interval for the *mean* of ``n_runs`` chi-square(``dof``) variables."""
    from scipy.stats import chi2

    lo = chi2.ppf(alpha / 2.0, n_runs * dof) / n_runs
    hi = chi2.ppf(1.0 - alpha / 2.0, n_runs * dof) / n_runs
    return float(lo), float(hi)


def average_nees(nees: Any) -> np.ndarray:
    """Mean over the Monte Carlo axis (the last one) of a ``(T, N)`` array of NEES values."""
    a = np.asarray(nees, float)
    return a.mean(axis=-1) if a.ndim > 1 else a


def fraction_inside(values: Any, dof: int, n_runs: int = 1, alpha: float = 0.05) -> float:
    """Share of the values that lie inside the acceptance interval (about ``1 - alpha`` for a consistent filter)."""
    lo, hi = nees_interval(dof, n_runs, alpha)
    v = np.asarray(values, float)
    return float(np.mean((v >= lo) & (v <= hi)))


def consistency_report(nees: Any, dof: int, alpha: float = 0.05) -> Dict[str, float]:
    """Summary of a ``(T, N)`` NEES array: time-average of the Monte Carlo mean, its interval, and whether it is inside.

    ``ratio`` is the average NEES divided by ``dof``: above 1 the filter is overconfident (covariance too small), below 1
    conservative.
    """
    a = np.asarray(nees, float)
    if a.ndim == 1:
        a = a[:, None]
    t, n = a.shape
    avg = float(a.mean())
    lo, hi = nees_interval(dof, n * t, alpha)  # the time average of independent runs is chi-square(N T dof) / (N T)
    return dict(average=avg, dof=float(dof), ratio=avg / dof, lower=lo, upper=hi, inside=float(lo <= avg <= hi), n_runs=float(n), n_times=float(t))


def is_consistent(nees: Any, dof: int, alpha: float = 0.05, tolerance: float = 0.0) -> bool:
    """Is the time-averaged NEES inside the acceptance interval, widened by a relative ``tolerance``?

    The ESKF is an *approximate* filter (first-order linearisation), so tests allow a small tolerance; a gross inconsistency
    (ratio far from 1) still fails.
    """
    r = consistency_report(nees, dof, alpha)
    return bool(r["lower"] * (1.0 - tolerance) <= r["average"] <= r["upper"] * (1.0 + tolerance))
