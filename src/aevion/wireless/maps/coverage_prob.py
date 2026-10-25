"""Coverage statistics and the classic UAV-base-station altitude/coverage trade-off (Al-Hourani et al., 2014).

For a UAV base station at altitude ``h`` serving ground users within horizontal radius ``R``, the mean path loss is
``PL(R, h) = FSPL(d) + P_LoS(theta) eta_LoS + (1 - P_LoS(theta)) eta_NLoS``, ``theta = atan(h / R)``. Raising the UAV increases
``P_LoS`` but also the distance, so the *maximum coverage radius* for a path-loss budget has an interior optimum altitude.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

from ..channel.air_to_ground import ENVIRONMENTS, los_probability
from ..propagation.base import fspl_db

__all__ = ["coverage_radius", "empirical_coverage", "mean_path_loss_db", "optimal_altitude"]


def mean_path_loss_db(radius_m: Any, altitude_m: float, freq_hz: float = 2.0e9, env: str = "urban") -> np.ndarray:
    """Probability-weighted Al-Hourani path loss at horizontal distance ``radius_m`` from a UAV at ``altitude_m``."""
    r = np.asarray(radius_m, float)
    d = np.hypot(r, altitude_m)
    theta = np.degrees(np.arctan2(altitude_m, np.maximum(r, 1e-9)))
    _, _, eta_los, eta_nlos = ENVIRONMENTS[env]
    p = los_probability(theta, env)
    return fspl_db(d, freq_hz) + p * eta_los + (1.0 - p) * eta_nlos


def coverage_radius(altitude_m: float, max_path_loss_db: float, freq_hz: float = 2.0e9, env: str = "urban", r_max: float = 50_000.0) -> float:
    """Largest horizontal radius with mean path loss <= ``max_path_loss_db`` (0 if the budget cannot be met even below the UAV)."""
    from scipy.optimize import brentq

    f = lambda r: float(mean_path_loss_db(r, altitude_m, freq_hz, env)) - max_path_loss_db  # noqa: E731
    if f(1e-3) > 0:
        return 0.0
    if f(r_max) <= 0:
        return r_max
    # the mean path loss is not monotone near the UAV; search from the far side for the outermost crossing
    grid = np.geomspace(1.0, r_max, 400)
    vals = np.array([f(r) for r in grid])
    inside = np.nonzero(vals <= 0)[0]
    if len(inside) == 0:  # the budget is met only within the first metre
        return float(brentq(f, 1e-3, grid[0]))
    k = int(inside.max())
    if k == len(grid) - 1:
        return r_max
    return float(brentq(f, grid[k], grid[k + 1]))


def optimal_altitude(
    max_path_loss_db: float, freq_hz: float = 2.0e9, env: str = "urban", h_range: Tuple[float, float] = (10.0, 3000.0)
) -> Tuple[float, float]:
    """``(altitude, radius)`` maximising the coverage radius for a path-loss budget (numerical search)."""
    from scipy.optimize import minimize_scalar

    hs = np.geomspace(h_range[0], h_range[1], 60)
    rs = np.array([coverage_radius(h, max_path_loss_db, freq_hz, env) for h in hs])
    i = int(np.argmax(rs))
    lo, hi = hs[max(i - 1, 0)], hs[min(i + 1, len(hs) - 1)]
    res = minimize_scalar(lambda h: -coverage_radius(h, max_path_loss_db, freq_hz, env), bounds=(lo, hi), method="bounded", options={"xatol": 0.5})
    return float(res.x), float(-res.fun)


def empirical_coverage(values: Any, threshold: float, weights: Any = None) -> float:
    """Fraction (optionally weighted) of samples at or above ``threshold``: the empirical coverage probability."""
    v = np.asarray(values, float)
    ok = (v >= threshold).astype(float)
    if weights is None:
        return float(ok.mean())
    w = np.asarray(weights, float)
    return float((ok * w).sum() / max(w.sum(), 1e-30))


def outage_probability_from_pathloss(max_path_loss_db: float, freq_hz: float, r: float, h: float, env: str = "urban") -> float:
    """Probability that a user at distance ``r`` is out of coverage under LoS/NLoS randomness (no shadowing): ``1 - P_LoS`` if only the LoS state meets the budget."""
    d = math.hypot(r, h)
    theta = math.degrees(math.atan2(h, max(r, 1e-9)))
    _, _, eta_los, eta_nlos = ENVIRONMENTS[env]
    base = float(fspl_db(d, freq_hz))
    p = float(los_probability(theta, env))
    ok_los, ok_nlos = base + eta_los <= max_path_loss_db, base + eta_nlos <= max_path_loss_db
    return float(1.0 - (p * ok_los + (1.0 - p) * ok_nlos))
