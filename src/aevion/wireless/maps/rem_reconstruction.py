"""Radio-map reconstruction from sparse measurements (what a UAV can collect by flying): IDW, RBF and ordinary kriging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Optional, Tuple

import numpy as np

__all__ = ["Reconstruction", "empirical_variogram", "fit_exponential_variogram", "idw", "kriging", "reconstruct", "rbf", "rmse"]


def _pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def idw(xy: Any, values: Any, query: Any, power: float = 2.0, k: Optional[int] = 12) -> np.ndarray:
    """Inverse-distance weighting using the ``k`` nearest samples (all if ``None``)."""
    xy, v, q = np.asarray(xy, float), np.asarray(values, float), np.asarray(query, float)
    d = _pairs(q, xy)
    if k is not None and k < len(xy):
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        dd = np.take_along_axis(d, idx, 1)
        vv = v[idx]
    else:
        dd, vv = d, np.broadcast_to(v, d.shape)
    w = 1.0 / np.maximum(dd, 1e-9) ** power
    return (w * vv).sum(axis=1) / w.sum(axis=1)


def rbf(xy: Any, values: Any, query: Any, kernel: str = "thin_plate_spline", smoothing: float = 0.0) -> np.ndarray:
    """Radial-basis-function interpolation (SciPy)."""
    from scipy.interpolate import RBFInterpolator  # imported lazily: it is a heavy import

    return RBFInterpolator(np.asarray(xy, float), np.asarray(values, float), kernel=kernel, smoothing=smoothing)(np.asarray(query, float))


def empirical_variogram(xy: Any, values: Any, n_bins: int = 12, max_dist: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Binned semivariance ``gamma(h) = 0.5 E[(z(x) - z(x+h))^2]``: ``(lags, gamma)``."""
    xy, v = np.asarray(xy, float), np.asarray(values, float)
    d = _pairs(xy, xy)
    iu = np.triu_indices(len(xy), 1)
    dist, sq = d[iu], 0.5 * (v[:, None] - v[None, :])[iu] ** 2
    hi = max_dist if max_dist is not None else dist.max() / 2.0
    edges = np.linspace(0.0, hi, n_bins + 1)
    lags, gam = [], []
    for a, b in pairwise(edges):
        m = (dist > a) & (dist <= b)
        if m.sum() >= 3:
            lags.append(dist[m].mean())
            gam.append(sq[m].mean())
    return np.array(lags), np.array(gam)


def fit_exponential_variogram(lags: Any, gamma: Any) -> Tuple[float, float, float]:
    """Fit ``gamma(h) = nugget + sill (1 - exp(-3 h / range))`` by grid search over the range and linear least squares; returns ``(nugget, sill, range)``."""
    h, g = np.asarray(lags, float), np.asarray(gamma, float)
    best = (np.inf, 0.0, float(g.max()) if len(g) else 1.0, float(h.max()) if len(h) else 1.0)
    for rng_ in np.linspace(max(h.min(), 1e-3), 3.0 * h.max(), 60):
        basis = 1.0 - np.exp(-3.0 * h / rng_)
        A = np.stack([np.ones_like(h), basis], axis=1)
        coef, *_ = np.linalg.lstsq(A, g, rcond=None)
        nug, sill = max(coef[0], 0.0), max(coef[1], 1e-9)
        err = float(np.sum((nug + sill * basis - g) ** 2))
        if err < best[0]:
            best = (err, nug, sill, float(rng_))
    return best[1], best[2], best[3]


def kriging(xy: Any, values: Any, query: Any, variogram: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with an exponential variogram. Returns ``(prediction, kriging_std)``; the variogram is fitted if not given."""
    xy, v, q = np.asarray(xy, float), np.asarray(values, float), np.asarray(query, float)
    if variogram is None:
        variogram = fit_exponential_variogram(*empirical_variogram(xy, v))
    nug, sill, rng_ = variogram

    def gam(h: np.ndarray) -> np.ndarray:
        return np.where(h > 0, nug + sill * (1.0 - np.exp(-3.0 * h / rng_)), 0.0)

    n = len(xy)
    K = np.ones((n + 1, n + 1))
    K[:n, :n] = gam(_pairs(xy, xy))
    K[n, n] = 0.0
    Kinv = np.linalg.pinv(K)
    g0 = np.ones((len(q), n + 1))
    g0[:, :n] = gam(_pairs(q, xy))
    lam = g0 @ Kinv.T  # (Q, n+1)
    pred = lam[:, :n] @ v
    var = np.maximum((lam * g0).sum(axis=1), 0.0)
    return pred, np.sqrt(var)


@dataclass
class Reconstruction:
    method: str
    prediction: np.ndarray
    std: Optional[np.ndarray]


def reconstruct(xy: Any, values: Any, query: Any, method: str = "kriging", **kw: Any) -> Reconstruction:
    """Interpolate measurements to ``query`` points with ``method`` in ``idw``, ``rbf``, ``kriging``."""
    if method == "idw":
        return Reconstruction(method, idw(xy, values, query, **kw), None)
    if method == "rbf":
        return Reconstruction(method, rbf(xy, values, query, **kw), None)
    if method == "kriging":
        p, s = kriging(xy, values, query, **kw)
        return Reconstruction(method, p, s)
    raise ValueError("method must be 'idw', 'rbf' or 'kriging'")


def rmse(pred: Any, truth: Any) -> float:
    d = np.asarray(pred, float) - np.asarray(truth, float)
    return float(math.sqrt(np.mean(d * d)))
