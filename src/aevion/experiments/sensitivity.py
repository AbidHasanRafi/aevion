"""Global sensitivity analysis: Morris elementary effects and Sobol variance-based indices.

Both methods separate *design* (the points in the unit hypercube at which the model is evaluated) from *analysis* (indices from the
outputs), so the evaluations can run anywhere (process pool, cluster, cache). :func:`morris` and :func:`sobol_indices` join the two.

**Morris** (Morris 1991; Campolongo, Cariboni and Saltelli 2007): ``r`` random one-factor-at-a-time trajectories; the elementary
effect of factor ``i`` is ``(f(x + Δe_i) − f(x)) / Δ``. ``μ*`` (mean absolute effect) ranks importance and ``σ`` flags nonlinearity or
interactions. Cost: ``r (d + 1)`` evaluations.

**Sobol** (Saltelli 2010 for first order, Jansen 1999 for total effect): from independent sample matrices ``A`` and ``B`` and the
hybrids ``AB_i`` (``A`` with column ``i`` from ``B``), ``S_i = E[f(B)(f(AB_i) − f(A))] / Var(f)`` and
``S_Ti = E[(f(A) − f(AB_i))²] / (2 Var(f))``. Cost: ``n (d + 2)`` evaluations. The tests recover the analytic indices of the
Ishigami and Sobol g-functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from .design import sobol as sobol_points

__all__ = ["morris_design", "morris_analyze", "morris", "MorrisResult", "saltelli_design", "sobol_analyze", "sobol_indices", "SobolResult"]


def _scale(u: np.ndarray, bounds: Optional[Sequence[Tuple[float, float]]]) -> np.ndarray:
    if bounds is None:
        return u
    b = np.asarray(bounds, float)
    if b.shape != (u.shape[1], 2) or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("bounds must be (d, 2) with upper > lower")
    return b[:, 0] + u * (b[:, 1] - b[:, 0])


def _evaluate(f: Callable[[np.ndarray], Any], X: np.ndarray, vectorized: bool) -> np.ndarray:
    y = np.asarray(f(X), float).reshape(-1) if vectorized else np.array([float(f(x)) for x in X])
    if y.shape != (len(X),):
        raise ValueError("the model must return one value per input point")
    if not np.all(np.isfinite(y)):
        raise ValueError("the model returned non-finite values")
    return y


# ------------------------------------------------------------------------------------------------ Morris
@dataclass
class MorrisResult:
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    effects: np.ndarray  # (r, d) elementary effects

    def ranking(self) -> np.ndarray:
        """Factor indices from most to least influential (by ``μ*``)."""
        return np.argsort(-self.mu_star)


def morris_design(d: int, r: int = 20, levels: int = 4, seed: int = 0) -> np.ndarray:
    """Stacked trajectories ``(r (d + 1), d)`` on a ``levels``-level grid with step ``Δ = levels / (2 (levels − 1))`` (``levels`` even)."""
    if d < 1 or r < 1:
        raise ValueError("d and r must be at least 1")
    if levels < 2 or levels % 2:
        raise ValueError("levels must be even and at least 2")
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    base_pool = grid[grid <= 1.0 - delta + 1e-12]
    out = []
    for _ in range(r):
        x = rng.choice(base_pool, d)
        traj = [x.copy()]
        for i in rng.permutation(d):
            up = x[i] + delta <= 1.0 + 1e-12
            down = x[i] - delta >= -1e-12
            step = delta if (up and (not down or rng.random() < 0.5)) else -delta
            x = x.copy()
            x[i] += step
            traj.append(x.copy())
        out.append(np.array(traj))
    return np.concatenate(out)


def morris_analyze(X: np.ndarray, y: np.ndarray, r: int, levels: int = 4) -> MorrisResult:
    """Elementary effects from the outputs ``y`` of the points ``X`` produced by :func:`morris_design` (inputs in the unit cube)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    d = X.shape[1]
    if len(X) != r * (d + 1) or len(y) != len(X):
        raise ValueError("X and y do not match a design with this r")
    effects = np.zeros((r, d))
    for t in range(r):
        P, v = X[t * (d + 1) : (t + 1) * (d + 1)], y[t * (d + 1) : (t + 1) * (d + 1)]
        for k in range(d):
            dx = P[k + 1] - P[k]
            i = int(np.argmax(np.abs(dx)))
            effects[t, i] = (v[k + 1] - v[k]) / dx[i]
    return MorrisResult(effects.mean(0), np.abs(effects).mean(0), effects.std(0, ddof=1) if r > 1 else np.zeros(d), effects)


def morris(
    f: Callable[[np.ndarray], Any], bounds: Sequence[Tuple[float, float]], r: int = 20, levels: int = 4, seed: int = 0, vectorized: bool = False
) -> MorrisResult:
    """Run the Morris screening on ``f`` over a box. Effects are per unit of the *unit cube* (a factor with a wider range shows a proportionally larger effect)."""
    X = morris_design(len(bounds), r, levels, seed)
    y = _evaluate(f, _scale(X, bounds), vectorized)
    return morris_analyze(X, y, r, levels)


# ------------------------------------------------------------------------------------------------ Sobol
@dataclass
class SobolResult:
    s1: np.ndarray
    st: np.ndarray
    s1_ci: np.ndarray  # (d, 2) bootstrap 95 % interval
    st_ci: np.ndarray
    variance: float
    n: int

    def ranking(self) -> np.ndarray:
        return np.argsort(-self.st)

    def interactions(self) -> np.ndarray:
        """``S_T − S_1``: the share of the variance a factor contributes only through interactions."""
        return self.st - self.s1


def saltelli_design(d: int, n: int, seed: int = 0) -> np.ndarray:
    """Stacked evaluation points ``[A; B; AB_1; …; AB_d]`` of shape ``((d + 2) n, d)`` in the unit cube (a scrambled Sobol sequence of dimension ``2 d``)."""
    if d < 1 or n < 2:
        raise ValueError("need d >= 1 and n >= 2")
    S = sobol_points(n, 2 * d, seed)
    A, B = S[:, :d], S[:, d:]
    blocks = [A, B]
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        blocks.append(AB)
    return np.concatenate(blocks)


def _estimators(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    var = float(np.var(np.concatenate([fA, fB]), ddof=1))
    if var <= 0:
        raise ValueError("the model output has zero variance over the sampled region")
    s1 = (fB[None, :] * (fAB - fA[None, :])).mean(axis=1) / var
    st = 0.5 * ((fA[None, :] - fAB) ** 2).mean(axis=1) / var
    return s1, st, var


def sobol_analyze(y: np.ndarray, d: int, n: int, n_boot: int = 200, seed: int = 0) -> SobolResult:
    """Indices and bootstrap intervals from the outputs ``y`` of the points of :func:`saltelli_design`."""
    y = np.asarray(y, float)
    if len(y) != (d + 2) * n:
        raise ValueError("y does not match a Saltelli design of this size")
    fA, fB = y[:n], y[n : 2 * n]
    fAB = y[2 * n :].reshape(d, n)
    s1, st, var = _estimators(fA, fB, fAB)
    rng = np.random.default_rng(seed)
    b1, bt = [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        try:
            a, t, _ = _estimators(fA[idx], fB[idx], fAB[:, idx])
        except ValueError:
            continue
        b1.append(a)
        bt.append(t)
    ci = lambda arr: np.percentile(np.array(arr), [2.5, 97.5], axis=0).T if arr else np.full((d, 2), np.nan)  # noqa: E731
    return SobolResult(s1, st, ci(b1), ci(bt), var, n)


def sobol_indices(
    f: Callable[[np.ndarray], Any], bounds: Sequence[Tuple[float, float]], n: int = 1024, seed: int = 0, vectorized: bool = False, n_boot: int = 200
) -> SobolResult:
    """First-order and total Sobol indices of ``f`` over a box with ``n (d + 2)`` evaluations."""
    d = len(bounds)
    X = saltelli_design(d, n, seed)
    y = _evaluate(f, _scale(X, bounds), vectorized)
    return sobol_analyze(y, d, n, n_boot, seed)
