"""Voronoi coverage control (Lloyd's algorithm).

Given a density ``φ`` over a region ``Q``, the locational cost ``H(p) = ∫_Q min_i ‖q − p_i‖² φ(q) dq`` measures how well agents at ``p`` cover it. Each
agent moving towards the density-weighted centroid of its Voronoi cell decreases ``H`` monotonically, and the fixed points are the centroidal Voronoi
configurations (Cortés, Martínez, Karatas and Bullo, *IEEE T-RA* 2004; Du, Faber and Gunzburger, *SIAM Review* 1999). The region is discretised on a
regular grid, which makes the cell, centroid and cost computations exact for the discrete measure: the tests compare them with closed forms
(for a uniform line segment the optimum is ``p_i = (2i − 1)/(2n)`` with ``H = 1/(12 n²)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

__all__ = ["CoverageDomain", "voronoi_assign", "coverage_cost", "centroids", "lloyd_step", "lloyd", "LloydResult", "cell_masses"]


class CoverageDomain:
    """A rectangular box sampled on a regular grid, optionally with a density and a mask (e.g. to exclude obstacles).

    ``bounds = (lo..., hi...)`` in 1, 2 or 3 dimensions; ``resolution`` is the cell edge. Grid points are cell centres, so the discrete measure of the
    whole box equals its volume exactly.
    """

    def __init__(
        self,
        bounds: Sequence[float],
        resolution: float,
        density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        mask: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        b = np.asarray(bounds, float)
        if b.ndim != 1 or len(b) % 2 or not (1 <= len(b) // 2 <= 3):
            raise ValueError("bounds must be (lo..., hi...) in 1, 2 or 3 dimensions")
        d = len(b) // 2
        lo, hi = b[:d], b[d:]
        if np.any(hi <= lo) or resolution <= 0:
            raise ValueError("bounds must satisfy hi > lo and resolution must be positive")
        n = np.maximum(np.round((hi - lo) / resolution).astype(int), 1)
        h = (hi - lo) / n
        axes = [lo[k] + (np.arange(n[k]) + 0.5) * h[k] for k in range(d)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
        w = np.full(len(grid), float(np.prod(h)))
        if density is not None:
            phi = np.asarray(density(grid), float)
            if phi.shape != (len(grid),) or np.any(phi < 0) or not np.all(np.isfinite(phi)):
                raise ValueError("density must return one finite, non-negative value per point")
            w = w * phi
        if mask is not None:
            keep = np.asarray(mask(grid), bool)
            grid, w = grid[keep], w[keep]
        if len(grid) == 0 or w.sum() <= 0:
            raise ValueError("the domain has no mass")
        self.points, self.weights, self.dim = grid, w, d
        self.lo, self.hi = lo, hi

    @property
    def mass(self) -> float:
        return float(self.weights.sum())


def voronoi_assign(sites: Any, points: Any) -> np.ndarray:
    """Index of the nearest site for every point (ties go to the lower index)."""
    s = np.asarray(sites, float)
    p = np.asarray(points, float)
    d2 = ((p[:, None, :] - s[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=1)


def coverage_cost(sites: Any, domain: CoverageDomain) -> float:
    """``H(p) = Σ_q min_i ‖q − p_i‖² w(q)`` over the discretised domain."""
    s = np.asarray(sites, float)
    d2 = ((domain.points[:, None, :] - s[None, :, :]) ** 2).sum(axis=-1).min(axis=1)
    return float((d2 * domain.weights).sum())


def cell_masses(sites: Any, domain: CoverageDomain) -> np.ndarray:
    """Mass (area × density) of every agent's Voronoi cell."""
    a = voronoi_assign(sites, domain.points)
    return np.bincount(a, weights=domain.weights, minlength=len(np.asarray(sites)))


def centroids(sites: Any, domain: CoverageDomain) -> np.ndarray:
    """Density-weighted centroid of each Voronoi cell (an agent with an empty cell keeps its own position)."""
    s = np.asarray(sites, float)
    a = voronoi_assign(s, domain.points)
    mass = np.bincount(a, weights=domain.weights, minlength=len(s))
    out = s.copy()
    for k in range(domain.dim):
        num = np.bincount(a, weights=domain.weights * domain.points[:, k], minlength=len(s))
        out[:, k] = np.where(mass > 0, num / np.maximum(mass, 1e-300), s[:, k])
    return out


def lloyd_step(sites: Any, domain: CoverageDomain, gain: float = 1.0, max_move: Optional[float] = None) -> np.ndarray:
    """Move every site a fraction ``gain`` of the way to its cell centroid (``gain = 1`` is Lloyd's algorithm), optionally limiting the step length."""
    s = np.asarray(sites, float)
    step = gain * (centroids(s, domain) - s)
    if max_move is not None:
        n = np.linalg.norm(step, axis=1, keepdims=True)
        step = step * np.minimum(1.0, max_move / np.maximum(n, 1e-12))
    return s + step


@dataclass
class LloydResult:
    sites: np.ndarray
    history: np.ndarray  # (iterations + 1, N, d)
    cost: np.ndarray  # (iterations + 1,)
    converged: bool
    iterations: int


def lloyd(
    sites: Any, domain: CoverageDomain, iterations: int = 100, gain: float = 1.0, tol: float = 1e-9, max_move: Optional[float] = None
) -> LloydResult:
    """Iterate :func:`lloyd_step` until the largest move falls below ``tol`` or ``iterations`` is reached."""
    s = np.asarray(sites, float).copy()
    hist: List[np.ndarray] = [s.copy()]
    cost = [coverage_cost(s, domain)]
    conv = False
    k = 0
    for k in range(1, int(iterations) + 1):  # noqa: B007  (k is the iteration count reported after the loop)
        new = lloyd_step(s, domain, gain, max_move)
        moved = float(np.max(np.linalg.norm(new - s, axis=1)))
        s = new
        hist.append(s.copy())
        cost.append(coverage_cost(s, domain))
        if moved < tol:
            conv = True
            break
    return LloydResult(s, np.stack(hist), np.asarray(cost), conv, k)
