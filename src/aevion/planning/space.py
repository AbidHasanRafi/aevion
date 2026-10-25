"""The space planners search: a voxelised, inflated occupancy grid with exact segment queries.

Obstacles are *inflated* by the vehicle radius plus a safety margin, so a planner can treat the vehicle as a point: a cell is
blocked when its centre is within ``inflation`` of an obstacle (``SDF <= inflation``). A segment is free when every voxel it
passes through is free (exact Amanatides & Woo traversal, the same walk the mapper uses), and a segment that leaves the grid is
blocked. Optionally a per-cell cost multiplier ``>= 1`` (see :meth:`PlanningSpace.clearance_cost`) makes planners prefer routes
with more clearance, while keeping the Euclidean heuristic admissible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba missing
    HAVE_NUMBA = False

    def njit(*a: Any, **k: Any) -> Any:  # type: ignore[no-redef]
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


__all__ = ["PlanningSpace", "segment_free_batch", "MAX_CELLS"]

MAX_CELLS = 64_000_000


@njit(cache=False)
def _segment_free(occ, lo, inv_res, a, b):  # pragma: no cover - compiled
    """True when the segment a -> b passes only through free voxels of ``occ`` (outside the grid counts as blocked)."""
    nx, ny, nz = occ.shape
    shape = np.array([nx, ny, nz])
    d = b - a
    for k in range(3):
        hi_k = lo[k] + shape[k] / inv_res
        if a[k] < lo[k] or a[k] >= hi_k or b[k] < lo[k] or b[k] >= hi_k:
            return False
    idx = np.empty(3, np.int64)
    last = np.empty(3, np.int64)
    step = np.empty(3, np.int64)
    t_max = np.empty(3)
    t_delta = np.empty(3)
    for k in range(3):
        c = int(np.floor((a[k] - lo[k]) * inv_res))
        e = int(np.floor((b[k] - lo[k]) * inv_res))
        idx[k] = min(max(c, 0), shape[k] - 1)
        last[k] = min(max(e, 0), shape[k] - 1)
        if d[k] > 0.0:
            step[k] = 1
            t_max[k] = ((lo[k] + (idx[k] + 1) / inv_res) - a[k]) / d[k]
            t_delta[k] = 1.0 / (inv_res * d[k])
        elif d[k] < 0.0:
            step[k] = -1
            t_max[k] = ((lo[k] + idx[k] / inv_res) - a[k]) / d[k]
            t_delta[k] = -1.0 / (inv_res * d[k])
        else:
            step[k] = 0
            t_max[k] = 1e300
            t_delta[k] = 1e300
    for _ in range(nx + ny + nz + 4):
        if occ[idx[0], idx[1], idx[2]]:
            return False
        if idx[0] == last[0] and idx[1] == last[1] and idx[2] == last[2]:
            return True
        k = 0
        if t_max[1] < t_max[k]:
            k = 1
        if t_max[2] < t_max[k]:
            k = 2
        idx[k] += step[k]
        t_max[k] += t_delta[k]
        if idx[k] < 0 or idx[k] >= shape[k]:
            return True  # numerical overshoot past the last voxel: everything before it was free
    return True


@njit(cache=False)
def _segments_free(occ, lo, inv_res, A, Bp):  # pragma: no cover - compiled
    out = np.empty(A.shape[0], np.bool_)
    for i in range(A.shape[0]):
        out[i] = _segment_free(occ, lo, inv_res, A[i], Bp[i])
    return out


def segment_free_batch(space: "PlanningSpace", a: Any, b: Any) -> np.ndarray:
    """Vectorised segment test for ``(N, 3)`` start and end points."""
    A = np.ascontiguousarray(np.atleast_2d(np.asarray(a, float)))
    Bp = np.ascontiguousarray(np.atleast_2d(np.asarray(b, float)))
    return np.asarray(_segments_free(space.occ, space.lo, 1.0 / space.res, A, Bp))


class PlanningSpace:
    """Voxel occupancy for planning. ``occ[i, j, k]`` is ``True`` for blocked cells; cell ``(i, j, k)`` spans
    ``lo + (i, j, k) * res`` to ``lo + (i + 1, j + 1, k + 1) * res``."""

    def __init__(self, occupied: np.ndarray, lo: Any, resolution: float, clearance: Optional[np.ndarray] = None) -> None:
        occ = np.ascontiguousarray(np.asarray(occupied, bool))
        if occ.ndim != 3 or min(occ.shape) < 2:
            raise ValueError("occupied must be a 3-D grid with at least 2 cells per axis")
        if resolution <= 0 or not np.isfinite(resolution):
            raise ValueError("resolution must be positive")
        if occ.size > MAX_CELLS:
            raise ValueError(f"the planning grid would have {occ.size:,} cells (limit {MAX_CELLS:,}); raise the resolution or shrink the bounds")
        self.occ = occ
        self.lo = np.asarray(lo, float).reshape(3)
        self.res = float(resolution)
        self.shape = occ.shape
        self.hi = self.lo + np.asarray(self.shape) * self.res
        self.clearance = clearance  # distance to the nearest obstacle at the cell centres [m] (None when unknown)
        self.cost_mult: Optional[np.ndarray] = None

    # ------------------------------------------------------------------------------------------ construction
    @classmethod
    def from_sdf(
        cls, sdf: Any, bounds: Optional[Sequence[float]] = None, resolution: Optional[float] = None, inflation: float = 0.0
    ) -> "PlanningSpace":
        """Block every cell centre whose signed distance is at most ``inflation``. Without ``bounds`` the SDF's own grid is used."""
        if bounds is None:
            lo = sdf.origin - 0.5 * sdf.spacing
            res = sdf.spacing
            d = sdf.values
        else:
            b = np.asarray(bounds, float)
            res = float(resolution or sdf.spacing)
            shape = np.ceil((b[3:] - b[:3]) / res - 1e-9).astype(int)
            if int(np.prod(shape)) > MAX_CELLS:
                raise ValueError("the planning grid would be too large; raise the resolution or shrink the bounds")
            ax = [b[k] + (np.arange(shape[k]) + 0.5) * res for k in range(3)]
            X, Y, Z = np.meshgrid(*ax, indexing="ij")
            d = sdf.distance(np.stack([X, Y, Z], -1).reshape(-1, 3)).reshape(tuple(shape))
            lo = b[:3]
        return cls(d <= inflation, lo, res, clearance=d)

    @classmethod
    def from_world(
        cls, world: Any, bounds: Optional[Sequence[float]] = None, resolution: float = 0.5, inflation: float = 0.5, ground: bool = True
    ) -> "PlanningSpace":
        """Voxelise a :class:`~aevion.environments.world.World` (primitives, meshes, terrain) with obstacles inflated by ``inflation``."""
        from ..environments.sdf import SignedDistanceField

        b = np.asarray(world.bounds if bounds is None else bounds, float)
        sdf = SignedDistanceField.from_world(world, bounds=b, spacing=resolution, ground=ground)
        return cls.from_sdf(sdf, None, None, inflation)

    @classmethod
    def from_occupancy(cls, grid: Any, inflation: float = 0.0, unknown_is_free: bool = True) -> "PlanningSpace":
        """From a :class:`~aevion.estimation.mapping.OccupancyGrid`; unknown space is free (optimistic) unless told otherwise."""
        sdf = grid.to_sdf(treat_unknown_as_free=unknown_is_free)
        return cls(sdf.values <= inflation, grid.lo, grid.res, clearance=sdf.values)

    @classmethod
    def empty(cls, bounds: Sequence[float], resolution: float = 0.5) -> "PlanningSpace":
        b = np.asarray(bounds, float)
        shape = np.ceil((b[3:] - b[:3]) / resolution - 1e-9).astype(int)
        return cls(np.zeros(tuple(shape), bool), b[:3], resolution)

    # ------------------------------------------------------------------------------------------ geometry
    def index(self, p: Any) -> np.ndarray:
        """Cell index of a world point (clipped into the grid)."""
        g = np.floor((np.asarray(p, float) - self.lo) / self.res).astype(np.int64)
        return np.clip(g, 0, np.asarray(self.shape) - 1)

    def inside(self, p: Any) -> np.ndarray:
        q = np.atleast_2d(np.asarray(p, float))
        return np.all((q >= self.lo) & (q < self.hi), axis=1) & np.isfinite(q).all(axis=1)

    def center(self, idx: Any) -> np.ndarray:
        return self.lo + (np.asarray(idx, float) + 0.5) * self.res

    def is_free(self, p: Any) -> np.ndarray:
        """Free (inside the grid and not blocked) for world points ``(N, 3)``."""
        q = np.atleast_2d(np.asarray(p, float))
        ins = self.inside(q)
        g = np.clip(np.floor((q - self.lo) / self.res).astype(np.int64), 0, np.asarray(self.shape) - 1)
        return ins & ~self.occ[g[:, 0], g[:, 1], g[:, 2]]

    def segment_free(self, a: Any, b: Any) -> bool:
        return bool(_segment_free(self.occ, self.lo, 1.0 / self.res, np.asarray(a, float), np.asarray(b, float)))

    def path_free(self, path: Any) -> bool:
        P = np.asarray(path, float)
        return bool(len(P) > 0 and np.all(segment_free_batch(self, P[:-1], P[1:]))) if len(P) > 1 else bool(self.is_free(P).all())

    def nearest_free(self, p: Any, max_radius: float = 5.0) -> Optional[np.ndarray]:
        """Centre of the free cell closest to ``p`` (the start or goal may lie inside the inflation), or ``None``."""
        c = self.index(p)
        r = max(int(np.ceil(max_radius / self.res)), 1)
        lo = np.maximum(c - r, 0)
        hi = np.minimum(c + r + 1, np.asarray(self.shape))
        sub = ~self.occ[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        if not sub.any():
            return None
        idx = np.argwhere(sub) + lo
        pts = self.center(idx)
        j = int(np.argmin(np.linalg.norm(pts - np.asarray(p, float), axis=1)))
        return pts[j]

    def clearance_cost(self, safe_distance: float, weight: float = 2.0) -> None:
        """Make cells nearer than ``safe_distance`` to an obstacle more expensive: ``1 + weight (1 - d / safe_distance)``."""
        if self.clearance is None:
            raise ValueError("this space has no clearance field (build it from an SDF, a world or a map)")
        deficit = np.clip(1.0 - np.asarray(self.clearance) / float(safe_distance), 0.0, 1.0)
        self.cost_mult = 1.0 + float(weight) * deficit

    def sample_free(self, n: int, rng: np.random.Generator, bounds: Optional[Sequence[float]] = None) -> np.ndarray:
        """``n`` uniformly distributed free points (rejection sampling in cells, then jitter inside the cell)."""
        lo = self.lo if bounds is None else np.maximum(np.asarray(bounds[:3], float), self.lo)
        hi = self.hi if bounds is None else np.minimum(np.asarray(bounds[3:], float), self.hi)
        out = np.empty((0, 3))
        for _ in range(60):
            need = n - len(out)
            if need <= 0:
                break
            cand = rng.uniform(lo, hi, (max(need * 3, 16), 3))
            out = np.vstack([out, cand[self.is_free(cand)]])
        if len(out) < n:
            raise RuntimeError("could not sample free space (the planning space is almost entirely blocked)")
        return out[:n]

    def to_dict(self) -> Dict[str, Any]:
        return dict(lo=self.lo.tolist(), res=self.res, shape=list(self.shape), blocked=int(self.occ.sum()))
