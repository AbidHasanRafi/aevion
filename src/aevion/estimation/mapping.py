"""Probabilistic voxel mapping: a log-odds occupancy grid built from range measurements.

Each range measurement is a ray from the sensor to its end point. Voxels the ray passes through are evidence of free space; the voxel
it ends in (when the sensor returned) is evidence of an obstacle. Evidence accumulates as log-odds ``l = log(p / (1 - p))``, which
turns Bayes' rule into an addition (Moravec & Elfes 1985; Hornung et al., OctoMap, 2013)::

    l_cell += l_occ   (hit)        l_cell += l_free   (pass-through)        clamped to [l_min, l_max]

The clamp lets the map change its mind when the world moves. Ray traversal is the exact voxel walk of Amanatides & Woo (1987),
compiled with Numba when available (a vectorised fine-sampling fallback otherwise). The grid converts to a
:class:`~aevion.environments.sdf.SignedDistanceField` so planners and controllers can query clearance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba missing
    HAVE_NUMBA = False

    def njit(*a: Any, **k: Any) -> Any:  # type: ignore[no-redef]
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


__all__ = ["OccupancyGrid", "MAX_VOXELS"]

MAX_VOXELS = 64_000_000  # ~256 MB of float32: refuse larger maps instead of exhausting memory


@njit(cache=False)
def _trace_rays(logodds, lo, inv_res, origins, ends, hits, l_free, l_occ, l_min, l_max):  # pragma: no cover - compiled
    nx, ny, nz = logodds.shape
    shape = np.array([nx, ny, nz])
    hi = np.empty(3)
    for a in range(3):
        hi[a] = lo[a] + shape[a] / inv_res
    n_marked = 0
    for i in range(origins.shape[0]):
        o = origins[i]
        d = ends[i] - o
        t0, t1 = 0.0, 1.0
        ok = True
        for a in range(3):  # clip the segment to the grid box (slab method)
            if abs(d[a]) < 1e-12:
                if o[a] < lo[a] or o[a] >= hi[a]:
                    ok = False
                    break
            else:
                ta, tb = (lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]
                if ta > tb:
                    ta, tb = tb, ta
                if ta > t0:
                    t0 = ta
                if tb < t1:
                    t1 = tb
                if t0 > t1:
                    ok = False
                    break
        if not ok or not (np.isfinite(t0) and np.isfinite(t1)):
            continue
        end_inside = hits[i] and t1 >= 1.0 - 1e-12  # the true end point is a hit inside the grid
        idx = np.empty(3, np.int64)
        step = np.empty(3, np.int64)
        t_max = np.empty(3)
        t_delta = np.empty(3)
        last = np.empty(3, np.int64)
        for a in range(3):
            g = (o[a] + t0 * d[a] - lo[a]) * inv_res
            c = int(np.floor(g))
            if c < 0:
                c = 0
            elif c > shape[a] - 1:
                c = shape[a] - 1
            idx[a] = c
            ge = (o[a] + t1 * d[a] - lo[a]) * inv_res
            ce = int(np.floor(ge))
            if ce < 0:
                ce = 0
            elif ce > shape[a] - 1:
                ce = shape[a] - 1
            last[a] = ce
            if d[a] > 0.0:
                step[a] = 1
                t_max[a] = ((lo[a] + (c + 1) / inv_res) - o[a]) / d[a]
                t_delta[a] = 1.0 / (inv_res * d[a])
            elif d[a] < 0.0:
                step[a] = -1
                t_max[a] = ((lo[a] + c / inv_res) - o[a]) / d[a]
                t_delta[a] = -1.0 / (inv_res * d[a])
            else:
                step[a] = 0
                t_max[a] = 1e300
                t_delta[a] = 1e300
        for _ in range(nx + ny + nz + 4):
            at_last = idx[0] == last[0] and idx[1] == last[1] and idx[2] == last[2]
            v = logodds[idx[0], idx[1], idx[2]]
            if at_last and end_inside:
                v += l_occ
            else:
                v += l_free
            if v > l_max:
                v = l_max
            elif v < l_min:
                v = l_min
            logodds[idx[0], idx[1], idx[2]] = v
            n_marked += 1
            if at_last:
                break
            a = 0  # advance across the nearest voxel boundary
            if t_max[1] < t_max[a]:
                a = 1
            if t_max[2] < t_max[a]:
                a = 2
            idx[a] += step[a]
            t_max[a] += t_delta[a]
            if idx[a] < 0 or idx[a] >= shape[a]:
                break
    return n_marked


class OccupancyGrid:
    """Log-odds occupancy grid over an axis-aligned box.

    ``bounds = (x0, y0, z0, x1, y1, z1)`` in the world frame; ``resolution`` is the voxel edge [m]. ``l_hit`` / ``l_miss`` are the
    log-odds added for a hit / a pass-through (defaults correspond to p = 0.7 and p = 0.4, the OctoMap values); the stored value
    is clamped to ``[l_min, l_max]`` (probabilities 0.12 and 0.97).
    """

    def __init__(
        self,
        bounds: Sequence[float],
        resolution: float = 0.25,
        l_hit: float = 0.85,
        l_miss: float = -0.4,
        l_min: float = -2.0,
        l_max: float = 3.5,
        occupied_threshold: float = 0.5,
    ) -> None:
        b = np.asarray(bounds, float)
        if b.shape != (6,) or np.any(b[3:] <= b[:3]) or not np.all(np.isfinite(b)):
            raise ValueError("bounds must be (x0, y0, z0, x1, y1, z1) with x1 > x0, y1 > y0, z1 > z0")
        if resolution <= 0.0 or not np.isfinite(resolution):
            raise ValueError("resolution must be positive")
        if l_hit <= 0.0 or l_miss >= 0.0 or l_min >= 0.0 or l_max <= 0.0:
            raise ValueError("need l_hit > 0 > l_miss and l_min < 0 < l_max")
        shape = np.ceil((b[3:] - b[:3]) / resolution - 1e-9).astype(int)
        if int(np.prod(shape)) > MAX_VOXELS:
            raise ValueError(f"the grid would have {int(np.prod(shape)):,} voxels (limit {MAX_VOXELS:,}); raise the resolution or shrink the bounds")
        self.lo, self.res = b[:3].copy(), float(resolution)
        self.shape = tuple(int(s) for s in shape)
        self.hi = self.lo + np.asarray(self.shape) * self.res
        self.l_hit, self.l_miss, self.l_min, self.l_max = float(l_hit), float(l_miss), float(l_min), float(l_max)
        self.thr = float(np.log(occupied_threshold / (1.0 - occupied_threshold)))
        self.logodds = np.zeros(self.shape, np.float32)
        self.n_updates = 0

    # ------------------------------------------------------------------------------------------ integration
    def integrate(self, origins: Any, ends: Any, hits: Any = None) -> int:
        """Integrate rays ``origin -> end``; ``hits[i]`` says whether the sensor returned at ``end`` (default: all hit).

        Returns the number of voxel updates. Rays with a non-finite end point are skipped.
        """
        o = np.atleast_2d(np.asarray(origins, float))
        e = np.atleast_2d(np.asarray(ends, float))
        if o.shape[0] == 1 and e.shape[0] > 1:
            o = np.broadcast_to(o, e.shape)
        if o.shape != e.shape or o.shape[1] != 3:
            raise ValueError("origins and ends must be (N, 3)")
        h = np.ones(len(e), bool) if hits is None else np.asarray(hits, bool).reshape(-1)
        good = np.isfinite(e).all(axis=1) & np.isfinite(o).all(axis=1)
        o, e, h = np.ascontiguousarray(o[good]), np.ascontiguousarray(e[good]), np.ascontiguousarray(h[good])
        if not len(e):
            return 0
        n = int(_trace_rays(self.logodds, self.lo, 1.0 / self.res, o, e, h, self.l_miss, self.l_hit, self.l_min, self.l_max))
        self.n_updates += n
        return n

    def insert_ranges(self, origin: Any, directions: Any, ranges: Any, valid: Any = None, max_range: Optional[float] = None) -> int:
        """Integrate a scan: unit ``directions`` (world frame), measured ``ranges`` and a ``valid`` mask (returns).

        A measurement at or beyond ``max_range`` (or marked invalid) is a *miss*: free space up to ``max_range`` and no obstacle.
        """
        d = np.atleast_2d(np.asarray(directions, float))
        r = np.asarray(ranges, float).reshape(-1)
        ok = np.isfinite(r) & (r > 0.0)
        if valid is not None:
            ok &= np.asarray(valid, bool).reshape(-1)
        if max_range is not None:
            ok &= r < max_range * (1.0 - 1e-6)
            r_eff = np.where(ok, r, max_range)
        else:
            r_eff = np.where(ok, r, 0.0)
        o = np.asarray(origin, float).reshape(1, 3)
        ends = o + d * r_eff[:, None]
        keep = r_eff > 0.0
        return self.integrate(np.broadcast_to(o, ends.shape)[keep], ends[keep], ok[keep])

    def insert_points(self, origin: Any, points: Any, max_range: Optional[float] = None) -> int:
        """Integrate world-frame hit ``points`` seen from ``origin`` (a point cloud)."""
        p = np.atleast_2d(np.asarray(points, float))
        o = np.asarray(origin, float).reshape(1, 3)
        d = p - o
        r = np.linalg.norm(d, axis=1)
        ok = np.isfinite(r) & (r > 1e-9)
        if max_range is not None:
            ok &= r < max_range
        return self.integrate(np.broadcast_to(o, p.shape)[ok], p[ok], np.ones(int(ok.sum()), bool))

    # ------------------------------------------------------------------------------------------ queries
    def index(self, points: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Voxel indices ``(N, 3)`` of world points and a mask of the points that lie inside the grid."""
        p = np.atleast_2d(np.asarray(points, float))
        g = np.floor((p - self.lo) / self.res).astype(np.int64)
        inside = np.all((g >= 0) & (g < np.asarray(self.shape)), axis=1) & np.isfinite(p).all(axis=1)
        return np.where(inside[:, None], g, 0), inside

    def probability(self, points: Any = None) -> np.ndarray:
        """Occupancy probability: of the whole grid, or at world points (unknown and outside points give 0.5)."""
        if points is None:
            return 1.0 / (1.0 + np.exp(-self.logodds.astype(float)))
        g, inside = self.index(points)
        l = np.where(inside, self.logodds[g[:, 0], g[:, 1], g[:, 2]], 0.0).astype(float)
        return 1.0 / (1.0 + np.exp(-l))

    def occupied(self, points: Any = None) -> np.ndarray:
        """Boolean occupancy of the grid, or of world points (outside the grid: ``False``)."""
        if points is None:
            return self.logodds > self.thr
        g, inside = self.index(points)
        return inside & (self.logodds[g[:, 0], g[:, 1], g[:, 2]] > self.thr)

    def known(self) -> np.ndarray:
        """Voxels that have received any evidence."""
        return self.logodds != 0.0

    def free(self) -> np.ndarray:
        return self.logodds < 0.0

    def coverage(self) -> float:
        """Fraction of voxels that have been observed."""
        return float(self.known().mean())

    def centers(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """World coordinates of voxel centres (of the ``mask`` voxels, default the occupied ones), ``(K, 3)``."""
        m = self.occupied() if mask is None else mask
        return self.lo + (np.argwhere(m) + 0.5) * self.res

    def entropy(self) -> float:
        """Total Shannon entropy [bits] of the map: how much remains to be learned (unknown voxels carry 1 bit)."""
        p = np.clip(self.probability(), 1e-6, 1.0 - 1e-6)
        return float(np.sum(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))

    def clear(self) -> None:
        self.logodds[:] = 0.0
        self.n_updates = 0

    # ------------------------------------------------------------------------------------------ distance fields
    def to_sdf(self, treat_unknown_as_free: bool = True) -> Any:
        """A :class:`~aevion.environments.sdf.SignedDistanceField` of the occupied voxels (negative inside obstacles).

        Unknown space is free by default (the optimistic assumption a replanner revises as it explores); pass ``False`` to treat
        it as an obstacle (conservative).
        """
        from ..environments.sdf import SignedDistanceField

        occ = self.occupied() if treat_unknown_as_free else (self.occupied() | ~self.known())
        return SignedDistanceField.from_occupancy(occ, self.lo + 0.5 * self.res, self.res)

    def distance(self, points: Any, treat_unknown_as_free: bool = True) -> np.ndarray:
        return self.to_sdf(treat_unknown_as_free).distance(points)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            lo=self.lo.tolist(), res=self.res, shape=list(self.shape), l_hit=self.l_hit, l_miss=self.l_miss, l_min=self.l_min, l_max=self.l_max
        )
