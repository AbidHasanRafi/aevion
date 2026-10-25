"""Signed-distance fields (SDF) and Euclidean distance transforms (ESDF) for planning, learning and fast collision checks.

* :class:`SignedDistanceField` stores distances on a regular grid (negative inside obstacles) with trilinear interpolation and the
  analytic gradient of the interpolant: ``distance(p)``, ``gradient(p)``, ``collides(p, radius)``. Queries outside the grid are clamped,
  so the field is constant outward and underestimates the distance far outside.
* :meth:`SignedDistanceField.from_world` samples the exact distance to boxes, spheres, vertical cylinders and meshes (BVH), and to the
  terrain. The terrain term is the vertical clearance ``z - ground(x, y)``, an upper bound on the true distance over slopes. The union of
  obstacles is the minimum of the signed distances (exact outside, a lower bound on penetration depth inside).
* :func:`euclidean_distance_transform` is the exact separable transform of Felzenszwalb & Huttenlocher (2012);
  :func:`esdf_from_occupancy` builds a signed ESDF from a boolean occupancy grid (for example from LiDAR mapping).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a: Any, **k: Any) -> Any:  # type: ignore[no-redef]
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


Vec = Union[Sequence[float], np.ndarray]

__all__ = ["SignedDistanceField", "euclidean_distance_transform", "esdf_from_occupancy", "box_sdf", "sphere_sdf", "cylinder_sdf"]

INF = 1e20
MAX_VOXELS = 64_000_000


# ------------------------------------------------------------------------------------------ transforms
@njit(cache=True)
def _edt_1d(f, v, z, out):  # pragma: no cover - compiled by numba when available
    """Lower envelope of parabolas (Felzenszwalb & Huttenlocher): ``out[q] = min_p f[p] + (q - p)^2``."""
    n = f.shape[0]
    k = 0
    v[0] = 0
    z[0] = -1e30
    z[1] = 1e30
    for q in range(1, n):
        while True:
            p = v[k]
            s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * q - 2.0 * p)
            if s <= z[k]:
                k -= 1
            else:
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = 1e30
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d = q - v[k]
        out[q] = d * d + f[v[k]]


@njit(cache=True)
def _edt_axis(grid, axis):  # pragma: no cover
    shp = grid.shape
    n = shp[axis]
    v = np.zeros(n, np.int64)
    z = np.zeros(n + 1)
    out = np.zeros(n)
    res = np.empty_like(grid)
    if axis == 0:
        for j in range(shp[1]):
            for k in range(shp[2]):
                _edt_1d(grid[:, j, k], v, z, out)
                res[:, j, k] = out
    elif axis == 1:
        for i in range(shp[0]):
            for k in range(shp[2]):
                _edt_1d(grid[i, :, k], v, z, out)
                res[i, :, k] = out
    else:
        for i in range(shp[0]):
            for j in range(shp[1]):
                _edt_1d(grid[i, j, :], v, z, out)
                res[i, j, :] = out
    return res


def euclidean_distance_transform(mask: np.ndarray, spacing: Vec = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Distance from every cell to the nearest ``True`` cell of ``mask`` (0 on the cells themselves), in length units.

    Isotropic grids use the O(n) Numba transform when available; anisotropic spacing (or no Numba) uses an exact O(n^2)-per-line pass.
    """
    m = np.asarray(mask, bool)
    if m.ndim != 3:
        raise ValueError("mask must be 3-D")
    if not m.any():
        return np.full(m.shape, np.inf)
    sp = np.broadcast_to(np.asarray(spacing, float), (3,))
    if HAVE_NUMBA and np.allclose(sp, sp[0]):
        g = np.where(m, 0.0, INF)
        for axis in range(3):
            g = _edt_axis(np.ascontiguousarray(g), axis)
        return np.sqrt(g) * float(sp[0])
    return _edt_anisotropic(m, sp)


def _edt_anisotropic(m: np.ndarray, spacing: Vec) -> np.ndarray:
    """Exact anisotropic transform: each pass uses its own spacing (squared distances accumulate in metres^2)."""
    g = np.where(m, 0.0, INF)
    for axis in range(3):
        s2 = float(spacing[axis]) ** 2
        n = g.shape[axis]
        gm = np.moveaxis(g, axis, -1)
        idx = np.arange(n)
        sq = s2 * (idx[:, None] - idx[None, :]) ** 2
        flat = gm.reshape(-1, n)
        out = np.empty_like(flat)
        step = max(1, 2_000_000 // (n * n))
        for s in range(0, len(flat), step):
            out[s : s + step] = np.min(flat[s : s + step, None, :] + sq[None], axis=-1)
        g = np.moveaxis(out.reshape(gm.shape), -1, axis)
    return np.sqrt(g)


def esdf_from_occupancy(occupied: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Signed ESDF [length units]: positive in free space (distance to the nearest occupied cell centre), negative inside (distance to the
    nearest free cell centre, minus half a cell so the surface lies between cells)."""
    occ = np.asarray(occupied, bool)
    if not occ.any():
        return np.full(occ.shape, np.inf)
    if occ.all():
        return np.full(occ.shape, -np.inf)
    out_d = euclidean_distance_transform(occ, (spacing,) * 3)
    in_d = euclidean_distance_transform(~occ, (spacing,) * 3)
    return np.where(occ, -(in_d - 0.5 * spacing), out_d - 0.5 * spacing)


# ------------------------------------------------------------------------------------ exact primitives
def sphere_sdf(p: np.ndarray, center: Vec, radius: float) -> np.ndarray:
    return np.linalg.norm(p - np.asarray(center, float), axis=-1) - radius


def box_sdf(p: np.ndarray, lo: Vec, hi: Vec) -> np.ndarray:
    c, h = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float)), 0.5 * (np.asarray(hi, float) - np.asarray(lo, float))
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def cylinder_sdf(p: np.ndarray, xy: Vec, z0: float, z1: float, radius: float) -> np.ndarray:
    """Finite vertical cylinder with flat caps."""
    dx = np.hypot(p[..., 0] - xy[0], p[..., 1] - xy[1]) - radius
    dz = np.abs(p[..., 2] - 0.5 * (z0 + z1)) - 0.5 * (z1 - z0)
    return np.hypot(np.maximum(dx, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(dx, dz), 0.0)


class SignedDistanceField:
    """Distances on a regular grid: ``values[i, j, k]`` at ``origin + (i, j, k) * spacing`` (negative inside obstacles)."""

    def __init__(self, values: np.ndarray, origin: Vec, spacing: float):
        v = np.asarray(values, float)
        if v.ndim != 3 or min(v.shape) < 2:
            raise ValueError("values must be a 3-D grid with at least 2 cells per axis")
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        self.values, self.origin, self.spacing = v, np.asarray(origin, float).reshape(3), float(spacing)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.values.shape  # type: ignore[return-value]

    @property
    def extent(self) -> np.ndarray:
        return np.concatenate([self.origin, self.origin + (np.array(self.shape) - 1) * self.spacing])

    # ------------------------------------------------------------------------------------ construction
    @classmethod
    def from_function(cls, fn: Any, bounds: Vec, spacing: float) -> "SignedDistanceField":
        b = np.asarray(bounds, float)
        axes = [np.arange(b[k], b[k + 3] + 1e-9, spacing) for k in range(3)]
        if int(np.prod([len(a) for a in axes])) > MAX_VOXELS:
            raise ValueError("SDF grid too large; raise the spacing or shrink the bounds")
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        return cls(fn(np.stack([X, Y, Z], -1)), b[:3], spacing)

    @classmethod
    def from_world(cls, world: Any, bounds: Optional[Vec] = None, spacing: float = 1.0, ground: bool = True) -> "SignedDistanceField":
        """Sample the world's obstacles (and the ground when ``ground``) on a grid over ``bounds`` (default: the world bounds)."""
        b = np.asarray(world.bounds if bounds is None else bounds, float)

        def field(p: np.ndarray) -> np.ndarray:
            d = np.full(p.shape[:-1], np.inf)
            flat = p.reshape(-1, 3)
            out = np.full(len(flat), np.inf)
            for bx in world.boxes:
                out = np.minimum(out, box_sdf(flat, bx[:3], bx[3:]))
            for s in world.spheres:
                out = np.minimum(out, sphere_sdf(flat, s[:3], s[3]))
            for c in world.cylinders:
                out = np.minimum(out, cylinder_sdf(flat, c[:2], c[2], c[3], c[4]))
            if world.meshes:
                out = np.minimum(out, world.mesh_index.signed_distance(flat))
            if ground:
                from ..core.backend import NumpyBackend

                gz = world.ground_height(NumpyBackend(), flat[:, 0], flat[:, 1])
                out = np.minimum(out, flat[:, 2] - gz)
            del d
            return out.reshape(p.shape[:-1])

        return cls.from_function(field, b, spacing)

    @classmethod
    def from_occupancy(cls, occupied: np.ndarray, origin: Vec, spacing: float) -> "SignedDistanceField":
        return cls(esdf_from_occupancy(occupied, spacing), origin, spacing)

    # ---------------------------------------------------------------------------------------- queries
    def _locate(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f = (np.asarray(p, float).reshape(-1, 3) - self.origin) / self.spacing
        dims = np.array(self.shape) - 1
        fc = np.clip(f, 0.0, dims - 1e-9)
        i0 = np.floor(fc).astype(int)
        return i0, fc - i0

    def distance(self, points: Any) -> np.ndarray:
        """Trilinearly interpolated distance at ``points`` ``(..., 3)``."""
        pts = np.asarray(points, float)
        i0, w = self._locate(pts)
        out = np.zeros(len(i0))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * (w[:, 2] if dz else 1 - w[:, 2])
                    out += wt * self.values[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return out.reshape(pts.shape[:-1])

    def gradient(self, points: Any) -> np.ndarray:
        """Exact gradient of the trilinear interpolant, ``(..., 3)`` (points away from the nearest obstacle)."""
        pts = np.asarray(points, float)
        i0, w = self._locate(pts)
        g = np.zeros((len(i0), 3))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    v = self.values[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
                    wx, wy, wz = (w[:, 0] if dx else 1 - w[:, 0]), (w[:, 1] if dy else 1 - w[:, 1]), (w[:, 2] if dz else 1 - w[:, 2])
                    sx, sy, sz = (1.0 if dx else -1.0), (1.0 if dy else -1.0), (1.0 if dz else -1.0)
                    g[:, 0] += sx * wy * wz * v
                    g[:, 1] += sy * wx * wz * v
                    g[:, 2] += sz * wx * wy * v
        return (g / self.spacing).reshape(pts.shape)

    def collides(self, points: Any, radius: float = 0.0) -> np.ndarray:
        """True where a sphere of ``radius`` at ``points`` touches an obstacle (``distance <= radius``)."""
        return self.distance(points) <= radius

    def to_dict(self) -> Dict[str, Any]:
        return dict(shape=list(self.shape), origin=self.origin.tolist(), spacing=self.spacing)
