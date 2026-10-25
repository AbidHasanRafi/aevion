"""The simulated world: physics environment plus geometry.

A :class:`World` owns the gravity model, atmosphere, wind and a set of static
geometry (terrain heightmap, axis-aligned boxes, spheres, vertical cylinders).
It supplies ground height queries and batched sphere-vs-geometry collision tests that
work on any backend, and exposes the raw arrays for ray-cast sensors and the viewer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from ..core.constants import Atmosphere, GravityModel
from .mesh import TriangleMesh
from .wind import NoWind, Wind, wind_from_dict


class World:
    def __init__(
        self,
        gravity: Optional[GravityModel] = None,
        atmosphere: Optional[Atmosphere] = None,
        wind: Optional[Wind] = None,
        bounds=(-100.0, -100.0, 0.0, 100.0, 100.0, 100.0),
        name: str = "empty",
    ):
        self.gravity = gravity or GravityModel()
        self.atmosphere = atmosphere or Atmosphere()
        self.wind = wind or NoWind()
        self.bounds = np.asarray(bounds, dtype=float)
        self.name = name
        self.weather: Optional[Any] = None  # see environments.weather.Weather
        self.origin: Optional[Any] = None  # geodesy.LocalFrame: where world (0, 0, 0) is on Earth (used by GNSS, magnetometer, sun)
        self.utc: Optional[str] = None  # ISO-8601 UTC time of simulation t = 0 (sun position, GNSS satellite geometry)
        self.actors: list = []  # (Actor, array name, row) triples: moving obstacles owning one primitive row each
        self.meshes: list = []  # TriangleMesh instances; queried through a BVH (mesh_index)
        self._mesh_index: Any = None
        self.boxes = np.zeros((0, 6))
        self.spheres = np.zeros((0, 4))
        self.cylinders = np.zeros((0, 5))
        self.box_colors = np.zeros((0, 3))
        self.sphere_colors = np.zeros((0, 3))
        self.cyl_colors = np.zeros((0, 3))
        # terrain heightmap: heights[iy, ix] at x = x0 + ix*res, y = y0 + iy*res
        self.heightmap: Optional[np.ndarray] = None
        self.hm_origin = (0.0, 0.0)
        self.hm_res = 1.0
        self._cache_key = None
        self._cache: Dict[Any, Any] = {}

    # --------------------------------------------------------------- building
    def add_box(self, center, size, color=None):
        c, s = np.asarray(center, float), np.asarray(size, float)
        self.boxes = np.vstack([self.boxes, np.concatenate([c - s / 2, c + s / 2])[None]])
        self.box_colors = np.vstack([self.box_colors, _color(color, len(self.boxes))])
        self._cache.clear()
        return self

    def add_box_corners(self, lo, hi, color=None):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return self.add_box((lo + hi) / 2, hi - lo, color)

    def add_sphere(self, center, radius, color=None):
        self.spheres = np.vstack([self.spheres, np.concatenate([np.asarray(center, float), [radius]])[None]])
        self.sphere_colors = np.vstack([self.sphere_colors, _color(color, len(self.spheres) + 7)])
        self._cache.clear()
        return self

    def add_cylinder(self, xy, radius, z0=0.0, z1=10.0, color=None):
        self.cylinders = np.vstack([self.cylinders, [xy[0], xy[1], z0, z1, radius]])
        self.cyl_colors = np.vstack([self.cyl_colors, _color(color, len(self.cylinders) + 3)])
        self._cache.clear()
        return self

    def add_actor(self, actor, _row=None):
        """Add a moving obstacle (see :mod:`aevion.environments.actors`); it owns one row of the primitive arrays, rewritten by :meth:`advance`."""
        color = actor.color if actor.color is not None else (0.95, 0.45, 0.15)
        if actor.shape == "box":
            n0 = len(self.boxes)
            self.add_box_corners(actor.primitive_row(0.0)[:3], actor.primitive_row(0.0)[3:], color)
            array, row = "boxes", n0
        elif actor.shape == "sphere":
            n0 = len(self.spheres)
            self.add_sphere(actor.primitive_row(0.0)[:3], actor.radius, color)
            array, row = "spheres", n0
        else:
            n0 = len(self.cylinders)
            r = actor.primitive_row(0.0)
            self.add_cylinder(r[:2], actor.radius, r[2], r[3], color)
            array, row = "cylinders", n0
        self.actors.append((actor, array, row))
        return self

    def advance(self, t):
        """Move every actor to its position at simulation time ``t`` (no-op for worlds without actors)."""
        if not self.actors:
            return
        for actor, array, row in self.actors:
            getattr(self, array)[row] = actor.primitive_row(t)
        self._cache.clear()

    def primitive_velocity(self, prim_ids, t):
        """Velocity ``(R, 3)`` of the surfaces with the given primitive ids at time ``t`` (zero for everything that is not an actor)."""
        ids = np.asarray(prim_ids)
        out = np.zeros(ids.shape + (3,))
        if not self.actors:
            return out
        k, s = len(self.boxes), len(self.spheres)
        base = {"boxes": 0, "spheres": k, "cylinders": k + s}
        for actor, array, row in self.actors:
            m = ids == base[array] + row
            if m.any():
                out[m] = actor.velocity(t)
        return out

    def add_mesh(self, mesh, color=None):
        """Add a :class:`~aevion.environments.mesh.TriangleMesh` (ray casting, collision and distance queries see it)."""
        if not isinstance(mesh, TriangleMesh):
            raise TypeError("add_mesh expects a TriangleMesh")
        self.meshes.append(mesh if color is None else mesh.with_color(color))
        self._mesh_index = None
        self._cache.clear()
        return self

    @property
    def mesh_index(self):
        """The BVH over all meshes (built on first use), or ``None`` for a world without meshes."""
        if not self.meshes:
            return None
        if self._mesh_index is None:
            from .mesh_index import MeshIndex

            self._mesh_index = MeshIndex(self.meshes)
        return self._mesh_index

    @property
    def n_mesh_triangles(self) -> int:
        return sum(m.n_faces for m in self.meshes)

    def prim_colors(self):
        """RGB colour of every primitive id (boxes, spheres, cylinders, then one row per mesh triangle) for renderers."""
        parts = [self.box_colors, self.sphere_colors, self.cyl_colors] + [np.tile(m.color, (m.n_faces, 1)) for m in self.meshes]
        return np.concatenate(parts, 0) if any(len(p) for p in parts) else np.zeros((0, 3))

    def mesh_response(self, pos, vel, collided, radius, restitution):
        """Push spheres out of the meshes and reflect the inward velocity (NumPy arrays; used by the fused-kernel path)."""
        idx = self.mesh_index
        if idx is None:
            return pos, vel, collided
        push, depth = idx.collide(pos, radius)
        hit = depth > 0.0
        if not hit.any():
            return pos, vel, collided
        n = push / np.maximum(depth, 1e-12)[:, None]
        vn = np.sum(vel * n, axis=1, keepdims=True)
        vel = np.where(hit[:, None] & (vn < 0.0), vel - (1.0 + restitution) * vn * n, vel)
        return pos + push, vel, np.asarray(collided) | hit

    def set_heightmap(self, heights, origin=(0.0, 0.0), resolution=1.0):
        self.heightmap = np.asarray(heights, dtype=float)
        self.hm_origin = (float(origin[0]), float(origin[1]))
        self.hm_res = float(resolution)
        self._cache.clear()
        return self

    @property
    def has_obstacles(self) -> bool:
        return len(self.boxes) + len(self.spheres) + len(self.cylinders) + len(self.meshes) > 0

    @property
    def n_obstacles(self) -> int:
        return len(self.boxes) + len(self.spheres) + len(self.cylinders)

    # ------------------------------------------------------------- simulation
    def reset(self, B, n, seed=None):
        self.wind.bind_world(self)
        self.wind.reset(B, n, seed)

    def _tensors(self, B):
        key = (B.name, B.device, str(getattr(B, "dtype", "")))
        if key not in self._cache:
            f = B.asarray
            self._cache[key] = dict(
                boxes=f(self.boxes),
                spheres=f(self.spheres),
                cyl=f(self.cylinders),
                hm=f(self.heightmap) if self.heightmap is not None else None,
            )
        return self._cache[key]

    def ground_height(self, B, x, y):
        """Terrain height under ``(x, y)`` (bilinear). Zero for a flat world."""
        if self.heightmap is None:
            return x * 0.0
        hm = self._tensors(B)["hm"]
        H, W = hm.shape
        fx = (x - self.hm_origin[0]) / self.hm_res
        fy = (y - self.hm_origin[1]) / self.hm_res
        fx = B.clip(fx, 0.0, W - 1.001)
        fy = B.clip(fy, 0.0, H - 1.001)
        if B.name == "numpy":
            ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        else:
            ix, iy = fx.floor().long(), fy.floor().long()
        tx, ty = fx - ix, fy - iy
        h00, h10 = hm[iy, ix], hm[iy, ix + 1]
        h01, h11 = hm[iy + 1, ix], hm[iy + 1, ix + 1]
        return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty

    def collide(self, B, pos, radius):
        """Sphere-vs-geometry test. Returns ``(hit, depth, normal)`` with shapes (N,), (N,), (N,3).

        ``normal`` points out of the obstacle; contributions from several obstacles are summed.
        """
        T = self._tensors(B)
        pos.shape[0]
        push = pos * 0.0
        r = radius
        eps = 1e-9

        if len(self.boxes):
            bmin, bmax = T["boxes"][:, :3][None], T["boxes"][:, 3:][None]  # (1,K,3)
            p = pos[:, None, :]
            c = B.minimum(B.maximum(p, bmin), bmax)
            d = p - c
            dist = B.sqrt(B.sum(d * d, axis=-1) + eps * eps)  # (N,K)
            outside_hit = dist < r
            inside = ((p > bmin) & (p < bmax)).all(-1)
            lo, hi = p - bmin, bmax - p
            cand = [lo[..., 0], lo[..., 1], lo[..., 2], hi[..., 0], hi[..., 1], hi[..., 2]]
            m = cand[0]
            for cc in cand[1:]:
                m = B.minimum(m, cc)
            oh = [(cc <= m + 1e-12) * 1.0 for cc in cand]
            in_normal = B.stack([oh[3] - oh[0], oh[4] - oh[1], oh[5] - oh[2]], -1)
            out_normal = d / dist[..., None]
            normal = B.where(inside[..., None], in_normal, out_normal)
            depth = B.where(inside, m + r, r - dist)
            depth = B.where(inside | outside_hit, depth, depth * 0.0)
            depth = B.maximum(depth, depth * 0.0)
            push = push + B.sum(normal * depth[..., None], axis=1)

        if len(self.spheres):
            sc, sr = T["spheres"][:, :3][None], T["spheres"][:, 3][None]
            d = pos[:, None, :] - sc
            dist = B.sqrt(B.sum(d * d, axis=-1) + eps * eps)
            depth = B.maximum(sr + r - dist, 0.0 * dist)
            push = push + B.sum(d / dist[..., None] * depth[..., None], axis=1)

        if len(self.cylinders):
            cy = T["cyl"][None]  # (1,C,5)
            pz = B.minimum(B.maximum(pos[:, None, 2], cy[..., 2]), cy[..., 3])
            d = B.stack([pos[:, None, 0] - cy[..., 0], pos[:, None, 1] - cy[..., 1], pos[:, None, 2] - pz], -1)
            dist = B.sqrt(B.sum(d * d, axis=-1) + eps * eps)
            depth = B.maximum(cy[..., 4] + r - dist, 0.0 * dist)
            push = push + B.sum(d / dist[..., None] * depth[..., None], axis=1)

        if self.meshes:
            mpush, _ = self.mesh_index.collide(B.to_numpy(pos), float(radius))
            push = push + B.asarray(mpush)

        mag = B.sqrt(B.sum(push * push, axis=-1) + eps * eps)
        hit = mag > 1e-6
        return hit, mag * hit, push / mag[..., None]

    def distance_to_nearest(self, pos: np.ndarray) -> np.ndarray:
        """Unsigned distance from points (NumPy, ``(N,3)``) to the nearest obstacle surface."""
        pos = np.asarray(pos, float).reshape(-1, 3)
        best = np.full(len(pos), np.inf)
        if len(self.boxes):
            c = np.clip(pos[:, None, :], self.boxes[None, :, :3], self.boxes[None, :, 3:])
            best = np.minimum(best, np.linalg.norm(pos[:, None, :] - c, axis=-1).min(1))
        if len(self.spheres):
            d = np.linalg.norm(pos[:, None, :] - self.spheres[None, :, :3], axis=-1) - self.spheres[None, :, 3]
            best = np.minimum(best, d.min(1))
        if len(self.cylinders):
            cy = self.cylinders
            pz = np.clip(pos[:, None, 2], cy[None, :, 2], cy[None, :, 3])
            d = (
                np.sqrt((pos[:, None, 0] - cy[None, :, 0]) ** 2 + (pos[:, None, 1] - cy[None, :, 1]) ** 2 + (pos[:, None, 2] - pz) ** 2)
                - cy[None, :, 4]
            )
            best = np.minimum(best, d.min(1))
        if self.meshes:
            best = np.minimum(best, self.mesh_index.nearest(pos)[0])
        return best

    def line_of_sight(self, a: np.ndarray, b: np.ndarray, n_samples: int = 0) -> np.ndarray:
        """Boolean mask: is the segment ``a -> b`` free of obstacles? (NumPy, exact for boxes/spheres/cylinders.)

        ``a`` and ``b`` are ``(N, 3)`` or broadcastable. Used for LoS/NLoS wireless links.
        """
        a = np.atleast_2d(np.asarray(a, float))
        b = np.atleast_2d(np.asarray(b, float))
        a, b = np.broadcast_arrays(a, b)
        d = b - a
        L = np.linalg.norm(d, axis=-1)
        dirn = d / np.maximum(L[:, None], 1e-12)
        from ..sensors.raycast import cast_rays_single_origin_batch

        t = cast_rays_single_origin_batch(self, a, dirn, max_range=None)
        blocked = t < L - 1e-6
        return ~blocked

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        d = dict(
            name=self.name,
            gravity=self.gravity.to_dict(),
            atmosphere=self.atmosphere.to_dict(),
            wind=self.wind.to_dict(),
            bounds=self.bounds.tolist(),
            boxes=self.boxes.tolist(),
            spheres=self.spheres.tolist(),
            cylinders=self.cylinders.tolist(),
            box_colors=self.box_colors.tolist(),
            sphere_colors=self.sphere_colors.tolist(),
            cyl_colors=self.cyl_colors.tolist(),
        )
        if self.heightmap is not None:
            d["heightmap"] = dict(origin=list(self.hm_origin), resolution=self.hm_res, heights=self.heightmap.tolist())
        if self.weather is not None:
            d["weather"] = self.weather.to_dict()
        if self.meshes:
            d["meshes"] = [m.to_dict() for m in self.meshes]
        if self.actors:
            d["actors"] = [dict(actor=a.to_dict(), array=arr, row=int(row)) for a, arr, row in self.actors]
        if self.origin is not None:
            d["origin"] = self.origin.to_dict()
        if self.utc is not None:
            d["utc"] = self.utc
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        """Rebuild a world from :meth:`to_dict` output. Hand-written dicts (e.g. from the viewer's scene editor) may omit gravity, atmosphere,
        wind and colours: those take their defaults, so ``{"bounds": ..., "boxes": [...]}`` is a complete world."""
        wind = wind_from_dict(d["wind"]) if d.get("wind") else NoWind()
        w = cls(GravityModel(**(d.get("gravity") or {})), Atmosphere(**(d.get("atmosphere") or {})), wind, d["bounds"], d.get("name", "world"))
        w.boxes = np.asarray(d.get("boxes", []), float).reshape(-1, 6)
        w.spheres = np.asarray(d.get("spheres", []), float).reshape(-1, 4)
        w.cylinders = np.asarray(d.get("cylinders", []), float).reshape(-1, 5)
        for attr, shapes, grey in (("box_colors", w.boxes, 0.75), ("sphere_colors", w.spheres, 0.8), ("cyl_colors", w.cylinders, 0.45)):
            given = d.get(attr)
            setattr(w, attr, np.asarray(given, float).reshape(-1, 3) if given is not None else np.full((len(shapes), 3), grey))
        if "heightmap" in d:
            h = d["heightmap"]
            w.set_heightmap(h["heights"], h["origin"], h["resolution"])
        if d.get("origin"):
            from .geodesy import LocalFrame

            w.origin = LocalFrame.from_dict(d["origin"])
        w.utc = d.get("utc")
        for ad in d.get("actors", []):  # the primitive rows are already in the arrays: re-attach, do not add
            from .actors import actor_from_dict

            w.actors.append((actor_from_dict(ad["actor"]), ad["array"], int(ad["row"])))
        for md in d.get("meshes", []):
            w.add_mesh(TriangleMesh.from_dict(md))
        if d.get("weather"):
            from .weather import Weather

            Weather.from_dict(d["weather"]).apply(w)
        return w

    # ------------------------------------------------------ procedural scenes
    @classmethod
    def random_forest(cls, n_trees=40, half_extent=30.0, radius=(0.3, 0.8), height=(6.0, 14.0), keep_clear=3.0, seed=0, **kw):
        """Cylindrical trunks scattered over a square, keeping a clear zone around the origin."""
        rng = np.random.default_rng(seed)
        w = cls(name="forest", bounds=(-half_extent, -half_extent, 0.0, half_extent, half_extent, 40.0), **kw)
        placed = 0
        while placed < n_trees:
            xy = rng.uniform(-half_extent, half_extent, 2)
            if np.linalg.norm(xy) < keep_clear:
                continue
            h = rng.uniform(*height)
            w.add_cylinder(xy, rng.uniform(*radius), 0.0, h, color=(0.25 + 0.1 * rng.random(), 0.45 + 0.2 * rng.random(), 0.2))
            placed += 1
        return w

    @classmethod
    def city(cls, blocks=(4, 4), block_size=20.0, street=12.0, height=(8.0, 45.0), seed=0, **kw):
        """Manhattan-style grid of rectangular buildings (useful for urban air-to-ground channels)."""
        rng = np.random.default_rng(seed)
        nx, ny = blocks
        pitch = block_size + street
        ex, ey = nx * pitch / 2, ny * pitch / 2
        w = cls(name="city", bounds=(-ex, -ey, 0.0, ex, ey, 120.0), **kw)
        for i in range(nx):
            for j in range(ny):
                cx = (i + 0.5) * pitch - ex
                cy = (j + 0.5) * pitch - ey
                h = rng.uniform(*height)
                sx, sy = block_size * rng.uniform(0.6, 0.95), block_size * rng.uniform(0.6, 0.95)
                g = rng.uniform(0.45, 0.8)
                w.add_box((cx, cy, h / 2), (sx, sy, h), color=(g, g, g + 0.05))
        return w

    @classmethod
    def corridor(cls, length=40.0, width=4.0, height=3.0, n_pillars=6, seed=0, **kw):
        """Enclosed corridor with pillars to weave between."""
        rng = np.random.default_rng(seed)
        w = cls(name="corridor", bounds=(-2.0, -width / 2 - 1, 0.0, length, width / 2 + 1, height + 1), **kw)
        t = 0.3
        w.add_box((length / 2, width / 2 + t / 2, height / 2), (length, t, height), color=(0.6, 0.6, 0.65))
        w.add_box((length / 2, -width / 2 - t / 2, height / 2), (length, t, height), color=(0.6, 0.6, 0.65))
        w.add_box((length / 2, 0.0, height + t / 2), (length, width, t), color=(0.7, 0.7, 0.72))
        for k in range(n_pillars):
            x = (k + 1) * length / (n_pillars + 1)
            y = rng.uniform(-width / 2 + 0.6, width / 2 - 0.6)
            w.add_cylinder((x, y), 0.25, 0.0, height, color=(0.8, 0.45, 0.2))
        return w

    @classmethod
    def hills(cls, half_extent=50.0, resolution=1.0, amplitude=6.0, scale=18.0, seed=0, **kw):
        """Smooth random terrain from filtered noise."""
        rng = np.random.default_rng(seed)
        n = int(2 * half_extent / resolution) + 1
        noise = rng.standard_normal((n, n))
        fx = np.fft.fftfreq(n, d=resolution)
        kx, ky = np.meshgrid(fx, fx)
        filt = np.exp(-((kx * kx + ky * ky) * (2 * math.pi * scale) ** 2) / 2.0)
        h = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
        h = (h - h.min()) / (h.max() - h.min() + 1e-12) * amplitude
        w = cls(name="hills", bounds=(-half_extent, -half_extent, 0.0, half_extent, half_extent, 60.0), **kw)
        w.set_heightmap(h, origin=(-half_extent, -half_extent), resolution=resolution)
        return w

    @classmethod
    def from_spec(cls, spec: str, seed=0, **kw):
        table: Dict[str, Any] = {"empty": cls, "forest": cls.random_forest, "city": cls.city, "corridor": cls.corridor, "hills": cls.hills}
        if spec not in table:
            raise ValueError(f"unknown world {spec!r}; choose from {sorted(table)}")
        return cls(**kw) if spec == "empty" else table[spec](seed=seed, **kw)


def _color(color, salt):
    if color is not None:
        return np.asarray(color, float)[None]
    r = np.random.default_rng(salt)
    g = 0.5 + 0.3 * r.random()
    return np.array([[g, g * 0.95, g * 0.9]])
