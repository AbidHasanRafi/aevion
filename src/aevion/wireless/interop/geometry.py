"""Export a :class:`~aevion.environments.world.World` as a Wavefront OBJ mesh for external ray tracers and visualisers.

Boxes become cuboids, cylinders 16-sided prisms, spheres low-resolution UV spheres and a terrain heightmap a triangulated grid.
Each object type is its own named group (``box``, ``cylinder``, ``sphere``, ``terrain``) so materials can be assigned per group
(e.g. concrete for ``box``) in tools such as Blender, Mitsuba or Sionna scene editors. No material properties are written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from ..._types import PathLike

__all__ = ["world_to_obj"]

_BOX_FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]


def _box_vertices(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])


def world_to_obj(world: Any, path: PathLike, cylinder_sides: int = 16, sphere_res: Tuple[int, int] = (12, 8)) -> Path:
    verts: List[np.ndarray] = []
    out: List[str] = []
    Face = Tuple[int, ...]
    n_v = 0

    def add(vs: np.ndarray) -> int:
        nonlocal n_v
        verts.append(np.asarray(vs, float))
        start = n_v + 1
        n_v += len(vs)
        return start

    def emit(group: str, faces: Sequence[Face]) -> None:
        out.append(f"g {group}")
        out.extend("f " + " ".join(map(str, f)) for f in faces)

    for b in np.asarray(world.boxes, float).reshape(-1, 6):
        s = add(_box_vertices(b[:3], b[3:]))
        emit("box", [tuple(s + i for i in f) for f in _BOX_FACES])
    for c in np.asarray(world.cylinders, float).reshape(-1, 5):
        x, y, z0, z1, r = c
        a = np.linspace(0, 2 * math.pi, cylinder_sides, endpoint=False)
        ring = np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1)
        s = add(np.vstack([np.c_[ring, np.full(cylinder_sides, z0)], np.c_[ring, np.full(cylinder_sides, z1)]]))
        k = cylinder_sides
        faces: List[Face] = [(s + i, s + (i + 1) % k, s + k + (i + 1) % k, s + k + i) for i in range(k)]
        faces += [tuple(s + i for i in range(k - 1, -1, -1)), tuple(s + k + i for i in range(k))]
        emit("cylinder", faces)
    for sp in np.asarray(world.spheres, float).reshape(-1, 4):
        cx, cy, cz, r = sp
        nu, nv = sphere_res
        pts: List[List[float]] = []
        faces = []
        for j in range(nv + 1):
            phi = math.pi * j / nv
            for i in range(nu):
                th = 2 * math.pi * i / nu
                pts.append([cx + r * math.sin(phi) * math.cos(th), cy + r * math.sin(phi) * math.sin(th), cz + r * math.cos(phi)])
        s = add(np.array(pts))
        for j in range(nv):
            for i in range(nu):
                v0, v1 = s + j * nu + i, s + j * nu + (i + 1) % nu
                faces.append((v0, v1, v1 + nu, v0 + nu))
        emit("sphere", faces)
    if getattr(world, "heightmap", None) is not None:
        h = np.asarray(world.heightmap, float)
        ny, nx = h.shape
        xs = world.hm_origin[0] + np.arange(nx) * world.hm_res
        ys = world.hm_origin[1] + np.arange(ny) * world.hm_res
        X, Y = np.meshgrid(xs, ys)
        s = add(np.stack([X.ravel(), Y.ravel(), h.ravel()], 1))
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = s + j * nx + i
                faces.append((v0, v0 + 1, v0 + nx + 1, v0 + nx))
        emit("terrain", faces)
    lines = [f"# Aevion world '{getattr(world, 'name', '')}' (metres, z up)"]
    if verts:
        lines.extend(f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in np.vstack(verts))
    lines.extend(out)
    p = Path(path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
