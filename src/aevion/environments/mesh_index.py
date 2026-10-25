"""A bounding-volume hierarchy over triangle meshes: ray casting, sphere collision, nearest-point and inside/outside queries.

The BVH is a binary tree of axis-aligned boxes built by median splits on the longest centroid axis (leaf size 4). Traversal kernels
are compiled with Numba when installed (multi-core, allocation-free); without Numba the same queries run through brute-force NumPy
references, which are also the oracle the tests compare the kernels against. Torch tensors are handled by the callers (the mesh pass
runs on the CPU).

Conventions: ``cast`` reports a hit only closer than ``tmax``; a miss gives ``t == tmax`` and ``tri == -1``. Normals face against the
ray. ``collide`` reports the deepest penetrating triangle per sphere, since summing contacts would double-count the shared edge of
the two triangles of a flat face.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

from .mesh import MeshError, TriangleMesh

try:  # optional acceleration
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba missing
    HAVE_NUMBA = False

    def njit(*a: Any, **k: Any) -> Any:  # type: ignore[no-redef]
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

    prange = range  # type: ignore[assignment,misc]

__all__ = ["MeshIndex", "closest_points_on_triangles", "HAVE_NUMBA"]

LEAF_SIZE = 4
STACK = 128
EPS_T = 1e-9


# ------------------------------------------------------------------------------------------------ build
def _build_py(tri_lo, tri_hi, cen, leaf_size):  # pragma: no cover - compiled by numba when available
    n = cen.shape[0]
    order = np.arange(n).astype(np.int32)
    cap = 2 * n + 1
    lo = np.empty((cap, 3))
    hi = np.empty((cap, 3))
    left = np.full(cap, -1, np.int32)
    right = np.full(cap, -1, np.int32)
    start = np.zeros(cap, np.int32)
    count = np.zeros(cap, np.int32)
    st_n = np.empty(STACK, np.int32)
    st_s = np.empty(STACK, np.int32)
    st_e = np.empty(STACK, np.int32)
    nn = 1
    sp = 0
    st_n[0], st_s[0], st_e[0] = 0, 0, n
    sp = 1
    while sp > 0:
        sp -= 1
        node, s, e = st_n[sp], st_s[sp], st_e[sp]
        for k in range(3):
            lo[node, k] = 1e300
            hi[node, k] = -1e300
        cmin = np.full(3, 1e300)
        cmax = np.full(3, -1e300)
        for j in range(s, e):
            t = order[j]
            for k in range(3):
                if tri_lo[t, k] < lo[node, k]:
                    lo[node, k] = tri_lo[t, k]
                if tri_hi[t, k] > hi[node, k]:
                    hi[node, k] = tri_hi[t, k]
                if cen[t, k] < cmin[k]:
                    cmin[k] = cen[t, k]
                if cen[t, k] > cmax[k]:
                    cmax[k] = cen[t, k]
        ext = cmax - cmin
        axis = 0
        if ext[1] > ext[axis]:
            axis = 1
        if ext[2] > ext[axis]:
            axis = 2
        if e - s <= leaf_size or ext[axis] <= 0.0:
            start[node] = s
            count[node] = e - s
            continue
        idx = order[s:e].copy()
        perm = np.argsort(cen[idx, axis])
        order[s:e] = idx[perm]
        mid = (s + e) // 2
        left[node] = nn
        right[node] = nn + 1
        st_n[sp], st_s[sp], st_e[sp] = nn, s, mid
        st_n[sp + 1], st_s[sp + 1], st_e[sp + 1] = nn + 1, mid, e
        sp += 2
        nn += 2
    return order, lo[:nn].copy(), hi[:nn].copy(), left[:nn].copy(), right[:nn].copy(), start[:nn].copy(), count[:nn].copy()


_build = njit(cache=True)(_build_py) if HAVE_NUMBA else _build_py


# ---------------------------------------------------------------------------------------- numba kernels
if HAVE_NUMBA:

    @njit(cache=True, error_model="numpy")
    def _ray_tri(ox, oy, oz, dx, dy, dz, V, F, t):
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        e1x, e1y, e1z = V[b, 0] - V[a, 0], V[b, 1] - V[a, 1], V[b, 2] - V[a, 2]
        e2x, e2y, e2z = V[c, 0] - V[a, 0], V[c, 1] - V[a, 1], V[c, 2] - V[a, 2]
        px, py, pz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
        det = e1x * px + e1y * py + e1z * pz
        if abs(det) < 1e-14:
            return -1.0
        inv = 1.0 / det
        tx, ty, tz = ox - V[a, 0], oy - V[a, 1], oz - V[a, 2]
        u = (tx * px + ty * py + tz * pz) * inv
        if u < -1e-9 or u > 1.0 + 1e-9:
            return -1.0
        qx, qy, qz = ty * e1z - tz * e1y, tz * e1x - tx * e1z, tx * e1y - ty * e1x
        v = (dx * qx + dy * qy + dz * qz) * inv
        if v < -1e-9 or u + v > 1.0 + 1e-9:
            return -1.0
        return (e2x * qx + e2y * qy + e2z * qz) * inv

    @njit(cache=True, error_model="numpy")
    def _slab(ox, oy, oz, ix, iy, iz, lo, hi, node, tmax):
        t1 = (lo[node, 0] - ox) * ix
        t2 = (hi[node, 0] - ox) * ix
        tn, tf = min(t1, t2), max(t1, t2)
        t1 = (lo[node, 1] - oy) * iy
        t2 = (hi[node, 1] - oy) * iy
        tn, tf = max(tn, min(t1, t2)), min(tf, max(t1, t2))
        t1 = (lo[node, 2] - oz) * iz
        t2 = (hi[node, 2] - oz) * iz
        tn, tf = max(tn, min(t1, t2)), min(tf, max(t1, t2))
        return tf >= max(tn, 0.0) and tn <= tmax

    @njit(parallel=True, cache=True, error_model="numpy")
    def _cast_nb(o, d, tmax, V, F, order, lo, hi, left, start, count, right, stacks, out_t, out_id):
        R = o.shape[0]
        nb = stacks.shape[0]
        chunk = (R + nb - 1) // nb
        for b in prange(nb):  # type: ignore[attr-defined]
            stack = stacks[b]
            for i in range(b * chunk, min((b + 1) * chunk, R)):
                ox, oy, oz = o[i, 0], o[i, 1], o[i, 2]
                dx, dy, dz = d[i, 0], d[i, 1], d[i, 2]
                # a zero component would give 0 * inf = nan when the origin lies exactly on a slab plane: use a tiny direction instead
                ix = 1.0 / (dx if abs(dx) > 1e-300 else 1e-300)
                iy = 1.0 / (dy if abs(dy) > 1e-300 else 1e-300)
                iz = 1.0 / (dz if abs(dz) > 1e-300 else 1e-300)
                best = tmax[i]
                bid = -1
                stack[0] = 0
                sp = 1
                while sp > 0:
                    sp -= 1
                    node = stack[sp]
                    if not _slab(ox, oy, oz, ix, iy, iz, lo, hi, node, best):
                        continue
                    if count[node] > 0:
                        for k in range(start[node], start[node] + count[node]):
                            tri = order[k]
                            t = _ray_tri(ox, oy, oz, dx, dy, dz, V, F, tri)
                            if t > EPS_T and t < best:
                                best = t
                                bid = tri
                    else:
                        stack[sp] = left[node]
                        stack[sp + 1] = right[node]
                        sp += 2
                out_t[i] = best
                out_id[i] = bid

    @njit(cache=True, error_model="numpy")
    def _closest_tri(px, py, pz, V, F, t):
        """Closest point on triangle ``t`` to ``p`` (Ericson, Real-Time Collision Detection 5.1.5)."""
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        ax, ay, az = V[a, 0], V[a, 1], V[a, 2]
        bx, by, bz = V[b, 0], V[b, 1], V[b, 2]
        cx, cy, cz = V[c, 0], V[c, 1], V[c, 2]
        abx, aby, abz = bx - ax, by - ay, bz - az
        acx, acy, acz = cx - ax, cy - ay, cz - az
        apx, apy, apz = px - ax, py - ay, pz - az
        d1 = abx * apx + aby * apy + abz * apz
        d2 = acx * apx + acy * apy + acz * apz
        if d1 <= 0.0 and d2 <= 0.0:
            return ax, ay, az
        bpx, bpy, bpz = px - bx, py - by, pz - bz
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0.0 and d4 <= d3:
            return bx, by, bz
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return ax + v * abx, ay + v * aby, az + v * abz
        cpx, cpy, cpz = px - cx, py - cy, pz - cz
        d5 = abx * cpx + aby * cpy + abz * cpz
        d6 = acx * cpx + acy * cpy + acz * cpz
        if d6 >= 0.0 and d5 <= d6:
            return cx, cy, cz
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return ax + w * acx, ay + w * acy, az + w * acz
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz)
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w

    @njit(cache=True, error_model="numpy")
    def _box_dist2(px, py, pz, lo, hi, node):
        dx = max(lo[node, 0] - px, 0.0, px - hi[node, 0])
        dy = max(lo[node, 1] - py, 0.0, py - hi[node, 1])
        dz = max(lo[node, 2] - pz, 0.0, pz - hi[node, 2])
        return dx * dx + dy * dy + dz * dz

    @njit(parallel=True, cache=True, error_model="numpy")
    def _nearest_nb(p, rmax, V, F, order, lo, hi, left, right, start, count, stacks, out_d, out_id, out_q):
        N = p.shape[0]
        nb = stacks.shape[0]
        chunk = (N + nb - 1) // nb
        for b in prange(nb):  # type: ignore[attr-defined]
            stack = stacks[b]
            for i in range(b * chunk, min((b + 1) * chunk, N)):
                px, py, pz = p[i, 0], p[i, 1], p[i, 2]
                best2 = rmax * rmax
                bid = -1
                bqx, bqy, bqz = px, py, pz
                stack[0] = 0
                sp = 1
                while sp > 0:
                    sp -= 1
                    node = stack[sp]
                    if _box_dist2(px, py, pz, lo, hi, node) > best2:
                        continue
                    if count[node] > 0:
                        for k in range(start[node], start[node] + count[node]):
                            tri = order[k]
                            qx, qy, qz = _closest_tri(px, py, pz, V, F, tri)
                            dd = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2
                            if dd < best2 or (bid == -1 and dd <= best2):
                                best2, bid, bqx, bqy, bqz = dd, tri, qx, qy, qz
                    else:
                        # visit the nearer child last so it is popped first
                        dl = _box_dist2(px, py, pz, lo, hi, left[node])
                        dr = _box_dist2(px, py, pz, lo, hi, right[node])
                        if dl < dr:
                            stack[sp] = right[node]
                            stack[sp + 1] = left[node]
                        else:
                            stack[sp] = left[node]
                            stack[sp + 1] = right[node]
                        sp += 2
                out_d[i] = np.sqrt(best2) if bid >= 0 else np.inf
                out_id[i] = bid
                out_q[i, 0], out_q[i, 1], out_q[i, 2] = bqx, bqy, bqz


# ------------------------------------------------------------------------------------- numpy references
def closest_points_on_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Vectorised closest points: ``p`` ``(N, 3)``, ``tri`` ``(M, 3, 3)`` -> ``(N, M, 3)`` (Ericson's region tests)."""
    a, b, c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]
    p = p[:, None, :]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, -1), np.sum(ac * ap, -1)
    bp = p - b
    d3, d4 = np.sum(ab * bp, -1), np.sum(ac * bp, -1)
    cp = p - c
    d5, d6 = np.sum(ab * cp, -1), np.sum(ac * cp, -1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        v_in, w_in = vb * denom, vc * denom
        out = a + ab * v_in[..., None] + ac * w_in[..., None]  # interior
        e_bc = b + (c - b) * (((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None])
        e_ac = a + ac * (d2 / (d2 - d6))[..., None]
        e_ab = a + ab * (d1 / (d1 - d3))[..., None]
    out = np.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None], e_bc, out)
    out = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], e_ac, out)
    out = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], e_ab, out)
    out = np.where(((d6 >= 0) & (d5 <= d6))[..., None], np.broadcast_to(c, out.shape), out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[..., None], np.broadcast_to(b, out.shape), out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[..., None], np.broadcast_to(a, out.shape), out)
    return out


def _ray_triangles_brute(o: np.ndarray, d: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Möller-Trumbore for ``R`` rays against ``M`` triangles -> ``(R, M)`` hit distances (``inf`` for misses)."""
    v0, e1, e2 = tri[None, :, 0], (tri[:, 1] - tri[:, 0])[None], (tri[:, 2] - tri[:, 0])[None]
    dd = d[:, None, :]
    pv = np.cross(dd, e2)
    det = np.sum(e1 * pv, -1)
    ok = np.abs(det) >= 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o[:, None, :] - v0
    u = np.sum(tv * pv, -1) * inv
    q = np.cross(tv, e1)
    v = np.sum(dd * q, -1) * inv
    t = np.sum(e2 * q, -1) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > EPS_T)
    return np.where(hit, t, np.inf)


# --------------------------------------------------------------------------------------------- the index
class MeshIndex:
    """BVH over ``meshes``. Triangle ids are global: mesh ``k`` owns ids ``offsets[k] : offsets[k] + n_faces_k``."""

    def __init__(self, meshes: Sequence[TriangleMesh], leaf_size: int = LEAF_SIZE):
        if not meshes:
            raise MeshError("MeshIndex needs at least one mesh")
        self.meshes = list(meshes)
        self.offsets = np.cumsum([0] + [m.n_faces for m in self.meshes])
        voff = np.cumsum([0] + [m.n_vertices for m in self.meshes[:-1]])
        self.vertices = np.ascontiguousarray(np.concatenate([m.vertices for m in self.meshes]))
        self.faces = np.ascontiguousarray(np.concatenate([m.faces + o for m, o in zip(self.meshes, voff)]).astype(np.int64))
        if not len(self.faces):
            raise MeshError("meshes contain no triangles")
        tri = self.vertices[self.faces]
        self.tri_lo, self.tri_hi = tri.min(1), tri.max(1)
        self.order, self.lo, self.hi, self.left, self.right, self.start, self.count = _build(
            self.tri_lo, self.tri_hi, 0.5 * (self.tri_lo + self.tri_hi), int(leaf_size)
        )
        self._stacks: Any = None

    @property
    def n_triangles(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> np.ndarray:
        return np.stack([self.lo[0], self.hi[0]])

    @property
    def depth(self) -> int:
        """Depth of the tree (diagnostic)."""
        best, stack = 0, [(0, 1)]
        while stack:
            n, dd = stack.pop()
            best = max(best, dd)
            if self.count[n] == 0:
                stack += [(int(self.left[n]), dd + 1), (int(self.right[n]), dd + 1)]
        return best

    def mesh_of(self, tri_ids: np.ndarray) -> np.ndarray:
        """Mesh index of each global triangle id (-1 for ids < 0)."""
        ids = np.asarray(tri_ids)
        return np.where(ids >= 0, np.searchsorted(self.offsets, ids, side="right") - 1, -1)

    def _stack_pool(self, n: int) -> np.ndarray:
        import numba

        nb = max(1, min(n, 4 * numba.get_num_threads()))
        if self._stacks is None or self._stacks.shape[0] < nb:
            self._stacks = np.empty((nb, STACK), np.int32)
        return self._stacks[:nb]

    # ----------------------------------------------------------------------------------------- queries
    def cast(self, origins: np.ndarray, dirs: np.ndarray, tmax: Any = 1e9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest hit of each ray closer than ``tmax``: ``(t, normal, tri)``; ``tri == -1`` and ``t == tmax`` on a miss."""
        o = np.ascontiguousarray(np.broadcast_to(np.asarray(origins, float), np.asarray(dirs).shape))
        d = np.ascontiguousarray(np.asarray(dirs, float))
        tm = np.ascontiguousarray(np.broadcast_to(np.asarray(tmax, float), (len(d),)))
        t, tid = tm.copy(), np.full(len(d), -1, np.int64)
        if len(d) == 0:
            return t, np.zeros((0, 3)), tid
        if HAVE_NUMBA:
            _cast_nb(
                o,
                d,
                tm,
                self.vertices,
                self.faces,
                self.order,
                self.lo,
                self.hi,
                self.left,
                self.start,
                self.count,
                self.right,
                self._stack_pool(len(d)),
                t,
                tid,
            )
        else:  # pragma: no cover - exercised only without numba
            tri = self.vertices[self.faces]
            for s in range(0, len(d), max(1, 2_000_000 // len(tri))):
                sl = slice(s, s + max(1, 2_000_000 // len(tri)))
                tt = _ray_triangles_brute(o[sl], d[sl], tri)
                k = np.argmin(tt, axis=1)
                best = tt[np.arange(len(k)), k]
                ok = best < tm[sl]
                t[sl] = np.where(ok, best, tm[sl])
                tid[sl] = np.where(ok, k, -1)
        return t, self.face_normal(tid, d), tid

    def face_normal(self, tri_ids: np.ndarray, dirs: Any = None) -> np.ndarray:
        """Unit geometric normal of the given triangles, flipped to face against ``dirs`` when given; zeros for id -1."""
        ids = np.asarray(tri_ids)
        safe = np.where(ids >= 0, ids, 0)
        v = self.vertices[self.faces[safe]]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-300)
        if dirs is not None:
            n = np.where((np.sum(n * np.asarray(dirs), axis=1) > 0)[:, None], -n, n)
        return n * (ids >= 0)[:, None]

    def nearest(self, points: np.ndarray, rmax: float = np.inf) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance, global triangle id and closest surface point for each query point (``inf`` / -1 beyond ``rmax``)."""
        p = np.ascontiguousarray(np.asarray(points, float).reshape(-1, 3))
        d, tid, q = np.full(len(p), np.inf), np.full(len(p), -1, np.int64), p.copy()
        if len(p) == 0:
            return d, tid, q
        if HAVE_NUMBA:
            r = 1e30 if not np.isfinite(rmax) else float(rmax)
            _nearest_nb(
                p,
                r,
                self.vertices,
                self.faces,
                self.order,
                self.lo,
                self.hi,
                self.left,
                self.right,
                self.start,
                self.count,
                self._stack_pool(len(p)),
                d,
                tid,
                q,
            )
        else:  # pragma: no cover
            tri = self.vertices[self.faces]
            for s in range(0, len(p), max(1, 1_000_000 // len(tri))):
                sl = slice(s, s + max(1, 1_000_000 // len(tri)))
                cp = closest_points_on_triangles(p[sl], tri)
                dist = np.linalg.norm(cp - p[sl][:, None, :], axis=-1)
                k = np.argmin(dist, axis=1)
                ar = np.arange(len(k))
                d[sl], tid[sl], q[sl] = dist[ar, k], k, cp[ar, k]
            far = d > rmax
            d[far], tid[far] = np.inf, -1
        return d, tid, q

    def collide(self, centers: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Push-out vector ``(N, 3)`` and penetration depth ``(N,)`` of spheres against the deepest triangle in contact.

        The push moves the sphere out of the surface along the outward face normal when the centre is behind the face (inside the
        solid): ``depth = radius + distance``; otherwise it points from the surface to the centre with ``depth = radius - distance``.
        Centres buried deeper than ``radius`` inside a closed mesh are found by a parity test (only for points inside the mesh's
        bounding box) and pushed out through the nearest face. Depth is 0 where there is no contact. Outward normals need
        counter-clockwise (outward) winding, which every constructor and loader here preserves.
        """
        c = np.ascontiguousarray(np.asarray(centers, float).reshape(-1, 3))
        dist, tid, q = self.nearest(c, rmax=radius)
        hit = tid >= 0
        out_n = self.face_normal(tid)
        away = c - q
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        behind = hit & (np.sum(away * out_n, axis=1) < -1e-12)
        direction = np.where(behind[:, None], out_n, np.where(norm > 1e-9, away / np.maximum(norm, 1e-300), out_n))
        depth = np.where(hit, np.where(behind, radius + dist, radius - dist), 0.0)
        lo, hi = self.bounds
        deep = ~hit & np.all((c >= lo) & (c <= hi), axis=1)
        if deep.any():
            idx = np.nonzero(deep)[0]
            idx = idx[self.contains(c[idx])]
            if len(idx):
                d2, t2, _ = self.nearest(c[idx])
                direction[idx] = self.face_normal(t2)
                depth[idx] = d2 + radius
        return direction * depth[:, None], depth

    def contains(self, points: np.ndarray, max_crossings: int = 64) -> np.ndarray:
        """Inside test by ray parity along a fixed skew direction (valid for closed, non-self-intersecting meshes)."""
        p = np.asarray(points, float).reshape(-1, 3)
        d = np.broadcast_to(np.array([0.5773502691896258, 0.5773502691896258 * 0.99, 0.5773502691896258 * 1.01]), p.shape)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        o, crossings, active = p.copy(), np.zeros(len(p), int), np.ones(len(p), bool)
        for _ in range(max_crossings):
            if not active.any():
                break
            idx = np.nonzero(active)[0]
            t, _, tid = self.cast(o[idx], d[idx], 1e9)
            hit = tid >= 0
            crossings[idx[hit]] += 1
            o[idx[hit]] += d[idx[hit]] * (t[hit][:, None] + 1e-7)
            active[idx[~hit]] = False
        return (crossings % 2) == 1

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Unsigned distance to the surface, negative inside (meaningful for closed meshes)."""
        p = np.asarray(points, float).reshape(-1, 3)
        d, _, _ = self.nearest(p)
        return np.where(self.contains(p), -d, d)
