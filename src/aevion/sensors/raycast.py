"""Ray casting against a :class:`~aevion.environments.world.World`.

Two implementations share one result format:

* a Numba-compiled, multi-core CPU kernel (used automatically when Numba is installed
  and the inputs are NumPy arrays), and
* a vectorised implementation written against the :class:`Backend` interface, so it runs
  on NumPy without Numba and on PyTorch tensors (including CUDA).

Primitive ids in :class:`RayHits.prim`: ``-1`` miss, ``-2`` terrain/ground,
``0..K-1`` boxes, ``K..K+S-1`` spheres, ``K+S..K+S+C-1`` cylinders, then one id per mesh triangle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.backend import NumpyBackend, is_tensor

try:  # optional acceleration
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

BIG = 1e9


@dataclass
class RayHits:
    t: Any  # hit distance (== max_range on miss)
    normal: Any  # (R, 3) surface normal (zeros on miss)
    prim: Any  # (R,) int primitive id

    @property
    def hit(self):
        return self.prim != -1


# --------------------------------------------------------------------------- numba
if HAVE_NUMBA:

    @njit(cache=True, fastmath=True)
    def _terrain_height(hm, ox, oy, res, x, y):
        H, W = hm.shape
        fx = (x - ox) / res
        fy = (y - oy) / res
        fx = min(max(fx, 0.0), W - 1.001)
        fy = min(max(fy, 0.0), H - 1.001)
        ix = int(fx)
        iy = int(fy)
        tx = fx - ix
        ty = fy - iy
        return (hm[iy, ix] * (1 - tx) + hm[iy, ix + 1] * tx) * (1 - ty) + (hm[iy + 1, ix] * (1 - tx) + hm[iy + 1, ix + 1] * tx) * ty

    @njit(parallel=True, fastmath=True, cache=True)
    def _cast_nb(o, d, boxes, spheres, cyls, hm, has_hm, hm_ox, hm_oy, hm_res, max_range, out_t, out_n, out_id):
        R = d.shape[0]
        K = boxes.shape[0]
        S = spheres.shape[0]
        C = cyls.shape[0]
        for i in prange(R):  # type: ignore[attr-defined]
            ox, oy, oz = o[i, 0], o[i, 1], o[i, 2]
            dx, dy, dz = d[i, 0], d[i, 1], d[i, 2]
            best = max_range
            bid = -1
            bnx, bny, bnz = 0.0, 0.0, 0.0

            # ---- ground / terrain
            if has_hm:
                step = hm_res * 0.5
                t_prev = 0.0
                f_prev = oz - _terrain_height(hm, hm_ox, hm_oy, hm_res, ox, oy)
                if f_prev > 0.0:
                    t = step
                    while t_prev < best:
                        tt = min(t, best)
                        f = (oz + dz * tt) - _terrain_height(hm, hm_ox, hm_oy, hm_res, ox + dx * tt, oy + dy * tt)
                        if f <= 0.0:
                            lo, hi = t_prev, tt
                            for _ in range(10):
                                mid = 0.5 * (lo + hi)
                                fm = (oz + dz * mid) - _terrain_height(hm, hm_ox, hm_oy, hm_res, ox + dx * mid, oy + dy * mid)
                                if fm > 0.0:
                                    lo = mid
                                else:
                                    hi = mid
                            th = 0.5 * (lo + hi)
                            if th < best:
                                best = th
                                bid = -2
                                px, py = ox + dx * th, oy + dy * th
                                e = hm_res * 0.5
                                gx = (
                                    _terrain_height(hm, hm_ox, hm_oy, hm_res, px + e, py) - _terrain_height(hm, hm_ox, hm_oy, hm_res, px - e, py)
                                ) / (2 * e)
                                gy = (
                                    _terrain_height(hm, hm_ox, hm_oy, hm_res, px, py + e) - _terrain_height(hm, hm_ox, hm_oy, hm_res, px, py - e)
                                ) / (2 * e)
                                nn = (gx * gx + gy * gy + 1.0) ** 0.5
                                bnx, bny, bnz = -gx / nn, -gy / nn, 1.0 / nn
                            break
                        t_prev = tt
                        if tt >= best:
                            break
                        t += step
            else:
                if dz < -1e-12 and oz > 0.0:
                    t = -oz / dz
                    if t < best:
                        best = t
                        bid = -2
                        bnx, bny, bnz = 0.0, 0.0, 1.0

            # ---- boxes (slab method)
            for b in range(K):
                tn = -1e30
                tf = 1e30
                ax = -1
                ok = True
                for k in range(3):
                    ok_k = o[i, k]
                    dk = d[i, k]
                    lo_b = boxes[b, k]
                    hi_b = boxes[b, 3 + k]
                    if abs(dk) < 1e-12:
                        if ok_k < lo_b or ok_k > hi_b:
                            ok = False
                            break
                    else:
                        a = (lo_b - ok_k) / dk
                        c = (hi_b - ok_k) / dk
                        if a > c:
                            a, c = c, a
                        if a > tn:
                            tn = a
                            ax = k
                        if c < tf:
                            tf = c
                        if tn > tf:
                            ok = False
                            break
                if ok and tn > 1e-9 and tn < best and ax >= 0:
                    best = tn
                    bid = b
                    bnx, bny, bnz = 0.0, 0.0, 0.0
                    sgn = -1.0 if d[i, ax] > 0 else 1.0
                    if ax == 0:
                        bnx = sgn
                    elif ax == 1:
                        bny = sgn
                    else:
                        bnz = sgn

            # ---- spheres
            for s in range(S):
                cx, cy, cz, r = spheres[s, 0], spheres[s, 1], spheres[s, 2], spheres[s, 3]
                lx, ly, lz = ox - cx, oy - cy, oz - cz
                bq = lx * dx + ly * dy + lz * dz
                cq = lx * lx + ly * ly + lz * lz - r * r
                disc = bq * bq - cq
                if disc >= 0.0:
                    sq = disc**0.5
                    t = -bq - sq
                    if t > 1e-9 and t < best:
                        best = t
                        bid = K + s
                        bnx, bny, bnz = (lx + dx * t) / r, (ly + dy * t) / r, (lz + dz * t) / r

            # ---- vertical cylinders
            for c in range(C):
                cx, cy, z0, z1, r = cyls[c, 0], cyls[c, 1], cyls[c, 2], cyls[c, 3], cyls[c, 4]
                lx, ly = ox - cx, oy - cy
                a = dx * dx + dy * dy
                if a > 1e-12:
                    bq = lx * dx + ly * dy
                    cq = lx * lx + ly * ly - r * r
                    disc = bq * bq - a * cq
                    if disc >= 0.0:
                        t = (-bq - disc**0.5) / a
                        if t > 1e-9 and t < best:
                            zz = oz + dz * t
                            if zz >= z0 and zz <= z1:
                                best = t
                                bid = K + S + c
                                bnx, bny, bnz = (lx + dx * t) / r, (ly + dy * t) / r, 0.0
                # caps
                if abs(dz) > 1e-12:
                    for cap in range(2):
                        zc = z1 if cap == 1 else z0
                        t = (zc - oz) / dz
                        if t > 1e-9 and t < best:
                            px, py = lx + dx * t, ly + dy * t
                            if px * px + py * py <= r * r:
                                # only count the cap that faces the ray
                                if (cap == 1 and dz < 0) or (cap == 0 and dz > 0):
                                    best = t
                                    bid = K + S + c
                                    bnx, bny, bnz = 0.0, 0.0, (1.0 if cap == 1 else -1.0)

            out_t[i] = best
            out_id[i] = bid
            out_n[i, 0] = bnx
            out_n[i, 1] = bny
            out_n[i, 2] = bnz


def _cast_numba(world, o, d, max_range):
    R = d.shape[0]
    out_t = np.empty(R)
    out_n = np.zeros((R, 3))
    out_id = np.empty(R, dtype=np.int64)
    has_hm = world.heightmap is not None
    hm = world.heightmap if has_hm else np.zeros((2, 2))
    _cast_nb(
        np.ascontiguousarray(o, dtype=np.float64),
        np.ascontiguousarray(d, dtype=np.float64),
        np.ascontiguousarray(world.boxes),
        np.ascontiguousarray(world.spheres),
        np.ascontiguousarray(world.cylinders),
        np.ascontiguousarray(hm),
        has_hm,
        world.hm_origin[0],
        world.hm_origin[1],
        world.hm_res,
        float(max_range),
        out_t,
        out_n,
        out_id,
    )
    return out_t, out_n, out_id


# ------------------------------------------------------------------ vectorised
def _cast_vectorized(B, world, o, d, max_range, chunk_elems=3_000_000):
    """Backend-agnostic implementation (NumPy or torch), chunked over rays."""
    R = d.shape[0]
    K, S, C = len(world.boxes), len(world.spheres), len(world.cylinders)
    T = world._tensors(B)
    outs_t, outs_n, outs_i = [], [], []
    per_ray = max(3 * max(K, S, C, 1), 1)
    step = max(int(chunk_elems // per_ray), 256)
    for s0 in range(0, R, step):
        oo, dd = o[s0 : s0 + step], d[s0 : s0 + step]
        n = dd.shape[0]
        best = B.full((n,), max_range)
        bid = B.full((n,), -1.0)
        bn = B.zeros(n, 3)

        def update(t, nrm, ids, valid):
            nonlocal best, bid, bn
            better = valid & (t < best) & (t > 1e-9)
            best = B.where(better, t, best)
            bid = B.where(better, ids, bid)
            bn = B.where(better[..., None], nrm, bn)

        # ground
        if world.heightmap is None:
            dz = dd[:, 2]
            safe = B.where(B.abs(dz) < 1e-12, B.full((n,), 1e-12), dz)
            t = -oo[:, 2] / safe
            valid = (dz < -1e-12) & (oo[:, 2] > 0.0)
            nrm = B.zeros(n, 3) + B.asarray([0.0, 0.0, 1.0])
            update(t, nrm, B.full((n,), -2.0), valid)
        else:
            res = world.hm_res
            ts = np.arange(res * 0.5, max_range + res * 0.5, res * 0.5)
            B.zeros(n)
            found = B.zeros(n) > 1.0
            tfound = B.full((n,), max_range)
            f_prev = oo[:, 2] - world.ground_height(B, oo[:, 0], oo[:, 1])
            above = f_prev > 0.0
            for tt in ts:
                p = oo + dd * float(tt)
                f = p[:, 2] - world.ground_height(B, p[:, 0], p[:, 1])
                cross = above & (~found) & (f <= 0.0)
                tfound = B.where(cross, B.full((n,), float(tt)), tfound)
                found = found | cross
                if bool(found.all()) if hasattr(found, "all") else False:
                    break
            nrm = B.zeros(n, 3) + B.asarray([0.0, 0.0, 1.0])
            update(tfound, nrm, B.full((n,), -2.0), found)

        # boxes
        if K:
            bmin, bmax = T["boxes"][:, :3][None], T["boxes"][:, 3:][None]
            o3, d3 = oo[:, None, :], dd[:, None, :]
            parallel = B.abs(d3) < 1e-12
            dsafe = B.where(parallel, B.full(d3.shape, 1e-12), d3)
            t1, t2 = (bmin - o3) / dsafe, (bmax - o3) / dsafe
            tnear, tfar = B.minimum(t1, t2), B.maximum(t1, t2)
            inside_slab = (o3 >= bmin) & (o3 <= bmax)
            tnear = B.where(parallel, B.where(inside_slab, B.full(tnear.shape, -1e30), B.full(tnear.shape, 1e30)), tnear)
            tfar = B.where(parallel, B.where(inside_slab, B.full(tfar.shape, 1e30), B.full(tfar.shape, -1e30)), tfar)
            tn = B.maximum(B.maximum(tnear[..., 0], tnear[..., 1]), tnear[..., 2])
            tf = B.minimum(B.minimum(tfar[..., 0], tfar[..., 1]), tfar[..., 2])
            valid = (tn <= tf) & (tn > 1e-9)
            tt = B.where(valid, tn, B.full(tn.shape, 1e30))  # (n,K)
            tbest = tt[:, 0]
            kbest = B.zeros(n)
            for k in range(1, K):
                better = tt[:, k] < tbest
                tbest = B.where(better, tt[:, k], tbest)
                kbest = B.where(better, B.full((n,), float(k)), kbest)
            # entering axis of the best box -> normal
            ki = B.to_numpy(kbest).astype(int)
            if B.name == "numpy":
                tn_sel = tnear[np.arange(n), ki]
            else:
                import torch

                tn_sel = tnear[torch.arange(n, device=tnear.device), torch.as_tensor(ki, device=tnear.device)]
            mx = B.maximum(B.maximum(tn_sel[:, 0], tn_sel[:, 1]), tn_sel[:, 2])
            onehot = B.stack([(tn_sel[:, a] >= mx) * 1.0 for a in range(3)], -1)
            nrm = -B.sign(dd) * onehot
            update(tbest, nrm, kbest, tbest < 1e29)

        # spheres
        if S:
            sc, sr = T["spheres"][:, :3][None], T["spheres"][:, 3][None]
            l = oo[:, None, :] - sc
            bq = B.sum(l * dd[:, None, :], axis=-1)
            cq = B.sum(l * l, axis=-1) - sr * sr
            disc = bq * bq - cq
            ok = disc >= 0.0
            t = -bq - B.sqrt(B.maximum(disc, disc * 0.0))
            t = B.where(ok & (t > 1e-9), t, B.full(t.shape, 1e30))
            tbest, sbest = t[:, 0], B.zeros(n)
            for s in range(1, S):
                better = t[:, s] < tbest
                tbest = B.where(better, t[:, s], tbest)
                sbest = B.where(better, B.full((n,), float(s)), sbest)
            si = B.to_numpy(sbest).astype(int)
            ctr = world.spheres[si, :3]
            rr = world.spheres[si, 3:4]
            hitp = oo + dd * tbest[:, None]
            nrm = (hitp - B.asarray(ctr)) / B.asarray(rr)
            update(tbest, nrm, sbest + K, tbest < 1e29)

        # vertical cylinders (side + caps)
        if C:
            T["cyl"]
            for c in range(C):
                cx_, cy_, z0, z1, r = [float(v) for v in world.cylinders[c]]
                lx, ly = oo[:, 0] - cx_, oo[:, 1] - cy_
                dx, dy, dz = dd[:, 0], dd[:, 1], dd[:, 2]
                a = dx * dx + dy * dy
                a_s = B.maximum(a, a * 0.0 + 1e-12)
                bq = lx * dx + ly * dy
                cq = lx * lx + ly * ly - r * r
                disc = bq * bq - a * cq
                t = (-bq - B.sqrt(B.maximum(disc, disc * 0.0))) / a_s
                zz = oo[:, 2] + dz * t
                valid = (a > 1e-12) & (disc >= 0.0) & (zz >= z0) & (zz <= z1)
                nrm = B.stack([(lx + dx * t) / r, (ly + dy * t) / r, t * 0.0], -1)
                update(t, nrm, B.full((n,), float(K + S + c)), valid)
                dz_s = B.where(B.abs(dz) < 1e-12, B.full((n,), 1e-12), dz)
                for zc, up in ((z1, 1.0), (z0, -1.0)):
                    t = (zc - oo[:, 2]) / dz_s
                    px, py = lx + dx * t, ly + dy * t
                    facing = (dz < 0.0) if up > 0 else (dz > 0.0)
                    valid = facing & (px * px + py * py <= r * r)
                    nrm = B.zeros(n, 3) + B.asarray([0.0, 0.0, up])
                    update(t, nrm, B.full((n,), float(K + S + c)), valid)

        outs_t.append(best)
        outs_n.append(bn)
        outs_i.append(bid)
    return B.cat(outs_t), B.cat(outs_n), B.cat(outs_i)


# ------------------------------------------------------------------ public API
def _add_meshes(world, o, d, max_range, t, n, i):
    """Merge triangle-mesh hits (BVH, CPU) into the primitive result arrays; returns new ``(t, n, i)`` in NumPy."""
    idx = world.mesh_index
    if idx is None:
        return t, n, i
    t, n, i = np.array(t, dtype=float), np.array(n, dtype=float), np.array(i, dtype=np.int64)
    tt, nn, tid = idx.cast(o, d, t)  # only hits closer than the primitive hit (or max_range) are reported
    hit = tid >= 0
    base = len(world.boxes) + len(world.spheres) + len(world.cylinders)
    t[hit], n[hit], i[hit] = tt[hit], nn[hit], base + tid[hit]
    return t, n, i


def cast_rays(world, origins, dirs, max_range: float = 100.0, backend=None) -> RayHits:
    """Cast rays. ``origins`` is ``(3,)`` or ``(R, 3)``; ``dirs`` is ``(R, 3)`` unit vectors.

    NumPy inputs use the Numba kernel if available (else the vectorised path); torch tensors
    always use the vectorised path on their own device.
    """
    if is_tensor(dirs):
        import torch

        from ..core.backend import TorchBackend

        B = backend if backend is not None else TorchBackend(str(dirs.device), dirs.dtype)
        o = origins if origins.ndim == 2 else origins[None, :].expand(dirs.shape[0], 3)
        t, n, i = _cast_vectorized(B, world, o, dirs, max_range)
        if world.meshes:
            import torch

            tn, nn_, ii = _add_meshes(world, B.to_numpy(o), B.to_numpy(dirs), max_range, B.to_numpy(t), B.to_numpy(n), B.to_numpy(i))
            t, n = torch.as_tensor(tn, device=t.device, dtype=t.dtype), torch.as_tensor(nn_, device=n.device, dtype=n.dtype)
            i = torch.as_tensor(ii, device=dirs.device)
        return RayHits(t, n, i.long())
    dirs = np.asarray(dirs, dtype=float)
    o = np.asarray(origins, dtype=float)
    if o.ndim == 1:
        o = np.broadcast_to(o, dirs.shape)
    if HAVE_NUMBA:
        t, n, i = _cast_numba(world, o, dirs, max_range)
    else:
        B = backend if backend is not None else NumpyBackend()
        t, n, i = _cast_vectorized(B, world, o, dirs, max_range)
        i = i.astype(np.int64)
    if world.meshes:
        t, n, i = _add_meshes(world, o, dirs, max_range, t, n, i)
    return RayHits(t, n, i)


def cast_rays_single_origin_batch(world, origins, dirs, max_range=None):
    """Distances only; ``max_range=None`` means unbounded. Used for line-of-sight tests."""
    hits = cast_rays(world, origins, dirs, BIG if max_range is None else max_range)
    return hits.t


def ray_aabb_brute(o, d, bmin, bmax):
    """Reference (slow, simple) ray/AABB distance used by the test-suite."""
    inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t1, t2 = (bmin - o) * inv, (bmax - o) * inv
    tn = np.minimum(t1, t2).max()
    tf = np.maximum(t1, t2).min()
    return tn if (tf >= max(tn, 0.0) and tn > 0) else np.inf
