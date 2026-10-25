"""Graph search on a :class:`~aevion.planning.space.PlanningSpace`: A-star, weighted A-star, Theta-star and D-star Lite.

* ``A*`` (Hart, Nilsson & Raphael 1968) finds a shortest path on the 6-, 18- or 26-connected voxel graph. With the Euclidean heuristic
  the result is optimal on the graph; ``weight > 1`` gives weighted A-star, whose cost is at most ``weight`` times the optimum. Diagonal
  moves that cut a corner between blocked cells are not allowed, so every graph edge is a collision-free segment of the continuous space.
* ``Theta*`` (Daniel et al. 2010) lets a node's parent be its grandparent when the straight segment between them is free, giving
  any-angle paths shorter than grid paths (not guaranteed optimal in continuous space).
* ``D* Lite`` (Koenig & Likhachev 2002) re-plans incrementally when cells change or the start moves.

The search is a Numba kernel with an array-based binary heap; without Numba the same code runs as plain Python (slowly).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .space import PlanningSpace, _segment_free, njit

Cell = Tuple[int, int, int]

__all__ = ["PlanResult", "astar", "theta_star", "DStarLite", "path_length", "neighbour_offsets"]


@dataclass
class PlanResult:
    """Outcome of a planner: ``path`` is ``(K, 3)`` world points (empty on failure)."""

    path: np.ndarray
    cost: float = math.inf
    success: bool = False
    expansions: int = 0
    time_s: float = 0.0
    reason: str = ""
    info: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


def _cell(x: Any) -> Cell:
    return (int(x[0]), int(x[1]), int(x[2]))


def path_length(path: Any) -> float:
    p = np.asarray(path, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def neighbour_offsets(connectivity: int = 26) -> np.ndarray:
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18 or 26")
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = abs(dx) + abs(dy) + abs(dz)
                if k == 0 or (connectivity == 6 and k > 1) or (connectivity == 18 and k > 2):
                    continue
                out.append((dx, dy, dz))
    return np.array(out, np.int64)


@njit(cache=False)
def _move_free(occ, x, y, z, dx, dy, dz):  # pragma: no cover - compiled
    """A move is free when its target and every voxel its unit-cube 'swept corner' touches are free (no corner cutting)."""
    nx, ny, nz = occ.shape
    for ax in range(2):
        for ay in range(2):
            for az in range(2):
                cx = x + (dx if ax == 1 else 0)
                cy = y + (dy if ay == 1 else 0)
                cz = z + (dz if az == 1 else 0)
                if cx < 0 or cy < 0 or cz < 0 or cx >= nx or cy >= ny or cz >= nz:
                    return False
                if occ[cx, cy, cz]:
                    return False
    return True


@njit(cache=False)
def _heap_push(hf, hi, size, f, i):  # pragma: no cover - compiled
    j = size
    hf[j] = f
    hi[j] = i
    while j > 0:
        p = (j - 1) // 2
        if hf[p] <= hf[j]:
            break
        tf, ti = hf[p], hi[p]
        hf[p], hi[p] = hf[j], hi[j]
        hf[j], hi[j] = tf, ti
        j = p
    return size + 1


@njit(cache=False)
def _heap_pop(hf, hi, size):  # pragma: no cover - compiled
    f0, i0 = hf[0], hi[0]
    size -= 1
    hf[0], hi[0] = hf[size], hi[size]
    j = 0
    while True:
        l = 2 * j + 1
        r = l + 1
        m = j
        if l < size and hf[l] < hf[m]:
            m = l
        if r < size and hf[r] < hf[m]:
            m = r
        if m == j:
            break
        tf, ti = hf[m], hi[m]
        hf[m], hi[m] = hf[j], hi[j]
        hf[j], hi[j] = tf, ti
        j = m
    return f0, i0, size


@njit(cache=False)
def _search(occ, cost_mult, use_cost, start, goal, offs, weight, theta, lo, inv_res, res, max_exp):  # pragma: no cover - compiled
    nx, ny, nz = occ.shape
    n = nx * ny * nz
    g = np.full(n, np.inf)
    parent = np.full(n, -1, np.int64)
    closed = np.zeros(n, np.bool_)
    cap = 1024
    hf = np.empty(cap)
    hi = np.empty(cap, np.int64)
    size = 0
    si = (start[0] * ny + start[1]) * nz + start[2]
    gi = (goal[0] * ny + goal[1]) * nz + goal[2]
    g[si] = 0.0
    parent[si] = si
    size = _heap_push(
        hf, hi, size, weight * res * math.sqrt(float((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2 + (start[2] - goal[2]) ** 2)), si
    )
    n_exp = 0
    found = False
    cx = np.empty(3, np.float64)
    cy = np.empty(3, np.float64)
    while size > 0:
        f0, cur, size = _heap_pop(hf, hi, size)
        if closed[cur]:
            continue
        closed[cur] = True
        n_exp += 1
        if cur == gi:
            found = True
            break
        if n_exp > max_exp:
            break
        z = cur % nz
        y = (cur // nz) % ny
        x = cur // (nz * ny)
        for k in range(offs.shape[0]):
            dx, dy, dz = offs[k, 0], offs[k, 1], offs[k, 2]
            xx, yy, zz = x + dx, y + dy, z + dz
            if xx < 0 or yy < 0 or zz < 0 or xx >= nx or yy >= ny or zz >= nz:
                continue
            if not _move_free(occ, x, y, z, dx, dy, dz):
                continue
            nb = (xx * ny + yy) * nz + zz
            if closed[nb]:
                continue
            step = res * math.sqrt(float(dx * dx + dy * dy + dz * dz))
            if use_cost:
                step *= 0.5 * (cost_mult[x, y, z] + cost_mult[xx, yy, zz])
            gn = g[cur] + step
            pn = cur
            if theta:  # any-angle: try to connect the neighbour straight to the current node's parent
                pp = parent[cur]
                if pp != cur:
                    px = pp // (nz * ny)
                    py = (pp // nz) % ny
                    pz = pp % nz
                    cx[0], cx[1], cx[2] = lo[0] + (px + 0.5) * res, lo[1] + (py + 0.5) * res, lo[2] + (pz + 0.5) * res
                    cy[0], cy[1], cy[2] = lo[0] + (xx + 0.5) * res, lo[1] + (yy + 0.5) * res, lo[2] + (zz + 0.5) * res
                    if _segment_free(occ, lo, inv_res, cx, cy):
                        gp = g[pp] + math.sqrt((cx[0] - cy[0]) ** 2 + (cx[1] - cy[1]) ** 2 + (cx[2] - cy[2]) ** 2)
                        if gp < gn:
                            gn, pn = gp, pp
            if gn < g[nb]:
                g[nb] = gn
                parent[nb] = pn
                h = res * math.sqrt(float((xx - goal[0]) ** 2 + (yy - goal[1]) ** 2 + (zz - goal[2]) ** 2))
                if size >= cap:
                    cap *= 2
                    nf = np.empty(cap)
                    ni = np.empty(cap, np.int64)
                    nf[:size] = hf[:size]
                    ni[:size] = hi[:size]
                    hf, hi = nf, ni
                size = _heap_push(hf, hi, size, gn + weight * h, nb)
    return found, parent, g, n_exp


def _solve(space: PlanningSpace, start: Any, goal: Any, connectivity: int, weight: float, theta: bool, max_expansions: int, snap: bool) -> PlanResult:
    t0 = time.perf_counter()
    s, e = np.asarray(start, float), np.asarray(goal, float)
    if not (np.isfinite(s).all() and np.isfinite(e).all()):
        return PlanResult(np.empty((0, 3)), reason="the start or goal is not finite")
    if weight < 1.0:
        raise ValueError("weight must be >= 1 (1 gives optimal A*)")
    if theta and space.cost_mult is not None:
        raise ValueError("Theta* ignores clearance weighting; use A* with a cost field, or inflate the obstacles")
    for name, p in (("start", s), ("goal", e)):
        if not space.inside(p)[0]:
            return PlanResult(np.empty((0, 3)), reason=f"the {name} lies outside the planning space")
    sc, gc = space.index(s), space.index(e)
    note = ""
    if space.occ[tuple(sc)] or space.occ[tuple(gc)]:
        if not snap:
            return PlanResult(np.empty((0, 3)), reason="the start or goal lies in an obstacle (inflation included)")
        ns, ng = space.nearest_free(s), space.nearest_free(e)
        if ns is None or ng is None:
            return PlanResult(np.empty((0, 3)), reason="no free cell near the start or goal")
        sc, gc = space.index(ns), space.index(ng)
        note = "start/goal moved to the nearest free cell"
    offs = neighbour_offsets(connectivity)
    use_cost = space.cost_mult is not None
    cm = space.cost_mult if use_cost else np.ones((1, 1, 1))
    found, parent, g, n_exp = _search(
        space.occ,
        np.ascontiguousarray(cm, dtype=np.float64),
        use_cost,
        sc,
        gc,
        offs,
        float(weight),
        bool(theta),
        space.lo,
        1.0 / space.res,
        space.res,
        int(max_expansions),
    )
    dt = time.perf_counter() - t0
    if not found:
        return PlanResult(
            np.empty((0, 3)), expansions=int(n_exp), time_s=dt, reason="no path" if n_exp <= max_expansions else "expansion limit reached"
        )
    ny, nz = space.shape[1], space.shape[2]
    gi = (int(gc[0]) * ny + int(gc[1])) * nz + int(gc[2])
    chain = [gi]
    while parent[chain[-1]] != chain[-1]:
        chain.append(int(parent[chain[-1]]))
    chain.reverse()
    cells = np.array([(c // (nz * ny), (c // nz) % ny, c % nz) for c in chain])
    pts = space.center(cells)
    # the exact start and goal replace the cell centres they snapped to
    if not note:
        pts[0], pts[-1] = s, e
    path = pts if len(pts) > 1 else np.vstack([pts, pts])
    return PlanResult(
        path,
        cost=float(g[gi]),
        success=True,
        expansions=int(n_exp),
        time_s=dt,
        reason=note or "ok",
        info=dict(connectivity=connectivity, weight=weight, theta=theta),
    )


def astar(
    space: PlanningSpace,
    start: Any,
    goal: Any,
    connectivity: int = 26,
    weight: float = 1.0,
    max_expansions: int = 5_000_000,
    snap_to_free: bool = False,
) -> PlanResult:
    """Shortest path on the voxel graph (``weight > 1``: weighted A\\*, cost <= weight x optimal)."""
    return _solve(space, start, goal, connectivity, weight, False, max_expansions, snap_to_free)


def theta_star(
    space: PlanningSpace, start: Any, goal: Any, connectivity: int = 26, max_expansions: int = 5_000_000, snap_to_free: bool = False
) -> PlanResult:
    """Any-angle path (parents may be non-adjacent when the straight segment between them is free)."""
    return _solve(space, start, goal, connectivity, 1.0, True, max_expansions, snap_to_free)


# ======================================================================================================== D* Lite
class DStarLite:
    """Incremental shortest-path search that repairs its solution when obstacles appear or disappear (Koenig & Likhachev 2002).

    The search runs *backwards* from the goal so that the start can move (the vehicle flies along the path and the map changes)
    without discarding what has been computed. Edges are the voxel graph's with Euclidean costs; a blocked cell has infinite cost.
    Use :meth:`update_cells` after the occupancy changed and :meth:`plan` to get the current path from the (possibly moved) start.
    """

    def __init__(self, space: PlanningSpace, start: Any, goal: Any, connectivity: int = 26) -> None:
        self.space = space
        self.offs = [tuple(int(v) for v in o) for o in neighbour_offsets(connectivity)]
        self.s = _cell(space.index(start))
        self.goal = _cell(space.index(goal))
        self.g: Dict[Cell, float] = {}
        self.rhs: Dict[Cell, float] = {self.goal: 0.0}
        self.k_m = 0.0
        self.last = self.s
        self.U: List[Tuple[float, float, Cell]] = []
        self._key_of: Dict[Cell, Tuple[float, float]] = {}
        self.expansions = 0
        self._insert(self.goal)

    # ---- helpers
    def _h(self, a: Cell, b: Cell) -> float:
        return self.space.res * math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _key(self, n: Cell) -> Tuple[float, float]:
        m = min(self.g.get(n, math.inf), self.rhs.get(n, math.inf))
        return (m + self._h(self.s, n) + self.k_m, m)

    def _insert(self, n: Cell) -> None:
        k = self._key(n)
        self._key_of[n] = k
        heapq.heappush(self.U, (k[0], k[1], n))

    def _remove(self, n: Cell) -> None:
        self._key_of.pop(n, None)  # lazy deletion: stale heap entries are skipped on pop

    def _in(self, n: Cell) -> bool:
        sh = self.space.shape
        return 0 <= n[0] < sh[0] and 0 <= n[1] < sh[1] and 0 <= n[2] < sh[2]

    def _cost(self, a: Cell, b: Cell) -> float:
        if not self._in(b) or not self._in(a):
            return math.inf
        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        if not _move_free(self.space.occ, a[0], a[1], a[2], d[0], d[1], d[2]):
            return math.inf
        c = self._h(a, b)
        if self.space.cost_mult is not None:
            c *= 0.5 * (self.space.cost_mult[a] + self.space.cost_mult[b])
        return c

    def _succ(self, n: Cell) -> Iterable[Cell]:
        for o in self.offs:
            m = (n[0] + o[0], n[1] + o[1], n[2] + o[2])
            if self._in(m):
                yield m

    def _update_vertex(self, u: Cell) -> None:
        if u != self.goal:
            self.rhs[u] = min((self._cost(u, s) + self.g.get(s, math.inf) for s in self._succ(u)), default=math.inf)
        self._remove(u)
        if self.g.get(u, math.inf) != self.rhs.get(u, math.inf):
            self._insert(u)

    def _compute(self, max_expansions: int = 5_000_000) -> None:
        while self.U:
            k, k2, u = self.U[0]
            if self._key_of.get(u) != (k, k2):  # stale
                heapq.heappop(self.U)
                continue
            top = (k, k2)
            if not (top < self._key(self.s) or self.rhs.get(self.s, math.inf) != self.g.get(self.s, math.inf)):
                break
            if self.expansions > max_expansions:
                break
            heapq.heappop(self.U)
            self._key_of.pop(u, None)
            self.expansions += 1
            new = self._key(u)
            if top < new:
                self._insert(u)
            elif self.g.get(u, math.inf) > self.rhs.get(u, math.inf):
                self.g[u] = self.rhs[u]
                for p in self._succ(u):
                    self._update_vertex(p)
            else:
                self.g[u] = math.inf
                for p in list(self._succ(u)) + [u]:
                    self._update_vertex(p)

    # ---- public
    def update_cells(self, cells: Iterable[Any], occupied: bool = True) -> None:
        """Mark voxel indices blocked (or free) and repair the solution."""
        changed = set()
        for c in cells:
            t = _cell(c)
            if self._in(t) and bool(self.space.occ[t]) != bool(occupied):
                self.space.occ[t] = bool(occupied)
                changed.add(t)
        affected = set()
        for c in changed:  # edges into or out of a changed cell (and the 2x2x2 corner checks around it) change
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        n = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if self._in(n):
                            affected.add(n)
        for u in affected:
            self._update_vertex(u)

    def move_start(self, new_start: Any) -> None:
        ns = _cell(self.space.index(new_start))
        self.k_m += self._h(self.last, ns)
        self.last = self.s = ns
        # keys are relative to the start: refresh the queue lazily by re-inserting what is queued
        pending = list(self._key_of)
        self.U = []
        self._key_of = {}
        for n in pending:
            self._insert(n)

    def plan(self, max_expansions: int = 5_000_000) -> PlanResult:
        t0 = time.perf_counter()
        before = self.expansions
        self._compute(max_expansions)
        if self.g.get(self.s, math.inf) == math.inf and self.rhs.get(self.s, math.inf) == math.inf:
            return PlanResult(np.empty((0, 3)), expansions=self.expansions - before, time_s=time.perf_counter() - t0, reason="no path")
        cur, chain = self.s, [self.s]
        for _ in range(int(np.prod(self.space.shape)) + 5):
            if cur == self.goal:
                break
            best, bc = None, math.inf
            for m in self._succ(cur):
                v = self._cost(cur, m) + self.g.get(m, math.inf)
                if v < bc:
                    best, bc = m, v
            if best is None or bc == math.inf:
                return PlanResult(np.empty((0, 3)), expansions=self.expansions - before, time_s=time.perf_counter() - t0, reason="no path")
            cur = best
            chain.append(cur)
        pts = self.space.center(np.array(chain))
        return PlanResult(
            pts if len(pts) > 1 else np.vstack([pts, pts]),
            cost=float(self.g.get(self.s, math.inf)),
            success=True,
            expansions=self.expansions - before,
            time_s=time.perf_counter() - t0,
            reason="ok",
        )
