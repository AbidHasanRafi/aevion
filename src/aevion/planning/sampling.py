"""Sampling-based planners: RRT-star (with informed sampling) and a reusable PRM roadmap.

* ``RRT*`` (Karaman & Frazzoli 2011) is asymptotically optimal: the path cost converges to the optimum as samples accumulate. It uses the
  shrinking connection radius ``r_n = min(γ (log n / n)^{1/3}, step)``, rewires the neighbourhood of every new node and, once a path
  exists, samples from the informed prolate spheroid (Gammell et al. 2014), which contains every shorter path.
* **PRM** (Kavraki et al. 1996) builds a roadmap once (free samples joined to their ``k`` nearest neighbours by free segments) and
  answers many start/goal queries with Dijkstra; edges can be re-validated after the map changes.

Both use the voxel space's exact segment test, so a returned path is collision-free in the inflated space.
"""

from __future__ import annotations

import math
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from .graph import PlanResult, path_length
from .space import PlanningSpace, segment_free_batch

__all__ = ["rrt_star", "PRM"]


def _informed_sample(rng: np.random.Generator, start: np.ndarray, goal: np.ndarray, c_best: float) -> np.ndarray:
    """A uniform sample of the prolate spheroid with foci ``start`` / ``goal`` and transverse diameter ``c_best``."""
    c_min = float(np.linalg.norm(goal - start))
    centre = 0.5 * (start + goal)
    a = c_best / 2.0
    b = math.sqrt(max(c_best**2 - c_min**2, 0.0)) / 2.0
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    ball = d * rng.random() ** (1.0 / 3.0)
    e1 = (goal - start) / max(c_min, 1e-12)
    # any orthonormal completion: the spheroid is rotationally symmetric about e1
    t = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = np.cross(e1, t)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return centre + (a * ball[0]) * e1 + (b * ball[1]) * e2 + (b * ball[2]) * e3


def rrt_star(
    space: PlanningSpace,
    start: Any,
    goal: Any,
    max_iter: int = 4000,
    step: Optional[float] = None,
    goal_bias: float = 0.05,
    gamma: Optional[float] = None,
    informed: bool = True,
    seed: int = 0,
    time_limit: Optional[float] = None,
    stop_at_first: bool = False,
) -> PlanResult:
    """Asymptotically optimal path from ``start`` to ``goal`` in ``space`` (a :class:`PlanningSpace`).

    ``step`` is the maximum edge length (default ``8 x`` the voxel size); ``gamma`` scales the connection radius (default from the
    free volume as in the paper). ``stop_at_first`` returns the first path found instead of continuing to improve it.
    """
    t0 = time.perf_counter()
    s, e = np.asarray(start, float), np.asarray(goal, float)
    if not (space.is_free(s)[0] and space.is_free(e)[0]):
        return PlanResult(np.empty((0, 3)), reason="the start or goal is not in free space")
    step = float(step if step is not None else 8.0 * space.res)
    if step <= 0:
        raise ValueError("step must be positive")
    rng = np.random.default_rng(seed)
    cap = max_iter + 2
    X = np.empty((cap, 3))
    parent = np.full(cap, -1, np.int64)
    cost = np.full(cap, np.inf)
    children: List[List[int]] = [[] for _ in range(cap)]
    X[0], cost[0], parent[0] = s, 0.0, 0
    n = 1
    free_vol = float((~space.occ).sum()) * space.res**3
    zeta3 = 4.0 / 3.0 * math.pi
    gamma = float(gamma) if gamma is not None else 2.0 * (1.0 + 1.0 / 3.0) ** (1.0 / 3.0) * (free_vol / zeta3) ** (1.0 / 3.0)
    best_cost, best_node = math.inf, -1
    first_at = None
    direct = float(np.linalg.norm(e - s))

    def try_goal(i: int) -> None:
        nonlocal best_cost, best_node
        d = float(np.linalg.norm(X[i] - e))
        if cost[i] + d < best_cost - 1e-12 and d <= step * 1.5 and space.segment_free(X[i], e):
            best_cost, best_node = cost[i] + d, i

    if direct <= step * 1.5 and space.segment_free(s, e):
        best_cost, best_node = direct, 0
        if stop_at_first:
            return PlanResult(np.vstack([s, e]), cost=direct, success=True, expansions=0, time_s=time.perf_counter() - t0, reason="ok")
    for it in range(max_iter):
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            break
        if rng.random() < goal_bias:
            q = e
        elif informed and np.isfinite(best_cost) and best_cost > direct * (1.0 + 1e-9):
            q = _informed_sample(rng, s, e, best_cost)
            if not space.inside(q)[0]:
                continue
        else:
            q = space.sample_free(1, rng)[0]
        d_all = np.linalg.norm(X[:n] - q, axis=1)
        j = int(np.argmin(d_all))
        dist = float(d_all[j])
        if dist < 1e-9:
            continue
        new = X[j] + (q - X[j]) * min(1.0, step / dist)
        if not space.is_free(new)[0] or not space.segment_free(X[j], new):
            continue
        r = min(gamma * (math.log(n + 1) / (n + 1)) ** (1.0 / 3.0), step * 2.0)
        r = max(r, space.res)
        d_new = np.linalg.norm(X[:n] - new, axis=1)
        near = np.nonzero(d_new <= r)[0]
        if j not in near:
            near = np.append(near, j)
        # choose the cheapest collision-free parent among the neighbours (cost-ordered, stop at the first valid one)
        cand_cost = cost[near] + d_new[near]
        order = np.argsort(cand_cost)
        ok = segment_free_batch(space, X[near[order]], np.tile(new, (len(order), 1)))
        if not ok.any():
            continue
        k = int(np.argmax(ok))
        par = int(near[order[k]])
        i = n
        X[i], parent[i], cost[i] = new, par, cand_cost[order[k]]
        children[par].append(i)
        n += 1
        # rewire: reroute neighbours through the new node when that is cheaper
        better = near[(cost[i] + d_new[near] < cost[near] - 1e-12) & (near != par)]
        if len(better):
            free = segment_free_batch(space, np.tile(new, (len(better), 1)), X[better])
            for jj, okj in zip(better, free):
                if not okj:
                    continue
                newc = cost[i] + d_new[jj]
                if newc >= cost[jj] - 1e-12:
                    continue
                children[parent[jj]].remove(int(jj))
                parent[jj] = i
                children[i].append(int(jj))
                delta = cost[jj] - newc
                stack = [int(jj)]
                while stack:  # the whole subtree gets cheaper by the same amount
                    u = stack.pop()
                    cost[u] -= delta
                    stack.extend(children[u])
        try_goal(i)
        if stop_at_first and best_node >= 0:
            first_at = it
            break
    # costs of nodes may have improved after the goal link was recorded: refresh it
    if n > 1:
        near_goal = np.nonzero(np.linalg.norm(X[:n] - e, axis=1) <= step * 1.5)[0]
        for gi in near_goal:
            try_goal(int(gi))
    dt = time.perf_counter() - t0
    if best_node < 0:
        return PlanResult(np.empty((0, 3)), expansions=n, time_s=dt, reason="no path found within the iteration budget", info=dict(nodes=n))
    chain = [best_node]
    while chain[-1] != 0:
        chain.append(int(parent[chain[-1]]))
    pts = np.vstack([X[c] for c in reversed(chain)] + [e])
    return PlanResult(
        pts,
        cost=path_length(pts),
        success=True,
        expansions=n,
        time_s=dt,
        reason="ok",
        info=dict(nodes=n, first_solution_iteration=first_at, gamma=gamma),
    )


class PRM:
    """A probabilistic roadmap over a :class:`PlanningSpace`.

    >>> prm = PRM(space, seed=0).build(n_samples=800, k=12)     # doctest: +SKIP
    >>> prm.query(start, goal).path                              # doctest: +SKIP
    """

    def __init__(self, space: PlanningSpace, seed: int = 0) -> None:
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.nodes = np.empty((0, 3))
        self.edges = np.empty((0, 2), np.int64)
        self.k = 12

    def build(self, n_samples: int = 800, k: int = 12) -> "PRM":
        self.k = int(k)
        self.nodes = self.space.sample_free(int(n_samples), self.rng)
        self._connect(np.arange(len(self.nodes)))
        return self

    def add_samples(self, m: int) -> "PRM":
        """Densify the roadmap with ``m`` more free samples."""
        new = self.space.sample_free(int(m), self.rng)
        first = len(self.nodes)
        self.nodes = np.vstack([self.nodes, new])
        self._connect(np.arange(first, len(self.nodes)))
        return self

    def _connect(self, which: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        tree = cKDTree(self.nodes)
        kk = min(self.k + 1, len(self.nodes))
        _, nb = tree.query(self.nodes[which], k=kk)
        pairs = np.array([(int(a), int(b)) for a, row in zip(which, np.atleast_2d(nb)) for b in row[1:] if a != b], np.int64).reshape(-1, 2)
        if not len(pairs):
            return
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        ok = segment_free_batch(self.space, self.nodes[pairs[:, 0]], self.nodes[pairs[:, 1]])
        self.edges = np.unique(np.vstack([self.edges, pairs[ok]]), axis=0) if len(self.edges) else pairs[ok]

    def revalidate(self) -> int:
        """Drop edges that are no longer free (after the occupancy changed); returns how many were removed."""
        if not len(self.edges):
            return 0
        ok = segment_free_batch(self.space, self.nodes[self.edges[:, 0]], self.nodes[self.edges[:, 1]])
        removed = int((~ok).sum())
        self.edges = self.edges[ok]
        return removed

    def _graph(self, extra: np.ndarray, extra_edges: List[Tuple[int, int]]) -> Any:
        from scipy.sparse import coo_matrix

        pts = np.vstack([self.nodes, extra]) if len(extra) else self.nodes
        e = np.vstack([self.edges, np.array(extra_edges, np.int64).reshape(-1, 2)]) if extra_edges else self.edges
        w = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1) if len(e) else np.empty(0)
        n = len(pts)
        return coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr(), pts

    def query(self, start: Any, goal: Any, k_connect: int = 10) -> PlanResult:
        from scipy.sparse.csgraph import dijkstra
        from scipy.spatial import cKDTree

        t0 = time.perf_counter()
        s, e = np.asarray(start, float), np.asarray(goal, float)
        if not len(self.nodes):
            return PlanResult(np.empty((0, 3)), reason="the roadmap is empty; call build() first")
        if not (self.space.is_free(s)[0] and self.space.is_free(e)[0]):
            return PlanResult(np.empty((0, 3)), reason="the start or goal is not in free space")
        tree = cKDTree(self.nodes)
        extra_edges: List[Tuple[int, int]] = []
        si, gi = len(self.nodes), len(self.nodes) + 1
        for idx, p in ((si, s), (gi, e)):
            _, nb = tree.query(p, k=min(k_connect, len(self.nodes)))
            nb = np.atleast_1d(nb)
            ok = segment_free_batch(self.space, np.tile(p, (len(nb), 1)), self.nodes[nb])
            extra_edges += [(idx, int(j)) for j, o in zip(nb, ok) if o]
        if self.space.segment_free(s, e):
            extra_edges.append((si, gi))
        G, pts = self._graph(np.vstack([s, e]), extra_edges)
        dist, pred = dijkstra(G, indices=si, return_predecessors=True)
        dt = time.perf_counter() - t0
        if not np.isfinite(dist[gi]):
            return PlanResult(np.empty((0, 3)), time_s=dt, reason="no path in the roadmap (add samples or raise k)")
        chain = [gi]
        while chain[-1] != si:
            chain.append(int(pred[chain[-1]]))
        path = pts[list(reversed(chain))]
        return PlanResult(
            path,
            cost=float(dist[gi]),
            success=True,
            expansions=len(self.nodes),
            time_s=dt,
            reason="ok",
            info=dict(nodes=len(self.nodes), edges=len(self.edges)),
        )
