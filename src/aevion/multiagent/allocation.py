"""Task allocation: Hungarian, auction and CBBA.

* :func:`hungarian` solves the linear assignment problem exactly (shortest augmenting paths with potentials, ``O(n² m)``); the tests
  compare it with ``scipy.optimize.linear_sum_assignment``.
* :func:`auction` is Bertsekas' forward auction with ε-scaling. It terminates with a total benefit within ``n ε`` of the optimum
  (Bertsekas 1988), which the tests verify.
* :func:`cbba` is the Consensus-Based Bundle Algorithm (Choi, Brunet and How, *IEEE T-RO* 2009): each agent greedily builds a bundle
  from its marginal scores, then agents exchange winning bids over a communication graph and resolve conflicts with the paper's rules.
  Assignments are conflict-free when the graph is connected, and for scores with diminishing marginal gains the result is at least
  50 % of the optimum (the paper's bound); the tests check both against brute-force enumeration.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["hungarian", "auction", "AuctionResult", "cbba", "CBBAResult", "TravelScore", "assignment_cost", "brute_force_allocation"]


def assignment_cost(C: Any, rows: Sequence[int], cols: Sequence[int]) -> float:
    C = np.asarray(C, float)
    return float(C[np.asarray(rows, int), np.asarray(cols, int)].sum()) if len(rows) else 0.0


# --------------------------------------------------------------------------------------------- Hungarian
def hungarian(cost: Any, maximize: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimal one-to-one assignment for an ``(n, m)`` matrix; every row of the smaller side is assigned.

    Returns ``(rows, cols, total)`` sorted by row. Entries equal to ``±inf`` (forbidden pairs) are allowed as long as a finite complete
    assignment exists; otherwise a ``ValueError`` says so.
    """
    C = np.array(cost, float)
    if C.ndim != 2:
        raise ValueError("cost must be a matrix")
    if np.isnan(C).any():
        raise ValueError("cost contains NaN")
    if maximize:
        C = -C
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), 0.0
    big = np.abs(C[np.isfinite(C)]).max() if np.isfinite(C).any() else 1.0
    BIG = (big + 1.0) * (m + n + 1) * 10.0
    W = np.where(np.isfinite(C), C, BIG)
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, int)  # p[j]: row matched to column j (1-based); 0 = free
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = W[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows = p[1:][p[1:] > 0] - 1
    cols = np.nonzero(p[1:] > 0)[0]
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    if not np.all(np.isfinite(C[rows, cols])):
        raise ValueError("no complete assignment avoids the forbidden (infinite) pairs")
    total = float(C[rows, cols].sum())
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols, (-total if maximize else total)


# --------------------------------------------------------------------------------------------- auction
@dataclass
class AuctionResult:
    rows: np.ndarray
    cols: np.ndarray
    total: float
    prices: np.ndarray
    eps: float
    bound: float  # the guaranteed distance from the optimum: m * eps
    iterations: int


def auction(benefit: Any, eps: Optional[float] = None, scaling: float = 5.0, max_iter: int = 1_000_000) -> AuctionResult:
    """Maximum-benefit assignment of ``n`` agents to ``m ≥ n`` objects by a forward auction with ε-scaling.

    The final assignment is within ``m ε`` of the optimum (``eps`` defaults to ``1 / (m + 1)``, which makes it exact for integer benefits). When
    ``n < m`` the problem is padded with zero-benefit dummy agents, which leaves the optimum unchanged and makes the ε-guarantee hold. ``scaling`` is the factor ε shrinks by between phases; larger problems converge much faster this way.
    """
    a = np.array(benefit, float)
    if a.ndim != 2 or a.shape[0] > a.shape[1]:
        raise ValueError("benefit must be (n, m) with n <= m")
    if not np.isfinite(a).all():
        raise ValueError("benefit must be finite (use the Hungarian method for forbidden pairs)")
    n_real, m = a.shape
    if n_real == 0:
        return AuctionResult(np.zeros(0, int), np.zeros(0, int), 0.0, np.zeros(m), 0.0, 0.0, 0)
    if n_real < m:  # a forward auction is only ε-optimal on a square problem: pad with zero-benefit dummy agents
        a = np.vstack([a, np.zeros((m - n_real, m))])
    n = a.shape[0]
    spread = float(a.max() - a.min()) or 1.0
    eps_final = float(eps) if eps is not None else 1.0 / (n + 1.0)
    prices = np.zeros(m)
    owner = -np.ones(m, int)
    assigned = -np.ones(n, int)
    cur = max(spread / 2.0, eps_final)
    iters = 0
    while True:
        assigned[:] = -1
        owner[:] = -1
        unassigned = list(range(n))
        while unassigned:
            iters += 1
            if iters > max_iter:
                raise RuntimeError("auction did not terminate; increase eps")
            i = unassigned.pop()
            val = a[i] - prices
            if m == 1:
                j, bid = 0, cur
            else:
                top2 = np.argpartition(-val, 1)[:2]
                j, k = (top2[0], top2[1]) if val[top2[0]] >= val[top2[1]] else (top2[1], top2[0])
                bid = val[j] - val[k] + cur
            prices[j] += bid
            if owner[j] >= 0:
                assigned[owner[j]] = -1
                unassigned.append(int(owner[j]))
            owner[j] = i
            assigned[i] = j
        if cur <= eps_final * (1 + 1e-12):
            break
        cur = max(cur / scaling, eps_final)
    rows = np.arange(n_real)
    cols = assigned[:n_real].copy()
    return AuctionResult(rows, cols, float(a[rows, cols].sum()), prices.copy(), cur, n * cur, iters)


# --------------------------------------------------------------------------------------------- CBBA
class TravelScore:
    """Time-discounted reward score for CBBA: ``Σ_j value_j · γ^{arrival time at j}`` along an agent's ordered path.

    ``γ^t`` makes the marginal gain of a task decrease as it is pushed later by other tasks, which is the diminishing-marginal-gain property
    CBBA's guarantee needs. Optional ``capability[i, j]`` (0/1) forbids agent ``i`` from taking task ``j``.
    """

    def __init__(
        self,
        agent_pos: Any,
        task_pos: Any,
        value: Any = 1.0,
        speed: float = 1.0,
        discount: float = 0.95,
        capability: Any = None,
        service_time: float = 0.0,
    ) -> None:
        self.ap = np.asarray(agent_pos, float)
        self.tp = np.asarray(task_pos, float)
        self.value = np.broadcast_to(np.asarray(value, float), (len(self.tp),)).copy()
        self.speed, self.gamma, self.service = float(speed), float(discount), float(service_time)
        self.cap = None if capability is None else np.asarray(capability, float) > 0

    def __call__(self, agent: int, path: Sequence[int]) -> float:
        pos, t, total = self.ap[agent], 0.0, 0.0
        for j in path:
            if self.cap is not None and not self.cap[agent, j]:
                return -np.inf
            t += float(np.linalg.norm(self.tp[j] - pos)) / self.speed
            total += self.value[j] * self.gamma**t
            t += self.service
            pos = self.tp[j]
        return total


@dataclass
class CBBAResult:
    paths: List[List[int]]
    score: float
    rounds: int
    converged: bool
    conflict_free: bool
    unassigned: List[int] = field(default_factory=list)

    def assignment(self) -> Dict[int, int]:
        """task -> agent"""
        return {j: i for i, p in enumerate(self.paths) for j in p}


def _best_insertion(score: Callable[[int, Sequence[int]], float], agent: int, path: List[int], j: int) -> Tuple[float, int]:
    base = score(agent, path)
    best, pos = -np.inf, 0
    for n in range(len(path) + 1):
        c = score(agent, path[:n] + [j] + path[n:]) - base
        if c > best:
            best, pos = c, n
    return best, pos


def cbba(
    score: Callable[[int, Sequence[int]], float],
    n_agents: int,
    n_tasks: int,
    adjacency: Any = None,
    max_bundle: Optional[int] = None,
    max_rounds: int = 500,
) -> CBBAResult:
    """Run CBBA with ``score(agent, path)`` (path = ordered task indices) over the communication graph ``adjacency`` (default: complete).

    ``adjacency[i, j] > 0`` means agent ``i`` receives agent ``j``'s bids each round. Stops when no agent changes its winner table for a
    full network diameter of rounds (here: ``n_agents`` consecutive quiet rounds, which is an upper bound on the diameter).
    """
    N, T = int(n_agents), int(n_tasks)
    A = np.ones((N, N)) if adjacency is None else np.asarray(adjacency, float)
    A = (A > 0) & ~np.eye(N, dtype=bool)
    Lt = T if max_bundle is None else int(max_bundle)
    y = np.zeros((N, T))
    z = -np.ones((N, T), int)
    s = np.zeros((N, N), int)
    bundle: List[List[int]] = [[] for _ in range(N)]
    path: List[List[int]] = [[] for _ in range(N)]

    def build(i: int) -> bool:
        changed = False
        while len(bundle[i]) < Lt:
            best_j, best_c, best_pos = -1, 0.0, 0
            for j in range(T):
                if j in bundle[i]:
                    continue
                c, pos = _best_insertion(score, i, path[i], j)
                if not np.isfinite(c) or c <= 1e-12:
                    continue
                if c > y[i, j] + 1e-12 or (abs(c - y[i, j]) <= 1e-12 and z[i, j] > i):  # outbid the current winner (ties go to the lower index)
                    if c > best_c + 1e-12:
                        best_j, best_c, best_pos = j, c, pos
            if best_j < 0:
                break
            bundle[i].append(best_j)
            path[i].insert(best_pos, best_j)
            y[i, best_j] = best_c
            z[i, best_j] = i
            changed = True
        return changed

    def release_from(i: int, j: int) -> None:
        k = bundle[i].index(j)
        for m in bundle[i][k:]:
            if z[i, m] == i:
                z[i, m] = -1
                y[i, m] = 0.0
        gone = set(bundle[i][k:])
        bundle[i] = bundle[i][:k]
        path[i] = [m for m in path[i] if m not in gone]

    def resolve(i: int, k: int, t: int, snap_y: np.ndarray, snap_z: np.ndarray, snap_s: np.ndarray) -> None:
        for j in range(T):
            zk, zi = snap_z[k, j], z[i, j]
            yk = snap_y[k, j]
            act = None  # "update" | "reset" | None
            if zk == k:
                if zi == i:
                    act = "update" if (yk > y[i, j] + 1e-12 or (abs(yk - y[i, j]) <= 1e-12 and k < i)) else None
                elif zi == k or zi == -1:
                    act = "update"
                else:
                    m = zi
                    act = "update" if (snap_s[k, m] > s[i, m] or yk > y[i, j] + 1e-12) else None
            elif zk == i:
                if zi == i:
                    act = None
                elif zi == k:
                    act = "reset"
                elif zi == -1:
                    act = None
                else:
                    m = zi
                    act = "reset" if snap_s[k, m] > s[i, m] else None
            elif zk == -1:
                if zi == i or zi == -1:
                    act = None
                elif zi == k:
                    act = "update"
                else:
                    m = zi
                    act = "update" if snap_s[k, m] > s[i, m] else None
            else:  # k believes a third agent m wins
                m = zk
                if zi == i:
                    act = "update" if (snap_s[k, m] > s[i, m] and yk > y[i, j] + 1e-12) else None
                elif zi == k:
                    act = "update" if snap_s[k, m] > s[i, m] else "reset"
                elif zi == m or zi == -1:
                    act = "update" if snap_s[k, m] > s[i, m] else None
                else:
                    n = zi
                    if (snap_s[k, m] > s[i, m] and snap_s[k, n] > s[i, n]) or (snap_s[k, m] > s[i, m] and yk > y[i, j] + 1e-12):
                        act = "update"
                    elif snap_s[k, n] > s[i, n] and s[i, m] > snap_s[k, m]:
                        act = "reset"
            if act == "update":
                prev = z[i, j]
                y[i, j], z[i, j] = yk, zk
                if prev == i and j in bundle[i]:
                    release_from(i, j)
            elif act == "reset":
                prev = z[i, j]
                y[i, j], z[i, j] = 0.0, -1
                if prev == i and j in bundle[i]:
                    release_from(i, j)
        # timestamps: direct contact refreshes k's entry; everything else is the freshest of the two views
        s[i, k] = t
        for m in range(N):
            if m != i and m != k:
                s[i, m] = max(s[i, m], snap_s[k, m])

    quiet, rounds = 0, 0
    for i in range(N):
        build(i)
    while quiet < N + 1 and rounds < max_rounds:
        rounds += 1
        snap_y, snap_z, snap_s = y.copy(), z.copy(), s.copy()
        before = z.copy()
        for i in range(N):
            for k in np.nonzero(A[i])[0]:
                resolve(i, int(k), rounds, snap_y, snap_z, snap_s)
        for i in range(N):
            build(i)
        quiet = quiet + 1 if np.array_equal(before, z) else 0
    converged = quiet >= N + 1
    claimed: Dict[int, int] = {}
    ok = True
    for i in range(N):
        for j in path[i]:
            if j in claimed:
                ok = False
            claimed[j] = i
    total = float(sum(score(i, path[i]) for i in range(N) if path[i]))
    return CBBAResult([list(p) for p in path], total, rounds, converged, ok, [j for j in range(T) if j not in claimed])


def brute_force_allocation(score: Callable[[int, Sequence[int]], float], n_agents: int, n_tasks: int) -> Tuple[float, List[List[int]]]:
    """Exhaustive optimum of the same objective (every assignment of tasks to agents, every order): for tests on small instances only."""
    best = -np.inf
    best_paths: List[List[int]] = [[] for _ in range(n_agents)]
    for owner in itertools.product(range(-1, n_agents), repeat=n_tasks):
        groups = [[j for j in range(n_tasks) if owner[j] == i] for i in range(n_agents)]
        total, valid = 0.0, True
        paths: List[List[int]] = []
        for i, g in enumerate(groups):
            if not g:
                paths.append([])
                continue
            bs, bp = -np.inf, None
            for perm in itertools.permutations(g):
                sc = score(i, list(perm))
                if sc > bs:
                    bs, bp = sc, list(perm)
            if not np.isfinite(bs):
                valid = False
                break
            total += bs
            paths.append(bp or [])
        if valid and total > best + 1e-12:
            best, best_paths = total, paths
    return best, best_paths
