"""Communication graphs: generators, Laplacian spectra and the weight matrices consensus algorithms use.

Everything here works on dense ``(N, N)`` adjacency matrices (``A[i, j] = 1`` when agent ``i`` hears agent ``j``). Undirected graphs have
symmetric matrices; :func:`adjacency` can also build directed ones (``"ring_directed"``). The algebraic connectivity ``λ₂`` of the Laplacian
``L = D − A`` governs how fast linear consensus converges (``|1 − εΛ₂|`` per step), so the graph tools and the algorithms are checked against
each other in the tests.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

__all__ = [
    "adjacency",
    "degrees",
    "laplacian",
    "algebraic_connectivity",
    "laplacian_spectrum",
    "is_connected",
    "components",
    "diameter",
    "metropolis_weights",
    "max_degree_weights",
    "disk_graph",
    "knn_graph",
    "neighbours",
]


def _check(A: Any, square: bool = True) -> np.ndarray:
    A = np.asarray(A, float)
    if A.ndim != 2 or (square and A.shape[0] != A.shape[1]):
        raise ValueError("an adjacency matrix must be square")
    if not np.all(np.isfinite(A)) or np.any(A < 0):
        raise ValueError("adjacency weights must be finite and non-negative")
    return A


def degrees(A: Any) -> np.ndarray:
    """In-degree of every node (row sums): how many neighbours each agent listens to."""
    A = _check(A)
    return (A - np.diag(np.diag(A))).sum(axis=1)


def laplacian(A: Any) -> np.ndarray:
    """``L = D − A`` with self-loops ignored; rows sum to zero."""
    A = _check(A)
    A = A - np.diag(np.diag(A))
    return np.diag(A.sum(axis=1)) - A


def laplacian_spectrum(A: Any) -> np.ndarray:
    """Eigenvalues of the Laplacian, sorted ascending (real for symmetric ``A``; real parts otherwise)."""
    L = laplacian(A)
    if np.allclose(L, L.T):
        return np.linalg.eigvalsh(L)
    return np.sort(np.real(np.linalg.eigvals(L)))


def algebraic_connectivity(A: Any) -> float:
    """Second-smallest Laplacian eigenvalue ``λ₂`` (Fiedler value). Positive exactly when an undirected graph is connected."""
    ev = laplacian_spectrum(A)
    return float(ev[1]) if len(ev) > 1 else 0.0


def components(A: Any) -> List[List[int]]:
    """Connected components (weakly connected for directed graphs), each a sorted list of node indices."""
    A = _check(A)
    n = A.shape[0]
    S = (A + A.T) > 0
    seen = np.zeros(n, bool)
    out: List[List[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(S[u])[0]:
                if not seen[v] and v != u:
                    seen[v] = True
                    stack.append(int(v))
        out.append(sorted(comp))
    return out


def is_connected(A: Any) -> bool:
    return len(components(A)) == 1


def diameter(A: Any) -> float:
    """Longest shortest path (in hops) of an undirected graph; ``inf`` when disconnected."""
    A = _check(A)
    n = A.shape[0]
    S = ((A + A.T) > 0) & ~np.eye(n, dtype=bool)
    worst = 0
    for s in range(n):
        dist = np.full(n, -1)
        dist[s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.nonzero(S[u])[0]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        nxt.append(int(v))
            frontier = nxt
        if (dist < 0).any():
            return float("inf")
        worst = max(worst, int(dist.max()))
    return float(worst)


def metropolis_weights(A: Any) -> np.ndarray:
    """Metropolis–Hastings weights ``w_ij = 1 / (1 + max(d_i, d_j))``: symmetric, doubly stochastic, needs only local degrees.

    Averaging with this matrix preserves the mean exactly and converges to it on every connected undirected graph.
    """
    A = _check(A)
    S = ((A + A.T) > 0) & ~np.eye(A.shape[0], dtype=bool)
    d = S.sum(axis=1)
    W = np.where(S, 1.0 / (1.0 + np.maximum(d[:, None], d[None, :])), 0.0)
    W[np.diag_indices_from(W)] = 1.0 - W.sum(axis=1)
    return W


def max_degree_weights(A: Any) -> np.ndarray:
    """``W = I − L / (d_max + 1)``: doubly stochastic for undirected graphs."""
    A = _check(A)
    S = ((A + A.T) > 0).astype(float)
    np.fill_diagonal(S, 0.0)
    return np.eye(A.shape[0]) - laplacian(S) / (S.sum(axis=1).max() + 1.0)


def disk_graph(pos: Any, radius: float) -> np.ndarray:
    """Agents hear every other agent within ``radius`` (the unit-disk model of a radio range)."""
    p = np.asarray(pos, float)
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    A = (d <= radius).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def knn_graph(pos: Any, k: int, symmetric: bool = True) -> np.ndarray:
    """Each agent hears its ``k`` nearest neighbours (symmetrised by default, so the graph is undirected)."""
    p = np.asarray(pos, float)
    n = len(p)
    k = int(min(max(k, 0), n - 1))
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    A = np.zeros((n, n))
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    A[np.arange(n)[:, None], idx] = 1.0
    return np.maximum(A, A.T) if symmetric else A


def adjacency(kind: str, n: int, *, p: float = 0.3, seed: int = 0, rows: Optional[int] = None) -> np.ndarray:
    """Standard topologies on ``n`` nodes: ``line``, ``ring``, ``ring_directed``, ``complete``, ``star``, ``grid`` and ``random`` (Erdős–Rényi ``G(n, p)``)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    A = np.zeros((n, n))
    i = np.arange(n)
    if kind == "line":
        A[i[:-1], i[1:]] = A[i[1:], i[:-1]] = 1.0
    elif kind == "ring":
        if n > 2:
            A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
        elif n == 2:
            A[0, 1] = A[1, 0] = 1.0
    elif kind == "ring_directed":
        A[i, (i + 1) % n] = 1.0
        np.fill_diagonal(A, 0.0)
    elif kind == "complete":
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)
    elif kind == "star":
        A[0, 1:] = A[1:, 0] = 1.0
    elif kind == "grid":
        r = rows or int(np.ceil(np.sqrt(n)))
        for a in range(n):
            ra, ca = divmod(a, r)
            for b in (a + 1, a + r):
                if b < n and (b != a + 1 or divmod(b, r)[0] == ra):
                    A[a, b] = A[b, a] = 1.0
    elif kind == "random":
        rng = np.random.default_rng(seed)
        U = np.triu(rng.random((n, n)) < p, 1).astype(float)
        A = U + U.T
    else:
        raise ValueError(f"unknown graph {kind!r}; choose from line, ring, ring_directed, complete, star, grid, random")
    return A


def neighbours(A: Any, i: int) -> np.ndarray:
    """Indices agent ``i`` listens to."""
    A = _check(A)
    return np.nonzero((A[i] > 0) & (np.arange(A.shape[0]) != i))[0]
