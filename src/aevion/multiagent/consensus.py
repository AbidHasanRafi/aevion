"""Consensus and formation algorithms on communication graphs.

**Linear average consensus** ``x⁺ = (I − εL) x`` (or ``x⁺ = W x`` with Metropolis weights) drives every agent to the mean of the
initial values when the undirected graph is connected and ``0 < ε < 2/λ_max(L)``. The error contracts by ``max_{i ≥ 2} |1 − ε λ_i|``
per step (:func:`consensus_rate`). :func:`lossy_consensus` adds delay and random packet loss with a zero-order hold, and
:func:`push_sum` (Kempe, Dobra, Gehrke 2003) reaches the average on a strongly connected directed graph, where ``I − εL`` is not
doubly stochastic.

**Displacement-based formation control** (Ren and Beard 2008) is consensus on the shifted positions ``p_i − d_i``:
``ṗ_i = −k Σ_j a_ij ((p_i − d_i) − (p_j − d_j))`` converges to ``p_i = c + d_i`` for a common ``c``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .graphs import laplacian, laplacian_spectrum, metropolis_weights

__all__ = [
    "max_step",
    "consensus_rate",
    "consensus_step",
    "average_consensus",
    "lossy_consensus",
    "push_sum",
    "formation_error",
    "formation_step",
    "disagreement",
]


def max_step(A: Any) -> float:
    """Largest stable step ``2 / λ_max(L)`` of ``x⁺ = (I − εL) x``."""
    ev = laplacian_spectrum(A)
    return float(2.0 / ev[-1]) if ev[-1] > 0 else float("inf")


def consensus_rate(A: Any, eps: Optional[float] = None) -> float:
    """Asymptotic contraction factor per step of ``x⁺ = (I − εL) x`` on the disagreement subspace (``ε`` defaults to ``1 / (d_max + 1)``)."""
    ev = laplacian_spectrum(A)
    if eps is None:
        eps = 1.0 / (np.max(np.diag(laplacian(A))) + 1.0)
    return float(np.max(np.abs(1.0 - eps * ev[1:]))) if len(ev) > 1 else 0.0


def consensus_step(x: Any, A: Any, eps: Optional[float] = None) -> np.ndarray:
    """One step of linear consensus on ``(N, ...)`` values (default step ``1 / (d_max + 1)``)."""
    x = np.asarray(x, float)
    L = laplacian(A)
    if eps is None:
        eps = 1.0 / (np.max(np.diag(L)) + 1.0)
    return x - eps * np.tensordot(L, x, axes=(1, 0))


def disagreement(x: Any) -> float:
    """Largest distance of any agent's value from the mean of all of them."""
    x = np.asarray(x, float).reshape(len(x), -1)
    return float(np.max(np.linalg.norm(x - x.mean(axis=0), axis=1)))


def average_consensus(x0: Any, A: Any, steps: int = 100, eps: Optional[float] = None, weights: str = "laplacian") -> np.ndarray:
    """Trajectory ``(steps + 1, N, ...)`` of linear consensus. ``weights="metropolis"`` uses the Metropolis matrix instead of ``I − εL``."""
    x = np.asarray(x0, float)
    out = [x.copy()]
    if weights == "metropolis":
        W = metropolis_weights(A)
        for _ in range(int(steps)):
            x = np.tensordot(W, x, axes=(1, 0))
            out.append(x.copy())
    elif weights == "laplacian":
        for _ in range(int(steps)):
            x = consensus_step(x, A, eps)
            out.append(x.copy())
    else:
        raise ValueError("weights must be 'laplacian' or 'metropolis'")
    return np.stack(out)


def lossy_consensus(x0: Any, A: Any, steps: int = 200, eps: Optional[float] = None, delay: int = 0, loss: float = 0.0, seed: int = 0) -> np.ndarray:
    """Consensus over links that are delayed by ``delay`` steps and drop each message with probability ``loss``.

    Every agent applies ``x_i += ε Σ_j a_ij (x̂_ij − x_i^d)`` where ``x̂_ij`` is the value agent ``i`` last *received* from ``j`` (a zero-order hold) and
    ``x_i^d`` is its own value from the same delayed time. With a pure delay the updates over the whole graph cancel, so the mean is preserved
    exactly; with packet loss the links become asymmetric and the mean drifts, though the agents still converge to a neighbourhood of it (the
    tests quantify this). The default step shrinks with the delay (stability needs ``ε λ_max τ`` small).
    """
    A = np.asarray(A, float)
    n = len(A)
    x = np.asarray(x0, float)
    if eps is None:
        eps = 1.0 / (np.max(np.diag(laplacian(A))) + 1.0) / (1.0 + delay)
    rng = np.random.default_rng(seed)
    extra = (1,) * (x.ndim - 1)
    link = (A > 0).reshape((n, n) + extra)
    hist = [x.copy()] * (delay + 1)  # hist[0] is the oldest
    recv = np.repeat(x[None], n, axis=0)  # recv[i, j]: what i last heard from j
    out = [x.copy()]
    for _ in range(int(steps)):
        sent = hist[0]
        ok = rng.random((n, n)) >= loss
        recv = np.where(ok.reshape((n, n) + extra), sent[None], recv)
        x = x + eps * np.where(link, recv - sent[:, None], 0.0).sum(axis=1)
        hist = hist[1:] + [x.copy()]
        out.append(x.copy())
    return np.stack(out)


def push_sum(x0: Any, A: Any, steps: int = 200) -> np.ndarray:
    """Push-sum on a (possibly directed) graph where ``A[i, j] = 1`` means ``i`` hears ``j``: returns the ratio estimates ``(steps + 1, N, ...)``.

    Each agent splits its value and its weight equally among itself and its out-neighbours; the ratio ``s_i / w_i`` converges to the
    average of the initial values if the graph is strongly connected. This is exact in exact arithmetic with no loss.
    """
    A = np.asarray(A, float)
    n = len(A)
    S = (A > 0) & ~np.eye(n, dtype=bool)
    out_deg = S.sum(axis=0)  # the number of agents each j sends to
    P = (S + np.eye(n)) / (out_deg + 1.0)[None, :]  # column stochastic
    s = np.asarray(x0, float).copy()
    w = np.ones(n)
    out = [s.copy()]
    for _ in range(int(steps)):
        s = np.tensordot(P, s, axes=(1, 0))
        w = P @ w
        out.append(s / w.reshape((n,) + (1,) * (s.ndim - 1)))
    return np.stack(out)


def formation_error(pos: Any, offsets: Any) -> float:
    """RMS deviation of the shape from the desired one, after removing the common translation: ``rms(p_i − d_i − mean(p − d))``."""
    e = np.asarray(pos, float) - np.asarray(offsets, float)
    e = e - e.mean(axis=0)
    return float(np.sqrt((e**2).sum(axis=1).mean()))


def formation_step(pos: Any, A: Any, offsets: Any, gain: float, dt: float, vel_ff: Any = 0.0) -> np.ndarray:
    """Velocity commands ``v_i = v_ff − k Σ_j a_ij ((p_i − d_i) − (p_j − d_j))`` of displacement-based formation control."""
    q = np.asarray(pos, float) - np.asarray(offsets, float)
    v = -gain * np.tensordot(laplacian(A), q, axes=(1, 0))
    return v + np.asarray(vel_ff, float)
