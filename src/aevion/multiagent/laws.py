"""Swarm laws: cooperative behaviours that act on each vehicle's reference through the lossy inter-agent link.

A law is configured in the ``swarm`` block (``laws: [{type: formation_keeping, ...}]``). Each vehicle knows its own state exactly and
only the last *received* position of the others, which may be old (delay), missing (loss, range) or frozen (a ``comm_blackout``
fault), so the laws degrade as a real decentralised swarm does.

``formation_keeping``  couples neighbours' tracking errors: a vehicle that lags its slot pulls its neighbours' references back, so the
                       shape holds when one vehicle is disturbed (a gust, a degraded rotor, a payload).
``rendezvous``         average consensus on position: vehicles meet at a common point.
``coverage``           Lloyd's algorithm on the believed positions: each vehicle moves towards the centroid of its Voronoi cell.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coverage import CoverageDomain, voronoi_assign
from .graphs import adjacency, disk_graph, knn_graph

__all__ = ["SwarmLaw", "FormationKeeping", "Rendezvous", "CoverageLaw", "static_graph"]


def static_graph(kind: str, offsets: Optional[np.ndarray], n: int, k: int = 3, radius: float = 3.0) -> np.ndarray:
    """Neighbour structure of a law: ``complete``, ``ring``, ``line``, or ``knn`` / ``disk`` computed on the formation offsets."""
    if kind in ("complete", "ring", "line", "star"):
        return adjacency(kind, n)
    if offsets is None:
        raise ValueError(f"graph '{kind}' needs a formation (vehicle.formation) to measure distances on")
    if kind == "knn":
        return knn_graph(offsets, k)
    if kind == "disk":
        return disk_graph(offsets, radius)
    raise ValueError(f"unknown graph '{kind}'; choose complete, ring, line, star, knn or disk")


class SwarmLaw:
    """Base class: ``apply`` returns the new ``(ref_pos, ref_vel)`` arrays of shape ``(N, 3)``."""

    name = "law"

    def bind(self, offsets: Optional[np.ndarray], dt: float) -> None:
        self.offsets, self.dt = offsets, dt

    def apply(self, t: float, pos: np.ndarray, known: np.ndarray, ref_pos: np.ndarray, ref_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def _limit(v: np.ndarray, vmax: float) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, vmax / np.maximum(n, 1e-12))


class FormationKeeping(SwarmLaw):
    """``ref_i = slot_i + κ Σ_j a_ij (ê_ij − e_i)`` with tracking errors ``e = p − slot`` (``ê_ij`` is what ``i`` believes about ``j``), shift capped at ``max_shift``."""

    name = "formation_keeping"

    def __init__(
        self, n: int, gain: float = 0.4, graph: str = "knn", k: int = 3, radius: float = 3.0, max_shift: float = 1.5, start: float = 0.0
    ) -> None:
        if gain < 0 or max_shift <= 0:
            raise ValueError("gain must be non-negative and max_shift positive")
        self.n, self.gain, self.graph, self.k, self.radius, self.max_shift, self.start = n, gain, graph, k, radius, max_shift, start
        self.A: Optional[np.ndarray] = None

    def bind(self, offsets: Optional[np.ndarray], dt: float) -> None:
        super().bind(offsets, dt)
        self.A = static_graph(self.graph, offsets, self.n, self.k, self.radius)

    def apply(self, t, pos, known, ref_pos, ref_vel):
        if t < self.start or self.A is None:
            return ref_pos, ref_vel
        slots = ref_pos  # the common trajectory plus each vehicle's offset: known to everybody
        e_own = pos - slots
        e_known = known - slots[None, :, :]  # e_known[i, j] = what i believes about j's error
        diff = (e_known - e_own[:, None, :]) * self.A[:, :, None]
        shift = _limit(self.gain * diff.sum(axis=1), self.max_shift)
        return ref_pos + shift, ref_vel


class Rendezvous(SwarmLaw):
    """Average consensus on position: ``ref_i = p_i + κ Σ_j a_ij (p̂_ij − p_i)``, step capped at ``max_step``; begins at time ``start``."""

    name = "rendezvous"

    def __init__(
        self, n: int, gain: float = 0.6, graph: str = "complete", k: int = 3, radius: float = 3.0, max_step: float = 2.0, start: float = 0.0
    ) -> None:
        self.n, self.gain, self.graph, self.k, self.radius, self.max_step, self.start = n, gain, graph, k, radius, max_step, start
        self.A: Optional[np.ndarray] = None

    def bind(self, offsets: Optional[np.ndarray], dt: float) -> None:
        super().bind(offsets, dt)
        self.A = static_graph(self.graph, offsets, self.n, self.k, self.radius)

    def apply(self, t, pos, known, ref_pos, ref_vel):
        if t < self.start or self.A is None:
            return ref_pos, ref_vel
        deg = np.maximum(self.A.sum(axis=1, keepdims=True), 1.0)
        mean = ((known * self.A[:, :, None]).sum(axis=1)) / deg
        step = _limit(self.gain * (mean - pos), self.max_step)
        return pos + step, np.zeros_like(ref_vel)


class CoverageLaw(SwarmLaw):
    """Lloyd coverage of the rectangle ``bounds = (x0, y0, x1, y1)`` at the reference altitude, with optional Gaussian ``hotspots`` of extra importance.

    Vehicle ``i`` builds the Voronoi diagram from its believed positions of the others, finds the centroid of its own cell and flies towards it.
    """

    name = "coverage"

    def __init__(
        self,
        n: int,
        bounds: Sequence[float] = (-10.0, -10.0, 10.0, 10.0),
        resolution: float = 0.5,
        hotspots: Optional[List[Dict[str, Any]]] = None,
        gain: float = 0.5,
        max_step: float = 2.0,
        start: float = 0.0,
    ) -> None:
        if len(bounds) != 4:
            raise ValueError("bounds must be (x0, y0, x1, y1)")
        hs: List[Dict[str, Any]] = [
            dict(center=list(h["center"]), sigma=float(h.get("sigma", 2.0)), weight=float(h.get("weight", 5.0))) for h in (hotspots or [])
        ]

        def density(q: np.ndarray) -> np.ndarray:
            phi = np.ones(len(q))
            for h in hs:
                phi = phi + h["weight"] * np.exp(-((q - np.asarray(h["center"])) ** 2).sum(axis=1) / (2 * h["sigma"] ** 2))
            return phi

        self.n, self.gain, self.max_step, self.start = n, gain, max_step, start
        self.domain = CoverageDomain(list(bounds), resolution, density=density if hs else None)

    def apply(self, t, pos, known, ref_pos, ref_vel):
        if t < self.start:
            return ref_pos, ref_vel
        d = self.domain
        out = pos.copy()
        for i in range(len(pos)):
            sites = known[i, :, :2].copy()
            sites[i] = pos[i, :2]
            cell = voronoi_assign(sites, d.points) == i
            m = d.weights[cell].sum()
            if m <= 0:
                continue
            c = (d.points[cell] * d.weights[cell, None]).sum(axis=0) / m
            out[i, :2] = pos[i, :2] + _limit(self.gain * (c - pos[i, :2]), self.max_step)
        out[:, 2] = ref_pos[:, 2]
        return out, np.zeros_like(ref_vel)
