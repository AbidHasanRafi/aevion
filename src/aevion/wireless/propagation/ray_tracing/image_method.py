"""Specular ray tracing by the image method: the direct path plus first-order reflections from the ground and from building faces.

For each reflecting plane the transmitter is mirrored; the reflected path is the straight line from the image to the receiver, valid
when it crosses the plane inside the face and both legs are free of other obstacles. Each path carries the free-space field
``(lambda / 4 pi d) exp(-j k d)`` over its unfolded length ``d`` times the Fresnel reflection coefficient. The coherent sum gives the
multipath channel gain, the power-delay profile gives the delay spread, and the LoS-to-rest power ratio gives a geometric Rician
K-factor.

Limitations: first order only, flat faces of axis-aligned boxes, no diffuse scattering and no diffraction (add
:func:`~aevion.wireless.propagation.knife_edge.world_diffraction_loss_db` for shadowed links); vertical cylinders and spheres block
rays but do not reflect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ...units import C_LIGHT
from .materials import complex_permittivity, fresnel_reflection

__all__ = ["PathSet", "trace_paths"]


@dataclass
class PathSet:
    """Propagation paths of ``N`` links; ``P`` columns: 0 = direct, 1 = ground, then box faces (6 per box in the order -x, +x, -y, +y, -z(skipped), +z)."""

    length_m: np.ndarray  # (N, P) unfolded path length
    field: np.ndarray  # (N, P) complex field amplitude relative to a unit-power isotropic transmitter, 0 where invalid
    valid: np.ndarray  # (N, P) bool
    labels: List[str]
    freq_hz: float

    @property
    def delay_s(self) -> np.ndarray:
        return self.length_m / C_LIGHT

    def power_gain_db(self, coherent: bool = True) -> np.ndarray:
        """Channel power gain (path gain, <= 0 dB): ``|sum_p a_p|^2`` (coherent) or ``sum_p |a_p|^2`` (incoherent)."""
        p = np.abs(self.field.sum(axis=1)) ** 2 if coherent else (np.abs(self.field) ** 2).sum(axis=1)
        return 10.0 * np.log10(np.maximum(p, 1e-30))

    def rms_delay_spread_s(self) -> np.ndarray:
        w = np.abs(self.field) ** 2
        tot = np.maximum(w.sum(axis=1), 1e-30)
        tau = self.delay_s
        m1 = (w * tau).sum(axis=1) / tot
        m2 = (w * tau**2).sum(axis=1) / tot
        return np.sqrt(np.maximum(m2 - m1**2, 0.0))

    def k_factor_db(self) -> np.ndarray:
        """Rician K from geometry: direct-path power over the sum of reflected power (``+inf`` if nothing is reflected, ``-inf`` if the direct path is blocked)."""
        los = np.abs(self.field[:, 0]) ** 2
        rest = (np.abs(self.field[:, 1:]) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_db = 10.0 * np.log10(np.where(rest > 0, los / np.where(rest > 0, rest, 1.0), 1.0))
        return np.where(los <= 0, -np.inf, np.where(rest <= 0, np.inf, ratio_db))

    def n_paths(self) -> np.ndarray:
        return self.valid.sum(axis=1)


def _los(world: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if world is None or not getattr(world, "has_obstacles", False) or len(a) == 0:
        return np.ones(len(a), bool)
    return np.asarray(world.line_of_sight(a, b), bool)


def trace_paths(
    world: Any,
    tx: Any,
    rx: Any,
    freq_hz: float,
    ground: bool = True,
    ground_material: str = "medium_dry_ground",
    wall_material: str = "concrete",
    polarization: str = "avg",
    include_roofs: bool = True,
) -> PathSet:
    """Direct and first-order reflected paths between ``tx`` and ``rx`` (``(N, 3)`` or broadcastable)."""
    tx, rx = np.broadcast_arrays(np.atleast_2d(np.asarray(tx, float)), np.atleast_2d(np.asarray(rx, float)))
    n = len(tx)
    lam = C_LIGHT / freq_hz
    k = 2.0 * math.pi / lam
    boxes = np.asarray(world.boxes, float).reshape(-1, 6) if world is not None else np.zeros((0, 6))

    labels = ["direct"] + (["ground"] if ground else [])
    for b in range(len(boxes)):
        labels += [f"box{b}:-x", f"box{b}:+x", f"box{b}:-y", f"box{b}:+y"] + ([f"box{b}:+z"] if include_roofs else [])
    n_p = len(labels)
    length = np.zeros((n, n_p))
    field = np.zeros((n, n_p), complex)
    valid = np.zeros((n, n_p), bool)

    d0 = np.linalg.norm(rx - tx, axis=1)
    free = _los(world, tx, rx)
    length[:, 0] = d0
    field[:, 0] = np.where(free, lam / (4.0 * math.pi * np.maximum(d0, 1e-3)) * np.exp(-1j * k * d0), 0.0)
    valid[:, 0] = free

    col = 1
    if ground:
        eps = complex_permittivity(ground_material, freq_hz)
        img = tx * np.array([1.0, 1.0, -1.0])
        d = np.linalg.norm(rx - img, axis=1)
        ok = (tx[:, 2] > 0) & (rx[:, 2] > 0)
        t = np.where(ok, -img[:, 2] / np.where(np.abs(rx[:, 2] - img[:, 2]) < 1e-12, 1.0, rx[:, 2] - img[:, 2]), 0.0)
        q = img + t[:, None] * (rx - img)
        q[:, 2] = 0.0
        cand = ok & (t > 0) & (t < 1)
        qq = q + np.array([0.0, 0.0, 1e-3])
        cand &= _los(world, tx, qq) & _los(world, qq, rx) if cand.any() else cand
        theta = np.arccos(np.clip((rx[:, 2] - img[:, 2]) / np.maximum(d, 1e-9), 0.0, 1.0))  # angle from the vertical normal
        gam = fresnel_reflection(theta, eps, polarization)
        length[:, col] = d
        field[:, col] = np.where(cand, lam / (4.0 * math.pi * np.maximum(d, 1e-3)) * gam * np.exp(-1j * k * d), 0.0)
        valid[:, col] = cand
        col += 1

    eps_w = complex_permittivity(wall_material, freq_hz)
    for _b, (lo, hi) in enumerate(zip(boxes[:, :3], boxes[:, 3:])):
        faces = [(0, lo[0], -1.0), (0, hi[0], 1.0), (1, lo[1], -1.0), (1, hi[1], 1.0)] + ([(2, hi[2], 1.0)] if include_roofs else [])
        for ax, pos, sgn in faces:
            nrm = np.zeros(3)
            nrm[ax] = sgn
            side_tx = (tx[:, ax] - pos) * sgn
            side_rx = (rx[:, ax] - pos) * sgn
            img = tx.copy()
            img[:, ax] = 2.0 * pos - tx[:, ax]
            denom = rx[:, ax] - img[:, ax]
            t = np.where(np.abs(denom) > 1e-12, (pos - img[:, ax]) / np.where(np.abs(denom) > 1e-12, denom, 1.0), -1.0)
            q = img + t[:, None] * (rx - img)
            other = [a for a in range(3) if a != ax]
            inside = np.ones(n, bool)
            for a in other:
                inside &= (q[:, a] >= lo[a] - 1e-9) & (q[:, a] <= hi[a] + 1e-9)
            cand = (side_tx > 1e-9) & (side_rx > 1e-9) & (t > 0) & (t < 1) & inside
            if cand.any():
                qq = q + 1e-3 * nrm
                cand &= _los(world, tx, qq) & _los(world, qq, rx)
            d = np.linalg.norm(rx - img, axis=1)
            cos_i = np.clip(np.abs(rx[:, ax] - img[:, ax]) / np.maximum(d, 1e-9), 0.0, 1.0)
            gam = fresnel_reflection(np.arccos(cos_i), eps_w, polarization)
            length[:, col] = d
            field[:, col] = np.where(cand, lam / (4.0 * math.pi * np.maximum(d, 1e-3)) * gam * np.exp(-1j * k * d), 0.0)
            valid[:, col] = cand
            col += 1
    return PathSet(length, field, valid, labels, freq_hz)
