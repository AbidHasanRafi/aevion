"""Diffraction over obstacles: ITU-R P.526 knife-edge loss and the Bullington equivalent-edge method on a 3-D world.

The single knife-edge approximation of the Fresnel-Kirchhoff integral is accurate for the *first* Fresnel-zone
obstruction (``nu > -0.78``); the multi-edge Bullington construction replaces every obstacle on the path by one equivalent
edge and tends to *under*-estimate loss for many closely spaced obstacles (it is the simple, vectorisable ITU option; it
is not a substitute for a full uniform-theory-of-diffraction solver).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..units import C_LIGHT

__all__ = ["bullington_loss_db", "fresnel_parameter", "knife_edge_loss_db", "path_profile", "world_diffraction_loss_db"]


def fresnel_parameter(h: Any, d1: Any, d2: Any, freq_hz: float) -> np.ndarray:
    """``nu = h sqrt(2 (d1 + d2) / (lambda d1 d2))``; ``h`` is the edge height above the direct ray (positive = obstructing)."""
    lam = C_LIGHT / freq_hz
    d1, d2 = np.maximum(np.asarray(d1, float), 1e-6), np.maximum(np.asarray(d2, float), 1e-6)
    return np.asarray(h, float) * np.sqrt(2.0 * (d1 + d2) / (lam * d1 * d2))


def knife_edge_loss_db(nu: Any) -> np.ndarray:
    """ITU-R P.526 single knife-edge diffraction loss ``J(nu)`` [dB] (0 for ``nu <= -0.78``)."""
    nu = np.asarray(nu, float)
    j = 6.9 + 20.0 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1.0) + nu - 0.1)
    return np.where(nu > -0.78, j, 0.0)


def bullington_loss_db(h_tx: Any, h_rx: Any, dist: Any, edge_s: Any, edge_z: Any, edge_mask: Any, freq_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bullington equivalent-edge loss for ``N`` links with up to ``M`` candidate edges each.

    ``edge_s``/``edge_z`` are ``(N, M)`` distances from the transmitter and absolute edge-top heights, ``edge_mask`` marks
    valid edges, ``dist`` the horizontal link length. Returns ``(loss_db, nu)``; links without edges get 0 dB.
    """
    h_tx, h_rx, D = (np.asarray(a, float) for a in (h_tx, h_rx, dist))
    s, z, m = np.asarray(edge_s, float), np.asarray(edge_z, float), np.asarray(edge_mask, bool)
    if s.shape[-1] == 0:  # no candidate edges at all
        zeros = np.zeros(len(D))
        return zeros, np.full(len(D), -np.inf)
    D_ = np.maximum(D, 1e-6)[:, None]
    m = m & (s > 1e-6) & (s < D_ - 1e-6)
    s_safe = np.where(m, s, 1.0)
    st = np.where(m, (z - h_tx[:, None]) / s_safe, -np.inf).max(axis=1)  # steepest ray from the transmitter
    sr = np.where(m, (z - h_rx[:, None]) / np.where(m, D_ - s, 1.0), -np.inf).max(axis=1)
    any_edge = m.any(axis=1)
    st, sr = np.where(any_edge, st, 0.0), np.where(any_edge, sr, 0.0)
    denom = np.where(np.abs(st + sr) < 1e-12, 1e-12, st + sr)
    s_b = np.clip((h_rx - h_tx + sr * D) / denom, 1e-3, np.maximum(D - 1e-3, 1e-3))
    z_b = h_tx + st * s_b
    h_eq = z_b - (h_tx * (D - s_b) + h_rx * s_b) / D_[:, 0]
    nu = fresnel_parameter(h_eq, s_b, np.maximum(D - s_b, 1e-6), freq_hz)
    loss = np.where(any_edge, knife_edge_loss_db(nu), 0.0)
    return loss, np.where(any_edge, nu, -np.inf)


def _segment_box_chords(a: np.ndarray, b: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal slab test of segments ``a->b`` (``(N, 3)``) against axis-aligned boxes ``(M, 6)``.

    Returns the entry/exit parameters ``t0, t1`` in ``[0, 1]`` and a hit mask, each ``(N, M)``.
    """
    d = (b - a)[:, None, :2]
    o = a[:, None, :2]
    lo, hi = boxes[None, :, :2], boxes[None, :, 3:5]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo, t_hi = (lo - o) / d, (hi - o) / d
    t_near = np.where(np.abs(d) < 1e-12, np.where((o >= lo) & (o <= hi), -np.inf, np.inf), np.minimum(t_lo, t_hi))
    t_far = np.where(np.abs(d) < 1e-12, np.where((o >= lo) & (o <= hi), np.inf, -np.inf), np.maximum(t_lo, t_hi))
    t0 = np.maximum(t_near.max(axis=-1), 0.0)
    t1 = np.minimum(t_far.min(axis=-1), 1.0)
    return t0, t1, t0 < t1


def path_profile(world: Any, tx: Any, rx: Any, terrain_samples: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Obstacle edges along each straight link through ``world``.

    Returns ``(dist, edge_s, edge_z, edge_mask)``: horizontal link length ``(N,)`` and, per candidate edge ``(N, M)``,
    the distance from the transmitter, the absolute top height and a validity mask. Candidates are the roofs of boxes
    (taken at the chord end that obstructs most) and ``terrain_samples`` points of the terrain profile.
    """
    tx, rx = np.broadcast_arrays(np.atleast_2d(np.asarray(tx, float)), np.atleast_2d(np.asarray(rx, float)))
    n = len(tx)
    dist = np.linalg.norm((rx - tx)[:, :2], axis=1)
    s_list, z_list, m_list = [], [], []
    boxes = np.asarray(world.boxes, float).reshape(-1, 6)
    if len(boxes):
        t0, t1, hit = _segment_box_chords(tx, rx, boxes)
        line0 = tx[:, 2:3] + t0 * (rx - tx)[:, 2:3]
        line1 = tx[:, 2:3] + t1 * (rx - tx)[:, 2:3]
        top = boxes[None, :, 5]
        use0 = (top - line0) >= (top - line1)  # the end where the ray is lower is the obstructing edge
        t_e = np.where(use0, t0, t1)
        s_list.append(t_e * dist[:, None])
        z_list.append(np.broadcast_to(top, t_e.shape))
        m_list.append(hit & (np.maximum(top - line0, top - line1) > -1e3))
    if getattr(world, "heightmap", None) is not None and terrain_samples > 0:
        u = np.linspace(0.0, 1.0, terrain_samples + 2)[1:-1]
        px = tx[:, None, 0] + u[None] * (rx - tx)[:, None, 0]
        py = tx[:, None, 1] + u[None] * (rx - tx)[:, None, 1]
        from ...core.backend import get_backend

        zt = np.asarray(world.ground_height(get_backend("numpy"), px, py), float)
        s_list.append(np.broadcast_to(u[None] * dist[:, None], zt.shape))
        z_list.append(zt)
        m_list.append(np.ones_like(zt, bool))
    if not s_list:
        return dist, np.zeros((n, 0)), np.zeros((n, 0)), np.zeros((n, 0), bool)
    return dist, np.concatenate(s_list, 1), np.concatenate(z_list, 1), np.concatenate(m_list, 1)


def world_diffraction_loss_db(
    world: Any, tx: Any, rx: Any, freq_hz: float, terrain_samples: int = 24, max_loss_db: Optional[float] = 60.0
) -> np.ndarray:
    """Bullington diffraction loss [dB] of straight links through ``world`` (0 for clear, unobstructed paths)."""
    tx2, rx2 = np.broadcast_arrays(np.atleast_2d(np.asarray(tx, float)), np.atleast_2d(np.asarray(rx, float)))
    dist, s, z, m = path_profile(world, tx2, rx2, terrain_samples)
    loss, _ = bullington_loss_db(tx2[:, 2], rx2[:, 2], dist, s, z, m, freq_hz)
    return np.minimum(loss, max_loss_db) if max_loss_db is not None else loss
