"""Observability analysis of a linearised, discrete-time navigation filter.

For ``δx_{k+1} = Φ_k δx_k`` and ``z_k = H_k δx_k`` the observability matrix over a window is::

    O = [ H_0 ;  H_1 Φ_0 ;  H_2 Φ_1 Φ_0 ;  ... ]

and the state is observable along a direction exactly when that direction is not in the null space of ``O`` (Chen, *Linear
System Theory and Design*; for INS see Hong, Chun & Lee 2005). The numerical null space comes from the SVD, with the columns
of ``O`` scaled by the typical size of each state so that metres, radians and bias units are comparable.

Typical use: build ``Φ`` from :meth:`ESKF.transition` and ``H`` from the measurement Jacobians, then ask whether yaw is
observable with and without a magnetometer.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["observability_matrix", "unobservable_subspace", "observable_rank", "direction_observability"]


def observability_matrix(Phis: Sequence[Any], Hs: Sequence[Any]) -> np.ndarray:
    """Stack ``H_k Φ_{k-1} ... Φ_0`` over the window. ``Hs`` has one more entry than ``Phis`` or the same number."""
    Phis = [np.asarray(p, float) for p in Phis]
    Hs = [np.atleast_2d(np.asarray(h, float)) for h in Hs]
    d = Hs[0].shape[1]
    rows, acc = [], np.eye(d)
    for k, H in enumerate(Hs):
        if k > 0:
            acc = Phis[k - 1] @ acc
        rows.append(H @ acc)
    return np.vstack(rows)


def _scaled_svd(O: np.ndarray, scale: Optional[Sequence[float]]) -> tuple:
    s = np.ones(O.shape[1]) if scale is None else np.asarray(scale, float)
    U, sv, Vt = np.linalg.svd(O * s[None, :], full_matrices=True)
    return sv, Vt, s


def observable_rank(O: np.ndarray, scale: Optional[Sequence[float]] = None, rel_tol: float = 1e-9) -> int:
    sv, _, _ = _scaled_svd(O, scale)
    return int(np.sum(sv > rel_tol * max(sv.max(), 1e-300)))


def unobservable_subspace(O: np.ndarray, scale: Optional[Sequence[float]] = None, rel_tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis (columns, in the *scaled* coordinates) of the unobservable directions."""
    sv, Vt, _ = _scaled_svd(O, scale)
    r = int(np.sum(sv > rel_tol * max(sv.max(), 1e-300)))
    return Vt[r:].T


def direction_observability(O: np.ndarray, direction: Sequence[float], scale: Optional[Sequence[float]] = None, rel_tol: float = 1e-9) -> float:
    """Share (0..1) of a state direction that is observable: 1 minus the squared norm of its projection on the null space."""
    _, _, s = _scaled_svd(O, scale)
    d = np.asarray(direction, float) / s
    d = d / np.linalg.norm(d)
    N = unobservable_subspace(O, scale, rel_tol)
    if N.shape[1] == 0:
        return 1.0
    return float(1.0 - np.sum((N.T @ d) ** 2))
