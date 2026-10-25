"""Control allocation: map a collective thrust and body torques to rotor speeds."""

from __future__ import annotations

import numpy as np

from ..vehicles.spec import VehicleSpec


class Allocator:
    """Pseudo-inverse allocation with collective-shift desaturation.

    ``[T, τx, τy, τz]^T = A w^2``. Solving for ``w^2`` and clipping to ``[w_min^2, w_max^2]``
    would corrupt the torques whenever a rotor saturates. Instead we first shift **all** rotors
    by a common amount (which only changes collective thrust for symmetric layouts) so that
    attitude control keeps priority over altitude.
    """

    def __init__(self, spec: VehicleSpec, B, w_max_scale: float = 1.0):
        self.B, self.spec = B, spec
        A = spec.allocation_matrix()
        self.A = B.asarray(A)
        self.pinv = B.asarray(np.linalg.pinv(A))  # (M, 4)
        self.u_min = spec.w_min**2
        self.u_max = (spec.w_max * w_max_scale) ** 2
        self.M = spec.n_rotors

    def wrench_to_u(self, thrust, torque):
        """Return squared rotor speeds ``(N, M)`` (unclipped)."""
        B = self.B
        wrench = B.cat([thrust.reshape(-1, 1), torque], axis=-1)
        return B.einsum("mk,nk->nm", self.pinv, wrench)

    def to_rotor_speeds(self, thrust, torque, desaturate: bool = True):
        B = self.B
        u = self.wrench_to_u(thrust, torque)
        if desaturate:
            hi_over = B.maximum(_rowmax(B, u) - self.u_max, 0.0 * thrust.reshape(-1))
            u = u - hi_over[:, None]
            lo_under = B.maximum(self.u_min - _rowmin(B, u), 0.0 * thrust.reshape(-1))
            u = u + lo_under[:, None]
        u = B.minimum(B.maximum(u, self.u_min * B.ones(1)), self.u_max * B.ones(1))
        return B.sqrt(u)

    def rotor_speeds_to_wrench(self, w):
        B = self.B
        out = B.einsum("km,nm->nk", self.A, w * w)
        return out[:, 0], out[:, 1:]

    def u_from_wrench_hover(self, mass, g):
        return mass * g


def _rowmax(B, x):
    m = x[:, 0]
    for i in range(1, x.shape[1]):
        m = B.maximum(m, x[:, i])
    return m


def _rowmin(B, x):
    m = x[:, 0]
    for i in range(1, x.shape[1]):
        m = B.minimum(m, x[:, i])
    return m


class FaultTolerantAllocator:
    """Bounded least-squares allocation that respects rotor failures and actuator limits.

    Solves ``min || W (A w² − y) ||`` subject to ``0 ≤ w² ≤ w_max²`` over the *healthy* rotors only, with
    yaw de-prioritised by ``yaw_weight``, the standard strategy for keeping a hexa/octocopter airborne after
    losing a motor (it sacrifices heading control to preserve thrust, roll and pitch). Runs on CPU, one small
    QP per vehicle, so it is intended for fault-injection studies rather than massive batches.
    """

    def __init__(self, spec: VehicleSpec, B, yaw_weight: float = 0.05):
        self.spec, self.B = spec, B
        self.A = spec.allocation_matrix()
        self.W = np.diag([1.0, 1.0, 1.0, yaw_weight])
        self.u_max = spec.w_max**2
        self.M = spec.n_rotors

    def to_rotor_speeds(self, thrust, torque, health):
        """``health`` is ``(N, M)`` (1 healthy, 0 dead). Returns rotor speed commands ``(N, M)``."""
        from scipy.optimize import lsq_linear

        B = self.B
        y = np.concatenate([B.to_numpy(thrust).reshape(-1, 1), B.to_numpy(torque)], axis=1)
        h = B.to_numpy(health)
        out = np.zeros((y.shape[0], self.M))
        for i in range(y.shape[0]):
            keep = np.nonzero(h[i] > 0.0)[0]
            A = self.A[:, keep] * h[i, keep]  # degraded rotors deliver proportionally less
            sol = lsq_linear(self.W @ A, self.W @ y[i], bounds=(0.0, self.u_max))
            out[i, keep] = np.sqrt(np.maximum(sol.x, 0.0))
        return B.asarray(out)
