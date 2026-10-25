"""Flight controllers: geometric SE(3), cascaded PID, body-rate, and LQR hover.

All controllers are batched over ``N`` vehicles, backend-agnostic (NumPy / torch) and
use *nominal* vehicle parameters (mass, inertia, kT, kQ) from the :class:`VehicleSpec`, so
simulations with randomised plant parameters expose genuine model mismatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..core import rotation as rot
from .allocation import Allocator
from .reference import Reference


def _auto_bandwidths(spec):
    tau = spec.motor_tau
    wn_att = float(np.clip(0.5 / max(tau, 1e-3), 8.0, 60.0))
    return wn_att, wn_att / 4.0


@dataclass
class Gains:
    wn_pos: Optional[tuple] = None  # position natural frequency per axis [rad/s]
    zeta_pos: float = 0.95
    wn_att: Optional[tuple] = None  # attitude natural frequency per axis [rad/s]
    zeta_att: float = 0.9
    ki_pos: float = 0.0  # integral gain in (1/s^2)-units of position loop
    int_limit: float = 2.0
    max_tilt_deg: float = 55.0


class GeometricController:
    """Geometric tracking controller on SO(3) × R^3 (Lee, Leok & McClamroch, 2010).

    Position/velocity errors define a desired thrust vector; the full attitude error on
    SO(3) (no Euler singularities) defines the body torque. Gains are expressed as natural
    frequencies and scaled by the vehicle's mass and inertia automatically.
    """

    def __init__(self, sim, gains: Gains | None = None):
        self.B, self.spec, self.sim = sim.B, sim.spec, sim
        B, spec = self.B, self.spec
        g = gains or Gains()
        wa, wp = _auto_bandwidths(spec)
        self.wn_pos = B.asarray(g.wn_pos if g.wn_pos is not None else (wp, wp, wp * 1.4))
        self.wn_att = B.asarray(g.wn_att if g.wn_att is not None else (wa, wa, wa * 0.5))
        self.zeta_pos, self.zeta_att = g.zeta_pos, g.zeta_att
        self.ki = g.ki_pos
        self.int_limit = g.int_limit
        self.tan_tilt = math.tan(math.radians(g.max_tilt_deg))
        self.m = spec.mass
        self.J = B.asarray(spec.inertia_matrix)
        self.g = sim.world.gravity.g
        self.alloc = Allocator(spec, B)
        self.integral: Any = None
        self.last: Dict[str, Any] = {}

    def reset(self, n=None):
        n = n or self.sim.n
        self.integral = self.B.zeros(n, 3)

    def wrench(self, state, ref: Reference, dt: float):
        """Compute collective thrust (N,) and body torque (N,3)."""
        B = self.B
        ref = ref.filled(B)
        if self.integral is None:
            self.reset(state.pos.shape[0])
        R = rot.to_matrix(B, state.quat)
        ep, ev = state.pos - ref.pos, state.vel - ref.vel
        acc_cmd = self.acceleration_command(state, ref, dt, ep, ev)
        F = self.m * (acc_cmd + B.asarray([0.0, 0.0, 1.0]) * self.g)

        # tilt limiting: bound horizontal force relative to vertical
        Fz = B.maximum(F[..., 2], 0.05 * self.m * self.g * B.ones(1))
        fxy = F[..., :2]
        nxy = B.sqrt(B.sum(fxy * fxy, axis=-1) + 1e-12)
        s = B.minimum(Fz * self.tan_tilt / nxy, B.ones(1))
        F = B.stack([F[..., 0] * s, F[..., 1] * s, Fz], -1)

        b3 = R[..., :, 2]
        thrust = B.maximum(B.sum(F * b3, axis=-1), 0.0 * Fz)
        b3d = F / B.norm(F, axis=-1, keepdims=True)
        b1c = B.stack([B.cos(ref.yaw), B.sin(ref.yaw), ref.yaw * 0.0], -1)
        b2d = B.cross(b3d, b1c)
        b2d = b2d / B.maximum(B.norm(b2d, axis=-1, keepdims=True), 1e-9 * B.ones(1))
        b1d = B.cross(b2d, b3d)
        Rd = B.stack([b1d, b2d, b3d], axis=-1)  # columns

        RdT_R = B.einsum("nji,njk->nik", Rd, R)
        eR = 0.5 * rot.vee(B, RdT_R - B.einsum("nji->nij", RdT_R))
        wd = B.stack([ref.yaw_rate * 0.0, ref.yaw_rate * 0.0, ref.yaw_rate], -1)
        ew = state.omega - B.einsum("nji,njk,nk->ni", R, Rd, wd)
        kR = self.wn_att**2
        kW = 2 * self.zeta_att * self.wn_att
        ang_acc = -kR * eR - kW * ew
        tau = self.torque_command(state, ang_acc, dt)
        self.last = dict(ep=ep, ev=ev, eR=eR, ew=ew, thrust=thrust, tau=tau)
        return thrust, tau

    # ---- the two laws that variants of this controller replace (translational loop and torque law)
    def acceleration_command(self, state, ref: Reference, dt: float, ep, ev):
        """Desired world-frame acceleration (before gravity compensation): the critically-damped PD(I) law with feed-forward."""
        B = self.B
        if self.ki != 0.0:
            self.integral = B.clip(self.integral + ep * dt, -self.int_limit, self.int_limit)
        return -(self.wn_pos**2) * ep - 2 * self.zeta_pos * self.wn_pos * ev - self.ki * self.integral + ref.acc

    def torque_command(self, state, ang_acc, dt: float):
        """Body torque that realises the desired angular acceleration: ``J α + ω × J ω`` with the nominal inertia."""
        B = self.B
        Jw = B.einsum("ij,nj->ni", self.J, state.omega)
        return B.einsum("ij,nj->ni", self.J, ang_acc) + B.cross(state.omega, Jw)

    def act(self, state, ref: Reference, dt: float):
        thrust, tau = self.wrench(state, ref, dt)
        return self.alloc.to_rotor_speeds(thrust, tau)


class RateController:
    """Inner body-rate loop: ``τ = J (Kp (ω* − ω) + Ki ∫ ) + ω × J ω`` with output allocation."""

    def __init__(self, sim, bandwidth=None, ki=0.0):
        self.B, self.spec, self.sim = sim.B, sim.spec, sim
        wa, _ = _auto_bandwidths(sim.spec)
        bw = bandwidth if bandwidth is not None else (wa, wa, wa * 0.6)
        self.kp = self.B.asarray(bw)
        self.ki = ki
        self.J = self.B.asarray(sim.spec.inertia_matrix)
        self.alloc = Allocator(sim.spec, self.B)
        self.integral: Any = None

    def reset(self, n=None):
        self.integral = self.B.zeros(n or self.sim.n, 3)

    def act(self, state, rate_des, thrust, dt):
        B = self.B
        if self.integral is None:
            self.reset(state.pos.shape[0])
        e = rate_des - state.omega
        self.integral = B.clip(self.integral + e * dt, -5.0, 5.0)
        acc = self.kp * e + self.ki * self.integral
        Jw = B.einsum("ij,nj->ni", self.J, state.omega)
        tau = B.einsum("ij,nj->ni", self.J, acc) + B.cross(state.omega, Jw)
        return self.alloc.to_rotor_speeds(thrust, tau)


class CascadedPID:
    """Classic cascade used in autopilots: position P → velocity PI → attitude P → rate PI.

    Gains are in physically interpretable units and the structure mirrors PX4/ArduPilot,
    which makes it the standard baseline for learning-based controllers.
    """

    def __init__(self, sim, kp_pos=1.0, kp_vel=3.0, ki_vel=0.8, kp_att=8.0, kp_rate=None, ki_rate=0.0, v_max=8.0, max_tilt_deg=45.0):
        self.B, self.spec, self.sim = sim.B, sim.spec, sim
        B = self.B
        self.kp_pos, self.kp_vel, self.ki_vel = kp_pos, kp_vel, ki_vel
        self.kp_att = B.asarray((kp_att, kp_att, kp_att * 0.5))
        self.v_max = v_max
        self.tan_tilt = math.tan(math.radians(max_tilt_deg))
        self.m, self.g = sim.spec.mass, sim.world.gravity.g
        self.rate = RateController(sim, kp_rate, ki_rate)
        self.vint: Any = None

    def reset(self, n=None):
        n = n or self.sim.n
        self.vint = self.B.zeros(n, 3)
        self.rate.reset(n)

    def act(self, state, ref: Reference, dt: float):
        B = self.B
        ref = ref.filled(B)
        if self.vint is None:
            self.reset(state.pos.shape[0])
        v_des = self.kp_pos * (ref.pos - state.pos) + ref.vel
        return self._velocity_stage(state, v_des, ref, dt)

    def act_velocity(self, state, v_des, yaw=None, dt: float = 0.01):
        """Skip the position loop and track a velocity setpoint directly (for velocity-command policies)."""
        B = self.B
        n = state.pos.shape[0]
        ref = Reference(pos=state.pos, vel=v_des, yaw=yaw if yaw is not None else B.zeros(n)).filled(B)
        if self.vint is None:
            self.reset(n)
        return self._velocity_stage(state, v_des, ref, dt)

    def _velocity_stage(self, state, v_des, ref, dt):
        B = self.B
        sp = B.norm(v_des, axis=-1, keepdims=True)
        v_des = v_des * B.minimum(self.v_max / B.maximum(sp, 1e-9 * B.ones(1)), B.ones(1))
        ev = v_des - state.vel
        self.vint = B.clip(self.vint + ev * dt, -3.0, 3.0)
        acc = self.kp_vel * ev + self.ki_vel * self.vint + ref.acc
        F = self.m * (acc + B.asarray([0.0, 0.0, 1.0]) * self.g)
        Fz = B.maximum(F[..., 2], 0.05 * self.m * self.g * B.ones(1))
        nxy = B.sqrt(F[..., 0] ** 2 + F[..., 1] ** 2 + 1e-12)
        s = B.minimum(Fz * self.tan_tilt / nxy, B.ones(1))
        F = B.stack([F[..., 0] * s, F[..., 1] * s, Fz], -1)

        R = rot.to_matrix(B, state.quat)
        thrust = B.maximum(B.sum(F * R[..., :, 2], axis=-1), 0.0 * Fz)
        b3d = F / B.norm(F, axis=-1, keepdims=True)
        b1c = B.stack([B.cos(ref.yaw), B.sin(ref.yaw), ref.yaw * 0.0], -1)
        b2d = B.cross(b3d, b1c)
        b2d = b2d / B.maximum(B.norm(b2d, axis=-1, keepdims=True), 1e-9 * B.ones(1))
        Rd = B.stack([B.cross(b2d, b3d), b2d, b3d], axis=-1)
        RdT_R = B.einsum("nji,njk->nik", Rd, R)
        eR = 0.5 * rot.vee(B, RdT_R - B.einsum("nji->nij", RdT_R))
        rate_des = -self.kp_att * eR
        return self.rate.act(state, rate_des, thrust, dt)


class LQRHover:
    """Linear-quadratic regulator about hover (12 states: p, v, roll/pitch/yaw, body rates).

    Small-angle model in the ENU/FLU convention, with inputs normalised by mass/inertia
    (``a_T = ΔT/m`` and angular accelerations ``α = J^{-1} τ``) so the weights are vehicle
    independent::

        p_dot = v,   v_dot = [ g θ, −g φ, a_T ],   (φ, θ, ψ)_dot = ω,   ω_dot = α

    Weights default to bandwidths derived from the motor time constant, which keeps the
    design inside the actuator's capability. The Riccati equation is solved with SciPy.
    """

    def __init__(self, sim, wn_att=None, wn_pos=None, wn_yaw=None):
        from scipy.linalg import solve_continuous_are

        self.B, self.spec, self.sim = sim.B, sim.spec, sim
        m, g = sim.spec.mass, sim.world.gravity.g
        wa, wp = _auto_bandwidths(sim.spec)
        wa = wn_att or wa * 0.7
        wp = wn_pos or wp * 0.7
        wy = wn_yaw or wa * 0.5
        A = np.zeros((12, 12))
        A[0:3, 3:6] = np.eye(3)
        A[3, 7] = g
        A[4, 6] = -g
        A[6:9, 9:12] = np.eye(3)
        Bm = np.zeros((12, 4))
        Bm[5, 0] = 1.0
        Bm[9:12, 1:4] = np.eye(3)
        Q = np.diag([wp**4] * 2 + [(1.3 * wp) ** 4] + [2 * wp**2] * 2 + [2 * (1.3 * wp) ** 2] + [wa**4] * 2 + [wy**4] + [2 * wa**2] * 2 + [2 * wy**2])
        Rm = np.eye(4)
        P = solve_continuous_are(A, Bm, Q, Rm)
        self.K = np.linalg.solve(Rm, Bm.T @ P)  # (4, 12) acting on normalised inputs
        self.Kt = self.B.asarray(self.K)
        self.alloc = Allocator(sim.spec, self.B)
        self.m, self.g = m, g
        self.J = self.B.asarray(sim.spec.inertia_matrix)

    def act(self, state, ref: Reference, dt: float = 0.0):
        B = self.B
        ref = ref.filled(B)
        eul = rot.to_euler(B, state.quat)
        dyaw = B.atan2(B.sin(eul[..., 2] - ref.yaw), B.cos(eul[..., 2] - ref.yaw))
        cy, sy = B.cos(eul[..., 2]), B.sin(eul[..., 2])
        ep, ev = state.pos - ref.pos, state.vel - ref.vel

        def to_yaw_frame(v):
            return B.stack([cy * v[..., 0] + sy * v[..., 1], -sy * v[..., 0] + cy * v[..., 1], v[..., 2]], -1)

        # acceleration feedforward through the small-angle equilibrium (θ* = a_x/g, φ* = −a_y/g)
        a_ff = to_yaw_frame(ref.acc)
        att_ff = B.stack([-a_ff[..., 1] / self.g, a_ff[..., 0] / self.g], -1)
        x = B.cat([to_yaw_frame(ep), to_yaw_frame(ev), eul[..., :2] - att_ff, dyaw[..., None], state.omega], axis=-1)
        u = -B.einsum("km,nm->nk", self.Kt, x)
        thrust = B.maximum(self.m * (self.g + u[:, 0] + ref.acc[..., 2]), 0.0 * u[:, 0])
        Jw = B.einsum("ij,nj->ni", self.J, state.omega)
        tau = B.einsum("ij,nj->ni", self.J, u[:, 1:]) + B.cross(state.omega, Jw)
        return self.alloc.to_rotor_speeds(thrust, tau)
