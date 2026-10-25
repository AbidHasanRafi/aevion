"""Advanced position controllers: model-predictive (linear and sampling-based), incremental, adaptive and sliding-mode.

Each controller is a variant of :class:`~aevion.control.controllers.GeometricController`: it shares the SO(3) attitude loop, tilt
limiting and rotor allocation and replaces one law:

=============================  ===================================================================================================
:class:`MPCController`         translational loop: constrained linear MPC on a per-axis double integrator (acceleration, tilt and
                               speed limits) solved by a batched ADMM QP, with a Kalman disturbance estimator (offset-free)
:class:`MPPIController`        translational loop: sampling-based path-integral control that can avoid obstacles in a signed-distance
                               field; any backend (NumPy / torch)
:class:`INDIController`        torque law: incremental nonlinear dynamic inversion (no inertia model needed)
:class:`L1AdaptiveController`  translational loop: L1 adaptive augmentation (state predictor, adaptation law, low-pass filter)
:class:`SlidingModeController` translational loop: sliding-mode law with a boundary layer, robust to bounded disturbances
=============================  ===================================================================================================

The predictive controllers re-plan at ``rate_hz`` (default 20 Hz) and hold the command between solves; the attitude loop runs at the
simulation rate.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from .controllers import Gains, GeometricController
from .reference import Reference

__all__ = [
    "BatchedBoxQP",
    "DisturbanceObserver",
    "MPCController",
    "MPPIController",
    "INDIController",
    "L1AdaptiveController",
    "SlidingModeController",
]


# ======================================================================================================== solver
class BatchedBoxQP:
    """ADMM solver for many QPs that share ``H`` and the constraint matrix ``A``: ``min ½uᵀHu + qᵀu  s.t.  l <= A u <= u_b``.

    Only ``q`` and the bounds differ between problems, so ``(H + σI + ρAᵀA)⁻¹`` is computed once and every iteration is a pair of
    matrix products over the whole batch (the OSQP iteration, Stellato et al. 2020, with a fixed ``ρ``).
    """

    def __init__(self, H: np.ndarray, A: np.ndarray, rho: float = 1.0, sigma: float = 1e-6, alpha: float = 1.6) -> None:
        self.H, self.A = np.asarray(H, float), np.asarray(A, float)
        self.rho, self.sigma, self.alpha = float(rho), float(sigma), float(alpha)
        n = self.H.shape[0]
        self.Kinv = np.linalg.inv(self.H + self.sigma * np.eye(n) + self.rho * self.A.T @ self.A)
        self.n, self.m = n, self.A.shape[0]

    def solve(
        self, q: np.ndarray, lo: np.ndarray, hi: np.ndarray, x0: Optional[np.ndarray] = None, y0: Optional[np.ndarray] = None, iters: int = 80
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve a batch: ``q`` is ``(M, n)``, bounds ``(M, m)``. Returns ``(x, y, primal_residual)``."""
        M = q.shape[0]
        x = np.zeros((M, self.n)) if x0 is None else x0.copy()
        y = np.zeros((M, self.m)) if y0 is None else y0.copy()
        z = np.clip(x @ self.A.T, lo, hi)
        for _ in range(iters):
            xt = (self.sigma * x - q + (self.rho * z - y) @ self.A) @ self.Kinv
            zt = xt @ self.A.T
            x = self.alpha * xt + (1.0 - self.alpha) * x
            zr = self.alpha * zt + (1.0 - self.alpha) * z
            z_new = np.clip(zr + y / self.rho, lo, hi)
            y = y + self.rho * (zr - z_new)
            z = z_new
        return x, y, float(np.max(np.abs(x @ self.A.T - z)))


class DisturbanceObserver:
    """Steady-state Kalman filter for ``[p, v, d]`` per axis (a constant acceleration disturbance ``d``), batched.

    The model is ``p⁺ = p + v dt + ½ (u + d) dt²``, ``v⁺ = v + (u + d) dt``, ``d⁺ = d`` driven by process noise, measuring ``p`` and
    ``v``. The gain is the solution of the discrete Riccati equation, so a constant disturbance (wind, mass error, drag) is
    estimated without bias and the predictive controller becomes *offset-free*.
    """

    def __init__(self, sigma_d: float = 0.8, sigma_p: float = 0.05, sigma_v: float = 0.05, limit: float = 6.0) -> None:
        self.sigma_d, self.sigma_p, self.sigma_v, self.limit = sigma_d, sigma_p, sigma_v, float(limit)
        self._dt: Optional[float] = None
        self.reset(1)

    def reset(self, n: int) -> None:
        self.x = np.zeros((n, 3, 3))  # (vehicle, axis, [p, v, d])
        self.u_prev = np.zeros((n, 3))
        self.started = False

    def _gains(self, dt: float) -> None:
        from scipy.linalg import solve_discrete_are

        A = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Q = np.diag([1e-10, 1e-8, (self.sigma_d**2) * dt])
        R = np.diag([self.sigma_p**2, self.sigma_v**2])
        P = solve_discrete_are(A.T, H.T, Q, R)
        self.K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        self.A = A
        self.Bv = np.array([0.5 * dt * dt, dt, 0.0])
        self._dt = dt

    def update(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        """Fuse the latest position and velocity; returns the disturbance estimate ``(n, 3)``."""
        if self._dt is None or abs(dt - self._dt) > 1e-12:
            self._gains(dt)
        if not self.started or self.x.shape[0] != pos.shape[0]:
            self.x = np.zeros((pos.shape[0], 3, 3))
            self.x[:, :, 0], self.x[:, :, 1] = pos, vel
            self.started = True
        xp = self.x @ self.A.T + self.u_prev[:, :, None] * self.Bv[None, None, :]
        y = np.stack([pos, vel], axis=-1)
        self.x = xp + (y - xp[:, :, 0:2]) @ self.K.T
        self.x[:, :, 2] = np.clip(self.x[:, :, 2], -self.limit, self.limit)
        return self.x[:, :, 2]

    def command_applied(self, u: np.ndarray) -> None:
        self.u_prev = np.asarray(u, float)

    @property
    def disturbance(self) -> np.ndarray:
        return self.x[:, :, 2]


class _Predictive(GeometricController):
    """Shared machinery: the controller re-plans at ``rate_hz`` and holds its acceleration between solves."""

    def __init__(
        self,
        sim: Any,
        gains: Optional[Gains] = None,
        rate_hz: float = 20.0,
        max_tilt_deg: float = 40.0,
        a_xy_max: Optional[float] = None,
        a_z_range: Optional[Tuple[float, float]] = None,
        v_max: float = 6.0,
    ) -> None:
        super().__init__(sim, gains)
        self.rate_hz = float(rate_hz)
        g = self.g
        tilt = math.radians(max_tilt_deg)
        self.a_xy = float(a_xy_max) if a_xy_max is not None else g * math.tan(tilt) / math.sqrt(2.0)
        if a_z_range is None:
            up = 0.8 * (self.spec.max_thrust() / self.spec.mass - g)
            a_z_range = (-0.7 * g, max(up, 0.5))
        self.a_z = (float(a_z_range[0]), float(a_z_range[1]))
        self.v_max = float(v_max)
        self._clock = 0.0
        self._next = 0.0
        self._acc_hold: Optional[np.ndarray] = None
        self.solves = 0
        self.N, self.h = 20, 0.1  # overwritten by the subclasses
        self._traj: Any = None
        self._offsets: Any = None
        self._t_now = 0.0
        self.obs: Optional[DisturbanceObserver] = None

    def reset(self, n: Optional[int] = None) -> None:
        super().reset(n)
        self._clock = self._next = self._t_now = 0.0
        self._acc_hold = None
        self.solves = 0
        if self.obs is not None:
            self.obs.reset(n or self.sim.n)

    def wrench(self, state: Any, ref: Reference, dt: float) -> Any:
        out = super().wrench(state, ref, dt)
        if self.obs is not None and self._acc_hold is not None:
            self.obs.command_applied(self.B.to_numpy(self._acc_hold))  # the observer needs the acceleration that was commanded
        return out

    @property
    def disturbance_estimate(self) -> Optional[np.ndarray]:
        """The estimated constant acceleration disturbance ``(n, 3)`` (None without an observer)."""
        return None if self.obs is None else self.obs.disturbance.copy()

    def attach_trajectory(self, traj: Any, offsets: Any = None) -> None:
        """Let the controller preview the reference exactly (the engine does this for deterministic trajectories) instead of
        extrapolating it from ``pos, vel, acc, jerk``; stateful trajectories such as missions are left to the extrapolation."""
        self._traj = None if hasattr(traj, "observe") else traj
        self._offsets = None if offsets is None else np.asarray(offsets, float)

    def _due(self, dt: float) -> bool:
        due = self._acc_hold is None or self._clock + 1e-9 >= self._next
        if due:
            self._next = self._clock + 1.0 / self.rate_hz
        self._t_now = self._clock
        self._clock += dt
        return due

    def _preview(self, ref: Reference) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference position, velocity and acceleration over the horizon, ``(n, 3, N)`` each."""
        B = self.B
        r = ref.filled(B)
        n = r.pos.shape[0]
        tau = (np.arange(self.N) + 1.0) * self.h
        if self._traj is not None:
            samples = [self._traj.sample(self._t_now + float(t)) for t in tau]
            P = np.stack([np.asarray(x[0], float) for x in samples], axis=-1)  # (3, N)
            V = np.stack([np.asarray(x[1], float) for x in samples], axis=-1)
            A = np.stack([np.asarray(x[2], float) for x in samples], axis=-1)
            off = 0.0 if self._offsets is None else self._offsets[:, :, None]
            return P[None] + off, V[None] * np.ones((n, 1, 1)), A[None] * np.ones((n, 1, 1))
        p, v, a, j = (B.to_numpy(x) for x in (r.pos, r.vel, r.acc, r.jerk))
        T = tau[None, None, :]
        rp = p[:, :, None] + v[:, :, None] * T + 0.5 * a[:, :, None] * T**2 + j[:, :, None] * T**3 / 6.0
        rv = v[:, :, None] + a[:, :, None] * T + 0.5 * j[:, :, None] * T**2
        ra = a[:, :, None] + j[:, :, None] * T
        return rp, rv, ra

    def _acc_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-self.a_xy, -self.a_xy, self.a_z[0]]), np.array([self.a_xy, self.a_xy, self.a_z[1]])


# ======================================================================================================== MPC
class MPCController(_Predictive):
    """Constrained linear MPC for the translational loop (acceleration command), one double integrator per axis.

    Over a horizon of ``horizon`` steps of ``step`` seconds it minimises the tracking cost of position and velocity against the
    *previewed* reference (Taylor-expanded from ``pos, vel, acc, jerk``), a control cost, and a terminal cost, subject to
    acceleration limits (tilt and thrust) and a speed limit. The problem is a small condensed QP per axis, solved for all
    vehicles and axes at once with :class:`BatchedBoxQP`, warm-started from the previous solution. With ``observer=True`` a
    Kalman filter estimates a constant acceleration disturbance, which is added to the prediction (offset-free tracking under
    wind or a wrong mass).
    """

    def __init__(
        self,
        sim: Any,
        horizon: int = 20,
        step: float = 0.1,
        q_pos: float = 4.0,
        q_vel: float = 1.0,
        r_acc: float = 0.05,
        terminal_scale: float = 6.0,
        observer: bool = True,
        gains: Optional[Gains] = None,
        iterations: int = 60,
        **kw: Any,
    ) -> None:
        super().__init__(sim, gains, **kw)
        if horizon < 2 or step <= 0:
            raise ValueError("need horizon >= 2 and step > 0")
        N, h = int(horizon), float(step)
        self.N, self.h, self.iters = N, h, int(iterations)
        k = np.arange(N)
        G_v = np.tril(np.full((N, N), h))
        G_p = np.tril(h * h * ((k[:, None] - k[None, :]) + 0.5))
        qp_w = np.full(N, float(q_pos))
        qv_w = np.full(N, float(q_vel))
        qp_w[-1] *= terminal_scale
        qv_w[-1] *= terminal_scale
        self.G_p, self.G_v, self.qp_w, self.qv_w, self.r = G_p, G_v, qp_w, qv_w, float(r_acc)
        Hm = 2.0 * (G_p.T @ (qp_w[:, None] * G_p) + G_v.T @ (qv_w[:, None] * G_v) + self.r * np.eye(N))
        self.qp = BatchedBoxQP(Hm, np.vstack([np.eye(N), G_v]), rho=2.0)
        self.obs = DisturbanceObserver() if observer else None
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def reset(self, n: Optional[int] = None) -> None:
        super().reset(n)
        self._x = self._y = None

    def acceleration_command(self, state: Any, ref: Reference, dt: float, ep: Any, ev: Any) -> Any:
        B = self.B
        p0, v0 = B.to_numpy(state.pos), B.to_numpy(state.vel)
        n = p0.shape[0]
        d = self.obs.update(p0, v0, dt) if self.obs is not None else np.zeros((n, 3))
        if not self._due(dt):
            return B.asarray(self._acc_hold)
        rp, rv, ra = self._preview(ref)
        N = self.N
        k1 = (np.arange(N) + 1.0) * self.h
        # effective acceleration w = u + d acts on the integrators; the controller chooses u = w - d
        p0v = p0[:, :, None] + k1[None, None, :] * v0[:, :, None]
        q = 2.0 * (
            (self.qp_w[None, None, :] * (p0v - rp)).reshape(-1, N) @ self.G_p
            + (self.qv_w[None, None, :] * (v0[:, :, None] - rv)).reshape(-1, N) @ self.G_v
            - self.r * (ra + d[:, :, None]).reshape(-1, N)  # the effort term penalises u = w - d, so its target for w is a_ref + d
        )
        lo_u, hi_u = self._acc_bounds()
        lo_w = (lo_u[None, :] + d).reshape(-1, 1) * np.ones((1, N))
        hi_w = (hi_u[None, :] + d).reshape(-1, 1) * np.ones((1, N))
        v_lo = (-self.v_max - v0).reshape(-1, 1) * np.ones((1, N))
        v_hi = (self.v_max - v0).reshape(-1, 1) * np.ones((1, N))
        lo, hi = np.hstack([lo_w, v_lo]), np.hstack([hi_w, v_hi])
        x0: Optional[np.ndarray] = None
        y0: Optional[np.ndarray] = None
        if self._x is not None and self._x.shape[0] == q.shape[0]:  # warm start: shift the previous plan by one step
            x0 = np.hstack([self._x[:, 1:], self._x[:, -1:]])
            y0 = self._y
        x, y, _ = self.qp.solve(q, lo, hi, x0, y0, self.iters)
        self._x, self._y = x, y
        self.solves += 1
        w0 = x[:, 0].reshape(n, 3)
        u0 = np.clip(w0 - d, lo_u[None, :], hi_u[None, :])
        hold = 1.0 / self.rate_hz  # the command is held until the next solve: do not let it carry the speed out of its box meanwhile
        u0 = np.clip(u0, (-self.v_max - v0) / hold - d, (self.v_max - v0) / hold - d)
        u0 = np.clip(u0, lo_u[None, :], hi_u[None, :])
        self._acc_hold = u0
        return B.asarray(u0)


# ======================================================================================================== MPPI
class MPPIController(_Predictive):
    """Model-predictive path-integral control (Williams et al. 2016) for the translational loop.

    ``samples`` perturbed acceleration sequences are rolled out through a double-integrator model for every vehicle at once
    (backend arrays, so it runs on CUDA with the torch backend), costed, and averaged with weights ``exp(-(S - S_min) / (λ (S̄ - S_min)))`` (the temperature is relative to the spread of the sampled costs, so ``λ`` is scale-free).
    Because the cost is evaluated by simulation it can contain terms a QP cannot: with ``sdf`` the rollouts are penalised for
    approaching obstacles (``obstacle_weight``), so the controller flies around them without a separately planned path.
    """

    def __init__(
        self,
        sim: Any,
        horizon: int = 20,
        step: float = 0.1,
        samples: int = 512,
        sigma: float = 1.0,
        temperature: float = 0.1,
        q_pos: float = 4.0,
        q_vel: float = 1.0,
        r_acc: float = 0.02,
        sdf: Any = None,
        clearance: float = 1.5,
        obstacle_weight: float = 200.0,
        seed: int = 0,
        observer: bool = True,
        gains: Optional[Gains] = None,
        **kw: Any,
    ) -> None:
        kw.setdefault("rate_hz", 30.0)
        super().__init__(sim, gains, **kw)
        self.obs = DisturbanceObserver() if observer else None
        if horizon < 2 or samples < 2 or step <= 0 or temperature <= 0:
            raise ValueError("need horizon >= 2, samples >= 2, step > 0 and temperature > 0")
        self.N, self.h, self.K = int(horizon), float(step), int(samples)
        self.sigma, self.lam = float(sigma), float(temperature)
        self.qp, self.qv, self.r = float(q_pos), float(q_vel), float(r_acc)
        self.sdf, self.clear, self.w_obs = sdf, float(clearance), float(obstacle_weight)
        self.gen = self.B.make_rng(seed)
        self.U: Any = None
        lo, hi = self._acc_bounds()
        self.lo, self.hi = self.B.asarray(lo), self.B.asarray(hi)

    def reset(self, n: Optional[int] = None) -> None:
        super().reset(n)
        self.U = None

    def _sdf_cost(self, pos: Any) -> Any:
        B = self.B
        flat = B.to_numpy(pos).reshape(-1, 3)
        d = self.sdf.distance(flat).reshape(tuple(pos.shape[:-1]))
        pen = np.maximum(self.clear - d, 0.0) ** 2 + 50.0 * (d < 0.0)
        return B.asarray(pen)

    def acceleration_command(self, state: Any, ref: Reference, dt: float, ep: Any, ev: Any) -> Any:
        B = self.B
        n = state.pos.shape[0]
        d_np = self.obs.update(B.to_numpy(state.pos), B.to_numpy(state.vel), dt) if self.obs is not None else np.zeros((n, 3))
        if not self._due(dt):
            return self._acc_hold
        D = B.asarray(d_np)[:, None, :]  # constant acceleration disturbance, added to every sampled rollout
        N, K, h = self.N, self.K, self.h
        if self.U is None or self.U.shape[0] != n:
            self.U = B.zeros(n, N, 3)
        rp_np, rv_np, ra_np = self._preview(ref)
        RP, RV, RA = B.asarray(rp_np), B.asarray(rv_np), B.asarray(ra_np)  # (n, 3, N)
        eps = self.sigma * B.randn((n, K, N, 3), self.gen)
        eps = eps * B.ones(1)  # keep dtype/device of the backend
        V = B.clip(self.U[:, None, :, :] + eps, self.lo, self.hi)  # sampled controls, within the acceleration limits
        p = state.pos[:, None, :] * B.ones(1, K, 1)
        v = state.vel[:, None, :] * B.ones(1, K, 1)
        cost = B.zeros(n, K)
        for k in range(N):
            rp, rv, ra = RP[:, :, k], RV[:, :, k], RA[:, :, k]
            u = V[:, :, k, :]
            p = p + v * h + 0.5 * (u + D) * h * h
            v = v + (u + D) * h
            speed2 = B.sum(v * v, axis=-1)
            cost = cost + self.qp * B.sum((p - rp[:, None, :]) ** 2, axis=-1) + self.qv * B.sum((v - rv[:, None, :]) ** 2, axis=-1)
            cost = cost + self.r * B.sum((u - ra[:, None, :]) ** 2, axis=-1) + 5.0 * B.maximum(speed2 - self.v_max**2, 0.0 * speed2)
            if self.sdf is not None:
                cost = cost + self.w_obs * self._sdf_cost(p)
        cmin = cost.min(dim=1, keepdim=True).values if B.name == "torch" else cost.min(axis=1, keepdims=True)
        spread = cost.mean(dim=1, keepdim=True) - cmin if B.name == "torch" else cost.mean(axis=1, keepdims=True) - cmin
        w = B.exp(-(cost - cmin) / (self.lam * spread + 1e-9))  # temperature relative to the cost spread: scale-free tuning
        w = w / B.sum(w, axis=1, keepdims=True)
        self.U = B.clip(self.U + B.einsum("nk,nkhc->nhc", w, eps), self.lo, self.hi)
        u0 = self.U[:, 0, :]
        self.U = B.cat([self.U[:, 1:, :], self.U[:, -1:, :]], axis=1)  # receding horizon: shift the plan
        self._acc_hold = u0
        self.solves += 1
        return u0


# ======================================================================================================== INDI
class INDIController(GeometricController):
    """Incremental nonlinear dynamic inversion for the torque law (Smeur, Chu & de Croon 2016; Sieberling et al. 2010).

    Instead of inverting the inertia model, INDI computes the *change* of torque needed to turn the measured angular acceleration
    into the desired one: ``τ = τ_prev + J (α_des − α_meas)``. Model error in ``J`` then only scales a correction, and unmodelled
    constant torques (a shifted centre of mass, a motor imbalance) are cancelled because they appear in ``α_meas``. ``α_meas``
    is the low-passed derivative of the angular rate (cutoff ``alpha_cutoff_hz``), and the previous torque is the one the rotors are *actually* producing (from the measured rotor speeds, so saturation cannot wind
    it up), passed through the same low-pass filter, so both signals carry the same delay (the key to INDI's robustness).
    """

    def __init__(self, sim: Any, gains: Optional[Gains] = None, alpha_cutoff_hz: float = 25.0, actuator_tau: Optional[float] = None) -> None:
        super().__init__(sim, gains)
        self.fc = float(alpha_cutoff_hz)
        self.act_tau = float(actuator_tau if actuator_tau is not None else max(self.spec.motor_tau, 1e-3))
        self._om_prev: Any = None
        self._alpha_f: Any = None
        self._tau_act: Any = None
        self._tau_mot: Any = None
        self._c_f: Any = None
        self._w_prev: Any = None
        self._spin = self.B.asarray(np.asarray(self.spec.rotor_spin, float))

    def reset(self, n: Optional[int] = None) -> None:
        super().reset(n)
        self._om_prev = self._alpha_f = self._tau_act = self._tau_mot = self._c_f = self._w_prev = None

    def torque_command(self, state: Any, ang_acc: Any, dt: float) -> Any:
        B = self.B
        om = state.omega
        nominal = super().torque_command(state, ang_acc, dt)
        gyro_term = nominal - B.einsum("ij,nj->ni", self.J, ang_acc)  # omega x J omega
        rotor = getattr(state, "rotor", None)
        if self._om_prev is None or dt <= 0.0:
            self._om_prev, self._alpha_f, self._tau_act, self._tau_mot = B.clone(om), om * 0.0, B.clone(nominal), B.clone(nominal)
            self._c_f = B.clone(gyro_term)
            return nominal
        a = 1.0 - math.exp(-2.0 * math.pi * self.fc * dt)
        self._alpha_f = self._alpha_f + a * ((om - self._om_prev) / dt - self._alpha_f)
        self._om_prev = B.clone(om)
        if rotor is not None:  # the torque the rotors are actually producing, from the measured rotor speeds (no wind-up under saturation)
            delivered = self.alloc.rotor_speeds_to_wrench(rotor)[1]
            if self.spec.rotor_inertia > 0.0 and self._w_prev is not None:  # plus the yaw reaction of accelerating the rotors: -sum s_i J_r dw_i/dt
                wdot = (rotor - self._w_prev) / dt
                react = -B.sum(self._spin * wdot, axis=-1) * self.spec.rotor_inertia
                delivered = delivered + B.stack([react * 0.0, react * 0.0, react], axis=-1)
            self._w_prev = B.clone(rotor)
        else:  # no actuator feedback (estimated state): model the rotors as a first-order lag behind the previous command
            delivered = self._tau_mot
        self._tau_act = self._tau_act + a * (delivered - self._tau_act)  # filtered with the SAME filter as the angular acceleration
        self._c_f = self._c_f + a * (gyro_term - self._c_f)
        tau = self._tau_act + B.einsum("ij,nj->ni", self.J, ang_acc - self._alpha_f) + (gyro_term - self._c_f)
        self._tau_mot = self._tau_mot + (dt / self.act_tau) * (tau - self._tau_mot)
        return tau


# ======================================================================================================== L1
class L1AdaptiveController(GeometricController):
    """L1 adaptive augmentation of the position loop (Hovakimyan & Cao 2010).

    A state predictor ``v̂' = u + σ̂ - a_s (v̂ - v)`` shadows the measured velocity; the prediction error drives a
    *piecewise-constant* estimate ``σ̂ = -Φ⁻¹ e^{-a_s T} ṽ`` of the matched uncertainty (wind, drag, mass error, actuator lag);
    and the compensation ``-σ̂`` passes through a low-pass filter of bandwidth ``omega_c``, which decouples the fast adaptation
    from the control loop (the defining property of L1: guaranteed transient performance with fast adaptation).
    """

    def __init__(self, sim: Any, gains: Optional[Gains] = None, predictor_pole: float = 15.0, omega_c: float = 8.0, limit: float = 6.0) -> None:
        super().__init__(sim, gains)
        self.a_s, self.wc, self.limit = float(predictor_pole), float(omega_c), float(limit)
        self._vhat: Any = None
        self._sigma: Any = None
        self._comp: Any = None
        self._u_prev: Any = None

    def reset(self, n: Optional[int] = None) -> None:
        super().reset(n)
        self._vhat = self._sigma = self._comp = self._u_prev = None

    @property
    def uncertainty_estimate(self) -> Any:
        return self._sigma

    def acceleration_command(self, state: Any, ref: Reference, dt: float, ep: Any, ev: Any) -> Any:
        B = self.B
        base = super().acceleration_command(state, ref, dt, ep, ev)
        v = state.vel
        if self._vhat is None or dt <= 0.0:
            self._vhat, self._sigma, self._comp, self._u_prev = B.clone(v), v * 0.0, v * 0.0, base
            return base
        e_a = math.exp(-self.a_s * dt)
        phi = (1.0 - e_a) / self.a_s
        v_tilde = self._vhat - v
        self._sigma = B.clip(-(e_a / phi) * v_tilde, -self.limit, self.limit)  # piecewise-constant adaptation
        self._vhat = self._vhat + dt * (self._u_prev + self._sigma - self.a_s * v_tilde)  # state predictor
        a = 1.0 - math.exp(-self.wc * dt)  # low-pass C(s) = wc / (s + wc)
        self._comp = self._comp + a * (-self._sigma - self._comp)
        u = base + self._comp
        self._u_prev = u
        return u


# ======================================================================================================== sliding mode
class SlidingModeController(GeometricController):
    """Sliding-mode translational loop with a boundary layer.

    With the sliding variable ``s = ė + λ e`` the acceleration command is ``a_ref − λ ė − K tanh(s / φ)``. For any disturbance
    bounded by ``K`` the error enters the boundary layer ``|s| < φ`` in finite time and stays there, so tracking is robust to
    matched uncertainty. The price: ``K`` must exceed the disturbance acceleration (a 9 m/s wind needs about 4 m/s²; below that the
    error is large), and a small ``φ`` makes the command chatter.
    """

    def __init__(self, sim: Any, gains: Optional[Gains] = None, lam: Optional[float] = None, k_switch: float = 6.0, boundary: float = 0.8) -> None:
        super().__init__(sim, gains)
        self.lam = float(lam) if lam is not None else float(np.mean(self.B.to_numpy(self.wn_pos)))
        self.k_sw, self.phi = float(k_switch), float(boundary)

    def acceleration_command(self, state: Any, ref: Reference, dt: float, ep: Any, ev: Any) -> Any:
        B = self.B
        s = ev + self.lam * ep
        return ref.acc - self.lam * ev - self.k_sw * B.tanh(s / self.phi)
