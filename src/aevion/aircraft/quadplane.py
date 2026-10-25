"""Vertical-take-off-and-landing aircraft: a lift-plus-cruise quad-plane and a tilt-rotor variant, with a blended controller.

A quad-plane is a fixed-wing aircraft (the aerodynamics of :class:`~aevion.aircraft.fixedwing.FixedWing`) with four vertical lift
rotors and a pusher propeller. The tilt-rotor variant has no pusher: the same four rotors tilt from vertical (hover) to horizontal
(cruise) about the body y axis. Both are reduced-order: point thrust per rotor, first-order rotor and tilt-servo lags, no
rotor-wing interference, slipstream or gyroscopic torque.

:class:`QuadPlaneController` flies the whole envelope with one structure. The attitude loop computes the moments it needs and
allocates them to the surfaces in proportion to dynamic pressure (``λ`` rising from 0 to 1 between ``v_lo`` and ``v_hi``) and to
the rotors with ``1 - λ``. The rotor collective supplies the weight the wing does not carry (``m (g + a_z) - L_wing``), and the
pusher or tilt follows a rate-limited airspeed reference, so a transition is a ramp of the airspeed command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from ..core import rotation as rot
from .fixedwing import G0, AircraftParams, AircraftState, B, FixedWing

__all__ = ["QuadPlaneParams", "QuadPlane", "TiltRotorPlane", "QuadPlaneController", "TiltRotorController"]


@dataclass
class QuadPlaneParams:
    aero: AircraftParams = field(default_factory=lambda: AircraftParams(mass=12.5, Jx=1.0, Jy=1.3, Jz=2.1, Jxz=0.12))
    rotor_positions: Tuple[Tuple[float, float, float], ...] = (
        (0.55, -0.85, 0.0),
        (-0.55, 0.85, 0.0),
        (0.55, 0.85, 0.0),
        (-0.55, -0.85, 0.0),
    )  # FLU [m]
    rotor_spin: Tuple[float, ...] = (1.0, 1.0, -1.0, -1.0)
    thrust_max: float = 42.0  # per lift rotor [N]: 1.37 thrust-to-weight in hover
    kq_over_kt: float = 0.016  # reaction torque per unit thrust [m]
    tau_rotor: float = 0.06
    tilt_rate: float = math.radians(40.0)  # tilt servo speed [rad/s]
    parasite_cd_rotors_off: float = 0.012  # extra drag coefficient of the idle lift rotors and booms


class QuadPlane(FixedWing):
    """Fixed-wing aerodynamics plus four lift rotors and a pusher. Controls: ``(δe, δa, δr, δt, T₁, T₂, T₃, T₄)``."""

    def __init__(self, params: Optional[QuadPlaneParams] = None, n: int = 1, dt: float = 0.005, wind: Any = None) -> None:
        self.qp = params or QuadPlaneParams()
        super().__init__(self.qp.aero, n, dt, wind)
        self.n_controls = 8
        self._lo, self._hi, self._tau = self._actuator_limits()
        self.pos_r = np.asarray(self.qp.rotor_positions, float)
        self.spin = np.asarray(self.qp.rotor_spin, float)

    def _actuator_limits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi, tau = super()._actuator_limits()
        q = self.qp
        return (np.concatenate([lo, np.zeros(4)]), np.concatenate([hi, np.full(4, q.thrust_max)]), np.concatenate([tau, np.full(4, q.tau_rotor)]))

    def _rotor_wrench(self, T: np.ndarray, tilt: Any = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Force and moment of the lift rotors in body axes (thrust along ``[sin τ, 0, cos τ]``)."""
        tilt_arr = np.asarray(tilt, float) * np.ones(T.shape[0])
        axis = np.stack([np.sin(tilt_arr), np.zeros_like(tilt_arr), np.cos(tilt_arr)], axis=1)  # (n, 3)
        Ftot = T.sum(axis=1)[:, None] * axis
        # moment = sum_i r_i x (T_i axis) + reaction torque -s_i kq T_i axis
        M = np.zeros_like(Ftot)
        for i in range(T.shape[1]):
            f = T[:, i : i + 1] * axis
            M += np.cross(self.pos_r[i][None, :], f) - self.spin[i] * self.qp.kq_over_kt * f
        return Ftot, M

    def aero_only(self, vel: np.ndarray, quat: np.ndarray, omega: np.ndarray, u: np.ndarray, wind: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """The wing, tail and pusher (no lift rotors): what the fixed-wing part alone does."""
        F, M, info = super().forces_moments(vel, quat, omega, u[:, :4], wind)
        return F, M, info

    def forces_moments(
        self, vel: np.ndarray, quat: np.ndarray, omega: np.ndarray, u: np.ndarray, wind: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        F, M, info = self.aero_only(vel, quat, omega, u, wind)
        # idle rotors and booms add parasite drag along the airflow
        qbar = 0.5 * self.p.rho * info["airspeed"] ** 2 * self.p.S
        R = rot.to_matrix(B, quat)
        va = np.einsum("nji,nj->ni", R, vel - wind)
        spd = np.maximum(np.linalg.norm(va, axis=1), 1e-6)
        F = F - (qbar * self.qp.parasite_cd_rotors_off / spd)[:, None] * va
        Fr, Mr = self._rotor_wrench(np.clip(u[:, 4:8], 0.0, self.qp.thrust_max), self._tilt(u))
        info["rotor_thrust"] = np.clip(u[:, 4:8], 0.0, self.qp.thrust_max).sum(axis=1)
        return F + Fr, M + Mr, info

    def _tilt(self, u: np.ndarray) -> Any:
        return 0.0

    def hover_thrust(self) -> float:
        return self.p.mass * G0 / 4.0


class TiltRotorPlane(QuadPlane):
    """Four rotors that tilt from vertical (hover) to horizontal (cruise) about the body y axis; no pusher.

    Controls: ``(δe, δa, δr, δt [unused], T₁..T₄, τ)`` with the tilt command ``τ`` in ``[-0.6, π/2]`` rad (0 = vertical, thrust up;
    π/2 = horizontal, thrust forward; slightly negative tilts the thrust aft to brake).
    The tilt servo has a rate limit, modelled as the first-order lag of the other actuators plus ``tilt_rate`` clipping.
    """

    def __init__(self, params: Optional[QuadPlaneParams] = None, n: int = 1, dt: float = 0.005, wind: Any = None) -> None:
        super().__init__(params, n, dt, wind)
        self.n_controls = 9
        self._lo, self._hi, self._tau = self._actuator_limits()

    def _actuator_limits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi, tau = super()._actuator_limits()
        hi = hi.copy()
        hi[3] = 0.0  # no pusher
        return np.concatenate([lo, [-0.6]]), np.concatenate([hi, [math.pi / 2.0]]), np.concatenate([tau, [0.5]])  # tilting aft brakes

    def _tilt(self, u: np.ndarray) -> Any:
        return u[:, 8]


class QuadPlaneController:
    """One controller for hover, transition and cruise (see the module documentation).

    ``cruise_airspeed`` [m/s] is where the wing carries the aircraft alone; ``v_lo`` / ``v_hi`` bound the blending of the
    control surfaces against the rotors; ``accel`` is the longitudinal acceleration limit of the airspeed reference.
    """

    def __init__(
        self,
        model: QuadPlane,
        cruise_airspeed: float = 30.0,
        v_lo: float = 8.0,
        v_hi: float = 20.0,
        accel: float = 1.2,
        w_att: float = 9.0,
        zeta: float = 0.8,
        w_alt: float = 1.0,
        hover_pitch_deg: float = 0.0,
    ) -> None:
        self.m = model
        self.Va_cruise, self.v_lo, self.v_hi, self.accel = float(cruise_airspeed), float(v_lo), float(v_hi), float(accel)
        self.kp, self.kd = w_att**2, 2.0 * zeta * w_att
        self.ki = 0.6 * w_att**2
        self.kp_h, self.kd_h = w_alt**2, 2.0 * 0.9 * w_alt
        self.kp_psi = (0.6 * w_att) ** 2
        self.hover_pitch = math.radians(hover_pitch_deg)
        p = model.p
        # cruise pitch attitude (nose up) that makes the wing carry the weight, from the level-flight lift balance
        q = 0.5 * p.rho * cruise_airspeed**2 * p.S
        cl = p.mass * G0 / q
        self.theta_cruise = max((cl - p.C_L_0) / p.C_L_alpha, 0.0)
        self.tilt_max = math.pi / 2.0
        # rotor allocation matrix [sum T; tau_x; tau_y; tau_z] = A T
        pos, spin = model.pos_r, model.spin
        self.A = np.stack([np.ones(4), pos[:, 1], -pos[:, 0], -spin * model.qp.kq_over_kt])
        self.Apinv = np.linalg.pinv(self.A)
        self.reset(model.n)

    def reset(self, n: Optional[int] = None) -> None:
        n = n or self.m.n
        self.Va_ref = np.zeros(n)
        self.i_V = np.zeros(n)
        self.i_att = np.zeros(
            (n, 2)
        )  # roll and pitch integrators (angular acceleration): they remove the offset that the wing's pitch stiffness leaves
        self.started = False

    def _lambda(self, Va: np.ndarray) -> np.ndarray:
        return np.clip((Va - self.v_lo) / (self.v_hi - self.v_lo), 0.0, 1.0)

    def update(self, st: AircraftState, heading: Any, altitude: Any, airspeed: Any, dt: Optional[float] = None) -> np.ndarray:
        m, p = self.m, self.m.p
        dt = m.dt if dt is None else dt
        n = st.n
        one = np.ones(n)
        psi_c, h_c, Va_c = np.asarray(heading, float) * one, np.asarray(altitude, float) * one, np.asarray(airspeed, float) * one
        Va, alpha, beta = m.airspeed_alpha_beta(st)
        if not self.started:
            self.Va_ref = Va.copy()
            self.started = True
        self.Va_ref = self.Va_ref + np.clip(Va_c - self.Va_ref, -self.accel * dt, self.accel * dt)  # rate-limited reference
        eul = m.euler(st)
        phi, theta_up, psi = eul[:, 0], -eul[:, 1], eul[:, 2]
        pr, q_up, r = st.omega[:, 0], -st.omega[:, 1], st.omega[:, 2]
        # attitude loop: angular accelerations -> body moments (FLU)
        s = np.clip((Va - self.v_lo) / max(self.Va_cruise - self.v_lo, 1e-6), 0.0, 1.0) ** 2
        theta_c = self.hover_pitch + (self.theta_cruise - self.hover_pitch) * s
        self.i_att[:, 0] = np.clip(self.i_att[:, 0] + (0.0 - phi) * dt, -0.15, 0.15)
        self.i_att[:, 1] = np.clip(self.i_att[:, 1] + (theta_c - theta_up) * dt, -0.15, 0.15)
        alpha_phi = self.kp * (0.0 - phi) - self.kd * pr + self.ki * self.i_att[:, 0]
        alpha_th = self.kp * (theta_c - theta_up) - self.kd * q_up + self.ki * self.i_att[:, 1]
        alpha_psi = self.kp_psi * ((psi_c - psi + math.pi) % (2.0 * math.pi) - math.pi) - 0.7 * self.kd * r
        lam = self._lambda(Va)
        tau_z_lim = 0.25 * m.qp.kq_over_kt * m.qp.thrust_max * 2.0 + lam * 50.0  # the rotors' yaw authority is small: do not ask for more
        tau = np.stack(
            [p.Jx * alpha_phi, -p.Jy * alpha_th, np.clip(p.Jz * alpha_psi, -tau_z_lim, tau_z_lim)], axis=1
        )  # FLU: a nose-up moment is negative about +y
        qS = 0.5 * p.rho * np.maximum(Va, 3.0) ** 2 * p.S
        da = lam * tau[:, 0] / (qS * p.b * p.C_l_delta_a)
        de = lam * (-tau[:, 1]) / (qS * p.c * p.C_m_delta_e)
        dr = lam * (-tau[:, 2]) / (qS * p.b * p.C_n_delta_r)
        # vertical force: the rotors supply what the wing does not
        h, vz = st.pos[:, 2], st.vel[:, 2]
        a_z = np.clip(
            self.kp_h * (h_c - h) - self.kd_h * vz, -1.5, 1.5
        )  # a thrust-to-weight of 1.4 leaves little spare thrust: keep attitude authority
        u_probe = st.controls.copy()
        u_probe[:, 4:8] = 0.0
        F_aero, _, _ = m.forces_moments(st.vel, st.quat, st.omega, u_probe, m._wind_at(st.t, st.pos))
        R = rot.to_matrix(B, st.quat)
        Fz_world = np.einsum("nij,nj->ni", R, F_aero)[:, 2]  # vertical force of everything except the lift rotors
        Rzz = np.maximum(R[:, 2, 2], 0.5)
        return self._allocate(st, tau, lam, da, de, dr, a_z, Fz_world, Rzz, Va, dt)

    def _desaturate(self, wrench: np.ndarray) -> np.ndarray:
        """Rotor thrusts for a wrench within their limits, with priorities: the collective first, then roll and pitch, then yaw.

        The roll/pitch part is scaled down (never below zero) until every thrust fits, and the yaw part gets whatever room is left."""
        Tmax = self.m.qp.thrust_max
        n = wrench.shape[0]
        base = np.maximum(wrench[:, 0:1], 0.0).clip(0.0, 3.6 * Tmax) / 4.0 * np.ones((1, 4))  # keep 10 % of the collective as attitude margin
        z = np.zeros((n, 1))
        rp = np.concatenate([z, wrench[:, 1:3], z], axis=1) @ self.Apinv.T
        yaw = np.concatenate([z, z, z, wrench[:, 3:4]], axis=1) @ self.Apinv.T

        def fit(cur: np.ndarray, part: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                room = np.where(part > 0.0, (Tmax - cur) / part, np.where(part < 0.0, -cur / part, np.inf))
            return np.clip(np.min(room, axis=1, keepdims=True), 0.0, 1.0)

        s_rp = fit(base, rp)
        cur = base + s_rp * rp
        s_y = fit(cur, yaw)
        return np.clip(cur + s_y * yaw, 0.0, Tmax)

    def _allocate(
        self,
        st: AircraftState,
        tau: np.ndarray,
        lam: np.ndarray,
        da: np.ndarray,
        de: np.ndarray,
        dr: np.ndarray,
        a_z: np.ndarray,
        Fz_world: np.ndarray,
        Rzz: np.ndarray,
        Va: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        m, p = self.m, self.m.p
        T_tot = np.maximum(p.mass * (G0 + a_z) - Fz_world, 0.0) / Rzz
        wrench = np.stack([T_tot, (1.0 - lam) * tau[:, 0], (1.0 - lam) * tau[:, 1], (1.0 - lam) * tau[:, 2]], axis=1)
        T = self._desaturate(wrench)
        # pusher: airspeed reference tracking with a drag feed-forward
        e_V = self.Va_ref - Va
        self.i_V = np.clip(self.i_V + e_V * dt, -20.0, 20.0)
        drag = (
            0.5 * p.rho * self.Va_ref**2 * p.S * (p.C_D_p + 0.02)
        )  # drag at the *reference* speed: using the measured speed would feed speed back positively
        thrust_ff = drag + p.mass * np.clip(
            self.accel * np.sign(e_V) * (np.abs(e_V) > 0.3), -self.accel, self.accel
        )  # drag plus the force of the reference acceleration
        dt_cmd = np.sqrt(np.maximum(2.0 * thrust_ff / (p.rho * p.S_prop * p.C_prop) + Va**2, 0.0)) / p.k_motor
        dt_cmd = np.clip(dt_cmd + 0.04 * e_V + 0.01 * self.i_V, 0.0, 1.0)
        u = np.zeros((st.n, m.n_controls))
        u[:, 0], u[:, 1], u[:, 2], u[:, 3] = de, da, dr, dt_cmd
        u[:, 4:8] = T
        return u


class TiltRotorController(QuadPlaneController):
    """The same blended controller for a tilt-rotor: with no pusher the *rotors* provide the vertical and the forward force.

    The vertical force the wing does not carry ``T_z`` and the forward force needed to follow the airspeed reference
    ``F_x = m a_cmd + D`` give the rotor thrust ``T = sqrt(T_z² + F_x²)`` and the tilt ``τ = atan2(F_x, T_z)``: vertical in hover,
    rotating forward as the aircraft accelerates, horizontal in cruise (where ``T_z -> 0``).
    """

    def _allocate(
        self,
        st: AircraftState,
        tau: np.ndarray,
        lam: np.ndarray,
        da: np.ndarray,
        de: np.ndarray,
        dr: np.ndarray,
        a_z: np.ndarray,
        Fz_world: np.ndarray,
        Rzz: np.ndarray,
        Va: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        m, p = self.m, self.m.p
        T_z = np.maximum(p.mass * (G0 + a_z) - Fz_world, 0.0) / Rzz
        e_V = self.Va_ref - Va
        a_cmd = np.clip(0.8 * e_V, -self.accel, self.accel)
        drag = 0.5 * p.rho * self.Va_ref**2 * p.S * (p.C_D_p + 0.02 + self.m.qp.parasite_cd_rotors_off)  # at the reference speed (see above)
        F_x = p.mass * a_cmd + drag
        T_tot = np.sqrt(T_z**2 + F_x**2)
        tilt = np.clip(np.arctan2(F_x, np.maximum(T_z, 1e-3)), -0.6, math.pi / 2.0)
        wrench = np.stack([T_tot, (1.0 - lam) * tau[:, 0], (1.0 - lam) * tau[:, 1], (1.0 - lam) * tau[:, 2]], axis=1)
        T = self._desaturate(wrench)
        u = np.zeros((st.n, m.n_controls))
        u[:, 0], u[:, 1], u[:, 2] = de, da, dr
        u[:, 4:8] = T
        u[:, 8] = tilt
        return u
