"""A six-degree-of-freedom fixed-wing aircraft with stability-derivative aerodynamics.

The model follows Beard & McLain, *Small Unmanned Aircraft* (2012), chapter 4: forces and moments are built from coefficients that
are linear in angle of attack ``α``, sideslip ``β``, body rates and surface deflections, with a smooth transition to flat-plate lift
beyond stall, plus a propeller thrust model and a full inertia tensor (including ``J_xz``). The default parameters are the
Aerosonde-like UAV of the book's Appendix E.

Conventions: world ENU, body FLU. The aerodynamic formulas use the textbook forward-right-down axes and are converted at the boundary
(``diag(1, -1, -1)``), so the coefficient signs in :class:`AircraftParams` are the published ones. Controls are
``(δe, δa, δr, δt)``: elevator, aileron and rudder deflections [rad] (positive elevator pitches the nose down) and throttle in
``[0, 1]``, with first-order actuator lags and limits.

.. note::
   Reduced-order model: no ground effect, propeller torque or slipstream, and the ground is a plane that stops the aircraft
   (``crashed``) rather than a landing-gear model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core import rotation as rot
from ..core.backend import NumpyBackend

__all__ = ["AircraftParams", "AircraftState", "FixedWing", "AEROSONDE"]

B = NumpyBackend()
G0 = 9.80665
T_FRD = np.diag([1.0, -1.0, -1.0])  # FLU <-> FRD (its own inverse)


@dataclass
class AircraftParams:
    """Mass, geometry and non-dimensional coefficients (Beard & McLain 2012, Appendix E.1: an Aerosonde-like UAV)."""

    mass: float = 11.0
    Jx: float = 0.8244
    Jy: float = 1.135
    Jz: float = 1.759
    Jxz: float = 0.1204
    S: float = 0.55  # wing area [m^2]
    b: float = 2.8956  # span [m]
    c: float = 0.18994  # mean chord [m]
    S_prop: float = 0.2027
    k_motor: float = 80.0
    C_prop: float = 1.0
    e: float = 0.9  # Oswald efficiency
    rho: float = 1.2682
    # lift, drag, pitch
    C_L_0: float = 0.23
    C_D_0: float = 0.043
    C_m_0: float = 0.0135
    C_L_alpha: float = 5.61
    C_D_alpha: float = 0.030
    C_m_alpha: float = -2.74
    C_L_q: float = 7.95
    C_D_q: float = 0.0
    C_m_q: float = -38.21
    C_L_delta_e: float = 0.13
    C_D_delta_e: float = 0.0135
    C_m_delta_e: float = -0.99
    M_stall: float = 50.0  # blending rate of the stall model
    alpha0: float = 0.4712  # stall angle [rad]
    C_D_p: float = 0.043  # parasitic drag of the nonlinear polar (the book lists 0 and uses C_D0 only in its linear model; the polar needs it)
    # lateral
    C_Y_0: float = 0.0
    C_l_0: float = 0.0
    C_n_0: float = 0.0
    C_Y_beta: float = -0.83
    C_l_beta: float = -0.13
    C_n_beta: float = 0.073
    C_Y_p: float = 0.0
    C_l_p: float = -0.51
    C_n_p: float = -0.069
    C_Y_r: float = 0.0
    C_l_r: float = 0.25
    C_n_r: float = -0.095
    C_Y_delta_a: float = 0.075
    C_l_delta_a: float = 0.17
    C_n_delta_a: float = -0.011
    C_Y_delta_r: float = 0.19
    C_l_delta_r: float = 0.0024
    C_n_delta_r: float = -0.069
    # actuators
    tau_surface: float = 0.06
    tau_throttle: float = 0.25
    delta_max: float = math.radians(30.0)

    @property
    def AR(self) -> float:
        return self.b**2 / self.S

    @property
    def inertia(self) -> np.ndarray:
        """Inertia tensor in FLU axes (the book's FRD tensor transformed)."""
        J = np.array([[self.Jx, 0.0, -self.Jxz], [0.0, self.Jy, 0.0], [-self.Jxz, 0.0, self.Jz]])
        return T_FRD @ J @ T_FRD

    def replace(self, **kw: Any) -> "AircraftParams":
        return replace(self, **kw)

    def to_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


AEROSONDE = AircraftParams()


@dataclass
class AircraftState:
    """State of ``n`` aircraft. ``controls`` are the actual (lagged) surface deflections and throttle."""

    pos: np.ndarray
    vel: np.ndarray  # world velocity [m/s]
    quat: np.ndarray  # body (FLU) -> world (ENU), [w, x, y, z]
    omega: np.ndarray  # body rates (FLU) [rad/s]
    controls: np.ndarray  # (n, 4): delta_e, delta_a, delta_r, delta_t
    t: float = 0.0
    crashed: Optional[np.ndarray] = None
    # diagnostics of the last evaluation
    airspeed: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    acc: Optional[np.ndarray] = None
    f_body: Optional[np.ndarray] = None
    thrust: Optional[np.ndarray] = None
    soc: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def copy(self) -> "AircraftState":
        return AircraftState(
            np.array(self.pos, copy=True),
            np.array(self.vel, copy=True),
            np.array(self.quat, copy=True),
            np.array(self.omega, copy=True),
            np.array(self.controls, copy=True),
            t=self.t,
            crashed=None if self.crashed is None else self.crashed.copy(),
        )


class FixedWing:
    """Batched fixed-wing dynamics (``n`` aircraft) integrated with RK4.

    ``wind`` is a world-frame wind vector ``(3,)`` or ``(n, 3)`` [m/s], or a callable ``wind(t, pos) -> (n, 3)``.
    """

    def __init__(self, params: Optional[AircraftParams] = None, n: int = 1, dt: float = 0.01, wind: Any = None) -> None:
        self.p = params or AircraftParams()
        self.n, self.dt = int(n), float(dt)
        self.wind = wind
        self.B = B
        self.J = self.p.inertia
        self.Jinv = np.linalg.inv(self.J)
        self.ground_z = 0.0
        self.n_controls = 4
        self.mass = self.p.mass
        self._lo, self._hi, self._tau = self._actuator_limits()

    def _actuator_limits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lower and upper limits and first-order time constants of every control channel."""
        p = self.p
        lo = np.array([-p.delta_max, -p.delta_max, -p.delta_max, 0.0])
        hi = np.array([p.delta_max, p.delta_max, p.delta_max, 1.0])
        return lo, hi, np.array([p.tau_surface] * 3 + [p.tau_throttle])

    # ---------------------------------------------------------------------------------------- set-up
    def reset(
        self,
        pos: Any = (0.0, 0.0, 100.0),
        airspeed: float = 35.0,
        heading: float = 0.0,
        euler: Optional[Any] = None,
        flight_path: float = 0.0,
        controls: Any = None,
    ) -> AircraftState:
        """Straight flight at ``airspeed`` (m/s) on ``heading`` (radians counter-clockwise from east, the ENU yaw) with the
        given flight-path angle; ``euler`` = (roll, pitch, yaw) overrides the attitude (pitch defaults to the flight-path angle)."""
        n = self.n
        pos = np.ones((n, 1)) * np.asarray(pos, float)
        if euler is None:
            eul = np.array([0.0, flight_path, heading])
        else:
            eul = np.asarray(euler, float)
        q = rot.from_euler(B, np.full(n, eul[0]), np.full(n, eul[1]), np.full(n, eul[2]))
        vb = np.array([airspeed, 0.0, 0.0])
        vel = rot.rotate(B, q, np.ones((n, 1)) * vb)
        u = np.ones((n, self.n_controls)) * (np.zeros(self.n_controls) if controls is None else np.asarray(controls, float))
        st = AircraftState(pos, vel, q, np.zeros((n, 3)), u, 0.0, np.zeros(n, bool))
        self._diagnose(st, u)
        return st

    # ---------------------------------------------------------------------------------------- aerodynamics
    def _wind_at(self, t: float, pos: np.ndarray) -> np.ndarray:
        if self.wind is None:
            return np.zeros_like(pos)
        if callable(self.wind):
            return np.asarray(self.wind(t, pos), float)
        return np.ones((pos.shape[0], 1)) * np.asarray(self.wind, float)

    def sigma(self, alpha: np.ndarray) -> np.ndarray:
        p = self.p
        a, m = p.alpha0, p.M_stall
        e1, e2 = np.exp(np.clip(-m * (alpha - a), -50, 50)), np.exp(np.clip(m * (alpha + a), -50, 50))
        return (1.0 + e1 + e2) / ((1.0 + e1) * (1.0 + e2))

    def C_L(self, alpha: np.ndarray) -> np.ndarray:
        p, s = self.p, self.sigma(alpha)
        return (1.0 - s) * (p.C_L_0 + p.C_L_alpha * alpha) + s * (2.0 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))

    def C_D(self, alpha: np.ndarray) -> np.ndarray:
        p = self.p
        return p.C_D_p + (p.C_L_0 + p.C_L_alpha * alpha) ** 2 / (math.pi * p.e * p.AR)

    def forces_moments(
        self, vel: np.ndarray, quat: np.ndarray, omega: np.ndarray, u: np.ndarray, wind: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Aerodynamic + propulsive force [N] and moment [N m] in the FLU body axes (gravity excluded)."""
        p = self.p
        R = rot.to_matrix(B, quat)
        va_flu = np.einsum("nji,nj->ni", R, vel - wind)  # air-relative velocity in body axes
        va_frd = va_flu @ T_FRD
        Va = np.maximum(np.linalg.norm(va_frd, axis=1), 0.5)
        alpha = np.arctan2(va_frd[:, 2], va_frd[:, 0])
        beta = np.arcsin(np.clip(va_frd[:, 1] / Va, -1.0, 1.0))
        pqr = omega @ T_FRD
        pr, qr, rr = pqr[:, 0], pqr[:, 1], pqr[:, 2]
        de, da, dr, dt = u[:, 0], u[:, 1], u[:, 2], np.clip(u[:, 3], 0.0, 1.0)
        qbar = 0.5 * p.rho * Va**2 * p.S
        cq, bq = p.c / (2.0 * Va), p.b / (2.0 * Va)
        CL = self.C_L(alpha) + p.C_L_q * cq * qr + p.C_L_delta_e * de
        CD = self.C_D(alpha) + p.C_D_q * cq * qr + p.C_D_delta_e * de
        lift, drag = qbar * CL, qbar * CD
        fx = -drag * np.cos(alpha) + lift * np.sin(alpha)
        fz = -drag * np.sin(alpha) - lift * np.cos(alpha)
        fy = qbar * (p.C_Y_0 + p.C_Y_beta * beta + p.C_Y_p * bq * pr + p.C_Y_r * bq * rr + p.C_Y_delta_a * da + p.C_Y_delta_r * dr)
        thrust = 0.5 * p.rho * p.S_prop * p.C_prop * ((p.k_motor * dt) ** 2 - Va**2)
        thrust = np.where(dt > 0.0, thrust, 0.0)
        ell = qbar * p.b * (p.C_l_0 + p.C_l_beta * beta + p.C_l_p * bq * pr + p.C_l_r * bq * rr + p.C_l_delta_a * da + p.C_l_delta_r * dr)
        mm = qbar * p.c * (p.C_m_0 + p.C_m_alpha * alpha + p.C_m_q * cq * qr + p.C_m_delta_e * de)
        nn = qbar * p.b * (p.C_n_0 + p.C_n_beta * beta + p.C_n_p * bq * pr + p.C_n_r * bq * rr + p.C_n_delta_a * da + p.C_n_delta_r * dr)
        F_frd = np.stack([fx + thrust, fy, fz], axis=1)
        M_frd = np.stack([ell, mm, nn], axis=1)
        info = dict(airspeed=Va, alpha=alpha, beta=beta, thrust=thrust, lift=lift, drag=drag, CL=CL)
        return F_frd @ T_FRD, M_frd @ T_FRD, info

    # ---------------------------------------------------------------------------------------- dynamics
    def derivatives(self, t: float, pos: np.ndarray, vel: np.ndarray, quat: np.ndarray, omega: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, ...]:
        wind = self._wind_at(t, pos)
        F, M, _ = self.forces_moments(vel, quat, omega, u, wind)
        R = rot.to_matrix(B, quat)
        acc = np.einsum("nij,nj->ni", R, F) / self.mass + np.array([0.0, 0.0, -G0])
        Jw = omega @ self.J.T
        omdot = (M - np.cross(omega, Jw)) @ self.Jinv.T
        return vel, acc, rot.deriv(B, quat, omega), omdot

    def _diagnose(self, st: AircraftState, u: np.ndarray) -> None:
        wind = self._wind_at(st.t, st.pos)
        F, M, info = self.forces_moments(st.vel, st.quat, st.omega, u, wind)
        R = rot.to_matrix(B, st.quat)
        st.airspeed, st.alpha, st.beta, st.thrust = info["airspeed"], info["alpha"], info["beta"], info["thrust"]
        st.f_body = F / self.mass
        st.acc = np.einsum("nij,nj->ni", R, st.f_body) + np.array([0.0, 0.0, -G0])

    def step(self, st: AircraftState, command: Any) -> AircraftState:
        """Advance one ``dt``. ``command`` is ``(n, 4)`` or ``(4,)`` of ``(δe, δa, δr, δt)``."""
        dt = self.dt
        cmd = np.clip(np.ones((self.n, 1)) * np.asarray(command, float), self._lo, self._hi)
        u = st.controls + (1.0 - np.exp(-dt / self._tau)) * (cmd - st.controls)  # exact first-order lag over the step
        x = (st.pos, st.vel, st.quat, st.omega)

        def f(y: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
            return self.derivatives(st.t, y[0], y[1], y[2], y[3], u)

        def add(y: Tuple[np.ndarray, ...], d: Tuple[np.ndarray, ...], h: float) -> Tuple[np.ndarray, ...]:
            return tuple(a + h * b for a, b in zip(y, d))

        k1 = f(x)
        k2 = f(add(x, k1, 0.5 * dt))
        k3 = f(add(x, k2, 0.5 * dt))
        k4 = f(add(x, k3, dt))
        new = tuple(a + dt / 6.0 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4))
        pos, vel, quat, omega = new
        quat = rot.qnormalize(B, quat)
        prev = st.crashed.copy() if st.crashed is not None else np.zeros(self.n, bool)
        crashed = prev | (pos[:, 2] <= self.ground_z)  # touching the ground stops the aircraft (no landing-gear model)
        on_ground = np.stack([pos[:, 0], pos[:, 1], np.full(self.n, self.ground_z)], axis=1)
        pos = np.where(crashed[:, None], np.where(prev[:, None], st.pos, on_ground), pos)
        vel = np.where(crashed[:, None], 0.0, vel)
        omega = np.where(crashed[:, None], 0.0, omega)
        out = AircraftState(pos, vel, quat, omega, u, st.t + dt, crashed)
        self._diagnose(out, u)
        return out

    # ---------------------------------------------------------------------------------------- helpers
    def euler(self, st: AircraftState) -> np.ndarray:
        """(roll, pitch, yaw) in the ENU/FLU convention [rad]."""
        return rot.to_euler(B, st.quat)

    def airspeed_alpha_beta(self, st: AircraftState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        wind = self._wind_at(st.t, st.pos)
        _, _, info = self.forces_moments(st.vel, st.quat, st.omega, st.controls, wind)
        return info["airspeed"], info["alpha"], info["beta"]

    def course(self, st: AircraftState) -> np.ndarray:
        """Ground course, counter-clockwise from east [rad]."""
        return np.arctan2(st.vel[:, 1], st.vel[:, 0])

    def flight_path_angle(self, st: AircraftState) -> np.ndarray:
        return np.arcsin(np.clip(st.vel[:, 2] / np.maximum(np.linalg.norm(st.vel, axis=1), 1e-9), -1, 1))

    def total_energy(self, st: AircraftState) -> np.ndarray:
        """Specific total energy ``g h + v²/2`` [J/kg] (conserved without drag and thrust)."""
        return G0 * st.pos[:, 2] + 0.5 * np.sum(st.vel**2, axis=1)
