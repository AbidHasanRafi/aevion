"""Successive-loop-closure autopilot and path following for a fixed-wing aircraft.

The structure follows Beard & McLain (2012, chapter 6): inner loops stabilise attitude and outer loops command the inner ones, each
designed on a low-order model and separated from the next by about a decade of bandwidth::

    course χ → bank φ_c → aileron (PD on roll rate)      airspeed Va → throttle (PI + trim feed-forward)
    altitude h → pitch θ_c → elevator (PD on pitch rate)  yaw damper: rudder from yaw rate (Dutch-roll damping)

Gains are computed, not hand-tuned: :class:`Autopilot` derives the roll, pitch and airspeed model coefficients from the aircraft's
aerodynamic derivatives at the trim point and places the closed-loop poles for the requested natural frequency and damping.
Angles follow the package's ENU/FLU conventions (courses counter-clockwise from east; nose-up pitch is ``-θ`` of the Euler convention).
:class:`PathFollower` tracks a polyline with a cross-track-error law (Nelson et al. 2007) and half-plane segment switching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .fixedwing import AircraftState, FixedWing
from .trim import TrimResult

__all__ = ["Autopilot", "PathFollower", "AutopilotGains", "wrap_pi"]


def wrap_pi(a: Any) -> Any:
    return (np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class AutopilotGains:
    kp_phi: float
    kd_phi: float
    kp_chi: float
    ki_chi: float
    kp_theta: float
    kd_theta: float
    kp_h: float
    ki_h: float
    kp_V: float
    ki_V: float
    k_yaw_damper: float
    phi_max: float
    theta_max: float
    trim_throttle: float


class Autopilot:
    """Attitude, course, altitude and airspeed hold for ``n`` aircraft about a trim condition.

    Bandwidths (rad/s): ``w_roll`` of the bank loop, ``w_pitch`` of the pitch loop (default: 30 % above the airframe's short-period
    frequency), ``w_V`` of the airspeed loop and ``w_alt`` of the altitude loop; the course loop is ``separation`` times slower than
    the bank loop it commands. ``zeta`` is the design damping ratio.
    """

    def __init__(
        self,
        model: FixedWing,
        trim: TrimResult,
        w_roll: float = 7.0,
        w_pitch: Optional[float] = None,
        w_V: float = 0.9,
        w_alt: float = 0.25,
        zeta: float = 0.8,
        separation: float = 8.0,
        zeta_outer: float = 0.9,
        phi_max_deg: float = 40.0,
        theta_max_deg: float = 25.0,
        yaw_damper: float = 0.5,
    ) -> None:
        self.m = model
        self.trim = trim
        p, Va = model.p, trim.airspeed
        rho, S, b, c = p.rho, p.S, p.b, p.c
        qS = 0.5 * rho * Va**2 * S
        # roll: phi'' = -a1 phi' + bphi * delta_a  (FRD roll, positive right wing down, equals the Euler roll of this package)
        a_phi1 = -qS * b * (b / (2.0 * Va)) * p.C_l_p / p.Jx
        b_phi = qS * b * p.C_l_delta_a / p.Jx
        if 2.0 * zeta * w_roll > a_phi1:  # the usual second-order design (pole placement)
            kp_phi = w_roll**2 / b_phi
            kd_phi = (2.0 * zeta * w_roll - a_phi1) / b_phi
        else:  # the roll subsidence is faster than the requested loop: treat roll rate as a lag and close a first-order loop
            kp_phi = w_roll * a_phi1 / b_phi
            kd_phi = 0.0
        w_chi = w_roll / separation
        kp_chi = 2.0 * zeta_outer * w_chi * Va / 9.80665
        ki_chi = w_chi**2 * Va / 9.80665
        # pitch (nose-up attitude): th'' = -a1 th' - a2 th + a3 delta_e
        a_th1 = -(rho * Va**2 * c * S / (2.0 * p.Jy)) * p.C_m_q * c / (2.0 * Va)
        a_th2 = -(rho * Va**2 * c * S / (2.0 * p.Jy)) * p.C_m_alpha
        a_th3 = (rho * Va**2 * c * S / (2.0 * p.Jy)) * p.C_m_delta_e
        if w_pitch is None:  # the pitch loop must be faster than the airframe's own short-period oscillation
            w_pitch = 1.3 * math.sqrt(max(a_th2, 1.0))
        self.w_pitch = w_pitch
        kp_th = (w_pitch**2 - a_th2) / a_th3
        kd_th = (2.0 * zeta * w_pitch - a_th1) / a_th3
        k_th_dc = kp_th * a_th3 / (a_th2 + kp_th * a_th3)
        w_h = w_alt  # the altitude loop is slow (it must stay below the airspeed/energy dynamics), not tied to the pitch bandwidth
        kp_h = 2.0 * zeta_outer * w_h / (k_th_dc * Va)
        ki_h = w_h**2 / (k_th_dc * Va)
        # airspeed from throttle: dV = -aV1 dV + aV2 d_delta_t
        dt_star = float(trim.controls[3])
        a_V1 = (
            rho * Va * S / p.mass * (p.C_D_0 + p.C_D_alpha * trim.alpha + p.C_D_delta_e * float(trim.controls[0]))
            + rho * p.S_prop * p.C_prop * Va / p.mass
        )
        a_V2 = rho * p.S_prop * p.C_prop * p.k_motor**2 * dt_star / p.mass
        kp_V = (2.0 * zeta_outer * w_V - a_V1) / a_V2
        ki_V = w_V**2 / a_V2
        self.g = AutopilotGains(
            kp_phi,
            kd_phi,
            kp_chi,
            ki_chi,
            kp_th,
            kd_th,
            kp_h,
            ki_h,
            kp_V,
            ki_V,
            yaw_damper,
            math.radians(phi_max_deg),
            math.radians(theta_max_deg),
            dt_star,
        )
        self.coefficients = dict(a_phi1=a_phi1, b_phi=b_phi, a_th1=a_th1, a_th2=a_th2, a_th3=a_th3, a_V1=a_V1, a_V2=a_V2)
        self.reset(model.n)

    def reset(self, n: Optional[int] = None) -> None:
        n = n or self.m.n
        self.i_chi = np.zeros(n)
        self.i_h = np.zeros(n)
        self.i_V = np.zeros(n)
        self.theta_c = np.zeros(n)
        self.phi_c = np.zeros(n)

    def update(self, st: AircraftState, course: Any, altitude: Any, airspeed: Any, dt: Optional[float] = None) -> np.ndarray:
        """Controls ``(δe, δa, δr, δt)`` for the commanded ground course [rad], altitude [m] and airspeed [m/s] (``(n,)`` or scalars)."""
        g, dt = self.g, (self.m.dt if dt is None else dt)
        n = st.n
        one = np.ones(n)
        chi_c, h_c, Va_c = np.asarray(course, float) * one, np.asarray(altitude, float) * one, np.asarray(airspeed, float) * one
        Va, _, _ = self.m.airspeed_alpha_beta(st)
        eul = self.m.euler(st)
        phi = eul[:, 0]
        theta_up = -eul[:, 1]
        p, q_up, r_frd = st.omega[:, 0], -st.omega[:, 1], -st.omega[:, 2]
        chi = self.m.course(st)
        # course -> bank (a left turn, a positive course error, needs a negative roll angle)
        e_chi = wrap_pi(chi_c - chi)
        # conditional integration: no wind-up while the bank limit is active or the error is large
        free_chi = (np.abs(self.phi_c) < 0.9 * g.phi_max) & (np.abs(e_chi) < math.radians(15.0))
        self.i_chi = np.where(
            free_chi, np.clip(self.i_chi + e_chi * dt, -0.5 * g.phi_max / max(g.ki_chi, 1e-9), 0.5 * g.phi_max / max(g.ki_chi, 1e-9)), self.i_chi
        )
        self.phi_c = np.clip(-(g.kp_chi * e_chi + g.ki_chi * self.i_chi), -g.phi_max, g.phi_max)
        delta_a = np.clip(g.kp_phi * (self.phi_c - phi) - g.kd_phi * p, -self.m.p.delta_max, self.m.p.delta_max)
        # altitude -> pitch
        e_h = h_c - st.pos[:, 2]
        lim = 0.5 * g.theta_max / max(g.ki_h, 1e-9)
        free_h = np.abs(e_h) < 12.0  # integrate only near the target: a big altitude step must not wind the integrator up
        self.i_h = np.where(free_h, np.clip(self.i_h + e_h * dt, -lim, lim), self.i_h)
        d_theta = np.clip(g.kp_h * e_h + g.ki_h * self.i_h, -g.theta_max, g.theta_max)
        self.theta_c = d_theta + self.trim.pitch_up
        delta_e = np.clip(g.kp_theta * (self.theta_c - theta_up) - g.kd_theta * q_up, -self.m.p.delta_max, self.m.p.delta_max)
        # airspeed -> throttle (with the trim throttle as feed-forward)
        e_V = Va_c - Va
        self.i_V = np.clip(self.i_V + e_V * dt, -0.5 / max(g.ki_V, 1e-9), 0.5 / max(g.ki_V, 1e-9))
        # feed-forward of the climb power: a climb angle gamma needs extra thrust m g gamma, which removes most of the altitude-airspeed coupling
        ff = self.m.p.mass * 9.80665 * d_theta / max(self.m.p.rho * self.m.p.S_prop * self.m.p.C_prop * self.m.p.k_motor**2 * g.trim_throttle, 1e-9)
        delta_t = np.clip(g.trim_throttle + ff + g.kp_V * e_V + g.ki_V * self.i_V, 0.0, 1.0)
        delta_r = np.clip(g.k_yaw_damper * r_frd, -self.m.p.delta_max, self.m.p.delta_max)
        return np.stack([delta_e, delta_a, delta_r, delta_t], axis=1)


class PathFollower:
    """Follows a polyline of 3-D waypoints at constant airspeed: course from the cross-track error, altitude from the segment.

    ``chi_inf`` is the approach angle to the line far away (radians); ``k_path`` how fast it is reduced with the cross-track
    error; a segment ends when the aircraft crosses the half-plane through its end point (the straight-line-path algorithm of
    Beard & McLain, algorithm 5). With ``turn_radius > 0`` the switch happens ``R tan(Δχ/2)`` *before* the corner, so a turn of
    that radius rolls out on the next leg instead of overshooting it.
    """

    def __init__(
        self,
        waypoints: Any,
        airspeed: float = 35.0,
        chi_inf: float = math.radians(60.0),
        k_path: float = 0.015,
        loop: bool = False,
        turn_radius: float = 0.0,
    ) -> None:
        self.turn_radius = float(turn_radius)
        self.W = np.asarray(waypoints, float)
        if self.W.ndim != 2 or self.W.shape[1] != 3 or len(self.W) < 2:
            raise ValueError("need at least two 3-D waypoints")
        self.Va, self.chi_inf, self.k, self.loop = float(airspeed), float(chi_inf), float(k_path), bool(loop)
        self.i = 0
        self.finished = False

    def reset(self) -> None:
        self.i, self.finished = 0, False

    def _segment(self) -> Tuple[np.ndarray, np.ndarray]:
        j = (self.i + 1) % len(self.W)
        return self.W[self.i], self.W[j]

    def command(self, pos: np.ndarray) -> Tuple[float, float, float]:
        """Course [rad], altitude [m] and airspeed for the position ``(3,)`` of one aircraft; advances the segment pointer."""
        a, b = self._segment()
        d = b - a
        L = float(np.linalg.norm(d[:2]))
        if L < 1e-9:
            self.i = (self.i + 1) % len(self.W)
            return self.command(pos)
        if float(np.dot(pos[:2] - b[:2] + self._lead(a, b) * d[:2] / L, d[:2])) > 0.0:  # beyond the half-plane of the end point: next segment
            if self.i + 2 < len(self.W) or self.loop:
                self.i = (self.i + 1) % len(self.W)
                a, b = self._segment()
                d, L = b - a, float(np.linalg.norm((b - a)[:2]))
            else:
                self.finished = True
        chi_path = math.atan2(d[1], d[0])
        # cross-track error: positive to the left of the path (counter-clockwise side)
        rel = pos[:2] - a[:2]
        e_y = float(-rel[0] * math.sin(chi_path) + rel[1] * math.cos(chi_path))
        chi_c = chi_path - self.chi_inf * (2.0 / math.pi) * math.atan(self.k * e_y)
        s = float(np.clip(np.dot(pos[:2] - a[:2], d[:2]) / max(L**2, 1e-9), 0.0, 1.0))
        return chi_c, float(a[2] + s * (b[2] - a[2])), self.Va

    def _lead(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance before the corner at which to start the turn so that a turn of radius ``turn_radius`` rolls out on the next leg:
        ``R tan(Δχ / 2)`` (the tangent length of the fillet), limited to 45 % of the segment."""
        if self.turn_radius <= 0.0:
            return 0.0
        j = self.W[(np.where((b == self.W).all(axis=1))[0][0] + 1) % len(self.W)] if (b == self.W).all(axis=1).any() else b
        d1, d2 = b[:2] - a[:2], j[:2] - b[:2]
        if np.linalg.norm(d2) < 1e-9:
            return 0.0
        dchi = abs(float(wrap_pi(math.atan2(d2[1], d2[0]) - math.atan2(d1[1], d1[0]))))
        return float(min(self.turn_radius * math.tan(0.5 * dchi), 0.45 * np.linalg.norm(d1)))

    def cross_track_error(self, pos: np.ndarray) -> float:
        a, b = self._segment()
        d = b - a
        chi_path = math.atan2(d[1], d[0])
        rel = pos[:2] - a[:2]
        return float(-rel[0] * math.sin(chi_path) + rel[1] * math.cos(chi_path))
