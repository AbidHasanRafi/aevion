"""Trim, linearisation and flight-dynamic modes of a :class:`~aevion.aircraft.fixedwing.FixedWing`.

**Trim** finds the constant-control flight condition for an airspeed ``Va``, flight-path angle ``γ`` and turn radius ``R`` (straight
flight when ``R`` is ``None``): the angle of attack, bank, pitch and control deflections for which the accelerations vanish
(straight) or equal the centripetal acceleration of the circle. The unknowns are solved with SciPy's Newton iterations.

**Linearisation** differentiates the dynamics numerically about the trim in body velocity ``(u, v, w)``, Euler angles and body
rates ``(p, q, r)``. For symmetric flight the longitudinal ``(u, w, q, θ)`` and lateral ``(v, p, r, φ)`` blocks decouple; their
eigenvalues are the classical modes (phugoid, short period, roll subsidence, spiral, Dutch roll).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

from ..core import rotation as rot
from .fixedwing import AircraftState, B, FixedWing

__all__ = ["TrimResult", "trim", "linearize", "modes", "euler_rates", "state_from_trim"]


@dataclass
class TrimResult:
    airspeed: float
    gamma: float
    radius: Optional[float]
    alpha: float
    beta: float
    phi: float
    theta: float  # Euler pitch of the ENU/FLU convention (positive = nose down); see ``pitch_up``
    controls: np.ndarray  # (delta_e, delta_a, delta_r, delta_t)
    residual: float
    converged: bool

    @property
    def pitch_up(self) -> float:
        """Nose-up pitch attitude [rad] (the aerospace convention)."""
        return -self.theta

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            airspeed=self.airspeed,
            gamma=self.gamma,
            radius=self.radius,
            alpha=self.alpha,
            beta=self.beta,
            phi=self.phi,
            theta=self.theta,
            delta_e=float(self.controls[0]),
            delta_a=float(self.controls[1]),
            delta_r=float(self.controls[2]),
            delta_t=float(self.controls[3]),
            residual=self.residual,
            converged=self.converged,
        )


def _body_velocity(Va: float, alpha: float, beta: float) -> np.ndarray:
    """Air-relative velocity in FLU body axes from airspeed, angle of attack and sideslip (FRD ``(cα cβ, sβ, sα cβ)``)."""
    return np.array([Va * math.cos(alpha) * math.cos(beta), -Va * math.sin(beta), -Va * math.sin(alpha) * math.cos(beta)])


def state_from_trim(model: FixedWing, t: TrimResult, pos: Any = (0.0, 0.0, 100.0), heading: float = 0.0) -> AircraftState:
    """The aircraft state (and actual controls) at the trim point."""
    n = model.n
    q = rot.from_euler(B, np.full(n, t.phi), np.full(n, t.theta), np.full(n, heading))
    vb = _body_velocity(t.airspeed, t.alpha, t.beta)
    vel = rot.rotate(B, q, np.ones((n, 1)) * vb)
    omega = np.zeros((n, 3))
    if t.radius is not None:
        psi_dot = t.airspeed * math.cos(t.gamma) / t.radius
        omega = np.einsum("nji,j->ni", rot.to_matrix(B, q), np.array([0.0, 0.0, psi_dot]))  # body rates of a steady turn
    st = AircraftState(np.ones((n, 1)) * np.asarray(pos, float), vel, q, omega, np.ones((n, 1)) * t.controls, 0.0, np.zeros(n, bool))
    model._diagnose(st, st.controls)
    return st


def trim(model: FixedWing, airspeed: float, gamma: float = 0.0, radius: Optional[float] = None, beta: float = 0.0) -> TrimResult:
    """Trim for ``airspeed`` [m/s], flight-path angle ``gamma`` [rad] and (optionally) the radius of a level or helical turn [m].
    Positive ``radius`` turns left (counter-clockwise seen from above)."""
    one = FixedWing(model.p, 1, model.dt, None)  # trim is a flight condition relative to the air: the wind does not enter

    def unpack(x: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        if radius is None:  # in FLU axes a positive Euler pitch is nose DOWN, so a nose-up attitude alpha + gamma is Euler pitch -(alpha + gamma)
            return x[0], 0.0, -(x[0] + gamma), np.array([x[1], 0.0, 0.0, x[2]])
        return x[0], x[1], x[2], np.array([x[3], x[4], x[5], x[6]])

    def residual(x: np.ndarray) -> np.ndarray:
        alpha, phi, theta, u = unpack(x)
        q = rot.from_euler(B, np.array([phi]), np.array([theta]), np.array([0.0]))
        vb = _body_velocity(airspeed, alpha, beta)
        vel = rot.rotate(B, q, vb[None, :])
        om_w = np.array([0.0, 0.0, 0.0 if radius is None else airspeed * math.cos(gamma) / radius])
        omega = np.einsum("nji,j->ni", rot.to_matrix(B, q), om_w)
        d = one.derivatives(0.0, np.zeros((1, 3)), vel, q, omega, u[None, :])
        centripetal = np.cross(om_w, vel[0])
        r = [*(d[1][0] - centripetal), *d[3][0]]  # world acceleration and body angular acceleration
        r.append(vel[0, 2] - airspeed * math.sin(gamma))  # the flight-path angle
        return np.array(r)

    if radius is None:
        x0 = np.array([0.05, 0.0, 0.5])

        # equations: ax, az (world: ax, ay, az) and pitch acceleration; solve the 3 unknowns by least squares on the 3 relevant residuals
        def f3(x: np.ndarray) -> np.ndarray:
            r = residual(x)
            return np.array([r[0], r[2], r[4]])  # forward (east) and vertical acceleration, pitch acceleration (heading east)

        x, info, ier, _ = fsolve(f3, x0, full_output=True, xtol=1e-12)
        res = float(np.linalg.norm(residual(x)))
        alpha, phi, theta, u = unpack(x)
    else:
        x0 = np.array([0.05, 0.3, 0.05, 0.0, 0.0, 0.0, 0.5])
        # 7 unknowns, 7 independent residuals: ax, ay, az (3), body angular accelerations (3), flight-path angle (1)
        x, info, ier, _ = fsolve(lambda x: residual(x), x0, full_output=True, xtol=1e-12)
        res = float(np.linalg.norm(residual(x)))
        alpha, phi, theta, u = unpack(x)
    return TrimResult(airspeed, gamma, radius, float(alpha), float(beta), float(phi), float(theta), u, res, bool(ier == 1 and res < 1e-6))


# ======================================================================================================== linearisation
def euler_rates(euler: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """``(φ̇, θ̇, ψ̇)`` from the Euler angles (roll, pitch, yaw: ZYX) and the body rates ``(p, q, r)``."""
    ph, th = euler[0], euler[1]
    sp, cp, tt, ct = math.sin(ph), math.cos(ph), math.tan(th), math.cos(th)
    return np.array([omega[0] + sp * tt * omega[1] + cp * tt * omega[2], cp * omega[1] - sp * omega[2], sp / ct * omega[1] + cp / ct * omega[2]])


def _flow(model: FixedWing, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """State derivative of ``x = (u, v, w, p, q, r, φ, θ, ψ)`` (body velocity, body rates, Euler angles) for controls ``u``."""
    vb, om, eul = x[0:3], x[3:6], x[6:9]
    q = rot.from_euler(B, np.array([eul[0]]), np.array([eul[1]]), np.array([eul[2]]))
    R = rot.to_matrix(B, q)[0]
    vel = (R @ vb)[None, :]
    d = model.derivatives(0.0, np.zeros((1, 3)), vel, q, om[None, :], u[None, :])
    vb_dot = R.T @ d[1][0] - np.cross(om, vb)
    return np.concatenate([vb_dot, d[3][0], euler_rates(eul, om)])


def linearize(model: FixedWing, t: TrimResult, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    """Numerical Jacobians ``A = ∂f/∂x`` (9×9) and ``Bm = ∂f/∂u`` (9×4) at the trim, plus the longitudinal and lateral blocks.

    The state order is ``(u, v, w, p, q, r, φ, θ, ψ)`` in FLU body axes; ``A_lon`` is ``(u, w, q, θ)`` and ``A_lat`` is
    ``(v, p, r, φ)``. For a turning trim the blocks are only approximately decoupled.
    """
    one = FixedWing(model.p, 1, model.dt, None)
    st = state_from_trim(one, t)
    eul = rot.to_euler(B, st.quat)[0]
    x0 = np.concatenate([rot.rotate_inv(B, st.quat, st.vel)[0], st.omega[0], eul])
    u0 = np.asarray(t.controls, float)
    A = np.zeros((9, 9))
    for i in range(9):
        h = eps * max(1.0, abs(x0[i]))
        dx = np.zeros(9)
        dx[i] = h
        A[:, i] = (_flow(one, x0 + dx, u0) - _flow(one, x0 - dx, u0)) / (2.0 * h)
    Bm = np.zeros((9, 4))
    for j in range(4):
        h = eps
        du = np.zeros(4)
        du[j] = h
        Bm[:, j] = (_flow(one, x0, u0 + du) - _flow(one, x0, u0 - du)) / (2.0 * h)
    lon, lat = [0, 2, 4, 7], [1, 3, 5, 6]
    return dict(
        A=A,
        B=Bm,
        A_lon=A[np.ix_(lon, lon)],
        A_lat=A[np.ix_(lat, lat)],
        B_lon=Bm[lon][:, [0, 3]],
        B_lat=Bm[lat][:, [1, 2]],
        x0=x0,
        f0=_flow(one, x0, u0),
    )


def _pairs(eig: np.ndarray) -> list:
    out = []
    for lam in eig:
        if lam.imag > 1e-9:
            wn = abs(lam)
            out.append(dict(eigenvalue=lam, wn=float(wn), zeta=float(-lam.real / wn), period=float(2.0 * math.pi / lam.imag)))
    return out


def modes(lin: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """Classify the eigenvalues of the decoupled blocks.

    Longitudinal: ``short_period`` is the fastest complex pair; ``phugoid`` is what remains of the slow dynamics - a complex pair
    (frequency, damping) or, when the propeller's speed damping makes it overdamped, two real roots (``overdamped=True``).
    Lateral: ``roll`` (the fastest real root), ``spiral`` (the slowest real root) and ``dutch_roll`` (the complex pair).
    """
    out: Dict[str, Dict[str, Any]] = {}
    elon = np.linalg.eigvals(lin["A_lon"])
    pairs = sorted(_pairs(elon), key=lambda d: d["wn"])
    if pairs and abs(pairs[-1]["eigenvalue"]) > 1.0:
        out["short_period"] = pairs[-1]
        rest = [l for l in elon if abs(l - pairs[-1]["eigenvalue"]) > 1e-9 and abs(l - np.conj(pairs[-1]["eigenvalue"])) > 1e-9]
        slow = list(pairs[:-1])
        if slow:
            out["phugoid"] = slow[0]
        elif rest:
            re = sorted(float(l.real) for l in rest)
            out["phugoid"] = dict(
                eigenvalues=re,
                overdamped=True,
                wn=float(math.sqrt(abs(re[0] * re[-1]))),
                zeta=float(-(re[0] + re[-1]) / (2.0 * math.sqrt(abs(re[0] * re[-1])))),
            )
    elat = np.linalg.eigvals(lin["A_lat"])
    real = sorted(float(l.real) for l in elat if abs(l.imag) < 1e-9)
    dr = _pairs(elat)
    if dr:
        out["dutch_roll"] = dr[0]
    if real:
        out["roll"] = dict(eigenvalue=real[0], time_constant=-1.0 / real[0])
        if len(real) > 1:
            out["spiral"] = dict(eigenvalue=real[-1], time_constant=(-1.0 / real[-1]) if real[-1] != 0 else math.inf)
    return out
