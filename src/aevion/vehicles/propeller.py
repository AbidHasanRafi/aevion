"""Propeller aerodynamics by blade-element momentum theory (BEMT), and a database of generic propellers.

BEMT combines two views of a rotor (Leishman, *Principles of Helicopter Aerodynamics*, 2nd ed., ch. 3 and 8; Adkins & Liebeck 1994).
The blade-element view computes the lift and drag of each spanwise section from its local angle of attack; the momentum view requires
the same thrust to accelerate the air through the annulus the section sweeps. Equating the two at each radius gives the induced
inflow, and integrating over the span gives thrust ``T``, torque ``Q`` and power ``P = Q Ω``. With Prandtl's tip-loss factor the model
reproduces the figure of merit and the thrust loss with axial inflow of real propellers well enough to derive a simulation's ``kT``,
``kQ`` and inflow gain from a blade geometry.

Non-dimensional performance uses the usual convention (``n`` rev/s, diameter ``D``)::

    J = V / (n D)    C_T = T / (ρ n² D⁴)    C_P = P / (ρ n³ D⁵) = 2π C_Q    η = J C_T / C_P

and the static figure of merit is ``FM = T^{3/2} / sqrt(2 ρ A) / P`` (ideal over actual power).

.. note::
   The airfoil is a generic thin cambered section (linear lift curve, quadratic drag polar, smooth stall). The named propellers in
   :data:`PROPELLERS` are generic geometries for the listed diameter and pitch, tuned to fall in the range of published small-propeller
   coefficients; they are not manufacturer data. For quantitative work, fit ``kT`` and ``kQ`` to a thrust-stand measurement.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["Airfoil", "Propeller", "PropellerPoint", "PROPELLERS", "get_propeller", "rotor_parameters", "RHO_SL"]

RHO_SL = 1.225
INCH = 0.0254


@dataclass
class Airfoil:
    """Thin-airfoil section: ``C_l = a (α - α₀)`` up to stall, ``C_d = C_d0 + k C_l²`` (plus a flat-plate post-stall rise)."""

    cl_alpha: float = 5.5  # lift-curve slope [1/rad] at the low Reynolds numbers of small propellers
    alpha0: float = math.radians(-3.0)  # zero-lift angle of a cambered section
    cl_max: float = 1.3
    cd0: float = 0.022
    k: float = 0.045
    alpha_stall: float = math.radians(14.0)

    def coefficients(self, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lift and drag coefficients at angle of attack ``alpha`` [rad]; defined for every angle (deep stall and reverse flow)."""
        a = np.asarray(alpha, float)
        d = a - self.alpha0
        ad = np.abs(d)
        ds = self.alpha_stall - self.alpha0  # angle from zero-lift at which the section stalls
        attached = np.sign(d) * np.minimum(self.cl_alpha * ad, self.cl_max)
        # past stall the lift falls smoothly to zero at 90 degrees from zero-lift, as for a plate normal to the flow
        frac = np.clip((ad - ds) / (math.pi / 2.0 - ds), 0.0, 1.0)
        stalled = np.sign(d) * self.cl_max * np.cos(0.5 * math.pi * frac)
        cl = np.where(ad <= ds, attached, stalled)
        cd = self.cd0 + self.k * cl**2 + np.where(ad > ds, 1.2 * np.sin(np.clip(ad - ds, 0.0, math.pi / 2.0)) ** 2, 0.0)
        return cl, cd


@dataclass
class PropellerPoint:
    """One operating point: ``T`` [N], ``Q`` [N m], ``P`` [W] and the non-dimensional coefficients."""

    T: float
    Q: float
    P: float
    J: float
    CT: float
    CP: float
    eta: float
    figure_of_merit: float
    rpm: float
    v_induced_mean: float


@dataclass
class Propeller:
    """A fixed-pitch propeller described by its blade geometry.

    ``diameter`` and ``pitch`` are in metres (the geometric pitch gives the blade angle ``β(r) = atan(p / (2π r))``, i.e. a
    helical blade), ``chord_ratio`` is the chord at 75 % radius divided by the radius, ``hub_ratio`` the root cut-out.
    """

    diameter: float
    pitch: float
    blades: int = 2
    chord_ratio: float = 0.16
    hub_ratio: float = 0.15
    airfoil: Airfoil = field(default_factory=Airfoil)
    name: str = ""
    n_stations: int = 40

    def __post_init__(self) -> None:
        if self.diameter <= 0 or self.pitch <= 0 or self.blades < 1 or not 0.0 < self.hub_ratio < 0.5 or self.chord_ratio <= 0:
            raise ValueError("need positive diameter, pitch and chord ratio, at least one blade and 0 < hub_ratio < 0.5")

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def disk_area(self) -> float:
        return math.pi * self.radius**2

    def _chord(self, x: np.ndarray) -> np.ndarray:
        """Chord distribution (as a fraction of R): tapered, widest near 0.45 R, thin at the root and the tip."""
        shape = np.sin(math.pi * np.clip((x - 0.05) / 1.0, 0.0, 1.0)) ** 0.6 + 0.12
        ref = float(np.interp(0.75, x, shape))
        return self.chord_ratio * shape / ref

    def _solve(self, omega: float, v_axial: float, rho: float) -> Tuple[float, float, float]:
        """Integrate the blade elements at angular speed ``omega`` [rad/s] and axial speed ``v_axial``: returns ``(T, Q, v_i_mean)``."""
        R, Nb, af = self.radius, self.blades, self.airfoil
        x = np.linspace(self.hub_ratio, 1.0, self.n_stations + 1)
        x = 0.5 * (x[1:] + x[:-1])
        dx = (1.0 - self.hub_ratio) / self.n_stations
        r = x * R
        c = self._chord(x) * R
        beta = np.arctan2(self.pitch, 2.0 * math.pi * r)
        UT = omega * r
        vi = np.full_like(r, 0.3 * math.sqrt(abs(omega * R) ** 2 * 0.002 + 1e-6))  # first guess; the iteration is insensitive to it
        for _ in range(200):
            UP = v_axial + vi
            phi = np.arctan2(UP, UT)
            sphi = np.maximum(np.abs(np.sin(phi)), 1e-4)
            # Prandtl tip (and root) loss
            f_tip = Nb * (1.0 - x) / (2.0 * sphi)
            f_root = Nb * (x - self.hub_ratio) / (2.0 * sphi * np.maximum(x, 1e-3))
            F = (2.0 / math.pi) * np.arccos(np.clip(np.exp(-f_tip), 0.0, 1.0)) * (2.0 / math.pi) * np.arccos(np.clip(np.exp(-f_root), 0.0, 1.0))
            F = np.maximum(F, 1e-3)
            alpha = beta - phi
            cl, cd = af.coefficients(alpha)
            U2 = UT**2 + UP**2
            dT_dr = 0.5 * rho * U2 * c * (cl * np.cos(phi) - cd * np.sin(phi)) * Nb  # blade element
            # momentum: dT/dr = 4 pi r rho (V + vi) vi F  ->  solve vi from the blade-element thrust
            rhs = dT_dr / (4.0 * math.pi * r * rho * F)
            disc = (0.5 * v_axial) ** 2 + np.maximum(rhs, -((0.5 * v_axial) ** 2) * 0.999)
            vi_new = -0.5 * v_axial + np.sqrt(np.maximum(disc, 0.0))
            vi_new = np.where(rhs >= 0, vi_new, np.minimum(vi_new, 0.0))
            vi_new = np.where(np.isfinite(vi_new), vi_new, 0.0)
            if np.max(np.abs(vi_new - vi)) < 1e-7 * max(omega * R, 1.0):
                vi = vi_new
                break
            vi = 0.5 * vi + 0.5 * vi_new  # damped fixed-point iteration
        UP = v_axial + vi
        phi = np.arctan2(UP, UT)
        alpha = beta - phi
        cl, cd = af.coefficients(alpha)
        U2 = UT**2 + UP**2
        dT = 0.5 * rho * U2 * c * (cl * np.cos(phi) - cd * np.sin(phi)) * Nb
        dQ = 0.5 * rho * U2 * c * (cl * np.sin(phi) + cd * np.cos(phi)) * Nb * r
        T = float(np.sum(dT) * dx * R)
        Q = float(np.sum(dQ) * dx * R)
        area_w = 2.0 * math.pi * x * dx * R**2
        vi_mean = float(np.sum(vi * area_w) / np.sum(area_w))
        return T, Q, vi_mean

    def operating_point(self, rpm: float, v_axial: float = 0.0, rho: float = RHO_SL) -> PropellerPoint:
        """Performance at ``rpm`` and axial (climb / forward-axis) speed ``v_axial`` [m/s] in air of density ``rho``."""
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        omega = rpm * 2.0 * math.pi / 60.0
        T, Q, vi = self._solve(omega, v_axial, rho)
        n = rpm / 60.0
        D = self.diameter
        P = Q * omega
        J = v_axial / (n * D)
        CT = T / (rho * n**2 * D**4)
        CP = P / (rho * n**3 * D**5)
        eta = J * CT / CP if CP > 1e-12 else 0.0
        fm = (abs(T) ** 1.5 / math.sqrt(2.0 * rho * self.disk_area)) / P if (T > 0 and P > 1e-9 and v_axial == 0.0) else 0.0
        return PropellerPoint(T, Q, P, J, CT, CP, eta, fm, rpm, vi)

    def static(self, rpm: float, rho: float = RHO_SL) -> PropellerPoint:
        return self.operating_point(rpm, 0.0, rho)

    def sweep_advance_ratio(self, rpm: float, J: np.ndarray, rho: float = RHO_SL) -> Dict[str, np.ndarray]:
        """Coefficients over a range of advance ratios at fixed ``rpm`` (``J`` is converted to an axial speed)."""
        out: Dict[str, list] = {k: [] for k in ("J", "CT", "CP", "eta")}
        for j in np.atleast_1d(J):
            pt = self.operating_point(rpm, float(j) * (rpm / 60.0) * self.diameter, rho)
            out["J"].append(pt.J)
            out["CT"].append(pt.CT)
            out["CP"].append(pt.CP)
            out["eta"].append(pt.eta)
        return {k: np.array(v) for k, v in out.items()}

    def zero_thrust_advance_ratio(self, rpm: float = 6000.0, rho: float = RHO_SL) -> float:
        """The advance ratio at which thrust vanishes (the windmilling limit), by bisection."""
        lo, hi = 0.0, 3.0 * self.pitch / self.diameter
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.operating_point(rpm, mid * (rpm / 60.0) * self.diameter, rho).T > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# ======================================================================================================== database
def _designation(name: str) -> Tuple[float, float, int]:
    m = re.fullmatch(r"\s*([\d.]+)\s*[x×]\s*([\d.]+)(?:\s*[x×]\s*(\d))?\s*", name)
    if not m:
        raise ValueError(f"cannot read the propeller designation {name!r}; write it as DIAMETERxPITCH in inches, e.g. '10x4.5' or '5x4.3x3'")
    return float(m.group(1)), float(m.group(2)), int(m.group(3) or 2)


# diameter x pitch [inch] (x blades). chord_ratio is the c(0.75R)/R used for the generic geometry of that size class
_PRESETS: Dict[str, Dict[str, float]] = {
    "3x2": dict(chord_ratio=0.20),
    "5x3": dict(chord_ratio=0.19),
    "5x4.5": dict(chord_ratio=0.19),
    "5x4.3x3": dict(chord_ratio=0.17),
    "7x4": dict(chord_ratio=0.17),
    "8x4.5": dict(chord_ratio=0.16),
    "9x4.7": dict(chord_ratio=0.16),
    "10x4.5": dict(chord_ratio=0.15),
    "11x4.7": dict(chord_ratio=0.15),
    "12x6": dict(chord_ratio=0.15),
    "13x6.5": dict(chord_ratio=0.14),
    "15x5.5": dict(chord_ratio=0.13),
    "18x6.1": dict(chord_ratio=0.12),
}


def get_propeller(name: str) -> Propeller:
    """A generic propeller from a designation ``'DxP'`` or ``'DxPxB'`` (inches), using the tuned geometry of its size class."""
    d, p, b = _designation(name)
    key = f"{d:g}x{p:g}" + (f"x{b}" if b != 2 else "")
    chord = _PRESETS.get(key, {}).get("chord_ratio")
    if chord is None:  # unlisted size: chord ratio decreasing slowly with diameter, as for real propellers
        chord = float(np.clip(0.23 - 0.0055 * d, 0.11, 0.21))
    return Propeller(d * INCH, p * INCH, blades=b, chord_ratio=chord, name=key)


PROPELLERS: Dict[str, Propeller] = {k: get_propeller(k) for k in _PRESETS}


# ======================================================================================================== simulation parameters
def rotor_parameters(prop: Propeller, rpm_ref: Optional[float] = None, rho: float = RHO_SL, blade_mass: Optional[float] = None) -> Dict[str, float]:
    """The thrust, torque and inflow coefficients the multirotor model uses, derived from a blade geometry.

    The simulator's rotor law is ``T = kT ω²`` and ``Q = kQ ω²`` (with ``ω`` in rad/s) and ``T`` falls with the climb speed
    ``vc`` as ``1 - g_in vc / (ω R)``. They follow from the static BEMT point at ``rpm_ref`` (default: where the tip speed is
    ``0.55 Mach``) and from the slope of ``C_T`` with the advance ratio. Also returned: the rotor radius, an estimate of the rotor
    inertia (``N_b m_b R² / 3`` with a blade mass estimated from the size, or given), and the figure of merit.
    """
    R = prop.radius
    if rpm_ref is None:
        rpm_ref = 0.55 * 340.0 / R * 60.0 / (2.0 * math.pi)
    st = prop.static(rpm_ref, rho)
    n = rpm_ref / 60.0
    w = 2.0 * math.pi * n
    kT = st.T / w**2
    kQ = st.Q / w**2
    dj = 0.02
    p2 = prop.operating_point(rpm_ref, dj * n * prop.diameter, rho)
    slope = -(p2.CT - st.CT) / (st.CT * dj)  # s in CT(J) ~ CT0 (1 - s J)
    inflow_gain = slope * math.pi  # J = pi V / (omega R)
    mb = blade_mass if blade_mass is not None else 0.045 * prop.diameter**2 * 1.0 / max(prop.blades, 1) * 4.0  # a few grams for a 5" blade
    J_r = prop.blades * mb * R**2 / 3.0
    return dict(
        kT=kT,
        kQ=kQ,
        rotor_radius=R,
        inflow_gain=inflow_gain,
        rotor_inertia=J_r,
        figure_of_merit=st.figure_of_merit,
        rpm_ref=rpm_ref,
        CT0=st.CT,
        CP0=st.CP,
    )
