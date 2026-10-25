"""Electromagnetic properties of building and ground materials and Fresnel reflection coefficients.

Relative permittivity ``eps' = a f^b`` and conductivity ``sigma = c f^d`` [S/m] with ``f`` in GHz, following ITU-R P.2040
(building materials, Table 3) and P.527 (ground). The values are the recommendation's fitted constants as remembered by the
authors of this module; check them against the recommendation if a specific material matters for your result.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

__all__ = ["MATERIALS", "complex_permittivity", "fresnel_reflection"]

EPS0 = 8.8541878128e-12

#: material -> (a, b, c, d)
MATERIALS: Dict[str, Tuple[float, float, float, float]] = {
    "concrete": (5.31, 0.0, 0.0326, 0.8095),
    "brick": (3.75, 0.0, 0.038, 0.0),
    "plasterboard": (2.94, 0.0, 0.0116, 0.7076),
    "wood": (1.99, 0.0, 0.0047, 1.0718),
    "glass": (6.27, 0.0, 0.0043, 0.8158),
    "chipboard": (2.58, 0.0, 0.0217, 0.7800),
    "metal": (1.0, 0.0, 1.0e7, 0.0),
    "very_dry_ground": (3.0, 0.0, 0.00015, 2.52),
    "medium_dry_ground": (15.0, -0.1, 0.035, 1.63),
    "wet_ground": (30.0, -0.4, 0.15, 1.30),
}


def complex_permittivity(material: str, freq_hz: float) -> complex:
    """``eps_c = eps' - j sigma / (2 pi f eps0)``."""
    if material not in MATERIALS:
        raise KeyError(f"unknown material {material!r}; choose from {sorted(MATERIALS)}")
    a, b, c, d = MATERIALS[material]
    f_ghz = freq_hz / 1e9
    eps_r = a * f_ghz**b
    sigma = c * f_ghz**d
    return complex(eps_r, -sigma / (2.0 * math.pi * freq_hz * EPS0))


def fresnel_reflection(theta_i: Any, eps_c: complex, polarization: str = "avg") -> np.ndarray:
    """Fresnel reflection coefficient at incidence angle ``theta_i`` [rad, from the surface normal].

    ``polarization``: ``te`` (perpendicular), ``tm`` (parallel) or ``avg`` (power-average magnitude of the two, a polarisation-agnostic
    choice; the phase of the TE coefficient is kept). Returns a complex array.
    """
    th = np.asarray(theta_i, float)
    c, s2 = np.cos(th), np.sin(th) ** 2
    root = np.sqrt(eps_c - s2)
    te = (c - root) / (c + root)
    tm = (eps_c * c - root) / (eps_c * c + root)
    if polarization == "te":
        return te
    if polarization == "tm":
        return tm
    if polarization == "avg":
        mag = np.sqrt(0.5 * (np.abs(te) ** 2 + np.abs(tm) ** 2))
        return mag * np.exp(1j * np.angle(te))
    raise ValueError("polarization must be 'te', 'tm' or 'avg'")
