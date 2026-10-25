"""Phased-array geometry, steering vectors and array gain.

Element positions are expressed in wavelengths in the antenna frame (x boresight, y left, z up). A direction ``(az, el)``
has the unit vector ``u = [cos el cos az, cos el sin az, sin el]`` and the steering vector is ``a = exp(j 2 pi p . u)``.
Weights are normalised to unit norm, so a perfectly steered N-element array has an array gain of ``10 log10 N``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .patterns import GAIN_FLOOR_DBI, AntennaPattern, Patch3GPP

__all__ = ["PhasedArray", "UniformLinearArray", "UniformPlanarArray", "direction_vector"]


def direction_vector(az: Any, el: Any) -> np.ndarray:
    az, el = np.broadcast_arrays(np.asarray(az, float), np.asarray(el, float))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


class PhasedArray:
    """Array of identical elements at arbitrary positions (in wavelengths)."""

    def __init__(self, positions_wl: Any, element: Optional[AntennaPattern] = None):
        p = np.asarray(positions_wl, float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 1:
            raise ValueError("positions must have shape (N, 3)")
        self.positions = p
        self.element = element or Patch3GPP()

    @property
    def n_elements(self) -> int:
        return len(self.positions)

    def steering_vector(self, az: Any, el: Any) -> np.ndarray:
        """``(..., N)`` complex steering vectors."""
        u = direction_vector(az, el)
        return np.exp(2j * math.pi * (u @ self.positions.T))

    def beam_weights(self, az: float, el: float) -> np.ndarray:
        """Conjugate-matched (maximum-gain) unit-norm weights pointing at ``(az, el)``."""
        a = self.steering_vector(az, el)
        return np.conj(a) / math.sqrt(self.n_elements)

    def array_factor_db(self, weights: Any, az: Any, el: Any) -> np.ndarray:
        """``10 log10 |w^T a|^2`` towards ``(az, el)`` (no element pattern); ``w`` is the conjugate-matched convention above."""
        w = np.asarray(weights, complex)
        a = self.steering_vector(az, el)
        return 10.0 * np.log10(np.maximum(np.abs(a @ w) ** 2, 1e-12))

    def gain_dbi(self, weights: Any, az: Any, el: Any) -> np.ndarray:
        """Total gain: element pattern plus array factor [dBi]."""
        return np.maximum(self.element.gain_dbi(az, el) + self.array_factor_db(weights, az, el), GAIN_FLOOR_DBI)

    def half_power_beamwidth_deg(self, axis: str = "az", steer_deg: float = 0.0) -> float:
        """Numerical -3 dB beamwidth of the array factor around a steering angle (broadside = 0)."""
        s = math.radians(steer_deg)
        w = self.beam_weights(s, 0.0) if axis == "az" else self.beam_weights(0.0, s)
        grid = np.radians(np.linspace(-90.0, 90.0, 18001))
        af = self.array_factor_db(w, grid, 0.0) if axis == "az" else self.array_factor_db(w, 0.0, grid)
        above = np.nonzero(af >= af.max() - 3.0103)[0]
        lo, hi = above.min(), above.max()
        return float(np.degrees(grid[hi] - grid[lo]))

    def describe(self) -> Dict[str, Any]:
        return dict(n_elements=self.n_elements, element=self.element.describe(), peak_array_gain_db=round(10 * math.log10(self.n_elements), 2))


class UniformLinearArray(PhasedArray):
    """``n`` elements along the y axis (a horizontal array: steers in azimuth), spacing in wavelengths (0.5 = half wavelength)."""

    def __init__(self, n: int = 8, spacing_wl: float = 0.5, element: Optional[AntennaPattern] = None):
        if n < 1:
            raise ValueError("n must be >= 1")
        y = (np.arange(n) - (n - 1) / 2.0) * spacing_wl
        super().__init__(np.stack([np.zeros(n), y, np.zeros(n)], axis=1), element)
        self.n, self.spacing_wl = n, spacing_wl


class UniformPlanarArray(PhasedArray):
    """``ny x nz`` elements in the y-z plane (steers in azimuth and elevation)."""

    def __init__(self, ny: int = 4, nz: int = 4, dy_wl: float = 0.5, dz_wl: float = 0.5, element: Optional[AntennaPattern] = None):
        if ny < 1 or nz < 1:
            raise ValueError("ny and nz must be >= 1")
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy_wl
        z = (np.arange(nz) - (nz - 1) / 2.0) * dz_wl
        Y, Z = np.meshgrid(y, z, indexing="ij")
        super().__init__(np.stack([np.zeros(Y.size), Y.ravel(), Z.ravel()], axis=1), element)
        self.ny, self.nz = ny, nz
