"""Reconfigurable intelligent surface (RIS) geometry: a planar array of passive reflecting elements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np

from ..units import C_LIGHT

__all__ = ["RisPanel"]


@dataclass
class RisPanel:
    """``nx x ny`` elements of size ``dx x dy`` (metres) centred at ``center``, facing along ``normal`` (the side the waves arrive on).

    ``reflection_amplitude`` is ``|Gamma|`` (1 = lossless; practical varactor/PIN designs reach 0.7-0.95).
    """

    center: Any
    normal: Any = (0.0, 0.0, 1.0)
    nx: int = 16
    ny: int = 16
    dx: float = 0.03
    dy: float = 0.03
    reflection_amplitude: float = 0.9
    phase_bits: int = 0  # 0 = continuous phase control, b >= 1: 2^b discrete phase levels
    up: Any = field(default=(0.0, 1.0, 0.0), repr=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, float)
        n = np.asarray(self.normal, float)
        self.normal = n / np.linalg.norm(n)
        if self.nx < 1 or self.ny < 1 or self.dx <= 0 or self.dy <= 0:
            raise ValueError("a RIS needs at least one element of positive size")
        if not 0.0 < self.reflection_amplitude <= 1.0:
            raise ValueError("reflection_amplitude must be in (0, 1]")

    @property
    def n_elements(self) -> int:
        return self.nx * self.ny

    @property
    def area_m2(self) -> float:
        return self.n_elements * self.dx * self.dy

    def _axes(self) -> Tuple[np.ndarray, np.ndarray]:
        up = np.asarray(self.up, float)
        u = np.cross(up, self.normal)
        if np.linalg.norm(u) < 1e-6:  # normal parallel to 'up': pick any perpendicular axis
            u = np.cross(np.array([1.0, 0.0, 0.0]), self.normal)
        u = u / np.linalg.norm(u)
        v = np.cross(self.normal, u)
        return u, v

    def element_positions(self) -> np.ndarray:
        """``(N, 3)`` world positions of the element centres."""
        u, v = self._axes()
        ix = (np.arange(self.nx) - (self.nx - 1) / 2.0) * self.dx
        iy = (np.arange(self.ny) - (self.ny - 1) / 2.0) * self.dy
        X, Y = np.meshgrid(ix, iy, indexing="ij")
        return self.center[None, :] + X.reshape(-1, 1) * u[None, :] + Y.reshape(-1, 1) * v[None, :]

    def far_field_distance_m(self, freq_hz: float) -> float:
        """Fraunhofer distance ``2 D^2 / lambda`` of the panel diagonal ``D``: closer than this the spherical-wave model must be used."""
        lam = C_LIGHT / freq_hz
        d = math.hypot(self.nx * self.dx, self.ny * self.dy)
        return 2.0 * d * d / lam

    def element_gain_linear(self, freq_hz: float) -> float:
        """Isotropic-aperture element gain ``4 pi dx dy / lambda^2`` (the effective-area gain of one element)."""
        lam = C_LIGHT / freq_hz
        return 4.0 * math.pi * self.dx * self.dy / lam**2
