"""Beam codebooks for analog/hybrid beam management (5G NR style SSB beam sweeping)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .arrays import PhasedArray, UniformLinearArray, UniformPlanarArray

__all__ = ["Codebook", "dft_codebook", "grid_codebook"]


@dataclass
class Codebook:
    """A set of unit-norm beams with their pointing directions."""

    weights: np.ndarray  # (B, N)
    az: np.ndarray  # (B,) radians
    el: np.ndarray  # (B,) radians

    def __len__(self) -> int:
        return len(self.az)

    def best_beam(self, array: PhasedArray, az: Any, el: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Index and gain [array-factor dB] of the best beam towards each ``(az, el)`` (arrays of any common shape)."""
        a = array.steering_vector(az, el)  # (..., N)
        g = np.abs(a @ self.weights.T) ** 2  # (..., B)
        idx = g.argmax(axis=-1)
        return idx, 10.0 * np.log10(np.maximum(np.take_along_axis(g, idx[..., None], -1)[..., 0], 1e-12))


def grid_codebook(array: PhasedArray, az_deg: Any, el_deg: Any = (0.0,)) -> Codebook:
    """Beams pointing at every combination of the given azimuth/elevation angles."""
    az, el = np.meshgrid(np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float)), indexing="ij")
    az, el = az.ravel(), el.ravel()
    w = np.stack([array.beam_weights(a, e) for a, e in zip(az, el)])
    return Codebook(w, az, el)


def dft_codebook(array: PhasedArray, oversample: int = 1, fov_deg: float = 120.0) -> Codebook:
    """DFT-spaced beams: uniform in the *sine* domain (the natural orthogonal grid of a half-wavelength array).

    ULA: ``oversample * N`` azimuth beams within the field of view; UPA: the product grid over azimuth and elevation.
    """
    half = math.sin(math.radians(fov_deg / 2.0))
    if isinstance(array, UniformLinearArray):
        n = max(1, round(oversample * array.n * half))
        s = np.linspace(-half, half, n) if n > 1 else np.array([0.0])
        return grid_codebook(array, np.degrees(np.arcsin(s)))
    if isinstance(array, UniformPlanarArray):
        ny, nz = max(1, round(oversample * array.ny * half)), max(1, round(oversample * array.nz * half))
        sy = np.linspace(-half, half, ny) if ny > 1 else np.array([0.0])
        sz = np.linspace(-half, half, nz) if nz > 1 else np.array([0.0])
        return grid_codebook(array, np.degrees(np.arcsin(sy)), np.degrees(np.arcsin(sz)))
    raise TypeError("dft_codebook needs a UniformLinearArray or UniformPlanarArray")
