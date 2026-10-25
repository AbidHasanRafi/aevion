"""Common interface of large-scale propagation models.

A model maps link geometry to a **median path loss** (dB), a LoS probability and a shadow-fading standard deviation.
Randomness (LoS draws, shadowing) is applied by the caller, so every model is deterministic and testable; use
:func:`realise` to draw a full sample.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..units import C_LIGHT


@dataclass(frozen=True)
class LinkGeometry:
    """Geometry of transmitter/receiver pairs (all arrays broadcast to a common shape)."""

    d2d: np.ndarray  # horizontal distance [m]
    d3d: np.ndarray  # slant distance [m]
    h_bs: np.ndarray  # height of the base station / terrestrial node above ground [m]
    h_ut: np.ndarray  # height of the user terminal (the UAV) above ground [m]

    @classmethod
    def from_positions(cls, bs: Any, ut: Any, ground_z: float = 0.0) -> "LinkGeometry":
        """From ``(..., 3)`` positions; heights are measured above ``ground_z``."""
        bs, ut = np.broadcast_arrays(np.asarray(bs, float), np.asarray(ut, float))
        dv = ut - bs
        d2d = np.linalg.norm(dv[..., :2], axis=-1)
        d3d = np.linalg.norm(dv, axis=-1)
        return cls(d2d, d3d, np.maximum(bs[..., 2] - ground_z, 0.0), np.maximum(ut[..., 2] - ground_z, 0.0))

    @classmethod
    def from_distances(cls, d2d: Any, h_bs: Any, h_ut: Any) -> "LinkGeometry":
        d2d, h_bs, h_ut = np.broadcast_arrays(np.asarray(d2d, float), np.asarray(h_bs, float), np.asarray(h_ut, float))
        return cls(d2d, np.hypot(d2d, h_ut - h_bs), h_bs, h_ut)

    @property
    def elevation_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(np.abs(self.h_ut - self.h_bs), np.maximum(self.d2d, 1e-6)))


class PathLossModel(ABC):
    """Large-scale model: median path loss for given LoS state, LoS probability and shadowing spread."""

    name: str = "model"
    #: inclusive ranges where the model is specified; evaluation outside them still returns a value (clamped inputs) but
    #: :meth:`validity` reports it so callers and tests can flag extrapolation
    freq_range_ghz: Tuple[float, float] = (0.0, 1e3)
    d2d_range_m: Tuple[float, float] = (1.0, 1e5)
    h_ut_range_m: Tuple[float, float] = (0.0, 1e5)  # specified terminal (UAV) heights above ground

    def __init__(self, freq_hz: float = 2.0e9):
        if freq_hz <= 0:
            raise ValueError("freq_hz must be positive")
        self.freq_hz = float(freq_hz)

    @property
    def fc_ghz(self) -> float:
        return self.freq_hz / 1e9

    @abstractmethod
    def path_loss_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        """Median path loss [dB] for the given LoS flags (bool array broadcastable to the geometry)."""

    @abstractmethod
    def los_probability(self, g: LinkGeometry) -> np.ndarray:
        """Probability that the link is line-of-sight."""

    @abstractmethod
    def shadowing_std_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        """Standard deviation of log-normal shadow fading [dB]."""

    def validity(self, g: LinkGeometry) -> np.ndarray:
        """Boolean mask: is the link inside the model's specified distance and frequency range?"""
        lo, hi = self.d2d_range_m
        ok = (g.d2d >= lo) & (g.d2d <= hi)
        h_lo, h_hi = self.h_ut_range_m
        ok = ok & (g.h_ut >= h_lo - 1e-9) & (g.h_ut <= h_hi + 1e-9)
        f_lo, f_hi = self.freq_range_ghz
        return ok & (f_lo <= self.fc_ghz <= f_hi)

    def mean_path_loss_db(self, g: LinkGeometry) -> np.ndarray:
        """Probability-weighted path loss (in the linear power domain), a deterministic summary of the two states."""
        p = self.los_probability(g)
        lin = p * 10 ** (-self.path_loss_db(g, True) / 10) + (1 - p) * 10 ** (-self.path_loss_db(g, False) / 10)
        return -10.0 * np.log10(lin)

    def describe(self) -> Dict[str, Any]:
        return dict(
            name=self.name, freq_hz=self.freq_hz, freq_range_ghz=self.freq_range_ghz, d2d_range_m=self.d2d_range_m, h_ut_range_m=self.h_ut_range_m
        )


def realise(
    model: PathLossModel, g: LinkGeometry, rng: np.random.Generator, los: Optional[Any] = None, shadowing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one realisation: ``(path_loss_db, los)``. ``los=None`` draws LoS from the model's probability."""
    if los is None:
        los = rng.random(np.shape(g.d3d)) < model.los_probability(g)
    los = np.broadcast_to(np.asarray(los, bool), np.shape(g.d3d))
    pl = model.path_loss_db(g, los)
    if shadowing:
        pl = pl + rng.normal(size=pl.shape) * model.shadowing_std_db(g, los)
    return pl, los


def breakpoint_distance(h_bs: Any, h_ut: Any, freq_hz: float, h_e: float = 1.0) -> np.ndarray:
    """Effective-height breakpoint distance ``4 h'_BS h'_UT f_c / c`` (3GPP TR 38.901, notes of Table 7.4.1-1)."""
    hb = np.maximum(np.asarray(h_bs, float) - h_e, 0.1)
    hu = np.maximum(np.asarray(h_ut, float) - h_e, 0.1)
    return 4.0 * hb * hu * freq_hz / C_LIGHT


def _log10(x: Any) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, float), 1e-9))


def _clip_d(d: Any, lo: float = 1.0) -> np.ndarray:
    return np.maximum(np.asarray(d, float), lo)


def fspl_db(d3d: Any, freq_hz: float) -> np.ndarray:
    """Free-space path loss ``20 log10(4 pi d f / c)`` [dB]."""
    return 20.0 * _log10(4.0 * math.pi * _clip_d(d3d, 1e-3) * freq_hz / C_LIGHT)
