"""RIS phase configuration: optimal alignment, discrete quantisation and the resulting gain loss."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..units import C_LIGHT
from .surface import RisPanel

__all__ = ["optimal_phases", "quantization_loss_db", "quantize_phases"]


def optimal_phases(tx_pos: Any, rx_pos: Any, ris: RisPanel, freq_hz: float) -> np.ndarray:
    """Phases that make every element's contribution arrive in phase at ``rx_pos``: ``phi_n = k (d1_n + d2_n) mod 2 pi``."""
    k = 2.0 * math.pi * freq_hz / C_LIGHT
    el = ris.element_positions()
    d = np.linalg.norm(el - np.asarray(tx_pos, float), axis=-1) + np.linalg.norm(el - np.asarray(rx_pos, float), axis=-1)
    return np.mod(k * d, 2.0 * math.pi)


def quantize_phases(phases: Any, bits: int) -> np.ndarray:
    """Round to the nearest of ``2^bits`` equally spaced phase states."""
    if bits < 1:
        raise ValueError("bits must be >= 1")
    step = 2.0 * math.pi / 2**bits
    return np.mod(np.round(np.asarray(phases, float) / step) * step, 2.0 * math.pi)


def quantization_loss_db(bits: int) -> float:
    """Gain loss of b-bit phase quantisation for many elements: ``20 log10(sinc(pi / 2^b))`` with ``sinc(x) = sin(x) / x`` (0.9 dB at b = 2, 0.22 dB at b = 3)."""
    x = math.pi / 2**bits
    return float(-20.0 * math.log10(math.sin(x) / x))
