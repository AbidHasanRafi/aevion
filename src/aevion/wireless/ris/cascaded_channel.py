"""Cascaded transmitter -> RIS -> receiver channel (spherical-wave, valid in the near and far field).

Each element ``n`` contributes a field ``h_n = sqrt(P_t G_t G_r G_e) |Gamma| (lambda / 4 pi)^2 sqrt(F_n) / (d1_n d2_n)
exp(-j k (d1_n + d2_n) + j phi_n)``, where ``F_n = cos(theta_in) cos(theta_out)`` models the projected aperture (zero for rays arriving
or leaving from behind the panel), ``d1, d2`` are the transmitter-element and element-receiver distances and ``phi_n`` the programmed
phase. The received power is ``|sum_n h_n|^2``; with optimal phases the sum is coherent and grows as ``N^2`` (Tang et al.,
IEEE TWC 20(5), 2021).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from ..units import C_LIGHT, dbm2w, w2dbm
from .phase_opt import optimal_phases, quantize_phases
from .surface import RisPanel

__all__ = ["ris_channel", "ris_received_power_dbm"]


def ris_channel(
    tx_pos: Any,
    rx_pos: Any,
    ris: RisPanel,
    freq_hz: float,
    phases: Optional[Any] = None,
    gt_linear: float = 1.0,
    gr_linear: float = 1.0,
) -> np.ndarray:
    """Complex *amplitude* gain of the RIS-reflected path (``|h|^2`` is the power gain), shape ``(...,)`` for batched ``rx_pos (..., 3)``."""
    lam = C_LIGHT / freq_hz
    k = 2.0 * math.pi / lam
    el = ris.element_positions()  # (N, 3)
    tx = np.asarray(tx_pos, float)
    rx = np.asarray(rx_pos, float)
    d1 = np.linalg.norm(el - tx, axis=-1)  # (N,)
    d2 = np.linalg.norm(rx[..., None, :] - el, axis=-1)  # (..., N)
    cos_in = np.clip(((tx - el) @ ris.normal) / d1, 0.0, 1.0)
    cos_out = np.clip(np.einsum("...nc,c->...n", rx[..., None, :] - el, ris.normal) / d2, 0.0, 1.0)
    f = np.sqrt(cos_in * cos_out)
    g_e = ris.element_gain_linear(freq_hz)
    amp = math.sqrt(gt_linear * gr_linear * g_e) * ris.reflection_amplitude * (lam / (4.0 * math.pi)) ** 2 * f / (d1 * d2)
    if phases is None:
        phases = optimal_phases(tx, rx if rx.ndim == 1 else rx.reshape(-1, 3)[0], ris, freq_hz)
    return np.sum(amp * np.exp(-1j * k * (d1 + d2) + 1j * np.asarray(phases)), axis=-1)


def ris_received_power_dbm(
    tx_pos: Any,
    rx_pos: Any,
    ris: RisPanel,
    freq_hz: float,
    tx_power_dbm: float = 30.0,
    gt_dbi: float = 0.0,
    gr_dbi: float = 0.0,
    mode: str = "optimal",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Received power [dBm] via the RIS. ``mode``: ``optimal`` (continuous phases aligned at ``rx_pos`` for a single receiver),
    ``quantized`` (``ris.phase_bits``), ``random`` (uniform random phases: the unconfigured surface) or ``off`` (all zero phases)."""
    tx = np.asarray(tx_pos, float)
    rx = np.asarray(rx_pos, float)
    n = ris.n_elements
    if mode == "optimal":
        phases = optimal_phases(tx, rx if rx.ndim == 1 else rx.reshape(-1, 3)[0], ris, freq_hz)
    elif mode == "quantized":
        phases = quantize_phases(optimal_phases(tx, rx if rx.ndim == 1 else rx.reshape(-1, 3)[0], ris, freq_hz), max(ris.phase_bits, 1))
    elif mode == "random":
        phases = (rng or np.random.default_rng()).uniform(0.0, 2.0 * math.pi, size=n)
    elif mode == "off":
        phases = np.zeros(n)
    else:
        raise ValueError("mode must be 'optimal', 'quantized', 'random' or 'off'")
    h = ris_channel(tx, rx, ris, freq_hz, phases, 10 ** (gt_dbi / 10), 10 ** (gr_dbi / 10))
    return w2dbm(dbm2w(tx_power_dbm) * np.abs(h) ** 2)


def scaling_summary(ris: RisPanel, tx_pos: Any, rx_pos: Any, freq_hz: float) -> Dict[str, float]:
    """Power of the optimal-phase RIS path relative to the unconfigured (random-phase) mean, in dB: the *RIS gain* (about ``10 log10 N``)."""
    opt = ris_received_power_dbm(tx_pos, rx_pos, ris, freq_hz, mode="optimal")
    rng = np.random.default_rng(0)
    rnd = np.mean([10 ** (ris_received_power_dbm(tx_pos, rx_pos, ris, freq_hz, mode="random", rng=rng) / 10) for _ in range(200)])
    return dict(
        optimal_dbm=float(opt), random_mean_dbm=float(10 * math.log10(rnd)), gain_db=float(opt - 10 * math.log10(rnd)), n_elements=ris.n_elements
    )
