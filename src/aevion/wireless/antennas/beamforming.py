"""Beamforming weights and a beam-management state machine for a moving UAV."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .arrays import PhasedArray
from .codebooks import Codebook

__all__ = ["BeamManager", "BeamState", "mmse_weights", "mrt_weights", "pointing_loss_db", "zf_weights"]


def mrt_weights(h: Any) -> np.ndarray:
    """Maximum-ratio transmission ``w = h* / ||h||`` for a channel row vector ``h`` (``(..., N)``)."""
    h = np.asarray(h, complex)
    return np.conj(h) / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-30)


def zf_weights(H: Any) -> np.ndarray:
    """Zero-forcing precoder for a ``(K, N)`` multi-user channel: ``W = H^H (H H^H)^-1`` with unit-norm columns, shape ``(N, K)``."""
    H = np.asarray(H, complex)
    W = H.conj().T @ np.linalg.pinv(H @ H.conj().T)
    return W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-30)


def mmse_weights(H: Any, snr_linear: float) -> np.ndarray:
    """Regularised ZF (MMSE) precoder ``W = H^H (H H^H + K / snr I)^-1``, unit-norm columns; tends to ZF at high SNR, MRT at low."""
    H = np.asarray(H, complex)
    k = H.shape[0]
    W = H.conj().T @ np.linalg.inv(H @ H.conj().T + (k / max(snr_linear, 1e-12)) * np.eye(k))
    return W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-30)


def pointing_loss_db(error_rad: Any, beamwidth_rad: float, cap_db: float = 30.0) -> np.ndarray:
    """Gain lost by pointing ``error_rad`` away from the beam centre: ``min(12 (err / beamwidth)^2, cap)`` [dB]."""
    return np.minimum(12.0 * (np.asarray(error_rad, float) / max(beamwidth_rad, 1e-9)) ** 2, cap_db)


@dataclass
class BeamState:
    gain_db: np.ndarray  # array-factor gain towards the true direction with the currently selected beam
    beam: np.ndarray  # selected beam index per link
    swept: np.ndarray  # a new sweep finished at this step
    in_sweep: np.ndarray  # the link is currently sweeping (no data): treat as unavailable
    misalignment_deg: np.ndarray  # angle between the selected beam centre and the true direction


class BeamManager:
    """Periodic beam sweeping with tracking between sweeps (NR-style P1 procedure, abstracted).

    Every ``sweep_period_s`` each link measures all codebook beams (with ``meas_noise_db`` of RSRP noise) and selects the
    strongest; the sweep takes ``sweep_duration_s`` during which the link carries no data. Between sweeps the beam is fixed, so a
    manoeuvring UAV drifts out of the beam: that misalignment is the cost of a long sweep period.
    """

    def __init__(
        self,
        array: PhasedArray,
        codebook: Codebook,
        n_links: int,
        sweep_period_s: float = 0.16,
        sweep_duration_s: float = 0.005,
        meas_noise_db: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.array, self.codebook, self.n = array, codebook, n_links
        self.period, self.duration, self.noise = sweep_period_s, sweep_duration_s, meas_noise_db
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        self.beam = np.zeros(self.n, int)
        self._next_sweep = np.zeros(self.n)
        self._sweep_end = np.full(self.n, -1.0)

    def step(self, t: float, az: Any, el: Any) -> BeamState:
        az = np.broadcast_to(np.asarray(az, float), (self.n,))
        el = np.broadcast_to(np.asarray(el, float), (self.n,))
        due = t >= self._next_sweep - 1e-12
        swept = np.zeros(self.n, bool)
        if np.any(due):
            a = self.array.steering_vector(az, el)  # (L, N)
            g_db = 10.0 * np.log10(np.maximum(np.abs(a @ self.codebook.weights.T) ** 2, 1e-12))  # (L, B)
            g_db = g_db + self.rng.normal(0.0, self.noise, size=g_db.shape)
            self.beam = np.where(due, g_db.argmax(axis=1), self.beam)
            self._next_sweep = np.where(due, t + self.period, self._next_sweep)
            self._sweep_end = np.where(due, t + self.duration, self._sweep_end)
            swept = due
        w = self.codebook.weights[self.beam]  # (L, N)
        a = self.array.steering_vector(az, el)
        gain = 10.0 * np.log10(np.maximum(np.abs(np.sum(a * w, axis=1)) ** 2, 1e-12))
        c_az, c_el = self.codebook.az[self.beam], self.codebook.el[self.beam]
        cosang = np.cos(el) * np.cos(c_el) * np.cos(az - c_az) + np.sin(el) * np.sin(c_el)
        mis = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        return BeamState(gain, self.beam.copy(), swept, t < self._sweep_end - 1e-12, mis)

    @property
    def sweep_overhead(self) -> float:
        """Fraction of time lost to sweeping."""
        return min(self.duration / max(self.period, 1e-9), 1.0)

    @staticmethod
    def max_array_gain_db(n_elements: int) -> float:
        return 10.0 * math.log10(n_elements)
