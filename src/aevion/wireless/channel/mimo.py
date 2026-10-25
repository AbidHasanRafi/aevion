"""Narrowband MIMO channel matrices and capacity.

``H`` has shape ``(..., Nr, Nt)``: ``y = H x + n``. The geometric model is

``H = sqrt(K/(K+1)) a_r(az_r, el_r) a_t(az_t, el_t)^H + sqrt(1/(K+1)) R_r^(1/2) G R_t^(1/2)``

with a rank-one specular LoS term and a Kronecker-correlated Rayleigh scattered term ``G ~ CN(0, 1)``: a UAV link with a strong
LoS (large K) is nearly rank-one and cannot multiplex, which is the main way MIMO behaves differently for aerial users.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..antennas.arrays import PhasedArray

__all__ = [
    "condition_number_db",
    "ergodic_capacity",
    "exponential_correlation",
    "geometric_channel",
    "mimo_capacity",
    "rayleigh_channel",
    "water_filling_capacity",
]


def _sqrtm_psd(r: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(r)
    return (v * np.sqrt(np.maximum(w, 0.0))) @ v.conj().T


def exponential_correlation(n: int, rho: float) -> np.ndarray:
    """Exponential correlation matrix ``R_ij = rho^|i-j|`` (``0 <= rho < 1``): a standard model of closely spaced elements."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def rayleigh_channel(
    nr: int, nt: int, rng: np.random.Generator, r_rx: Optional[Any] = None, r_tx: Optional[Any] = None, size: Optional[int] = None
) -> np.ndarray:
    """Kronecker-correlated Rayleigh channel(s): shape ``(Nr, Nt)`` or ``(size, Nr, Nt)``."""
    shape = (nr, nt) if size is None else (size, nr, nt)
    g = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / math.sqrt(2.0)
    if r_rx is not None:
        g = _sqrtm_psd(np.asarray(r_rx, complex)) @ g
    if r_tx is not None:
        g = g @ _sqrtm_psd(np.asarray(r_tx, complex))
    return g


def geometric_channel(
    tx: PhasedArray,
    rx: PhasedArray,
    az_t: float,
    el_t: float,
    az_r: float,
    el_r: float,
    k_factor_db: float,
    rng: np.random.Generator,
    r_rx: Optional[Any] = None,
    r_tx: Optional[Any] = None,
) -> np.ndarray:
    """One ``(Nr, Nt)`` realisation with a specular LoS between the given departure/arrival angles plus correlated scatter."""
    k = 10.0 ** (k_factor_db / 10.0)
    a_t = tx.steering_vector(az_t, el_t)
    a_r = rx.steering_vector(az_r, el_r)
    los = np.outer(a_r, np.conj(a_t))
    g = rayleigh_channel(rx.n_elements, tx.n_elements, rng, r_rx, r_tx)
    return math.sqrt(k / (k + 1.0)) * los + math.sqrt(1.0 / (k + 1.0)) * g


def mimo_capacity(h: Any, snr_linear: float) -> np.ndarray:
    """Capacity [bit/s/Hz] with equal power allocation: ``log2 det(I + snr / Nt H H^H)``; ``h`` may be a batch ``(..., Nr, Nt)``."""
    h = np.asarray(h, complex)
    nt = h.shape[-1]
    m = np.eye(h.shape[-2]) + (snr_linear / nt) * (h @ np.swapaxes(h.conj(), -1, -2))
    sign, logdet = np.linalg.slogdet(m)
    return logdet / math.log(2.0)


def water_filling_capacity(h: Any, snr_linear: float) -> float:
    """Capacity [bit/s/Hz] with water-filling over the eigenmodes (transmitter knows ``H``); ``snr_linear`` = total transmit power / noise."""
    g = np.linalg.svd(np.asarray(h, complex), compute_uv=False) ** 2  # eigenvalues of H H^H, descending
    g = g[g > 1e-12]
    if len(g) == 0:
        return 0.0
    for k in range(len(g), 0, -1):  # use the k strongest modes; accept the largest k whose weakest mode gets positive power
        inv = 1.0 / g[:k]
        mu = (snr_linear + inv.sum()) / k
        p = mu - inv
        if p[-1] > 0:
            return float(np.sum(np.log2(1.0 + g[:k] * p)))
    return float(np.log2(1.0 + snr_linear * g[0]))


def ergodic_capacity(
    nr: int, nt: int, snr_linear: float, rng: np.random.Generator, n_samples: int = 2000, r_rx: Optional[Any] = None, r_tx: Optional[Any] = None
) -> float:
    """Monte-Carlo ergodic capacity of a correlated Rayleigh channel [bit/s/Hz]."""
    h = rayleigh_channel(nr, nt, rng, r_rx, r_tx, size=n_samples)
    return float(np.mean(mimo_capacity(h, snr_linear)))


def condition_number_db(h: Any) -> np.ndarray:
    """Channel condition number ``20 log10(sigma_max / sigma_min)`` [dB]: large values mean a near-rank-deficient (LoS) channel."""
    s = np.linalg.svd(np.asarray(h, complex), compute_uv=False)
    return 20.0 * np.log10(np.maximum(s[..., 0], 1e-30) / np.maximum(s[..., -1], 1e-30))
