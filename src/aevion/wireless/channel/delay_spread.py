"""RMS delay spread, power-delay profiles and frequency selectivity.

Delay spread statistics follow the log-normal model of 3GPP TR 38.901 Table 7.5-6: ``lg(DS/1 s) ~ N(mu, sigma^2)`` with
``mu, sigma`` depending on scenario, LoS state and carrier frequency (GHz). The coefficients are transcribed from the table;
see the module note in :mod:`aevion.wireless.propagation.tr38901` about verifying them against the standard.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

__all__ = ["coherence_bandwidth_hz", "delay_spread_params", "exponential_pdp", "frequency_response", "rms_delay_spread", "sample_delay_spread_s"]


def delay_spread_params(scenario: str, los: bool, fc_ghz: float) -> Tuple[float, float]:
    """``(mu_lgDS, sigma_lgDS)`` of ``lg(DS [s])`` for ``scenario`` in ``uma, umi, rma, inh``."""
    s = scenario.lower()
    f = max(fc_ghz, 0.5)
    lg = math.log10
    table = {
        ("uma", True): (-6.955 - 0.0963 * lg(f), 0.66),
        ("uma", False): (-6.28 - 0.204 * lg(f), 0.39),
        ("umi", True): (-0.24 * lg(1 + f) - 7.14, 0.38),
        ("umi", False): (-0.24 * lg(1 + f) - 6.83, 0.28),
        ("rma", True): (-7.49, 0.55),
        ("rma", False): (-7.43, 0.48),
        ("inh", True): (-0.01 * lg(1 + f) - 7.692, 0.18),
        ("inh", False): (-0.28 * lg(1 + f) - 7.173, 0.10),
    }
    key = (s, bool(los))
    if key not in table:
        raise KeyError(f"unknown scenario {scenario!r}; choose from uma, umi, rma, inh")
    return table[key]


def sample_delay_spread_s(scenario: str, los: Any, fc_ghz: float, rng: np.random.Generator, size: Optional[Any] = None) -> np.ndarray:
    """Draw RMS delay spreads [s]; ``los`` may be a bool array (one draw per element)."""
    los = np.asarray(los, bool)
    size = los.shape if size is None else size
    mu_l, sg_l = delay_spread_params(scenario, True, fc_ghz)
    mu_n, sg_n = delay_spread_params(scenario, False, fc_ghz)
    mu, sg = np.where(los, mu_l, mu_n), np.where(los, sg_l, sg_n)
    return 10.0 ** (mu + sg * rng.normal(size=size))


def rms_delay_spread(delays_s: Any, powers: Any) -> np.ndarray:
    """RMS delay spread of a power-delay profile ``(..., L)``: ``sqrt(E[tau^2] - E[tau]^2)`` with power weights."""
    t, p = np.asarray(delays_s, float), np.asarray(powers, float)
    w = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-30)
    m1 = (w * t).sum(axis=-1)
    m2 = (w * t * t).sum(axis=-1)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def coherence_bandwidth_hz(rms_ds_s: Any, correlation: float = 0.5) -> np.ndarray:
    """Coherence bandwidth: ``1 / (5 tau_rms)`` for frequency correlation 0.5, ``1 / (50 tau_rms)`` for 0.9 (Rappaport)."""
    k = {0.5: 5.0, 0.9: 50.0}.get(correlation)
    if k is None:
        raise ValueError("correlation must be 0.5 or 0.9")
    return 1.0 / (k * np.maximum(np.asarray(rms_ds_s, float), 1e-15))


def exponential_pdp(rms_ds_s: float, n_taps: int = 16, span: float = 5.0, k_factor_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Tapped-delay-line approximation of an exponential PDP with the given RMS delay spread.

    Taps are uniform on ``[0, span * rms]``; for an exponential PDP ``tau_rms = tau_decay``. With ``k_factor_db`` a LoS
    component is added in the first tap. Returns ``(delays [s], powers)`` normalised to unit total power.
    """
    tau = np.linspace(0.0, span * rms_ds_s, n_taps)
    p = np.exp(-tau / max(rms_ds_s, 1e-15))
    p = p / p.sum()
    if k_factor_db is not None:
        k = 10.0 ** (k_factor_db / 10.0)
        p = p / (k + 1.0)
        p[0] += k / (k + 1.0)
    return tau, p


def frequency_response(delays_s: Any, powers: Any, freqs_hz: Any, rng: np.random.Generator) -> np.ndarray:
    """One frequency response ``H(f) = sum_l a_l exp(-j 2 pi f tau_l)`` with Rayleigh taps ``a_l ~ CN(0, p_l)`` (unit mean power)."""
    t, p, f = np.asarray(delays_s, float), np.asarray(powers, float), np.asarray(freqs_hz, float)
    a = np.sqrt(p / 2.0) * (rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape))
    return (a[None, :] * np.exp(-2j * math.pi * f[:, None] * t[None, :])).sum(axis=1)
