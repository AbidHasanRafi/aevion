"""Jamming physics for communication and GNSS links (for resilience studies of UAV navigation and communications).

* Received interference power with free-space propagation, duty cycle and spectral overlap;
* communication links: SINR with interference present and the anti-jam margin of a spread-spectrum system (processing gain);
* GNSS: effective carrier-to-noise density under interference, ``(C/N0)_eff^-1 = (C/N0)^-1 + (J/S) / (Q R_c)`` (Betz; ``Q = 1`` for a
  wideband noise source on BPSK C/A code, ``R_c = 1.023 Mchip/s``), and the loss-of-lock decision.

These are link-budget models: they tell a researcher how much interference a receiver tolerates and at which range it degrades,
which is what is needed to evaluate detection, estimation and control strategies that must survive such events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..propagation.base import fspl_db
from ..units import db2lin, lin2db

__all__ = [
    "GNSS_L1_HZ",
    "GNSS_NOMINAL_POWER_DBM",
    "Jammer",
    "anti_jam_margin_db",
    "gnss_cn0_effective_dbhz",
    "gnss_jam_to_signal_db",
    "jammer_range_for_denial_m",
    "sinr_with_jammer_db",
]

GNSS_L1_HZ = 1.57542e9
GNSS_NOMINAL_POWER_DBM = -128.5  # GPS C/A minimum received power on the ground, -158.5 dBW


@dataclass
class Jammer:
    """An interference source. ``duty`` is the fraction of time it is on; ``bandwidth_hz`` its spectral width."""

    pos: Any
    power_dbm: float = 30.0
    gain_dbi: float = 0.0
    freq_hz: float = GNSS_L1_HZ
    bandwidth_hz: float = 20e6
    duty: float = 1.0
    t_start: float = 0.0
    t_end: float = math.inf

    def active(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

    def received_dbm(self, rx_pos: Any, rx_gain_dbi: float = 0.0, rx_bandwidth_hz: Optional[float] = None) -> np.ndarray:
        """Interference power in the victim's receiver [dBm]: free-space loss, spectral overlap and duty-cycle averaged."""
        d = np.linalg.norm(np.asarray(rx_pos, float) - np.asarray(self.pos, float), axis=-1)
        rx = self.power_dbm + self.gain_dbi + rx_gain_dbi - fspl_db(np.maximum(d, 1.0), self.freq_hz)
        if rx_bandwidth_hz is not None:
            rx = rx + lin2db(min(rx_bandwidth_hz / self.bandwidth_hz, 1.0))
        return rx + lin2db(self.duty)


def sinr_with_jammer_db(signal_dbm: Any, noise_dbm: float, jammer_dbm: Any, interference_dbm: Any = None) -> np.ndarray:
    """SINR [dB] with noise, an interferer and optional other interference (all dBm)."""
    denom = db2lin(noise_dbm) + db2lin(jammer_dbm)
    if interference_dbm is not None:
        denom = denom + db2lin(interference_dbm)
    return lin2db(db2lin(signal_dbm) / denom)


def anti_jam_margin_db(processing_gain_db: float, required_snr_db: float, system_losses_db: float = 0.0) -> float:
    """Spread-spectrum anti-jam margin ``M = G_p - SNR_req - L``: the J/S the link tolerates (e.g. GPS C/A: 43 dB gain)."""
    return processing_gain_db - required_snr_db - system_losses_db


def gnss_jam_to_signal_db(jammer_dbm: Any, signal_dbm: float = GNSS_NOMINAL_POWER_DBM) -> np.ndarray:
    return np.asarray(jammer_dbm, float) - signal_dbm


def gnss_cn0_effective_dbhz(cn0_dbhz: float, js_db: Any, chip_rate_hz: float = 1.023e6, q: float = 1.0) -> np.ndarray:
    """Effective C/N0 [dB-Hz] under interference with jam-to-signal ratio ``js_db`` (a wideband noise source matched to the signal band)."""
    inv = 1.0 / db2lin(cn0_dbhz) + db2lin(js_db) / (q * chip_rate_hz)
    return lin2db(1.0 / inv)


def jammer_range_for_denial_m(
    jammer_power_dbm: float,
    cn0_dbhz: float = 43.5,
    lock_threshold_dbhz: float = 28.0,
    signal_dbm: float = GNSS_NOMINAL_POWER_DBM,
    freq_hz: float = GNSS_L1_HZ,
    gain_dbi: float = 0.0,
) -> float:
    """Distance within which such an interferer pushes a receiver below its lock threshold (free space)."""
    inv_target = 1.0 / db2lin(lock_threshold_dbhz) - 1.0 / db2lin(cn0_dbhz)
    if inv_target <= 0:
        return 0.0
    js_db = lin2db(inv_target * 1.023e6)
    loss = jammer_power_dbm + gain_dbi - (signal_dbm + js_db)
    return float(10 ** ((loss - 20.0 * math.log10(4.0 * math.pi * freq_hz / 299_792_458.0)) / 20.0))
