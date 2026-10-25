"""GNSS spoofing threat model for resilience research: capture condition and the induced position error.

A counterfeit-signal transmitter is *captured by the victim receiver* only when its power at the receiver exceeds the authentic
signal by a margin (typically 1-3 dB). The induced position error then grows at some rate. These models exist so that
detection (innovation monitoring, multi-sensor consistency), estimation and control strategies can be evaluated against a
defined threat; they do not model the signal-level attack itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..propagation.base import fspl_db
from .jamming import GNSS_L1_HZ, GNSS_NOMINAL_POWER_DBM

__all__ = ["capture_probability", "pull_offset", "spoofer_power_advantage_db"]


def spoofer_power_advantage_db(
    spoofer_pos: Any,
    victim_pos: Any,
    spoofer_power_dbm: float = 10.0,
    gain_dbi: float = 0.0,
    authentic_dbm: float = GNSS_NOMINAL_POWER_DBM,
    freq_hz: float = GNSS_L1_HZ,
) -> np.ndarray:
    """Counterfeit minus authentic received power [dB] at the victim (free space)."""
    d = np.linalg.norm(np.asarray(victim_pos, float) - np.asarray(spoofer_pos, float), axis=-1)
    return spoofer_power_dbm + gain_dbi - fspl_db(np.maximum(d, 1.0), freq_hz) - authentic_dbm


def capture_probability(advantage_db: Any, threshold_db: float = 2.0, steepness_db: float = 1.0) -> np.ndarray:
    """Probability of capture: logistic in the power advantage (0.5 at ``threshold_db``)."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(advantage_db, float) - threshold_db) / max(steepness_db, 1e-9)))


def pull_offset(t: Any, t_capture: float, rate_mps: float, max_offset_m: float = math.inf) -> np.ndarray:
    """Magnitude of the induced position error: ``min(rate * (t - t_capture), max)`` after capture, else 0."""
    return np.minimum(np.maximum(np.asarray(t, float) - t_capture, 0.0) * rate_mps, max_offset_m)
