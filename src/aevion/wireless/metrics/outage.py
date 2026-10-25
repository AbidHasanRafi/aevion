"""Outage and availability statistics of a link, from time series or analytically."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List

import numpy as np

__all__ = ["OutageStats", "availability", "outage_events", "outage_probability", "rayleigh_outage_probability_db"]


def outage_probability(sinr_db: Any, threshold_db: float = -6.0, axis: int = 0) -> np.ndarray:
    """Fraction of samples below the SINR threshold."""
    return np.mean(np.asarray(sinr_db, float) < threshold_db, axis=axis)


def availability(outage: Any, axis: int = 0) -> np.ndarray:
    """Fraction of time the link is *not* in outage (a boolean series)."""
    return 1.0 - np.mean(np.asarray(outage, bool), axis=axis)


def rayleigh_outage_probability_db(mean_snr_db: Any, threshold_db: Any) -> np.ndarray:
    """``P[SNR < thr]`` under Rayleigh fading: ``1 - exp(-thr / mean)`` (both in dB inputs)."""
    return 1.0 - np.exp(-(10 ** (np.asarray(threshold_db, float) / 10.0)) / (10 ** (np.asarray(mean_snr_db, float) / 10.0)))


@dataclass
class OutageStats:
    probability: float
    n_events: int
    mean_duration_s: float
    longest_s: float
    durations_s: List[float]
    time_to_first_s: float  # math.inf when there is no outage


def outage_events(outage: Any, dt: float) -> OutageStats:
    """Statistics of contiguous outage periods of a boolean series sampled every ``dt`` seconds."""
    o = np.asarray(outage, bool).ravel()
    if o.size == 0:
        return OutageStats(0.0, 0, 0.0, 0.0, [], math.inf)
    d = np.diff(np.concatenate([[0], o.astype(int), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    durs = ((ends - starts) * dt).tolist()
    first = float(starts[0] * dt) if len(starts) else math.inf
    return OutageStats(float(o.mean()), len(durs), float(np.mean(durs)) if durs else 0.0, float(max(durs)) if durs else 0.0, durs, first)
