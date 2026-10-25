"""Small-scale fading, shadowing and Doppler statistics."""

from .distributions import (
    nakagami_gain,
    nakagami_m_from_rician_k,
    rayleigh_gain,
    rayleigh_outage_probability,
    rician_gain,
    rician_k_from_nakagami_m,
    rician_outage_probability,
)
from .doppler import average_fade_duration, coherence_time, jakes_autocorrelation, jakes_spectrum, level_crossing_rate, max_doppler_hz
from .shadowing import DECORRELATION_DISTANCE_M, CorrelatedShadowing, decorrelation_distance
from .temporal import FadingProcess, JakesProcess

__all__ = [
    "DECORRELATION_DISTANCE_M",
    "CorrelatedShadowing",
    "FadingProcess",
    "JakesProcess",
    "average_fade_duration",
    "coherence_time",
    "decorrelation_distance",
    "jakes_autocorrelation",
    "jakes_spectrum",
    "level_crossing_rate",
    "max_doppler_hz",
    "nakagami_gain",
    "nakagami_m_from_rician_k",
    "rayleigh_gain",
    "rayleigh_outage_probability",
    "rician_gain",
    "rician_k_from_nakagami_m",
    "rician_outage_probability",
]
