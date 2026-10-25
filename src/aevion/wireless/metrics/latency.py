"""Latency budgets for control and telemetry links."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..phy.nr_numerology import Numerology

__all__ = ["LatencyBudget", "air_interface_latency_s", "md1_wait_s", "mm1_sojourn_s", "reliability_after_harq"]


def air_interface_latency_s(
    numerology: Numerology,
    n_transmissions: Any = 1,
    harq_rtt_slots: int = 8,
    scheduling_wait_slots: float = 1.5,
    tx_slots: int = 1,
    processing_slots: float = 2.0,
) -> np.ndarray:
    """Expected one-way air-interface latency: scheduling wait + transmission + processing + ``(n_tx - 1)`` HARQ round trips."""
    slot = numerology.slot_duration_s
    n = np.asarray(n_transmissions, float)
    return (scheduling_wait_slots + tx_slots + processing_slots + (n - 1.0) * harq_rtt_slots) * slot


def mm1_sojourn_s(arrival_rate: float, service_rate: float) -> float:
    """Mean time in an M/M/1 system (waiting + service): ``1 / (mu - lambda)``."""
    if arrival_rate >= service_rate:
        return float("inf")
    return 1.0 / (service_rate - arrival_rate)


def md1_wait_s(arrival_rate: float, service_time_s: float) -> float:
    """Mean *waiting* time in an M/D/1 queue (Pollaczek-Khinchine): ``rho T / (2 (1 - rho))``, ``rho = lambda T``."""
    rho = arrival_rate * service_time_s
    if rho >= 1.0:
        return float("inf")
    return rho * service_time_s / (2.0 * (1.0 - rho))


def reliability_after_harq(first_tx_bler: Any, n_transmissions: int) -> np.ndarray:
    """Delivery probability after up to ``n`` transmissions with independent errors: ``1 - bler^n`` (no combining gain)."""
    return 1.0 - np.asarray(first_tx_bler, float) ** n_transmissions


@dataclass
class LatencyBudget:
    """End-to-end command latency of a UAV control loop over a radio link (seconds)."""

    uplink_s: float
    network_s: float
    downlink_s: float
    processing_s: float

    @property
    def total_s(self) -> float:
        return self.uplink_s + self.network_s + self.downlink_s + self.processing_s

    def stable_for(self, max_delay_s: float) -> bool:
        """Is the loop within a controller's delay margin?"""
        return self.total_s <= max_delay_s
