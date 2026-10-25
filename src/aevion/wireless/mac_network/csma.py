"""IEEE 802.11 DCF saturation throughput (Bianchi's Markov-chain model, JSAC 18(3), 2000).

For ``n`` saturated stations with minimum contention window ``W`` and ``m`` backoff stages, the per-slot transmission
probability ``tau`` and conditional collision probability ``p`` solve the fixed point

``tau = 2 (1 - 2p) / ((1 - 2p)(W + 1) + p W (1 - (2p)^m))``, ``p = 1 - (1 - tau)^(n-1)``.

Normalised throughput follows from the slot-time accounting of successes, collisions and idle slots; the basic-access and
RTS/CTS timings are both provided. This is the right tool for **UAV swarm ad-hoc networks** that share one channel: it
predicts how aggregate throughput collapses with swarm size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["BianchiParams", "bianchi_fixed_point", "bianchi_throughput_bps", "collision_probability"]


@dataclass(frozen=True)
class BianchiParams:
    """802.11b-style timing used by Bianchi's paper (all times in microseconds, sizes in bits at ``rate_bps``)."""

    payload_bits: int = 8184
    mac_header_bits: int = 272
    phy_header_bits: int = 128
    ack_bits: int = 112
    rts_bits: int = 160
    cts_bits: int = 112
    rate_bps: float = 1e6
    slot_us: float = 50.0
    sifs_us: float = 28.0
    difs_us: float = 128.0
    prop_us: float = 1.0
    cw_min: int = 32
    backoff_stages: int = 3

    def _t(self, bits: int) -> float:
        return bits / self.rate_bps * 1e6

    @property
    def t_success_basic(self) -> float:
        h = self._t(self.phy_header_bits + self.mac_header_bits)
        ack = self._t(self.phy_header_bits + self.ack_bits)
        return h + self._t(self.payload_bits) + self.sifs_us + self.prop_us + ack + self.difs_us + self.prop_us

    @property
    def t_collision_basic(self) -> float:
        h = self._t(self.phy_header_bits + self.mac_header_bits)
        return h + self._t(self.payload_bits) + self.difs_us + self.prop_us

    @property
    def t_success_rts(self) -> float:
        h = self._t(self.phy_header_bits + self.mac_header_bits)
        rts = self._t(self.phy_header_bits + self.rts_bits)
        cts = self._t(self.phy_header_bits + self.cts_bits)
        ack = self._t(self.phy_header_bits + self.ack_bits)
        return (
            rts
            + self.sifs_us
            + self.prop_us
            + cts
            + self.sifs_us
            + self.prop_us
            + h
            + self._t(self.payload_bits)
            + self.sifs_us
            + self.prop_us
            + ack
            + self.difs_us
            + self.prop_us
        )

    @property
    def t_collision_rts(self) -> float:
        return self._t(self.phy_header_bits + self.rts_bits) + self.difs_us + self.prop_us


def bianchi_fixed_point(n: int, w: int = 32, m: int = 3) -> Tuple[float, float]:
    """``(tau, p)``: per-slot transmission and conditional collision probabilities for ``n`` saturated stations."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return 2.0 / (w + 1.0), 0.0

    def tau_of(p: float) -> float:
        if abs(p - 0.5) < 1e-9:  # removable singularity of the closed form
            p = 0.5 + 1e-9
        return 2.0 * (1.0 - 2.0 * p) / ((1.0 - 2.0 * p) * (w + 1.0) + p * w * (1.0 - (2.0 * p) ** m))

    from scipy.optimize import brentq

    p = brentq(lambda x: x - (1.0 - (1.0 - tau_of(x)) ** (n - 1)), 1e-12, 1.0 - 1e-12)
    return tau_of(p), p


def collision_probability(n: int, w: int = 32, m: int = 3) -> float:
    return bianchi_fixed_point(n, w, m)[1]


def bianchi_throughput_bps(n: int, params: Optional[BianchiParams] = None, rts_cts: bool = False) -> float:
    """Aggregate saturation throughput [bit/s] of ``n`` stations (payload bits only)."""
    params = params or BianchiParams()
    tau, _ = bianchi_fixed_point(n, params.cw_min, params.backoff_stages)
    p_tr = 1.0 - (1.0 - tau) ** n
    p_s = n * tau * (1.0 - tau) ** (n - 1) / p_tr
    ts = params.t_success_rts if rts_cts else params.t_success_basic
    tc = params.t_collision_rts if rts_cts else params.t_collision_basic
    mean_slot_us = (1.0 - p_tr) * params.slot_us + p_tr * p_s * ts + p_tr * (1.0 - p_s) * tc
    payload_per_slot = p_s * p_tr * params.payload_bits
    if math.isclose(mean_slot_us, 0.0):
        return 0.0
    return payload_per_slot / (mean_slot_us * 1e-6)
