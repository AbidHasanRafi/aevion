"""Link adaptation: delayed/quantised CQI feedback, outer-loop correction (OLLA) and HARQ with chase combining.

The loop works at *evaluation-interval* granularity (the network is evaluated at 10-100 Hz, not every slot): one call
represents ``dt / slot_duration`` slots at the current SINR. ACK/NACK counts are drawn as binomials and fed to OLLA, and the
goodput and latency are the analytic expectations for those slots, so the result is both statistically correct and cheap.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

import numpy as np

from .bler_curves import BlerModel
from .mcs_tables import McsTable, get_table, transport_block_bits
from .nr_numerology import Numerology

__all__ = ["HarqStats", "LaResult", "LinkAdaptation", "harq_statistics"]


@dataclass
class HarqStats:
    expected_transmissions: np.ndarray
    residual_bler: np.ndarray
    first_tx_bler: np.ndarray


def harq_statistics(model: BlerModel, sinr_db: Any, mcs: Any, max_tx: int = 4, combining: bool = True) -> HarqStats:
    """Expected transmissions and residual BLER with HARQ (chase combining adds ``10 log10 k`` dB at the k-th attempt)."""
    sinr = np.asarray(sinr_db, float)
    mcs = np.asarray(mcs, int)
    p_first = model.bler(sinr, mcs)
    reach = np.ones_like(p_first)  # probability that attempt k happens
    expected = np.zeros_like(p_first)
    for k in range(1, max_tx + 1):
        expected = expected + reach
        gain = 10.0 * math.log10(k) if combining else 0.0
        reach = reach * model.bler(sinr + gain, mcs)
    return HarqStats(expected, reach, p_first)


@dataclass
class LaResult:
    mcs: np.ndarray  # selected MCS (-1: link unusable)
    bler: np.ndarray  # first-transmission BLER at the true SINR
    residual_bler: np.ndarray  # after HARQ
    rate_bps: np.ndarray  # goodput
    latency_s: np.ndarray  # expected HARQ latency added to a transport block
    olla_offset_db: np.ndarray
    transport_block_bits: np.ndarray


class LinkAdaptation:
    """Per-link MCS selection with OLLA. ``n_links`` links are adapted independently."""

    def __init__(
        self,
        n_links: int,
        numerology: Optional[Numerology] = None,
        table: str = "mcs1",
        bler_model: Optional[BlerModel] = None,
        target_bler: float = 0.1,
        cqi_delay_s: float = 0.005,
        cqi_bias_db: float = 0.0,
        olla: bool = True,
        olla_step_db: float = 0.1,
        olla_limit_db: float = 6.0,
        max_harq_tx: int = 4,
        harq_rtt_s: float = 0.004,
        overhead: float = 0.14,
        layers: int = 1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n = n_links
        self.num = numerology if numerology is not None else Numerology(1)
        self.table: McsTable = get_table(table)
        self.model = bler_model or BlerModel(self.table)
        self.target, self.olla, self.step_db, self.limit = target_bler, olla, olla_step_db, olla_limit_db
        self.cqi_delay, self.max_tx, self.rtt, self.overhead, self.layers = cqi_delay_s, max_harq_tx, harq_rtt_s, overhead, layers
        self.cqi_bias = float(cqi_bias_db)  # measurement/implementation bias of the reported SINR (positive = optimistic)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        self.offset = np.zeros(self.n)
        self._hist: Deque[tuple] = deque()  # (time, sinr_db) pairs awaiting the feedback delay

    def _delayed_sinr(self, t: float, sinr_db: np.ndarray) -> np.ndarray:
        self._hist.append((t, sinr_db.copy()))
        out = self._hist[0][1]
        while len(self._hist) > 1 and self._hist[1][0] <= t - self.cqi_delay:
            self._hist.popleft()
            out = self._hist[0][1]
        return out

    def step(
        self,
        t: float,
        sinr_db: Any,
        dt: float,
        n_prb: int,
        slots_active: Optional[Any] = None,
        sinr_samples_db: Optional[Any] = None,
        sample_weights: Optional[Any] = None,
    ) -> LaResult:
        """Adapt and evaluate one interval.

        ``sinr_db`` is the SINR the link adapts to (the CQI report). ``slots_active`` (``(n,)`` or scalar, default all) limits the slots
        in which the link transmits (e.g. scheduler share). When the interval spans several fading coherence times the BLER must be
        the *average over the fading distribution*, not the value at one snapshot: pass ``sinr_samples_db`` ``(n, S)`` (SINR values
        spread over the fading distribution) and ``sample_weights`` ``(n, S)`` (summing to one); the error statistics are then the
        weighted means over those samples.
        """
        sinr = np.broadcast_to(np.asarray(sinr_db, float), (self.n,)).copy()
        reported = self._delayed_sinr(t, sinr)
        mcs = self.model.best_mcs(reported + self.cqi_bias + self.offset, self.target)  # selection uses the stale report plus the OLLA offset
        usable = mcs >= 0
        mcs_c = np.maximum(mcs, 0)
        if sinr_samples_db is None:
            harq = harq_statistics(self.model, sinr, mcs_c, self.max_tx)
        else:
            samples = np.asarray(sinr_samples_db, float).reshape(self.n, -1)
            w = np.ones_like(samples) / samples.shape[1] if sample_weights is None else np.asarray(sample_weights, float).reshape(self.n, -1)
            each = harq_statistics(self.model, samples, mcs_c[:, None], self.max_tx)
            harq = HarqStats((w * each.expected_transmissions).sum(1), (w * each.residual_bler).sum(1), (w * each.first_tx_bler).sum(1))
        n_slots = max(round(dt / self.num.slot_duration_s), 1)
        share = np.ones(self.n) if slots_active is None else np.clip(np.broadcast_to(np.asarray(slots_active, float), (self.n,)), 0.0, 1.0)
        tx_slots = np.maximum(np.round(n_slots * share), 0).astype(int)
        if self.olla:
            # Classic OLLA (ACK: +up, NACK: -down with up/down = target/(1-target)), aggregated over the interval's slots:
            # the offset moves by -step * (nack_fraction - target) / target, so it is stationary exactly at BLER == target.
            nack = self.rng.binomial(tx_slots, np.clip(harq.first_tx_bler, 0.0, 1.0))
            frac = np.where(tx_slots > 0, nack / np.maximum(tx_slots, 1), self.target)
            delta = -self.step_db * (frac - self.target) / max(self.target, 1e-6)
            self.offset = np.clip(self.offset + np.where(usable & (tx_slots > 0), delta, 0.0), -self.limit, self.limit)
        bits = np.array([transport_block_bits(self.table[int(m)], n_prb, layers=self.layers) for m in mcs_c]) * (1.0 - self.overhead)
        slots_per_s = 1.0 / self.num.slot_duration_s
        rate = np.where(usable, bits * slots_per_s * share * (1.0 - harq.residual_bler) / harq.expected_transmissions, 0.0)
        latency = np.where(usable, (harq.expected_transmissions - 1.0) * self.rtt, 0.0)
        return LaResult(mcs, harq.first_tx_bler, harq.residual_bler, rate, latency, self.offset.copy(), bits)
