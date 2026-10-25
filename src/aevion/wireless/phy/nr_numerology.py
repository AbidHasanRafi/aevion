"""5G NR numerology and resource-grid arithmetic (3GPP TS 38.211 / 38.101 / 38.104).

Tables of the maximum transmission bandwidth configuration ``N_RB`` (TS 38.101-1 Table 5.3.2-1 for FR1, TS 38.101-2
Table 5.3.2-1 for FR2) are transcribed for the common channel bandwidths; combinations that do not exist in the standard raise
``KeyError`` rather than being interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = ["MAX_PRB", "Numerology", "max_prb", "peak_data_rate_bps", "re_per_prb"]

#: (frequency range, SCS kHz) -> {channel bandwidth MHz: N_RB}
MAX_PRB: Dict[Tuple[str, int], Dict[int, int]] = {
    ("FR1", 15): {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 40: 216, 50: 270},
    ("FR1", 30): {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106, 50: 133, 60: 162, 70: 189, 80: 217, 90: 245, 100: 273},
    ("FR1", 60): {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 40: 51, 50: 65, 60: 79, 70: 93, 80: 107, 90: 121, 100: 135},
    ("FR2", 60): {50: 66, 100: 132, 200: 264},
    ("FR2", 120): {50: 32, 100: 66, 200: 132, 400: 264},
}


def max_prb(bandwidth_mhz: int, scs_khz: int, fr: str = "FR1") -> int:
    """Number of resource blocks for a channel bandwidth and subcarrier spacing."""
    try:
        return MAX_PRB[(fr.upper(), scs_khz)][bandwidth_mhz]
    except KeyError:
        valid = {k: sorted(v) for k, v in MAX_PRB.items()}
        raise KeyError(f"no NR configuration for {fr} {scs_khz} kHz SCS with {bandwidth_mhz} MHz; valid: {valid}") from None


@dataclass(frozen=True)
class Numerology:
    """NR numerology ``mu`` (0..3): subcarrier spacing ``15 * 2^mu`` kHz, slot duration ``1 / 2^mu`` ms."""

    mu: int = 1

    def __post_init__(self) -> None:
        if self.mu not in (0, 1, 2, 3, 4):
            raise ValueError("numerology mu must be 0..4")

    @property
    def scs_khz(self) -> int:
        return 15 * 2**self.mu

    @property
    def slot_duration_s(self) -> float:
        return 1e-3 / 2**self.mu

    @property
    def slots_per_subframe(self) -> int:
        return 2**self.mu

    @property
    def symbols_per_slot(self) -> int:
        return 14  # normal cyclic prefix

    @property
    def symbol_duration_s(self) -> float:
        return self.slot_duration_s / self.symbols_per_slot

    @property
    def prb_bandwidth_hz(self) -> float:
        return 12 * self.scs_khz * 1e3

    @property
    def frequency_range(self) -> str:
        return "FR2" if self.mu >= 2 else "FR1"

    def n_prb(self, bandwidth_mhz: int, fr: str | None = None) -> int:
        return max_prb(bandwidth_mhz, self.scs_khz, fr or ("FR2" if self.mu >= 3 else "FR1"))

    def occupied_bandwidth_hz(self, bandwidth_mhz: int, fr: str | None = None) -> float:
        return self.n_prb(bandwidth_mhz, fr) * self.prb_bandwidth_hz


def re_per_prb(symbols: int = 14, dmrs_re: int = 12, overhead_re: int = 0) -> int:
    """Data resource elements in one PRB over one slot: ``12 * symbols - DMRS - xOverhead`` (TS 38.214 5.1.3.2, before the 156 cap)."""
    return min(12 * symbols - dmrs_re - overhead_re, 156)


def peak_data_rate_bps(
    numerology: Numerology, n_prb: int, qm: int = 8, code_rate: float = 948 / 1024, layers: int = 1, overhead: float = 0.14
) -> float:
    """Approximate peak rate (TS 38.306 4.1.2): ``layers * Qm * R * N_PRB * 12 / T_s * (1 - overhead)`` with ``T_s = 1e-3 / (14 * 2^mu)``."""
    t_s = 1e-3 / (14 * 2**numerology.mu)
    return layers * qm * code_rate * n_prb * 12 / t_s * (1.0 - overhead)
