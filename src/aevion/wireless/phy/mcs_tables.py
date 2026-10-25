"""NR modulation-and-coding tables (3GPP TS 38.214) and transport-block arithmetic.

* ``MCS_TABLE_1``: Table 5.1.3.1-1 (up to 64-QAM), ``MCS_TABLE_2``: Table 5.1.3.1-2 (up to 256-QAM).
* ``CQI_TABLE_1`` / ``CQI_TABLE_2``: Tables 5.2.2.1-2 / 5.2.2.1-4 (CQI index -> modulation, code rate x 1024).

Each row is ``(modulation order Qm, target code rate x 1024)``. The transport-block size here is the *information bit count*
``N_info = N_RE * R * Qm * layers`` of TS 38.214 5.1.3.2 step 2, without the final quantisation to the standard's TBS table and
without LDPC segmentation overhead (both change the result by well under 1 %).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["CQI_TABLE_1", "CQI_TABLE_2", "MCS_TABLE_1", "MCS_TABLE_2", "McsEntry", "McsTable", "get_table", "transport_block_bits"]

MCS_TABLE_1: List[Tuple[int, float]] = [
    (2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379), (2, 449), (2, 526), (2, 602), (2, 679),
    (4, 340), (4, 378), (4, 434), (4, 490), (4, 553), (4, 616), (4, 658),
    (6, 438), (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873), (6, 910), (6, 948),
]  # fmt: skip

MCS_TABLE_2: List[Tuple[int, float]] = [
    (2, 120), (2, 193), (2, 308), (2, 449), (2, 602),
    (4, 378), (4, 434), (4, 490), (4, 553), (4, 616), (4, 658),
    (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873),
    (8, 682.5), (8, 711), (8, 754), (8, 797), (8, 841), (8, 885), (8, 916.5), (8, 948),
]  # fmt: skip

CQI_TABLE_1: List[Tuple[int, float]] = [
    (2, 78), (2, 120), (2, 193), (2, 308), (2, 449), (2, 602),
    (4, 378), (4, 490), (4, 616),
    (6, 466), (6, 567), (6, 666), (6, 772), (6, 873), (6, 948),
]  # fmt: skip

CQI_TABLE_2: List[Tuple[int, float]] = [
    (2, 78), (2, 193), (2, 449),
    (4, 378), (4, 490), (4, 616),
    (6, 466), (6, 567), (6, 666), (6, 772), (6, 873),
    (8, 711), (8, 797), (8, 885), (8, 948),
]  # fmt: skip


@dataclass(frozen=True)
class McsEntry:
    index: int
    qm: int
    rate: float  # target code rate (<= 1)

    @property
    def spectral_efficiency(self) -> float:
        """Information bits per resource element, ``Qm * R``."""
        return self.qm * self.rate


class McsTable:
    """An ordered MCS (or CQI) table with efficiency lookups."""

    def __init__(self, rows: Sequence[Tuple[int, float]], name: str = "mcs"):
        self.name = name
        self.entries = [McsEntry(i, int(q), float(r) / 1024.0) for i, (q, r) in enumerate(rows)]
        self.efficiency = np.array([e.spectral_efficiency for e in self.entries])
        # the standard's tables dip by < 0.005 bit/RE at modulation changes (e.g. MCS 16 -> 17); anything larger is an error
        if np.any(np.diff(self.efficiency) < -0.01):
            raise ValueError("table must have non-decreasing spectral efficiency")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> McsEntry:
        return self.entries[i]

    def highest_below(self, efficiency: float) -> int:
        """Index of the most efficient entry not exceeding ``efficiency`` (-1 when even the first is too demanding)."""
        return int(np.searchsorted(np.maximum.accumulate(self.efficiency), efficiency, side="right") - 1)


_TABLES = {
    "mcs1": McsTable(MCS_TABLE_1, "mcs1"),
    "mcs2": McsTable(MCS_TABLE_2, "mcs2"),
    "cqi1": McsTable(CQI_TABLE_1, "cqi1"),
    "cqi2": McsTable(CQI_TABLE_2, "cqi2"),
}


def get_table(name: str = "mcs1") -> McsTable:
    if name not in _TABLES:
        raise KeyError(f"unknown table {name!r}; choose from {sorted(_TABLES)}")
    return _TABLES[name]


def transport_block_bits(entry: McsEntry, n_prb: int, symbols: int = 14, dmrs_re: int = 12, overhead_re: int = 0, layers: int = 1) -> float:
    """Information bits carried by ``n_prb`` PRBs in one slot (TS 38.214 5.1.3.2 step 2, unquantised)."""
    n_re = min(12 * symbols - dmrs_re - overhead_re, 156) * n_prb
    return n_re * entry.spectral_efficiency * layers
