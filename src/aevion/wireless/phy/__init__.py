"""PHY abstraction: NR numerology, MCS/CQI tables, BLER curves, link adaptation with HARQ, OFDM effective-SINR mapping."""

from .bler_curves import BlerModel, attenuated_shannon_se, required_sinr_db
from .link_adaptation import HarqStats, LaResult, LinkAdaptation, harq_statistics
from .mcs_tables import CQI_TABLE_1, CQI_TABLE_2, MCS_TABLE_1, MCS_TABLE_2, McsEntry, McsTable, get_table, transport_block_bits
from .nr_numerology import MAX_PRB, Numerology, max_prb, peak_data_rate_bps, re_per_prb
from .ofdm_abstraction import EESM_BETA, eesm_effective_sinr, eesm_effective_sinr_db, per_rb_sinr_db

__all__ = [
    "CQI_TABLE_1",
    "CQI_TABLE_2",
    "EESM_BETA",
    "MAX_PRB",
    "MCS_TABLE_1",
    "MCS_TABLE_2",
    "BlerModel",
    "HarqStats",
    "LaResult",
    "LinkAdaptation",
    "McsEntry",
    "McsTable",
    "Numerology",
    "attenuated_shannon_se",
    "eesm_effective_sinr",
    "eesm_effective_sinr_db",
    "get_table",
    "harq_statistics",
    "max_prb",
    "peak_data_rate_bps",
    "per_rb_sinr_db",
    "re_per_prb",
    "required_sinr_db",
    "transport_block_bits",
]
