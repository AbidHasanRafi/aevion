"""Link-level abstraction: SINR to spectral efficiency and block error rate.

1. **Attenuated Shannon** (3GPP TR 36.942 Annex A): ``SE(sinr) = alpha * log2(1 + sinr / sinr_eff)``, clipped to ``[0, se_max]`` and
   zero below ``sinr_min``. It captures what a real modem achieves relative to the Shannon bound.
2. **BLER per MCS**: each MCS has a required SINR, obtained by inverting the same law at its spectral efficiency. The block error rate
   is a steep complementary-Gaussian curve of the SINR margin, ``BLER = Q((sinr_dB - sinr_req_dB) / sigma_dB)``, shifted so that
   ``BLER = bler_ref`` (10 %) at the required SINR. ``sigma_dB`` (about 0.5-1 dB for long LDPC blocks) sets the waterfall steepness.

This is a calibratable abstraction, not a bit-accurate link-level simulator. Where the exact waterfall matters, replace
:meth:`BlerModel.bler` with curves measured from a link-level simulation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..units import db2lin, lin2db
from .mcs_tables import McsTable

__all__ = ["BlerModel", "attenuated_shannon_se", "required_sinr_db"]


def attenuated_shannon_se(sinr_db: Any, alpha: float = 0.6, sinr_eff_db: float = 0.0, se_max: float = 4.4, sinr_min_db: float = -10.0) -> np.ndarray:
    """Spectral efficiency [bit/s/Hz] from the TR 36.942 attenuated Shannon law (defaults: 3GPP downlink, ``alpha = 0.6``, ``SE_max = 4.4``)."""
    s = np.asarray(sinr_db, float)
    se = alpha * np.log2(1.0 + db2lin(s - sinr_eff_db))
    return np.where(s < sinr_min_db, 0.0, np.minimum(se, se_max))


def required_sinr_db(se: Any, alpha: float = 0.6, sinr_eff_db: float = 0.0) -> np.ndarray:
    """Inverse of :func:`attenuated_shannon_se` (without the clipping): SINR needed for spectral efficiency ``se``."""
    se = np.asarray(se, float)
    return lin2db(2.0 ** (se / alpha) - 1.0) + sinr_eff_db


def _q(x: Any) -> np.ndarray:
    from scipy.special import erfc  # lazy: scipy.special is a heavy import

    return 0.5 * erfc(np.asarray(x, float) / math.sqrt(2.0))


class BlerModel:
    """BLER of every entry of an MCS table as a function of SINR [dB]."""

    def __init__(self, table: McsTable, alpha: float = 0.6, sinr_eff_db: float = 0.0, sigma_db: float = 0.7, bler_ref: float = 0.1):
        self.table, self.alpha, self.sinr_eff_db, self.sigma_db, self.bler_ref = table, alpha, sinr_eff_db, sigma_db, bler_ref
        self._req_ref = np.maximum.accumulate(required_sinr_db(table.efficiency, alpha, sinr_eff_db))  # SINR at which BLER == bler_ref (monotone)
        # Q((x - mu)/sigma) = bler_ref at x = req  =>  mu = req - sigma * Q^-1(bler_ref)
        from scipy.special import erfcinv

        self._shift = sigma_db * math.sqrt(2.0) * float(erfcinv(2.0 * bler_ref))
        self._mu = self._req_ref - self._shift

    def required_sinr_db(self, mcs: Any) -> np.ndarray:
        """SINR [dB] at which the MCS reaches the reference BLER."""
        return self._req_ref[np.asarray(mcs, int)]

    def bler(self, sinr_db: Any, mcs: Any) -> np.ndarray:
        """Block error rate of ``mcs`` at ``sinr_db`` (broadcast)."""
        mu = self._mu[np.asarray(mcs, int)]
        return _q((np.asarray(sinr_db, float) - mu) / self.sigma_db)

    def best_mcs(self, sinr_db: Any, target_bler: float = 0.1) -> np.ndarray:
        """Highest MCS whose BLER does not exceed ``target_bler`` at ``sinr_db`` (-1 when none qualifies)."""
        from scipy.special import erfcinv

        s = np.asarray(sinr_db, float)
        # BLER(mcs) <= target  <=>  sinr >= mu + sigma * Q^-1(target)
        thr = self._mu + self.sigma_db * math.sqrt(2.0) * float(erfcinv(2.0 * target_bler))
        return np.searchsorted(thr, s, side="right") - 1

    def goodput_se(self, sinr_db: Any, mcs: Any) -> np.ndarray:
        """Expected delivered bits per resource element: ``Qm R (1 - BLER)``."""
        return self.table.efficiency[np.asarray(mcs, int)] * (1.0 - self.bler(sinr_db, mcs))
