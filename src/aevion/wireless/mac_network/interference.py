"""SINR computation with serving-cell selection, fractional cell load and external interferers (jammers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..units import db2lin, lin2db

__all__ = ["SinrResult", "compute_sinr", "select_serving"]


@dataclass
class SinrResult:
    sinr_db: np.ndarray  # (N,)
    signal_dbm: np.ndarray  # (N,)
    interference_dbm: np.ndarray  # (N,) including jammers, excluding noise
    serving: np.ndarray  # (N,) cell index


def select_serving(rx_dbm: Any, bias_db: Optional[Any] = None) -> np.ndarray:
    """Strongest-cell (max biased RSRP) selection; ``bias_db`` is a per-cell range-expansion offset ``(B,)``."""
    r = np.asarray(rx_dbm, float)
    if bias_db is not None:
        r = r + np.asarray(bias_db, float)[None, :]
    return r.argmax(axis=1)


def compute_sinr(
    rx_dbm: Any,
    serving: Any,
    noise_dbm: float,
    load: Optional[Any] = None,
    jammer_dbm: Optional[Any] = None,
) -> SinrResult:
    """SINR of each user towards its serving cell.

    ``rx_dbm`` is ``(N, B)`` received power from every cell; ``load`` the fraction of resources each cell uses ``(B,)``
    (full buffer = 1: all other cells interfere continuously; lower values model ICIC/low traffic); ``jammer_dbm`` ``(N,)`` or
    ``(N, J)`` received jamming power added to the interference.
    """
    rx = np.asarray(rx_dbm, float)
    n, nb = rx.shape
    serving = np.asarray(serving, int)
    lin = db2lin(rx)
    load_v = np.ones(nb) if load is None else np.broadcast_to(np.asarray(load, float), (nb,))
    sig = lin[np.arange(n), serving]
    interf = (lin * load_v[None, :]).sum(axis=1) - sig * load_v[serving]
    if jammer_dbm is not None:
        j = db2lin(np.asarray(jammer_dbm, float))
        interf = interf + (j.reshape(n, -1).sum(axis=1) if j.ndim > 1 else j)
    noise = float(db2lin(noise_dbm))
    return SinrResult(lin2db(sig / (interf + noise)), lin2db(sig), lin2db(np.maximum(interf, 1e-30)), serving)
