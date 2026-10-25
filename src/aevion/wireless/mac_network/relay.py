"""UAV relaying between a source and a destination (decode-and-forward and amplify-and-forward).

Two-hop link ``S -> R -> D`` with the relay at the UAV position. Per-hop SNRs ``g1`` (S-R) and ``g2`` (R-D), linear:

* **DF, half duplex**: the two hops share time, ``R = B/2 min(log2(1 + g1), log2(1 + g2))``;
* **DF, full duplex ideal**: ``R = B min(...)``;
* **AF**: end-to-end SNR ``g1 g2 / (g1 + g2 + 1)``, ``R = B/2 log2(1 + g_af)``;
* **direct**: ``R = B log2(1 + g_sd)``; :func:`best_rate` picks whichever is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ..propagation.base import fspl_db
from ..units import db2lin, noise_power_dbm

__all__ = ["RelayLink", "best_rate", "best_relay_position", "relay_rate"]


def relay_rate(snr1_linear: Any, snr2_linear: Any, bandwidth_hz: float, scheme: str = "df", half_duplex: bool = True) -> np.ndarray:
    g1, g2 = np.asarray(snr1_linear, float), np.asarray(snr2_linear, float)
    k = 0.5 if half_duplex else 1.0
    if scheme == "df":
        return k * bandwidth_hz * np.minimum(np.log2(1.0 + g1), np.log2(1.0 + g2))
    if scheme == "af":
        return k * bandwidth_hz * np.log2(1.0 + g1 * g2 / (g1 + g2 + 1.0))
    raise ValueError("scheme must be 'df' or 'af'")


def best_rate(snr_direct_linear: Any, snr1_linear: Any, snr2_linear: Any, bandwidth_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(rate, relayed)``: the better of the direct link and DF relaying, and whether the relay was used."""
    direct = bandwidth_hz * np.log2(1.0 + np.asarray(snr_direct_linear, float))
    relayed = relay_rate(snr1_linear, snr2_linear, bandwidth_hz, "df")
    return np.maximum(direct, relayed), relayed > direct


@dataclass
class RelayLink:
    """Free-space two-hop relay budget (LoS air links), for planning the relay position."""

    source: Any
    destination: Any
    freq_hz: float = 2.4e9
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 23.0
    gain_dbi: float = 0.0
    noise_figure_db: float = 7.0
    extra_loss_direct_db: float = 0.0  # e.g. obstruction on the direct S-D path

    def _snr(self, a: Any, b: Any, extra_db: float = 0.0) -> np.ndarray:
        d = np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)
        rx = self.tx_power_dbm + 2 * self.gain_dbi - fspl_db(d, self.freq_hz) - extra_db
        return db2lin(rx - noise_power_dbm(self.bandwidth_hz, self.noise_figure_db))

    def rates(self, relay_pos: Any, scheme: str = "df") -> dict:
        r = np.asarray(relay_pos, float)
        g1, g2 = self._snr(self.source, r), self._snr(r, self.destination)
        g_sd = self._snr(self.source, self.destination, self.extra_loss_direct_db)
        return dict(
            direct_bps=self.bandwidth_hz * np.log2(1.0 + g_sd),
            relay_bps=relay_rate(g1, g2, self.bandwidth_hz, scheme),
            snr_sr_db=10 * np.log10(g1),
            snr_rd_db=10 * np.log10(g2),
        )


def best_relay_position(
    link: RelayLink, altitude: float, extent: Tuple[float, float, float, float], resolution: float = 10.0, scheme: str = "df"
) -> Tuple[np.ndarray, float]:
    """Grid-search the relay position maximising the end-to-end rate at a fixed altitude. ``extent = (xmin, xmax, ymin, ymax)``."""
    xs = np.arange(extent[0], extent[1] + 1e-9, resolution)
    ys = np.arange(extent[2], extent[3] + 1e-9, resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, altitude)], axis=-1)
    r = link.rates(pts, scheme)["relay_bps"]
    i = int(np.argmax(r))
    return pts[i], float(r[i])
