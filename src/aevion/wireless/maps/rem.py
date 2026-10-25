"""Radio environment maps (REM): best-server RSRP, SINR and rate over a horizontal plane at a UAV altitude."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..phy.bler_curves import attenuated_shannon_se

__all__ = ["RadioEnvironmentMap", "radio_environment_map"]


@dataclass
class RadioEnvironmentMap:
    xs: np.ndarray
    ys: np.ndarray
    altitude: float
    rsrp_dbm: np.ndarray  # (ny, nx) best-server received power
    sinr_db: np.ndarray
    rate_bps: np.ndarray
    serving: np.ndarray  # (ny, nx) best-server cell index
    los: np.ndarray  # (ny, nx) bool: LoS to the serving cell

    def coverage_fraction(self, sinr_threshold_db: float = -6.0) -> float:
        """Fraction of the area whose SINR is at least the threshold."""
        return float(np.mean(self.sinr_db >= sinr_threshold_db))

    def percentile(self, q: float, field: str = "sinr_db") -> float:
        return float(np.percentile(getattr(self, field), q))

    def handover_boundary_length(self, resolution: float) -> float:
        """Approximate total length [m] of cell borders (cells where the serving index changes), a proxy for handover exposure."""
        s = self.serving
        edges = np.count_nonzero(s[:, 1:] != s[:, :-1]) + np.count_nonzero(s[1:, :] != s[:-1, :])
        return float(edges * resolution)


def radio_environment_map(network: Any, altitude: float, extent: Sequence[float], resolution: float = 10.0) -> RadioEnvironmentMap:
    """Deterministic large-scale map of an :class:`~aevion.wireless.mac_network.nr_network.NrCellularNetwork`.

    Uses the model's median path loss with the LoS state decided by geometry (if the network has a world) or by
    probability > 0.5, no shadowing, no fading and a level UAV. ``extent = (xmin, xmax, ymin, ymax)``.
    """
    xs = np.arange(extent[0], extent[1] + 1e-9, resolution)
    ys = np.arange(extent[2], extent[3] + 1e-9, resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(altitude))], axis=-1)
    rsrp, los = network.large_scale_rsrp(pts)
    n = len(pts)
    serving = rsrp.argmax(axis=1)
    from ..mac_network.interference import compute_sinr

    sinr = compute_sinr(rsrp, serving, network.noise_dbm, np.full(network.n_cells, network.load))
    se = attenuated_shannon_se(sinr.sinr_db, se_max=7.4)
    rate = network.bandwidth_hz * se * network.layers
    idx = np.arange(n)
    shp = X.shape
    return RadioEnvironmentMap(
        xs,
        ys,
        float(altitude),
        rsrp[idx, serving].reshape(shp),
        sinr.sinr_db.reshape(shp),
        rate.reshape(shp),
        serving.reshape(shp),
        los[idx, serving].reshape(shp),
    )
