"""Wireless metrics of a finished run, computed from the ``link_*`` channels of an archive."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .age_of_information import average_aoi
from .outage import outage_events

__all__ = ["aoi_from_archive", "link_metrics"]


def link_metrics(archive: Any, outage_sinr_db: float = -6.0) -> Dict[str, np.ndarray]:
    """Per-vehicle link metrics (arrays of length ``N``); empty if the run has no wireless channels."""
    if "link_sinr_db" not in archive:
        return {}
    sinr = np.asarray(archive["link_sinr_db"], float)
    rate = np.asarray(archive["link_rate_mbps"], float)
    n = sinr.shape[1]
    t = np.asarray(archive.t, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    thr = (archive.meta.get("wireless") or {}).get("outage_db", outage_sinr_db)
    out: Dict[str, np.ndarray] = {}
    outage = np.asarray(archive["link_outage"], bool) if "link_outage" in archive else sinr < thr
    stats = [outage_events(outage[:, i], dt) for i in range(n)]
    out["link_outage_frac"] = np.array([s.probability for s in stats])
    out["link_n_outages"] = np.array([float(s.n_events) for s in stats])
    out["link_longest_outage_s"] = np.array([s.longest_s for s in stats])
    out["link_mean_sinr_db"] = sinr.mean(axis=0)
    out["link_mean_rate_mbps"] = rate.mean(axis=0)
    out["link_p5_rate_mbps"] = np.percentile(rate, 5, axis=0)
    if "link_los" in archive:
        out["link_los_frac"] = np.asarray(archive["link_los"], float).mean(axis=0)
    if "link_serving" in archive:
        s = np.asarray(archive["link_serving"])
        out["link_cell_changes"] = (np.diff(s, axis=0) != 0).sum(axis=0).astype(float)
    for key, name in (("link_n_handovers", "link_handovers"), ("link_n_rlf", "link_rlf")):
        if key in archive:
            out[name] = np.asarray(archive[key], float)[-1]
    if "link_bler" in archive:
        out["link_mean_bler"] = np.asarray(archive["link_bler"], float).mean(axis=0)
    return out


def aoi_from_archive(archive: Any, update_hz: float = 10.0, packet_bits: float = 8000.0, extra_delay_s: float = 0.005) -> Dict[str, np.ndarray]:
    """Mean and peak age of information [s] of periodic telemetry updates sent over the recorded link.

    An update generated at time ``t`` is delivered after ``packet_bits / rate + extra_delay_s`` if the link is not in outage then;
    otherwise it is lost. Needs the ``link_rate_mbps`` channel.
    """
    if "link_rate_mbps" not in archive:
        return {}
    t = np.asarray(archive.t, float)
    rate = np.asarray(archive["link_rate_mbps"], float) * 1e6
    n = rate.shape[1]
    gen = np.arange(t[0], t[-1], 1.0 / update_hz)
    mean, peak = np.zeros(n), np.zeros(n)
    for i in range(n):
        r = np.interp(gen, t, rate[:, i])
        ok = r > 0
        deliv = np.where(ok, gen + packet_bits / np.maximum(r, 1.0) + extra_delay_s, np.inf)
        res = average_aoi(gen, deliv, float(t[-1]), float(t[0]))
        mean[i], peak[i] = res["mean"], res["peak"]
    return dict(aoi_mean_s=mean, aoi_peak_s=peak)
