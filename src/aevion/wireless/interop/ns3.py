"""Export trajectories and link traces for ns-3 (and other network simulators).

* :func:`export_ns2_mobility`: an ns-2 style mobility trace that ns-3's ``Ns2MobilityHelper`` reads, with one node per vehicle and
  ``set X_/Y_/Z_`` commands at every exported time step (full 3-D, unlike ``setdest``);
* :func:`export_waypoints_csv`: ``t,node,x,y,z,vx,vy,vz`` for waypoint-based mobility models;
* :func:`export_link_trace_csv`: the recorded link quality (``link_*`` channels) per vehicle and time.

No ns-3 installation is needed to produce the files, and none is imported.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List

import numpy as np

from ..._types import PathLike

__all__ = ["export_link_trace_csv", "export_ns2_mobility", "export_waypoints_csv"]


def _steps(archive: Any, every_s: float) -> np.ndarray:
    t = np.asarray(archive.t, float)
    if every_s <= 0 or len(t) < 2:
        return np.arange(len(t))
    idx, last = [0], t[0]
    for i in range(1, len(t)):
        if t[i] - last >= every_s - 1e-9:
            idx.append(i)
            last = t[i]
    if idx[-1] != len(t) - 1:
        idx.append(len(t) - 1)
    return np.array(idx)


def export_ns2_mobility(archive: Any, path: PathLike, every_s: float = 0.1) -> Path:
    """Write an ns-2 mobility trace (one node per vehicle) sampled every ``every_s`` seconds. Returns the path."""
    pos = np.asarray(archive["pos"], float)
    t = np.asarray(archive.t, float)
    idx = _steps(archive, every_s)
    lines: List[str] = []
    for v in range(pos.shape[1]):
        for ax, name in enumerate("XYZ"):
            lines.append(f"$node_({v}) set {name}_ {pos[0, v, ax]:.4f}")
    for i in idx[1:]:
        for v in range(pos.shape[1]):
            for ax, name in enumerate("XYZ"):
                lines.append(f'$ns_ at {t[i] - t[0]:.4f} "$node_({v}) set {name}_ {pos[i, v, ax]:.4f}"')
    p = Path(path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def export_waypoints_csv(archive: Any, path: PathLike, every_s: float = 0.1) -> Path:
    pos, vel = np.asarray(archive["pos"], float), np.asarray(archive["vel"], float)
    t = np.asarray(archive.t, float)
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "node", "x", "y", "z", "vx", "vy", "vz"])
        for i in _steps(archive, every_s):
            for v in range(pos.shape[1]):
                w.writerow([f"{t[i] - t[0]:.4f}", v, *(f"{x:.4f}" for x in pos[i, v]), *(f"{x:.4f}" for x in vel[i, v])])
    return p


def export_link_trace_csv(archive: Any, path: PathLike, every_s: float = 0.0) -> Path:
    """Per vehicle and time: every recorded ``link_*`` channel (empty file with a header only when the run has none)."""
    names = sorted(k for k in archive.channels if k.startswith("link_"))
    t = np.asarray(archive.t, float)
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "node", *names])
        data = {k: np.asarray(archive[k]) for k in names}
        n = next((d.shape[1] for d in data.values() if d.ndim >= 2), 0)
        for i in _steps(archive, every_s):
            for v in range(n):
                w.writerow([f"{t[i] - t[0]:.4f}", v, *(f"{float(data[k][i, v]):.6g}" for k in names)])
    return p
