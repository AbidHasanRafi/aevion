"""UAV-to-UAV mesh: link budget, topology graph, shortest-path routing and multi-hop capacity.

Links are line-of-sight air-to-air (free-space) with optional obstruction by a :class:`~aevion.environments.world.World`.
Routing is Dijkstra over a configurable metric (hop count, expected transmission count ETX, or airtime). Multi-hop throughput
uses the standard single-channel contention bound: with ``h`` hops sharing one channel, ``R_e2e = 1 / sum_hops(1 / R_hop)``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..network import LinkReport
from ..phy.bler_curves import attenuated_shannon_se
from ..propagation.base import fspl_db
from ..units import noise_power_dbm

__all__ = ["MeshNetwork", "MeshReport", "connected_components", "dijkstra", "link_rate_matrix"]


def link_rate_matrix(
    pos: Any,
    freq_hz: float,
    bandwidth_hz: float,
    tx_power_dbm: float,
    noise_figure_db: float = 7.0,
    gain_dbi: float = 0.0,
    world: Any = None,
    nlos_loss_db: float = 25.0,
    min_sinr_db: float = -5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise ``(rate_bps (N,N), sinr_db (N,N))`` of UAV-to-UAV links; the diagonal is zero. Links under ``min_sinr_db`` have no rate."""
    p = np.asarray(pos, float)
    n = len(p)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    pl = fspl_db(np.maximum(d, 1.0), freq_hz)
    if world is not None and getattr(world, "has_obstacles", False):
        ii, jj = np.triu_indices(n, 1)
        los = world.line_of_sight(p[ii], p[jj])
        loss = np.zeros((n, n))
        loss[ii, jj] = loss[jj, ii] = np.where(los, 0.0, nlos_loss_db)
        pl = pl + loss
    sinr_db = tx_power_dbm + 2 * gain_dbi - pl - noise_power_dbm(bandwidth_hz, noise_figure_db)
    rate = bandwidth_hz * attenuated_shannon_se(sinr_db, se_max=6.0, sinr_min_db=min_sinr_db)
    np.fill_diagonal(rate, 0.0)
    np.fill_diagonal(sinr_db, -np.inf)
    return rate, sinr_db


def dijkstra(weights: np.ndarray, src: int) -> Tuple[np.ndarray, np.ndarray]:
    """Single-source shortest paths on a dense weight matrix (``inf`` = no edge). Returns ``(distance, predecessor)``."""
    n = len(weights)
    dist = np.full(n, np.inf)
    pred = np.full(n, -1, int)
    dist[src] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in np.nonzero(np.isfinite(weights[u]))[0]:
            nd = d + weights[u, v]
            if nd < dist[v] - 1e-15:
                dist[v], pred[v] = nd, u
                heapq.heappush(heap, (nd, int(v)))
    return dist, pred


def connected_components(adj: np.ndarray) -> np.ndarray:
    """Component label of every node of an undirected adjacency matrix."""
    n = len(adj)
    label = np.full(n, -1, int)
    c = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        stack = [s]
        label[s] = c
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u])[0]:
                if label[v] < 0:
                    label[v] = c
                    stack.append(int(v))
        c += 1
    return label


@dataclass
class MeshReport:
    connected: np.ndarray  # (N,) bool: node can reach the gateway
    hops: np.ndarray  # (N,) hops to the gateway (-1 unreachable)
    e2e_rate_bps: np.ndarray  # (N,) multi-hop rate to the gateway
    next_hop: np.ndarray  # (N,) next node towards the gateway (-1 none)
    n_components: int
    min_rate_bps: float
    adjacency: np.ndarray
    sinr_db: Optional[np.ndarray] = None  # full pairwise matrix (gateway last when external)


class MeshNetwork:
    """UAV swarm reaching a gateway (one of the UAVs, or a fixed ground point) over a multi-hop mesh."""

    def __init__(
        self,
        freq_hz: float = 2.4e9,
        bandwidth_hz: float = 20e6,
        tx_power_dbm: float = 20.0,
        noise_figure_db: float = 7.0,
        gain_dbi: float = 0.0,
        min_link_rate_bps: float = 1e6,
        metric: str = "airtime",
        gateway: Any = 0,
        world: Any = None,
        per_hop_loss_db: float = 0.0,
        sinr_outage_db: float = -5.0,
        seed: int = 0,
    ):
        if metric not in ("hops", "etx", "airtime"):
            raise ValueError("metric must be 'hops', 'etx' or 'airtime'")
        self.freq, self.bw, self.pt, self.nf, self.g = freq_hz, bandwidth_hz, tx_power_dbm, noise_figure_db, gain_dbi
        self.min_rate, self.metric, self.gateway, self.world = min_link_rate_bps, metric, gateway, world
        self.sinr_outage = sinr_outage_db
        self.per_hop_loss = per_hop_loss_db

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def _gateway_pos(self, pos: np.ndarray) -> Tuple[np.ndarray, int]:
        if isinstance(self.gateway, (int, np.integer)):
            return pos, int(self.gateway)
        g = np.asarray(self.gateway, float).reshape(1, 3)
        return np.vstack([pos, g]), len(pos)

    def analyse(self, uav_pos: Any) -> MeshReport:
        """Topology, routes and multi-hop rates of the swarm towards the gateway."""
        pos = np.atleast_2d(np.asarray(uav_pos, float))
        n = len(pos)
        allpos, gw = self._gateway_pos(pos)
        rate, sinr_m = link_rate_matrix(allpos, self.freq, self.bw, self.pt, self.nf, self.g, self.world, min_sinr_db=self.sinr_outage)
        rate = np.maximum(rate - self.per_hop_loss, 0.0) if self.per_hop_loss else rate
        usable = rate >= self.min_rate
        with np.errstate(divide="ignore"):
            if self.metric == "hops":
                w = np.where(usable, 1.0, np.inf)
            elif self.metric == "airtime":
                w = np.where(usable, 1.0 / np.maximum(rate, 1.0), np.inf)
            else:  # ETX = 1 / (1 - PER); PER from the rate margin over the minimum rate
                margin = np.clip(1.0 - self.min_rate / np.maximum(rate, 1.0), 1e-3, 0.999)
                w = np.where(usable, 1.0 / margin, np.inf)
        dist, pred = dijkstra(w, gw)
        N = len(allpos)
        hops = np.full(N, -1, int)
        e2e = np.zeros(N)
        nxt = np.full(N, -1, int)
        for s in range(N):
            if not np.isfinite(dist[s]):
                continue
            h, inv, u = 0, 0.0, s
            while u != gw:
                v = int(pred[u])
                inv += 1.0 / rate[u, v]
                h += 1
                u = v
                if h > N:  # defensive: cannot happen with Dijkstra predecessors
                    break
            hops[s], e2e[s] = h, (1.0 / inv if inv > 0 else math.inf)
            nxt[s] = pred[s]
        e2e[gw] = math.inf
        sel = slice(0, n)
        conn = np.isfinite(dist)[sel]
        finite = e2e[sel][np.isfinite(e2e[sel]) & (e2e[sel] > 0)]
        return MeshReport(
            conn,
            hops[sel],
            np.where(np.isfinite(e2e[sel]), e2e[sel], 0.0),
            nxt[sel],
            int(connected_components(usable).max() + 1),
            float(finite.min()) if len(finite) else 0.0,
            usable[:n, :n],
            sinr_m,
        )

    wants_state = False

    def evaluate(self, uav_pos: Any, **_: Any) -> LinkReport:
        """Engine interface: per-UAV view of the mesh (``rate`` = multi-hop rate to the gateway, ``serving`` = next hop)."""
        rep = self.analyse(uav_pos)
        n = len(rep.connected)
        idx = np.arange(n)
        nxt = np.where(rep.next_hop >= 0, rep.next_hop, idx)
        sinr = np.where(rep.next_hop >= 0, rep.sinr_db[idx, nxt], -np.inf) if rep.sinr_db is not None else np.full(n, -np.inf)
        sinr = np.where(np.isfinite(sinr), sinr, -50.0)
        rssi = sinr + noise_power_dbm(self.bw, self.nf)
        all_rx = (rep.sinr_db[:n, :n] if rep.sinr_db is not None else np.zeros((n, n))) + noise_power_dbm(self.bw, self.nf)
        extras = dict(hops=rep.hops.astype(float), connected=rep.connected.astype(float), n_components=np.full(n, float(rep.n_components)))
        return LinkReport(rssi, sinr, rep.e2e_rate_bps, nxt, rep.connected.copy(), ~rep.connected, np.nan_to_num(all_rx, neginf=-300.0), extras)

    def to_dict(self) -> Dict[str, Any]:
        return dict(freq_hz=self.freq, bandwidth_hz=self.bw, tx_power_dbm=self.pt, metric=self.metric, min_link_rate_bps=self.min_rate)
