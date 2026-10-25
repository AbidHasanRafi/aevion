"""Cellular and ad-hoc link evaluation for UAVs moving through a :class:`World`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .channel import AirToGroundChannel, sector_gain_dbi, thermal_noise_dbm


@dataclass
class BaseStation:
    pos: Sequence[float]
    tx_power_dbm: float = 43.0
    antenna: str = "sector"  # "sector" (down-tilted macro) or "isotropic"
    gain_dbi: float = 8.0
    tilt_deg: float = 10.0


@dataclass
class LinkReport:
    """Per-UAV link quality. Arrays have shape ``(N,)`` unless noted."""

    rssi_dbm: np.ndarray
    sinr_db: np.ndarray
    rate_bps: np.ndarray
    serving: np.ndarray
    los: np.ndarray
    outage: np.ndarray
    all_rx_dbm: np.ndarray  # (N, B)
    extras: Dict[str, np.ndarray] = field(default_factory=dict)  # further per-UAV channels, recorded as ``link_<name>``

    def spectral_efficiency(self, bandwidth_hz):
        return self.rate_bps / bandwidth_hz


class CellularNetwork:
    """A UAV acting as a *cellular-connected user*: it attaches to the strongest base station and
    sees all others as interference. Throughput follows Shannon capacity with an implementation loss.
    """

    def __init__(
        self,
        base_stations: Sequence[BaseStation],
        channel: Optional[AirToGroundChannel] = None,
        bandwidth_hz: float = 20e6,
        noise_figure_db: float = 7.0,
        sinr_outage_db: float = -6.0,
        rate_loss_db: float = 3.0,
        world=None,
        seed: int = 0,
        uav_antenna_gain_dbi: float = 0.0,
    ):
        self.bs = list(base_stations)
        self.channel = channel or AirToGroundChannel()
        self.bandwidth = bandwidth_hz
        self.noise_dbm = thermal_noise_dbm(bandwidth_hz, noise_figure_db)
        self.outage_db = sinr_outage_db
        self.rate_loss_db = rate_loss_db
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.uav_gain = uav_antenna_gain_dbi
        self.bs_pos = np.array([b.pos for b in self.bs], float)

    def reset(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(0 if seed is None else seed)

    def _tx_gain(self, b: BaseStation, uav_pos):
        if b.antenna == "isotropic":
            return np.full(uav_pos.shape[:-1], b.gain_dbi)
        d = uav_pos - np.asarray(b.pos, float)
        ang_below = np.degrees(np.arctan2(-d[..., 2], np.linalg.norm(d[..., :2], axis=-1) + 1e-9))  # >0 below horizon
        return sector_gain_dbi(ang_below, b.tilt_deg, g_max=b.gain_dbi)

    def evaluate(self, uav_pos) -> LinkReport:
        """``uav_pos`` is ``(N, 3)``."""
        P = np.atleast_2d(np.asarray(uav_pos, float))
        N, nb = len(P), len(self.bs)
        rx = np.empty((N, nb))
        los = np.empty((N, nb), bool)
        for j, b in enumerate(self.bs):
            pl, l, th = self.channel.path_loss_db(np.broadcast_to(self.bs_pos[j], P.shape), P, self.rng, self.world)
            fade = self.channel.fading_gain(l, th, self.rng)
            rx[:, j] = b.tx_power_dbm + self._tx_gain(b, P) + self.uav_gain - pl + 10 * np.log10(np.maximum(fade, 1e-6))
            los[:, j] = l
        serving = rx.argmax(1)
        rssi = rx[np.arange(N), serving]
        lin = 10 ** (rx / 10.0)
        sig = lin[np.arange(N), serving]
        interf = lin.sum(1) - sig
        noise = 10 ** (self.noise_dbm / 10.0)
        sinr = sig / (interf + noise)
        sinr_db = 10 * np.log10(sinr)
        rate = self.bandwidth * np.log2(1 + sinr / 10 ** (self.rate_loss_db / 10.0))
        outage = sinr_db < self.outage_db
        rate = np.where(outage, 0.0, rate)
        return LinkReport(rssi, sinr_db, rate, serving, los[np.arange(N), serving], outage, rx)

    def coverage_map(self, altitude: float, extent, resolution: float = 5.0, samples: int = 6):
        """Median SINR [dB] over a horizontal grid at ``altitude``. ``extent = (xmin, xmax, ymin, ymax)``."""
        xs = np.arange(extent[0], extent[1] + 1e-9, resolution)
        ys = np.arange(extent[2], extent[3] + 1e-9, resolution)
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, altitude)], -1)
        acc = []
        for _ in range(samples):
            acc.append(self.evaluate(pts).sinr_db)
        return xs, ys, np.median(np.stack(acc), 0).reshape(X.shape)

    def to_dict(self):
        return dict(
            bs=[
                dict(pos=list(map(float, b.pos)), tx_power_dbm=b.tx_power_dbm, antenna=b.antenna, gain_dbi=b.gain_dbi, tilt_deg=b.tilt_deg)
                for b in self.bs
            ],
            channel=self.channel.to_dict(),
            bandwidth_hz=self.bandwidth,
            noise_dbm=self.noise_dbm,
            outage_db=self.outage_db,
        )

    @classmethod
    def from_dict(cls, d, world=None, seed=0):
        return cls(
            [BaseStation(**b) for b in d["bs"]],
            AirToGroundChannel(**d["channel"]),
            d.get("bandwidth_hz", 20e6),
            sinr_outage_db=d.get("outage_db", -6.0),
            world=world,
            seed=seed,
        )


class AerialBaseStation:
    """A UAV serving ``K`` ground users (UAV-assisted network). Users share the bandwidth equally
    and attach if the received power exceeds a threshold; coverage and sum-rate follow."""

    def __init__(
        self,
        users: np.ndarray,
        channel: Optional[AirToGroundChannel] = None,
        tx_power_dbm: float = 30.0,
        bandwidth_hz: float = 20e6,
        noise_figure_db: float = 7.0,
        snr_threshold_db: float = 0.0,
        world=None,
        seed: int = 0,
        gain_dbi: float = 3.0,
    ):
        self.users = np.atleast_2d(np.asarray(users, float))
        self.channel = channel or AirToGroundChannel(los_mode="mean", shadowing=False)
        self.tx_dbm, self.bandwidth = tx_power_dbm, bandwidth_hz
        self.noise_dbm = thermal_noise_dbm(bandwidth_hz, noise_figure_db)
        self.thr = snr_threshold_db
        self.world, self.rng, self.gain = world, np.random.default_rng(seed), gain_dbi

    def evaluate(self, uav_pos):
        """``uav_pos`` ``(N,3)``. Returns dict with per-user SNR ``(N,K)``, coverage fraction, sum rate [bps]."""
        P = np.atleast_2d(np.asarray(uav_pos, float))
        N, K = len(P), len(self.users)
        tx = P[:, None, :]
        rx = self.users[None, :, :]
        pl, los, th = self.channel.path_loss_db(np.broadcast_to(tx, (N, K, 3)), np.broadcast_to(rx, (N, K, 3)), self.rng, self.world)
        fade = self.channel.fading_gain(los, th, self.rng)
        p_rx = self.tx_dbm + self.gain - pl + 10 * np.log10(np.maximum(fade, 1e-6))
        snr_db = p_rx - self.noise_dbm
        served = snr_db >= self.thr
        n_served = np.maximum(served.sum(1), 1)
        share = self.bandwidth / n_served[:, None]
        rate = np.where(served, share * np.log2(1 + 10 ** (snr_db / 10.0)), 0.0)
        return dict(snr_db=snr_db, coverage=served.mean(1), sum_rate_bps=rate.sum(1), user_rate_bps=rate, los=los)
