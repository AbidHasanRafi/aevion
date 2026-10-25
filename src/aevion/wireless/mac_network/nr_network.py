"""A cellular-connected UAV under a 5G-NR-style stack: the integrated network model.

One call evaluates one network interval ``dt``:

1. **Large-scale**: link geometry to the median path loss of a chosen model (TR 38.901 / TR 36.777 / Al-Hourani / ...), LoS state
   (probabilistic with spatial correlation, or geometric by ray casting through the world), spatially correlated shadowing
   (Gudmundson), optional blockage and knife-edge diffraction.
2. **Antennas**: cell sector pattern towards the UAV; the UAV antenna evaluated in its body frame from the actual attitude.
3. **Small-scale**: time-correlated Rician/Rayleigh fading driven by the UAV's own Doppler.
4. **Measurements and mobility**: layer-3-filtered RSRP drives A3 handover (hysteresis, time-to-trigger, interruption, failures,
   ping-pong) and radio-link-failure timers.
5. **Interference and jamming**: fractional cell load; optional jammers.
6. **PHY/MAC**: scheduler share against background users, CQI/MCS with OLLA, BLER and HARQ, then goodput and latency.

Results are deterministic for a fixed seed. The model abstracts the NR stack at interval granularity: no per-slot HARQ, channel
estimation errors or CSI-RS/PMI feedback (see the Wireless section of ``DOCUMENTATION.md``).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..antennas.patterns import AntennaPattern, GaussianBeam, Isotropic, Patch3GPP, SectorTR36814
from ..antennas.uav_orientation import UavAntenna
from ..channel.air_to_ground import rician_k_db
from ..channel.blockage import BlockageProcess
from ..channel.mobility import link_kinematics
from ..fading.distributions import nakagami_m_from_rician_k
from ..fading.doppler import coherence_time
from ..fading.shadowing import DECORRELATION_DISTANCE_M, CorrelatedShadowing
from ..fading.temporal import FadingProcess
from ..network import LinkReport
from ..phy.bler_curves import BlerModel
from ..phy.link_adaptation import LinkAdaptation
from ..phy.mcs_tables import get_table
from ..phy.nr_numerology import Numerology, max_prb
from ..propagation import LinkGeometry, PathLossModel, make_model
from ..propagation.base import fspl_db
from ..propagation.knife_edge import world_diffraction_loss_db
from ..units import db2lin, lin2db, noise_power_dbm
from .handover import HandoverManager
from .interference import compute_sinr
from .rlf import RadioLinkMonitor
from .scheduler import CellScheduler

__all__ = ["NrCellularNetwork", "build_cells"]

_PATTERNS = {"sector": SectorTR36814, "patch": Patch3GPP, "omni": Isotropic, "isotropic": Isotropic, "beam": GaussianBeam}


def _pattern(spec: Union[str, Dict[str, Any], AntennaPattern, None], default: str = "sector") -> AntennaPattern:
    if isinstance(spec, AntennaPattern):
        return spec
    if spec is None:
        spec = default
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    kind = spec.pop("type", default)
    if kind not in _PATTERNS:
        raise KeyError(f"unknown cell antenna {kind!r}; choose from {sorted(_PATTERNS)}")
    return _PATTERNS[kind](**spec)


def build_cells(sites: Optional[Sequence[Dict[str, Any]]], bs: Optional[Sequence[Dict[str, Any]]], default_power_dbm: float) -> List[Dict[str, Any]]:
    """Expand ``sites`` (with ``sectors``) or legacy ``bs`` entries into a flat list of cells ``{pos, yaw, power, antenna}``."""
    cells: List[Dict[str, Any]] = []
    for s in sites or []:
        s = dict(s)
        n_sec = int(s.get("sectors", 3))
        az0 = float(s.get("azimuth0_deg", 0.0))
        tilt = float(s.get("tilt_deg", 10.0))
        # a single-sector site is an omnidirectional (5 dBi) macro; multi-sector sites use the TR 36.814 down-tilted pattern
        ant = s.get("antenna", {"type": "omni", "gain_dbi": 5.0} if n_sec == 1 else {"type": "sector", "tilt_deg": tilt})
        for k in range(n_sec):
            cells.append(
                dict(
                    pos=np.asarray(s["pos"], float),
                    yaw=math.radians(az0 + k * 360.0 / n_sec),
                    power=float(s.get("tx_power_dbm", default_power_dbm)),
                    antenna=_pattern(ant),
                )
            )
    for b in bs or []:  # legacy single-cell entries (same keys as the simple CellularNetwork)
        b = dict(b)
        ant = (
            {"type": "omni", "gain_dbi": b.get("gain_dbi", 0.0)}
            if b.get("antenna") == "isotropic"
            else {"type": "sector", "g_max_dbi": b.get("gain_dbi", 8.0), "tilt_deg": b.get("tilt_deg", 10.0)}
        )
        cells.append(
            dict(
                pos=np.asarray(b["pos"], float),
                yaw=math.radians(float(b.get("azimuth_deg", 0.0))),
                power=float(b.get("tx_power_dbm", default_power_dbm)),
                antenna=_pattern(ant),
            )
        )
    if not cells:
        raise ValueError("a network needs 'sites' or 'bs'")
    return cells


def _fading_quantiles(m: np.ndarray, n_q: int = 12) -> np.ndarray:
    """``(n, n_q)`` equiprobable quantiles of unit-mean Nakagami-``m`` fading power (Gamma(m, 1/m)), renormalised to unit mean."""
    from scipy.special import gammaincinv  # lazy: scipy.special is a heavy import

    p = (np.arange(n_q) + 0.5) / n_q
    g = gammaincinv(np.asarray(m, float)[:, None], p[None, :]) / np.asarray(m, float)[:, None]
    return g / g.mean(axis=1, keepdims=True)


class NrCellularNetwork:
    """See the module docstring. Construct directly or from a configuration ``network: {type: nr_cellular, ...}``."""

    wants_state = True

    def __init__(
        self,
        sites: Optional[Sequence[Dict[str, Any]]] = None,
        bs: Optional[Sequence[Dict[str, Any]]] = None,
        world: Any = None,
        seed: int = 0,
        freq_hz: float = 3.5e9,
        bandwidth_mhz: int = 100,
        scs_khz: int = 30,
        tx_power_dbm: float = 46.0,
        noise_figure_db: float = 7.0,
        propagation: Union[str, Dict[str, Any], PathLossModel] = "tr36777",
        los_mode: str = "probabilistic",
        shadowing: bool = True,
        fading: bool = True,
        nlos_model: str = "statistical",
        uav_antenna: Union[str, Dict[str, Any]] = "dipole",
        load: float = 1.0,
        jammers: Optional[Sequence[Dict[str, Any]]] = None,
        handover: Union[bool, Dict[str, Any]] = True,
        rlf: Union[bool, Dict[str, Any]] = True,
        scheduler: str = "proportional_fair",
        background_users: int = 0,
        phy: str = "nr",
        mcs_table: str = "mcs1",
        target_bler: float = 0.1,
        layers: int = 1,
        blockage: Optional[Dict[str, Any]] = None,
        outage_sinr_db: float = -6.0,
        ground_z: float = 0.0,
    ):
        if los_mode not in ("probabilistic", "geometric", "mean"):
            raise ValueError("los_mode must be 'probabilistic', 'geometric' or 'mean'")
        if nlos_model not in ("statistical", "diffraction"):
            raise ValueError("nlos_model must be 'statistical' or 'diffraction'")
        if phy not in ("nr", "shannon"):
            raise ValueError("phy must be 'nr' or 'shannon'")
        self.cells = build_cells(sites, bs, tx_power_dbm)
        self.cell_pos = np.stack([c["pos"] for c in self.cells])
        self.cell_yaw = np.array([c["yaw"] for c in self.cells])
        self.cell_power = np.array([c["power"] for c in self.cells])
        self.n_cells = len(self.cells)
        self.world, self.seed = world, int(seed)
        self.freq, self.ground_z = float(freq_hz), float(ground_z)
        self.los_mode, self.use_shadowing, self.use_fading, self.nlos_model = los_mode, shadowing, fading, nlos_model
        self.numerology = Numerology({15: 0, 30: 1, 60: 2, 120: 3}[scs_khz])
        fr = "FR2" if freq_hz >= 24e9 else "FR1"
        self.n_prb = max_prb(int(bandwidth_mhz), scs_khz, fr)
        self.bandwidth_hz = self.n_prb * self.numerology.prb_bandwidth_hz
        self.noise_dbm = noise_power_dbm(self.bandwidth_hz, noise_figure_db)
        self.load = float(load)
        self.jammers = [dict(j) for j in (jammers or [])]
        self.phy_kind, self.mcs_table_name, self.target_bler, self.layers = phy, mcs_table, float(target_bler), int(layers)
        self.outage_sinr_db = float(outage_sinr_db)
        self.scheduler_policy, self.background_users = scheduler, int(background_users)
        self._ho_cfg = {} if handover is True else (None if handover is False else dict(handover))
        self._rlf_cfg = {} if rlf is True else (None if rlf is False else dict(rlf))
        self._blockage_cfg = dict(blockage) if blockage else None
        if isinstance(propagation, PathLossModel):
            self.model = propagation
        else:
            spec = {"model": propagation} if isinstance(propagation, str) else dict(propagation)
            name = spec.pop("model", "tr36777")
            self.model = make_model(name, freq_hz=self.freq, **spec)
        self.scenario = getattr(self.model, "scenario", "uma")[:3]
        self._warned_validity = False
        ua = {"preset": uav_antenna} if isinstance(uav_antenna, str) else dict(uav_antenna)
        self.uav_antenna = UavAntenna.preset(ua.pop("preset", "dipole"), **ua)
        self._uav_spec = uav_antenna
        self.last_ho_events: List[Any] = []
        self.blackout = False  # set by the comm_blackout fault: no service while True
        self.reset(seed)

    # ------------------------------------------------------------------------------------------------------ state
    def reset(self, seed: Optional[int] = None) -> None:
        self.seed = self.seed if seed is None else int(seed)
        ss = np.random.SeedSequence(self.seed)
        self._streams = [np.random.default_rng(s) for s in ss.spawn(8)]
        self._n: Optional[int] = None
        self._t_last: Optional[float] = None
        self._pos_prev: Optional[np.ndarray] = None
        self._clock = 0.0

    def _init_state(self, n: int) -> None:
        r = self._streams
        c = self.n_cells
        self._n = n
        self._shadow = CorrelatedShadowing((n, c), r[0])
        self._los_state = CorrelatedShadowing((n, c), r[1])
        self._fading = FadingProcess(n * c, r[2])
        self._blockage = BlockageProcess(n * c, rng=r[3], **self._blockage_cfg) if self._blockage_cfg else None
        self._sched = CellScheduler(c, self.scheduler_policy, self.background_users, n_slots=200, rng=r[4])
        self._sched.reset(n, r[4])
        self._ho = HandoverManager(n, c, **self._ho_cfg) if self._ho_cfg is not None else None
        self._rlf = RadioLinkMonitor(n, **self._rlf_cfg) if self._rlf_cfg is not None else None
        self._serving: Optional[np.ndarray] = None
        table = get_table(self.mcs_table_name)
        self._la = LinkAdaptation(n, self.numerology, self.mcs_table_name, BlerModel(table), self.target_bler, layers=self.layers, rng=r[5])
        self._jam_phase = r[6]

    # ----------------------------------------------------------------------------------------------- components
    def _geometry(self, pos: np.ndarray) -> LinkGeometry:
        bs = np.broadcast_to(self.cell_pos[None, :, :], (len(pos), self.n_cells, 3))
        ut = np.broadcast_to(pos[:, None, :], bs.shape)
        return LinkGeometry.from_positions(bs, ut, self.ground_z)

    def _los(self, g: LinkGeometry, pos: np.ndarray, moved: np.ndarray) -> tuple:
        p_los = self.model.los_probability(g)
        if self.los_mode == "mean":
            return p_los > 0.5, p_los
        if self.los_mode == "geometric":
            if self.world is None:
                raise ValueError("los_mode='geometric' needs a world")
            bs = np.broadcast_to(self.cell_pos[None, :, :], (len(pos), self.n_cells, 3)).reshape(-1, 3)
            ut = np.broadcast_to(pos[:, None, :], (len(pos), self.n_cells, 3)).reshape(-1, 3)
            return self.world.line_of_sight(bs, ut).reshape(p_los.shape), p_los
        d_corr = DECORRELATION_DISTANCE_M.get(self.scenario, (37.0, 50.0))[0]
        from scipy.special import ndtr  # lazy: scipy.special is a heavy import

        z = self._los_state.step(1.0, moved[:, None], d_corr)  # spatially persistent standard-normal field
        return ndtr(z) < p_los, p_los

    def _check_validity(self, g: LinkGeometry) -> None:
        """Warn once when links leave the propagation model's specified range (the usual mistake: a terrestrial model for a UAV)."""
        if self._warned_validity:
            return
        ok = self.model.validity(g)
        if ok.all():
            return
        self._warned_validity = True
        h_lo, h_hi = self.model.h_ut_range_m
        reason = (
            f"UAV height outside {h_lo:g}-{h_hi:g} m (use propagation 'tr36777' for aerial users)"
            if np.any((g.h_ut > h_hi) | (g.h_ut < h_lo))
            else "distance or frequency outside the specified range"
        )
        warnings.warn(
            f"propagation model '{self.model.name}' is used outside its validity range for {int((~ok).sum())} of {ok.size} links: {reason}",
            stacklevel=3,
        )

    def _jammer_dbm(self, pos: np.ndarray, t: float) -> Optional[np.ndarray]:
        if not self.jammers:
            return None
        out = np.full(len(pos), -300.0)
        lin = np.zeros(len(pos))
        for j in self.jammers:
            if not (j.get("t_start", 0.0) <= t <= j.get("t_end", math.inf)):
                continue
            d = np.linalg.norm(pos - np.asarray(j["pos"], float), axis=1)
            rx = float(j.get("power_dbm", 30.0)) + float(j.get("gain_dbi", 0.0)) - fspl_db(np.maximum(d, 1.0), float(j.get("freq_hz", self.freq)))
            bw_overlap = min(float(j.get("bandwidth_hz", self.bandwidth_hz)) / self.bandwidth_hz, 1.0)
            lin = lin + db2lin(rx) * bw_overlap
        return np.where(lin > 0, lin2db(lin), out)

    # -------------------------------------------------------------------------------------------------- evaluate
    def evaluate(
        self, uav_pos: Any, t: Optional[float] = None, vel: Optional[Any] = None, quat: Optional[Any] = None, dt: Optional[float] = None, **_: Any
    ) -> LinkReport:
        pos = np.atleast_2d(np.asarray(uav_pos, float))
        n, c = len(pos), self.n_cells
        if self._n != n:
            self._init_state(n)
        if dt is None:
            dt = 0.1 if t is None or self._t_last is None else max(t - self._t_last, 1e-6)
        t = self._clock + dt if t is None else float(t)
        self._clock, self._t_last = t, t
        prev = self._pos_prev if self._pos_prev is not None else pos
        moved = np.linalg.norm(pos - prev, axis=1)
        v = np.asarray(vel, float).reshape(n, 3) if vel is not None else (pos - prev) / max(dt, 1e-9)
        self._pos_prev = pos.copy()
        q = np.broadcast_to(np.array([1.0, 0, 0, 0]) if quat is None else np.asarray(quat, float), (n, 4))

        g = self._geometry(pos)
        self._check_validity(g)
        los, p_los = self._los(g, pos, moved)
        pl = self.model.path_loss_db(g, los)
        if self.nlos_model == "diffraction" and self.world is not None and getattr(self.world, "has_obstacles", False):
            bs = np.broadcast_to(self.cell_pos[None], (n, c, 3)).reshape(-1, 3)
            ut = np.broadcast_to(pos[:, None], (n, c, 3)).reshape(-1, 3)
            diff = world_diffraction_loss_db(self.world, bs, ut, self.freq).reshape(n, c)
            pl = np.where(los, pl, fspl_db(g.d3d, self.freq) + diff)  # NLoS = free space + Bullington diffraction over the actual obstacles
        if self.use_shadowing:
            sig = self.model.shadowing_std_db(g, los)
            d_corr = np.where(
                los, DECORRELATION_DISTANCE_M.get(self.scenario, (37.0, 50.0))[0], DECORRELATION_DISTANCE_M.get(self.scenario, (37.0, 50.0))[1]
            )
            pl = pl + self._shadow.step(sig, moved[:, None], d_corr)
        if self._blockage is not None:
            pl = pl + self._blockage.step(dt).reshape(n, c)

        # antennas: cell sector gain towards the UAV, UAV gain from its attitude
        dvec = pos[:, None, :] - self.cell_pos[None, :, :]  # cell -> UAV
        az_cell = np.arctan2(dvec[..., 1], dvec[..., 0]) - self.cell_yaw[None, :]
        el_cell = np.arctan2(dvec[..., 2], np.hypot(dvec[..., 0], dvec[..., 1]))
        g_cell = np.empty((n, c))
        for j, cell in enumerate(self.cells):
            g_cell[:, j] = cell["antenna"].gain_dbi(az_cell[:, j], el_cell[:, j])
        g_uav = self.uav_antenna.gain_dbi(q[:, None, :], -dvec)  # UAV -> cell direction in the world frame

        rsrp = self.cell_power[None, :] + g_cell + g_uav - pl  # large-scale received power (what L3 measurements average to)

        # small-scale fading from the UAV's own motion (per link)
        fade_db = np.zeros((n, c))
        k_db = rician_k_db(g.elevation_deg)
        kin = link_kinematics(self.cell_pos[None, :, :], pos[:, None, :], v[:, None, :], self.freq)
        if self.use_fading:
            gain = self._fading.step(
                los.reshape(-1), k_db.reshape(-1), kin.max_doppler_hz.reshape(-1), dt, np.abs(kin.doppler_shift_hz).reshape(-1)
            ).reshape(n, c)
            fade_db = lin2db(np.maximum(gain, 1e-6))
        rx_inst = rsrp + fade_db

        # mobility management on the filtered measurements
        jam = self._jammer_dbm(pos, t)
        serving_for_sinr = self._serving if self._serving is not None else rsrp.argmax(axis=1)
        sinr0 = compute_sinr(rx_inst, serving_for_sinr, self.noise_dbm, np.full(c, self.load), jam)
        interrupted = np.zeros(n, bool)
        if self._ho is not None:
            serving, interrupted = self._ho.step(t, rsrp, dt, sinr0.sinr_db)
        else:
            serving = rsrp.argmax(axis=1)
        rlf_down = np.zeros(n, bool)
        if self._rlf is not None:
            n_before = self._rlf.n_rlf.copy()
            sinr_pre = compute_sinr(rx_inst, serving, self.noise_dbm, np.full(c, self.load), jam).sinr_db
            rlf_down = self._rlf.step(t, sinr_pre, dt)
            new_fail = np.nonzero(self._rlf.n_rlf > n_before)[0]
            if len(new_fail) and self._ho is not None:
                self._ho.force_reselect(new_fail, rsrp)
                assert self._ho.serving is not None
                serving = self._ho.serving.copy()
        self._serving = serving
        sinr = compute_sinr(rx_inst, serving, self.noise_dbm, np.full(c, self.load), jam)
        idx = np.arange(n)

        # PHY / MAC. A network interval usually spans many fading coherence times, so link adaptation and the error statistics
        # use the SINR *averaged over the fading distribution* (the instantaneous snapshot only when the channel is slower than dt).
        sinr_slow = compute_sinr(rsrp, serving, self.noise_dbm, np.full(c, self.load), jam).sinr_db
        coh = np.asarray(coherence_time(np.maximum(kin.max_doppler_hz[idx, serving], 1e-6), 0.5))
        w_avg = np.clip(dt / coh, 0.0, 1.0) if self.use_fading else np.zeros(n)
        sinr_eff = (1.0 - w_avg) * sinr.sinr_db + w_avg * sinr_slow
        share = self._sched.shares(serving, db2lin(sinr_eff), self.bandwidth_hz)
        unavailable = interrupted | rlf_down
        if self.phy_kind == "nr":
            samples = weights = None
            if self.use_fading and np.any(w_avg > 0):
                kk = np.where(los[idx, serving], db2lin(k_db[idx, serving]), 0.0)
                gq = _fading_quantiles(nakagami_m_from_rician_k(kk))  # (n, Q): fading-power quantiles, unit mean
                q = gq.shape[1]
                samples = np.concatenate([sinr.sinr_db[:, None], sinr_slow[:, None] + lin2db(gq)], axis=1)
                weights = np.concatenate([(1.0 - w_avg)[:, None], np.repeat((w_avg / q)[:, None], q, axis=1)], axis=1)
            la = self._la.step(t, sinr_eff, dt, self.n_prb, slots_active=share, sinr_samples_db=samples, sample_weights=weights)
            rate, mcs, bler, latency = la.rate_bps, la.mcs.astype(float), la.bler, la.latency_s * 1e3
        else:
            rate = self.bandwidth_hz * np.log2(1.0 + db2lin(sinr_eff - 3.0)) * share
            mcs, bler, latency = np.full(n, -1.0), np.zeros(n), np.zeros(n)
        outage = (sinr_eff < self.outage_sinr_db) | unavailable | self.blackout
        rate = np.where(outage, 0.0, rate)

        extras: Dict[str, np.ndarray] = dict(
            rsrp_dbm=rsrp[idx, serving],
            cell_gain_dbi=g_cell[idx, serving],
            uav_gain_dbi=g_uav[idx, serving],
            path_loss_db=pl[idx, serving],
            fading_db=fade_db[idx, serving],
            k_factor_db=np.where(los[idx, serving], k_db[idx, serving], -np.inf).clip(-30.0),
            doppler_hz=kin.doppler_shift_hz[idx, serving],
            share=share,
            mcs=mcs,
            bler=bler,
            latency_ms=latency,
            n_handovers=(self._ho.n_ho.astype(float) if self._ho is not None else np.zeros(n)),
            ho_interrupted=interrupted.astype(float),
            rlf=rlf_down.astype(float),
            n_rlf=(self._rlf.n_rlf.astype(float) if self._rlf is not None else np.zeros(n)),
            interference_dbm=sinr.interference_dbm,
            sinr_eff_db=sinr_eff,
        )
        return LinkReport(rsrp[idx, serving], sinr.sinr_db, rate, serving, los[idx, serving], outage, rx_inst, extras)

    def large_scale_rsrp(self, pos: Any) -> tuple:
        """Deterministic large-scale received power ``(rsrp_dbm (N, C), los (N, C))``: median path loss, geometric LoS when a world
        exists (else probability > 0.5), a level UAV, no shadowing/fading/blockage. Stateless: safe to call between evaluations."""
        pos = np.atleast_2d(np.asarray(pos, float))
        g = self._geometry(pos)
        p_los = self.model.los_probability(g)
        if self.world is not None and getattr(self.world, "has_obstacles", False):
            bs = np.broadcast_to(self.cell_pos[None], (len(pos), self.n_cells, 3)).reshape(-1, 3)
            ut = np.broadcast_to(pos[:, None], (len(pos), self.n_cells, 3)).reshape(-1, 3)
            los = self.world.line_of_sight(bs, ut).reshape(p_los.shape)
        else:
            los = p_los > 0.5
        pl = self.model.path_loss_db(g, los)
        dvec = pos[:, None, :] - self.cell_pos[None, :, :]
        az = np.arctan2(dvec[..., 1], dvec[..., 0]) - self.cell_yaw[None, :]
        el = np.arctan2(dvec[..., 2], np.hypot(dvec[..., 0], dvec[..., 1]))
        g_cell = np.stack([cell["antenna"].gain_dbi(az[:, j], el[:, j]) for j, cell in enumerate(self.cells)], axis=1)
        level = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (len(pos), 1, 4))
        g_uav = self.uav_antenna.gain_dbi(level, -dvec)
        return self.cell_power[None, :] + g_cell + g_uav - pl, los

    def snapshot(self, pos: Any, rng: Optional[np.random.Generator] = None, shadowing: bool = True) -> Dict[str, np.ndarray]:
        """Cheap, **stateless** link evaluation for learning environments: large-scale received power (geometric LoS if the network has a
        world), optional independent log-normal shadowing, interference at the configured load, SINR and a Shannon-style rate.

        Unlike :meth:`evaluate` it keeps no per-UAV state (no handover timers, fading memory or link adaptation), so it can be called
        for thousands of parallel environments and any subset of them. Returns ``sinr_db``, ``rate_bps``, ``serving``, ``rsrp_dbm``
        and ``los`` arrays of length ``N`` plus ``direction`` ``(N, 3)``: the unit vector from the UAV to its serving cell.
        """
        pos = np.atleast_2d(np.asarray(pos, float))
        rsrp, los = self.large_scale_rsrp(pos)
        if shadowing and self.use_shadowing:
            g = self._geometry(pos)
            sig = self.model.shadowing_std_db(g, los)
            rsrp = rsrp - (rng or self._streams[7]).normal(size=rsrp.shape) * sig
        serving = rsrp.argmax(axis=1)
        sinr = compute_sinr(rsrp, serving, self.noise_dbm, np.full(self.n_cells, self.load))
        idx = np.arange(len(pos))
        rate = self.bandwidth_hz * self.layers * np.log2(1.0 + db2lin(sinr.sinr_db - 3.0))
        d = self.cell_pos[serving] - pos
        direction = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        return dict(sinr_db=sinr.sinr_db, rate_bps=rate, serving=serving, rsrp_dbm=rsrp[idx, serving], los=los[idx, serving], direction=direction)

    # ------------------------------------------------------------------------------------------------ reporting
    @property
    def handover_events(self) -> List[Any]:
        ho = getattr(self, "_ho", None)
        return list(ho.events) if ho is not None else []

    def coverage_map(self, altitude: float, extent: Sequence[float], resolution: float = 10.0, samples: int = 1) -> tuple:
        """Large-scale SINR [dB] on a horizontal grid (no fading, no mobility state): ``(xs, ys, sinr_db)``."""
        from ..maps.rem import radio_environment_map

        rem = radio_environment_map(self, altitude, extent, resolution)
        return rem.xs, rem.ys, rem.sinr_db

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            type="nr_cellular",
            freq_hz=self.freq,
            bandwidth_hz=self.bandwidth_hz,
            n_prb=self.n_prb,
            scs_khz=self.numerology.scs_khz,
            cells=[
                dict(pos=c["pos"].tolist(), yaw_deg=math.degrees(c["yaw"]), tx_power_dbm=c["power"], antenna=c["antenna"].describe())
                for c in self.cells
            ],
            propagation=self.model.describe(),
            los_mode=self.los_mode,
            shadowing=self.use_shadowing,
            fading=self.use_fading,
            handover=self._ho_cfg,
            rlf=self._rlf_cfg,
            scheduler=self.scheduler_policy,
            phy=self.phy_kind,
            noise_dbm=self.noise_dbm,
            outage_db=self.outage_sinr_db,
            load=self.load,
            jammers=self.jammers,
        )
