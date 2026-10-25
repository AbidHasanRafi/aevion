"""A GNSS receiver that works from pseudoranges: constellation geometry, signal errors, a least-squares fix and DOP.

The measurement chain follows a real receiver:

1. **Satellites.** Synthetic Walker-delta constellations with the published geometry of GPS, Galileo, GLONASS and BeiDou-MEO (plane
   count, inclination, orbit radius, satellites per plane) on circular orbits, rotated into the Earth-fixed frame by the sidereal angle
   of the simulation's UTC time. They are **not real almanacs**: identities and positions on a given date differ from the real systems,
   but visibility, elevation distribution and DOP statistics are representative.
2. **Visibility.** An elevation mask and, when the world has obstacles, line-of-sight ray casting towards each satellite. Blocked
   satellites are dropped or, with probability ``nlos_probability``, received through a reflection (a positive range bias).
3. **Pseudorange errors.** Geometric range, receiver clock bias, residual ionosphere (Klobuchar-like diurnal delay with obliquity
   mapping; ``iono_residual`` remains after correction), residual troposphere (Saastamoinen zenith delay, Niell-like mapping),
   satellite orbit and clock error (per-satellite Gauss-Markov, common to all vehicles), multipath (elevation-dependent Gauss-Markov)
   and thermal noise that follows ``C/N0``.
4. **Solution.** Weighted least squares for position and clock bias, Doppler range-rate least squares for velocity, and DOP from the
   geometry matrix. With fewer than four usable satellites (or GDOP > 100) there is no fix and no reading is published, so downstream
   filters see a gap.

Jamming (``gnss_jamming`` fault) lowers ``C/N0`` per vehicle through :attr:`GnssReceiver.interference_fn`, and satellites below the lock
threshold are lost. Spoofing (``gnss_spoofing``) drags the published solution through the sensor's ``filters``.

Fidelity: pseudorange level, single frequency; no carrier phase/RTK, ephemeris quantisation or ionospheric scintillation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..environments.geodesy import LocalFrame
from .base import Sensor

__all__ = ["Constellation", "CONSTELLATIONS", "GnssReceiver", "dop", "gmst_rad", "C_LIGHT", "OMEGA_EARTH"]

C_LIGHT = 299_792_458.0
OMEGA_EARTH = 7.2921151467e-5  # rad/s
MU_EARTH = 3.986004418e14
R_EARTH = 6_378_137.0
J2000_UNIX = 946_728_000.0


def gmst_rad(unix_s: float) -> float:
    """Greenwich mean sidereal angle [rad] at Unix time ``unix_s`` (low-precision IAU formula, ~0.1 arcsec per decade)."""
    d = (unix_s - J2000_UNIX) / 86400.0  # days since J2000 (kept small: a full Julian day number would quantise the angle to ~0.1 m on the orbit)
    return math.radians((280.46061837 + 360.98564736629 * d) % 360.0)


@dataclass(frozen=True)
class Constellation:
    """Walker-delta ``inclination: n_sats / n_planes / phasing`` constellation on circular orbits of radius ``semi_major_m``."""

    name: str
    n_planes: int
    per_plane: int
    inclination_deg: float
    semi_major_m: float
    phasing: int = 1
    raan0_deg: float = 0.0

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.per_plane

    @property
    def mean_motion(self) -> float:
        return math.sqrt(MU_EARTH / self.semi_major_m**3)

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.mean_motion

    def elements(self) -> Tuple[np.ndarray, np.ndarray]:
        """RAAN and initial mean anomaly [rad] of every satellite."""
        j, k = np.meshgrid(np.arange(self.n_planes), np.arange(self.per_plane), indexing="ij")
        raan = np.radians(self.raan0_deg + 360.0 * j / self.n_planes)
        m0 = np.radians(360.0 * k / self.per_plane + 360.0 * self.phasing * j / self.n_sats)
        return raan.reshape(-1), m0.reshape(-1)

    def ecef(self, unix_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Satellite positions and ECEF velocities ``(S, 3)`` at Unix time ``unix_s``."""
        raan, m0 = self.elements()
        u = m0 + self.mean_motion * (unix_s - J2000_UNIX)  # argument of latitude on a circular orbit
        inc = math.radians(self.inclination_deg)
        cu, su, ci, si = np.cos(u), np.sin(u), math.cos(inc), math.sin(inc)
        co, so = np.cos(raan), np.sin(raan)
        a, n = self.semi_major_m, self.mean_motion
        pos = a * np.stack([co * cu - so * ci * su, so * cu + co * ci * su, si * su], axis=-1)  # inertial frame
        vel = a * n * np.stack([-co * su - so * ci * cu, -so * su + co * ci * cu, si * cu], axis=-1)
        th = gmst_rad(unix_s)
        c, s = math.cos(th), math.sin(th)
        rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # inertial -> Earth-fixed
        pe = pos @ rot.T
        ve = vel @ rot.T - np.cross(np.array([0.0, 0.0, OMEGA_EARTH]), pe)  # velocity relative to the rotating frame
        return pe, ve


CONSTELLATIONS: Dict[str, Constellation] = {
    "gps": Constellation("gps", 6, 5, 55.0, 26_559_800.0, phasing=2),
    "galileo": Constellation("galileo", 3, 8, 56.0, 29_599_800.0, phasing=1, raan0_deg=45.0),
    "glonass": Constellation("glonass", 3, 8, 64.8, 25_507_000.0, phasing=1, raan0_deg=90.0),
    "beidou": Constellation("beidou", 3, 8, 55.0, 27_906_000.0, phasing=1, raan0_deg=20.0),
}


def dop(unit_vectors_enu: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Dilution of precision from line-of-sight unit vectors in ENU ``(N, S, 3)`` and the usable-satellite ``mask`` ``(N, S)``.

    ``H`` rows are ``[-e, 1]``; ``Q = (H^T H)^-1``; ``HDOP = sqrt(Q_ee + Q_nn)``, ``VDOP = sqrt(Q_uu)``, ``PDOP``, ``TDOP`` and ``GDOP`` follow.
    Vehicles with fewer than four satellites or a singular geometry get ``inf``.
    """
    H = np.concatenate([-unit_vectors_enu, np.ones(unit_vectors_enu.shape[:-1] + (1,))], axis=-1) * mask[..., None]
    A = np.einsum("nsi,nsj->nij", H, H)
    n = A.shape[0]
    out = {k: np.full(n, np.inf) for k in ("hdop", "vdop", "pdop", "tdop", "gdop")}
    ok = (mask.sum(1) >= 4) & (np.abs(np.linalg.det(A)) > 1e-9)
    if ok.any():
        Q = np.linalg.inv(A[ok])
        out["hdop"][ok] = np.sqrt(Q[:, 0, 0] + Q[:, 1, 1])
        out["vdop"][ok] = np.sqrt(Q[:, 2, 2])
        out["pdop"][ok] = np.sqrt(Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2])
        out["tdop"][ok] = np.sqrt(Q[:, 3, 3])
        out["gdop"][ok] = np.sqrt(np.trace(Q, axis1=1, axis2=2))
    return out


def _iono_obliquity(el: np.ndarray, h_shell: float = 350e3) -> np.ndarray:
    return 1.0 / np.sqrt(1.0 - (R_EARTH / (R_EARTH + h_shell) * np.cos(el)) ** 2)


def _tropo_mapping(el: np.ndarray) -> np.ndarray:
    return 1.0 / (np.sin(el) + 0.00143 / (np.tan(el) + 0.0455))


class GnssReceiver(Sensor):
    """Single-frequency GNSS receiver (see the module docstring). Output keys: ``pos`` (ENU, world frame), ``vel``, ``fix``, ``n_sat``,
    ``hdop``, ``vdop``, ``pdop``, ``gdop``, ``cn0`` (mean dB-Hz), ``clock_bias_m``, ``sat_el_deg`` and ``sat_used`` (per satellite)."""

    name = "gnss"

    def __init__(
        self,
        rate: float = 5.0,
        latency: float = 0.1,
        constellations: Sequence[str] = ("gps", "galileo"),
        elevation_mask_deg: float = 10.0,
        cn0_zenith_dbhz: float = 47.0,
        lock_threshold_dbhz: float = 28.0,
        noise_sigma_m: float = 0.35,
        doppler_sigma_mps: float = 0.03,
        iono_vertical_m: float = 4.0,
        iono_residual: float = 0.4,
        tropo_residual: float = 0.1,
        sisre_m: float = 0.7,
        multipath_sigma_m: float = 0.4,
        urban: bool = True,
        nlos_probability: float = 0.25,
        nlos_bias_m: Tuple[float, float] = (10.0, 60.0),
        clock_bias_std_m: float = 1.0e4,
        clock_drift_walk: float = 0.05,
        max_gdop: float = 100.0,
        ideal: bool = False,
        **kw: Any,
    ):
        super().__init__(rate=rate, latency=latency, **kw)
        unknown = [c for c in constellations if c not in CONSTELLATIONS]
        if unknown or not constellations:
            raise ValueError(f"constellations must be a non-empty subset of {sorted(CONSTELLATIONS)}; got {list(constellations)}")
        if not 0.0 <= elevation_mask_deg < 90.0:
            raise ValueError("elevation_mask_deg must be in [0, 90)")
        if not 0.0 <= nlos_probability <= 1.0:
            raise ValueError("nlos_probability must be in [0, 1]")
        self.constellations = tuple(constellations)
        self.mask_rad = math.radians(elevation_mask_deg)
        self.cn0_zenith, self.lock_dbhz = float(cn0_zenith_dbhz), float(lock_threshold_dbhz)
        self.sigma, self.sigma_dopp = float(noise_sigma_m), float(doppler_sigma_mps)
        self.iono_v, self.iono_res, self.tropo_res = float(iono_vertical_m), float(iono_residual), float(tropo_residual)
        self.sisre, self.mp_sigma = float(sisre_m), float(multipath_sigma_m)
        self.urban, self.nlos_p, self.nlos_bias = bool(urban), float(nlos_probability), tuple(float(x) for x in nlos_bias_m)
        self.clock_std, self.clock_walk, self.max_gdop, self.ideal = float(clock_bias_std_m), float(clock_drift_walk), float(max_gdop), bool(ideal)
        #: optional ``f(positions_enu (N,3)) -> penalty_db (N,)``: extra C/N0 loss per vehicle (set by the ``gnss_jamming`` fault)
        self.interference_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.sats = [CONSTELLATIONS[c] for c in self.constellations]
        self.n_sats = sum(c.n_sats for c in self.sats)

    # ------------------------------------------------------------------------------------------ lifecycle
    def on_bind(self) -> None:
        n, S = self.n, self.n_sats
        world = self.sim.world
        self.frame: LocalFrame = world.origin or LocalFrame(45.0, 0.0, 0.0)
        utc = world.utc or "2026-06-21T12:00:00Z"
        from ..environments.solar import _to_unix

        self.t0_unix = float(_to_unix(utc))
        rng = np.random.default_rng(int(self.B.to_numpy(self.B.rand((1,), self.gen))[0] * 2**31))
        self._rng = rng
        self.clock_bias = rng.normal(0.0, 0.0 if self.ideal else self.clock_std, n)
        self.clock_drift = np.zeros(n)
        self.sis = rng.normal(0.0, self.sisre, S) if not self.ideal else np.zeros(S)  # per-satellite orbit/clock error, common to all users
        self.mp = np.zeros((n, S))
        self._last_fix = np.zeros(n, bool)
        self._held: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------------------------------------- helpers
    def _satellites(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        pos, vel = [], []
        for c in self.sats:
            p, v = c.ecef(self.t0_unix + t)
            pos.append(p)
            vel.append(v)
        return np.concatenate(pos), np.concatenate(vel)

    def _blocked(self, usr_enu: np.ndarray, e_enu: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Boolean (N, S): is the line of sight to the satellite blocked by the world's geometry?"""
        world = self.sim.world
        blocked = np.zeros(visible.shape, bool)
        if not (self.urban and world.has_obstacles):
            return blocked
        from .raycast import cast_rays

        ii, jj = np.nonzero(visible)
        if len(ii) == 0:
            return blocked
        hit = cast_rays(world, usr_enu[ii], e_enu[ii, jj], 2000.0).hit
        blocked[ii, jj] = np.asarray(hit)
        return blocked

    def measure(self, state: Any) -> Optional[dict]:
        B, n, S, t = self.B, self.n, self.n_sats, float(state.t)
        pos_enu = B.to_numpy(state.pos).astype(float)
        vel_enu = B.to_numpy(state.vel).astype(float)
        fr = self.frame
        usr = fr.enu_to_ecef(pos_enu)
        v_usr = vel_enu @ fr.rotation
        sat, vsat = self._satellites(t)
        rel = sat[None, :, :] - usr[:, None, :]
        rng = np.linalg.norm(rel, axis=-1)
        e_ecef = rel / rng[..., None]
        e_enu = e_ecef @ fr.rotation.T
        el = np.arcsin(np.clip(e_enu[..., 2], -1.0, 1.0))
        visible = el >= self.mask_rad
        blocked = self._blocked(pos_enu, e_enu, visible)
        dt = self.dt
        rng_ = self._rng

        # ---- signal strength (antenna gain pattern + jamming); NLoS signals arrive weaker
        cn0 = self.cn0_zenith - 9.0 * (1.0 - np.sin(np.maximum(el, 0.0))) * 1.0
        cn0 = np.broadcast_to(cn0, (n, S)).copy()
        used = visible.copy()
        nlos = np.zeros((n, S), bool)
        if blocked.any():
            reflect = blocked & (rng_.random((n, S)) < self.nlos_p)
            nlos = reflect
            cn0 = cn0 - 10.0 * nlos
            used = used & (~blocked | reflect)
        if self.interference_fn is not None:
            cn0 = cn0 - np.asarray(self.interference_fn(pos_enu), float)[:, None]
        used = used & (cn0 >= self.lock_dbhz)

        # ---- pseudorange error terms
        if self.ideal:
            err = np.zeros((n, S))
            doppler_err = np.zeros((n, S))
            sigma = np.full((n, S), 1.0)
            sig_d = np.full((n, S), 1.0)
            clock_b, clock_d = np.zeros(n), np.zeros(n)
        else:
            self.sis = self.sis * math.exp(-dt / 3600.0) + rng_.normal(0.0, self.sisre * math.sqrt(1.0 - math.exp(-2.0 * dt / 3600.0)), S)
            mp_std = self.mp_sigma * (0.4 + 1.6 * np.exp(-np.degrees(np.maximum(el, 0.0)) / 15.0))
            k = math.exp(-dt / 10.0)
            self.mp = self.mp * k + rng_.normal(0.0, 1.0, (n, S)) * mp_std * math.sqrt(1.0 - k * k)
            local_h = ((t + self.t0_unix) % 86400.0) / 3600.0 + fr.lon / 15.0
            diurnal = 0.35 + 0.65 * max(math.cos(2.0 * math.pi * (local_h - 14.0) / 24.0), 0.0)
            iono = self.iono_res * self.iono_v * diurnal * _iono_obliquity(np.maximum(el, 0.05))
            alt = np.maximum(pos_enu[:, 2] + fr.alt, 0.0)[:, None]
            tropo = self.tropo_res * 2.4 * np.exp(-alt / 8400.0) * _tropo_mapping(np.maximum(el, 0.05))
            sigma = self.sigma * 10.0 ** (-(cn0 - 45.0) / 20.0)
            nlos_bias = np.where(nlos, rng_.uniform(self.nlos_bias[0], self.nlos_bias[1], (n, S)), 0.0)
            err = iono + tropo + self.sis[None, :] + self.mp + nlos_bias + rng_.normal(0.0, 1.0, (n, S)) * sigma
            self.clock_drift = self.clock_drift + rng_.normal(0.0, self.clock_walk * math.sqrt(dt), n)
            self.clock_bias = self.clock_bias + self.clock_drift * dt
            clock_b, clock_d = self.clock_bias, self.clock_drift
            sig_d = self.sigma_dopp * 10.0 ** (-(cn0 - 45.0) / 20.0)
            doppler_err = rng_.normal(0.0, 1.0, (n, S)) * sig_d
        pr = rng + clock_b[:, None] + err

        # ---- weighted least-squares position + clock bias (Gauss-Newton, started from the receiver's previous belief = truth here)
        w = np.where(used, 1.0 / np.maximum(sigma, 1e-6) ** 2, 0.0)
        x = np.concatenate([usr + 0.0, np.zeros((n, 1))], axis=1)
        for _ in range(6):
            r = sat[None, :, :] - x[:, None, :3]
            rr = np.linalg.norm(r, axis=-1)
            res = pr - rr - x[:, 3:4]
            H = np.concatenate([-r / rr[..., None], np.ones((n, S, 1))], axis=-1)
            A = np.einsum("nsi,ns,nsj->nij", H, w, H)
            b = np.einsum("nsi,ns,ns->ni", H, w, res)
            ok = used.sum(1) >= 4
            A[~ok] = np.eye(4)
            try:
                dx = np.linalg.solve(A + 1e-12 * np.eye(4), b[..., None])[..., 0]
            except np.linalg.LinAlgError:  # singular geometry for some vehicle: give up on this epoch for all vehicles' updates
                ok &= False
                dx = np.zeros((n, 4))
            x = x + dx
            if np.max(np.abs(dx[ok])) < 1e-4 if ok.any() else True:
                break
        # ---- Doppler velocity: range-rate = -e . (v_user - v_sat) + clock drift
        e_unit = (sat[None, :, :] - x[:, None, :3]) / np.linalg.norm(sat[None, :, :] - x[:, None, :3], axis=-1)[..., None]
        rate = -np.einsum("nsi,nsi->ns", e_unit, v_usr[:, None, :] - vsat[None, :, :]) + clock_d[:, None] + doppler_err
        base = np.einsum("nsi,nsi->ns", e_unit, vsat[None, :, :] * np.ones((n, 1, 1)))
        Hv = np.concatenate([-e_unit, np.ones((n, S, 1))], axis=-1)
        wd = np.where(used, 1.0 / np.maximum(sig_d, 1e-6) ** 2, 0.0)
        Av = np.einsum("nsi,ns,nsj->nij", Hv, wd, Hv)
        bv = np.einsum("nsi,ns,ns->ni", Hv, wd, rate - base)  # rate = -e.v_u + e.v_s + d  ->  rate - e.v_s = -e.v_u + d
        Av[~ok] = np.eye(4)
        sol_v = np.linalg.solve(Av + 1e-12 * np.eye(4), bv[..., None])[..., 0]
        v_fix_ecef = sol_v[:, :3]

        d = dop(e_enu, used)
        fix = ok & np.isfinite(d["gdop"]) & (d["gdop"] <= self.max_gdop)
        self._last_fix = fix
        if not fix.any():
            return None  # no vehicle has a fix: nothing to publish (downstream filters see a gap)
        pos_fix = fr.ecef_to_enu(x[:, :3])
        vel_fix = v_fix_ecef @ fr.rotation.T
        if self._held is None:
            self._held = (pos_fix.copy(), vel_fix.copy())
        self._held = (np.where(fix[:, None], pos_fix, self._held[0]), np.where(fix[:, None], vel_fix, self._held[1]))
        pos_out, vel_out = self._held  # a vehicle without a fix keeps reporting its last good solution (check ``fix``)
        return dict(
            pos=B.asarray(pos_out),
            vel=B.asarray(vel_out),
            fix=B.asarray(fix.astype(float)),
            n_sat=B.asarray(used.sum(1).astype(float)),
            hdop=B.asarray(np.where(np.isfinite(d["hdop"]), d["hdop"], 99.0)),
            vdop=B.asarray(np.where(np.isfinite(d["vdop"]), d["vdop"], 99.0)),
            pdop=B.asarray(np.where(np.isfinite(d["pdop"]), d["pdop"], 99.0)),
            gdop=B.asarray(np.where(np.isfinite(d["gdop"]), d["gdop"], 99.0)),
            cn0=B.asarray(np.where(used.any(1), (cn0 * used).sum(1) / np.maximum(used.sum(1), 1), 0.0)),
            clock_bias_m=B.asarray(x[:, 3]),
            sat_el_deg=np.degrees(el),
            sat_used=used,
            t=state.t,
        )

    def record_payload(self, reading: dict) -> Any:
        """Stored per sample: ENU position (3), velocity (3), fix, n_sat, hdop, pdop, cn0 -> ``(N, 11)``."""
        B = self.B
        cols = [reading["pos"], reading["vel"]] + [B.to_numpy(reading[k])[:, None] for k in ("fix", "n_sat", "hdop", "pdop", "cn0")]
        return np.concatenate([B.to_numpy(c) for c in cols], axis=1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "constellations": list(self.constellations),
            "elevation_mask_deg": math.degrees(self.mask_rad),
            "noise_sigma_m": self.sigma,
            "urban": self.urban,
            "ideal": self.ideal,
        }
