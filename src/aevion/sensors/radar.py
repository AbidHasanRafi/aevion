"""FMCW radar: detections with range, angle, radial velocity and SNR, and an optional range-Doppler map.

The waveform is defined by the carrier ``carrier_hz``, sweep ``bandwidth_hz``, chirp duration ``chirp_s`` and chirps per frame
``n_chirps``::

    range resolution      dR = c / (2 B)
    velocity resolution   dv = lambda / (2 N T_c)
    unambiguous velocity  v_max = lambda / (4 T_c)       (velocities beyond +-v_max wrap around, as on hardware)

Targets come from ray casting a grid of beams across the field of view. Each first-surface hit is a target with a radar cross-section
(a specular cos^2 lobe times ``rcs_m2`` for obstacles, ``ground_rcs_m2 * sin(grazing)`` for the ground), a radial velocity (vehicle
velocity along the line of sight plus the surface velocity of moving actors) and an SNR from the radar equation::

    Pr = Pt G^2 lambda^2 sigma / ((4 pi)^3 R^4),     SNR = Pr (N T_c) / (k T0 F L)

(matched-filter SNR after coherent integration of the ``N`` chirps, with noise figure ``F`` and losses ``L``). A target is detected when
its SNR exceeds ``snr_threshold_db``, and measurement errors follow the Cramér-Rao scaling ``sigma = resolution / sqrt(2 SNR)``.
:meth:`Radar.range_doppler` renders the power map after the range and Doppler FFTs (sinc^2 point-spread function in both axes, velocity
wrapped) over a thermal noise floor, and :func:`ca_cfar` is a cell-averaging CFAR detector for it.

Fidelity: one first-surface return per beam (no multipath, penetration or shadowing within a beam), point targets, no clutter statistics
beyond the ground return, no antenna sidelobes. Radial velocity is positive when the target approaches.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from ..core import rotation as rot
from .base import Sensor
from .raycast import cast_rays

__all__ = ["Radar", "ca_cfar", "C_LIGHT", "K_BOLTZMANN"]

C_LIGHT = 299_792_458.0
K_BOLTZMANN = 1.380649e-23


class Radar(Sensor):
    """Scanning FMCW radar mounted along body +x. Output ``detections`` ``(N, K, 6)`` = ``[range, azimuth, elevation, radial_velocity, snr_db, rcs]``,
    ``valid`` ``(N, K)`` (detected beams), and ``rd_map`` ``(N, n_doppler, n_range)`` [dB] when ``map=True``."""

    name = "radar"

    def __init__(
        self,
        rate: float = 20.0,
        carrier_hz: float = 77e9,
        bandwidth_hz: float = 1e9,
        chirp_s: float = 40e-6,
        n_chirps: int = 128,
        tx_power_dbm: float = 10.0,
        antenna_gain_dbi: float = 20.0,
        noise_figure_db: float = 12.0,
        losses_db: float = 6.0,
        snr_threshold_db: float = 13.0,
        fov_h_deg: float = 90.0,
        fov_v_deg: float = 20.0,
        beams_h: int = 15,
        beams_v: int = 3,
        max_range: float = 100.0,
        min_range: float = 0.5,
        rcs_m2: float = 10.0,
        ground_rcs_m2: float = 0.5,
        ground_clutter: bool = True,
        map: bool = False,
        n_range_bins: int = 128,
        **kw: Any,
    ):
        super().__init__(rate=rate, **kw)
        if bandwidth_hz <= 0 or chirp_s <= 0 or n_chirps < 2 or max_range <= min_range or beams_h < 1 or beams_v < 1:
            raise ValueError("need bandwidth_hz > 0, chirp_s > 0, n_chirps >= 2, max_range > min_range and at least one beam per axis")
        self.fc, self.bw, self.tc, self.n_chirps = float(carrier_hz), float(bandwidth_hz), float(chirp_s), int(n_chirps)
        self.pt_dbm, self.gain_dbi, self.nf_db, self.thr_db = (
            float(tx_power_dbm),
            float(antenna_gain_dbi),
            float(noise_figure_db),
            float(snr_threshold_db),
        )
        self.losses_db = float(losses_db)
        self.fov_h, self.fov_v = math.radians(fov_h_deg), math.radians(fov_v_deg)
        self.beams_h, self.beams_v = int(beams_h), int(beams_v)
        self.max_range, self.min_range = float(max_range), float(min_range)
        self.rcs, self.ground_rcs, self.ground_clutter = float(rcs_m2), float(ground_rcs_m2), bool(ground_clutter)
        self.want_map, self.n_range_bins = bool(map), int(n_range_bins)
        az = np.linspace(-self.fov_h / 2, self.fov_h / 2, self.beams_h) if self.beams_h > 1 else np.array([0.0])
        el = np.linspace(-self.fov_v / 2, self.fov_v / 2, self.beams_v) if self.beams_v > 1 else np.array([0.0])
        A, E = np.meshgrid(az, el, indexing="xy")
        self.beam_az, self.beam_el = A.reshape(-1), E.reshape(-1)
        self.dirs_body = np.stack(
            [np.cos(self.beam_el) * np.cos(self.beam_az), np.cos(self.beam_el) * np.sin(self.beam_az), np.sin(self.beam_el)], axis=-1
        )
        if self.extrinsics is not None:
            self.dirs_body = self.extrinsics.rotate(self.dirs_body)

    # ------------------------------------------------------------------------------------- derived figures
    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.fc

    @property
    def range_resolution(self) -> float:
        return C_LIGHT / (2.0 * self.bw)

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2.0 * self.n_chirps * self.tc)

    @property
    def max_unambiguous_velocity(self) -> float:
        return self.wavelength / (4.0 * self.tc)

    @property
    def noise_psd(self) -> float:
        """Effective noise power spectral density ``k T0 F L`` [W/Hz]."""
        return K_BOLTZMANN * 290.0 * 10.0 ** ((self.nf_db + self.losses_db) / 10.0)

    @property
    def integration_s(self) -> float:
        """Coherent integration time of one frame, ``N T_c``."""
        return self.n_chirps * self.tc

    def snr_db(self, rcs_m2: Any, range_m: Any) -> Any:
        """Radar-equation SNR [dB] of a point target (monostatic, one antenna gain for transmit and receive)."""
        pt = 1e-3 * 10.0 ** (self.pt_dbm / 10.0)
        g = 10.0 ** (self.gain_dbi / 10.0)
        pr = pt * g * g * self.wavelength**2 * np.asarray(rcs_m2, float) / ((4.0 * math.pi) ** 3 * np.maximum(np.asarray(range_m, float), 1e-3) ** 4)
        with np.errstate(divide="ignore"):  # a zero cross-section is -inf dB, not a warning
            return 10.0 * np.log10(pr * self.integration_s / self.noise_psd)

    def detection_range(self, rcs_m2: float) -> float:
        """Range [m] at which a target of this RCS drops to the detection threshold (radar equation solved for ``R``)."""
        pt = 1e-3 * 10.0 ** (self.pt_dbm / 10.0)
        g = 10.0 ** (self.gain_dbi / 10.0)
        snr = 10.0 ** (self.thr_db / 10.0)
        return float((pt * g * g * self.wavelength**2 * rcs_m2 * self.integration_s / ((4.0 * math.pi) ** 3 * self.noise_psd * snr)) ** 0.25)

    def wrap_velocity(self, v: Any) -> Any:
        vm = self.max_unambiguous_velocity
        return (np.asarray(v, float) + vm) % (2.0 * vm) - vm

    # --------------------------------------------------------------------------------------------- sensing
    def measure(self, state: Any) -> dict:
        B, n = self.B, self.n
        R = B.to_numpy(rot.to_matrix(B, state.quat))
        pos, vel = B.to_numpy(state.pos), B.to_numpy(state.vel)
        K = len(self.dirs_body)
        dw = np.einsum("nij,kj->nki", R, self.dirs_body)
        o = np.repeat(pos[:, None, :], K, axis=1).reshape(-1, 3)
        world = self.sim.world
        h = cast_rays(world, o, dw.reshape(-1, 3), self.max_range)
        t = np.asarray(h.t).reshape(n, K)
        prim = np.asarray(h.prim).reshape(n, K)
        nrm = np.asarray(h.normal).reshape(n, K, 3)
        hit = (prim != -1) & (t >= self.min_range)
        cosinc = np.clip(-np.sum(nrm * dw, axis=-1), 0.0, 1.0)
        ground = prim == -2
        rcs = np.where(
            ground, self.ground_rcs * cosinc + 1e-9, self.rcs * cosinc**2
        )  # ground backscatter ~ sin(grazing angle) = cos(angle to the vertical)
        if not self.ground_clutter:
            hit = hit & ~ground
        v_surf = world.primitive_velocity(prim, float(state.t)) if getattr(world, "actors", None) else np.zeros((n, K, 3))
        approach = np.einsum("nki,nki->nk", dw, vel[:, None, :] - v_surf)  # beam direction points to the target: positive when the range is closing
        snr = self.snr_db(rcs, t)
        detected = hit & (snr >= self.thr_db)
        snr_lin = 10.0 ** (snr / 10.0)
        sd = 1.0 / np.sqrt(2.0 * np.maximum(snr_lin, 1e-9))
        g = lambda: B.to_numpy(self.noise((n, K), 1.0))  # noqa: E731
        rng_m = t + g() * self.range_resolution * sd
        vel_m = self.wrap_velocity(approach + g() * self.velocity_resolution * sd)
        bw_h = 0.886 * self.wavelength / 0.05  # a few-centimetre-class aperture: 0.886 lambda / D, D = 5 cm gives ~4 deg at 77 GHz
        az_m = self.beam_az[None, :] + g() * (bw_h / 1.6) * sd
        el_m = self.beam_el[None, :] + g() * (bw_h / 1.6) * sd
        det = np.stack([rng_m, az_m, el_m, vel_m, snr, rcs], axis=-1)
        det = np.where(detected[..., None], det, 0.0)
        out = dict(detections=B.asarray(det), valid=detected, t=state.t)
        if self.want_map:
            out["rd_map"] = B.asarray(self.range_doppler(t, approach, snr, detected))
        return out

    def range_doppler(self, ranges: np.ndarray, approach: np.ndarray, snr_db: np.ndarray, detected: np.ndarray) -> np.ndarray:
        """Power map in dB over (Doppler bin, range bin) for every vehicle: thermal floor plus one sinc^2 blob per *target in view*.

        Axes: range bins ``0..max_range`` (``n_range_bins`` cells), Doppler bins ``-v_max..+v_max`` (``n_chirps`` cells, zero at the centre).
        Power is relative to the noise floor (0 dB = noise), i.e. a point target peaks at its SNR.
        """
        n, K = ranges.shape
        nr, nd = self.n_range_bins, self.n_chirps
        r_axis = (np.arange(nr) + 0.5) * (self.max_range / nr)
        vm = self.max_unambiguous_velocity
        v_axis = (np.arange(nd) - nd / 2.0) * (2.0 * vm / nd)
        maps = np.ones((n, nd, nr))  # linear power relative to noise
        dr, dv = self.range_resolution, self.velocity_resolution
        for i in range(n):
            for k in np.nonzero(detected[i])[0]:
                p = 10.0 ** (snr_db[i, k] / 10.0)
                vk = float(self.wrap_velocity(approach[i, k]))
                dvv = (v_axis - vk + vm) % (2.0 * vm) - vm  # distance on the wrapped velocity axis
                maps[i] += p * np.outer(np.sinc(dvv / dv) ** 2, np.sinc((r_axis - ranges[i, k]) / dr) ** 2)
        return 10.0 * np.log10(maps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "carrier_hz": self.fc,
            "bandwidth_hz": self.bw,
            "chirp_s": self.tc,
            "n_chirps": self.n_chirps,
            "max_range": self.max_range,
        }


def ca_cfar(power_db: np.ndarray, guard: Tuple[int, int] = (2, 2), train: Tuple[int, int] = (8, 8), pfa: float = 1e-4) -> np.ndarray:
    """Cell-averaging 2-D CFAR on a power map in dB (relative to noise): returns a boolean detection mask.

    The threshold is ``alpha * mean(training cells)`` with ``alpha = N (pfa^(-1/N) - 1)`` for ``N`` training cells (exponential noise power),
    so the false-alarm rate in pure noise equals ``pfa`` regardless of the noise level.
    """
    p = 10.0 ** (np.asarray(power_db, float) / 10.0)
    gd, gr = guard
    td, tr = train
    outer = (2 * (gd + td) + 1, 2 * (gr + tr) + 1)
    inner = (2 * gd + 1, 2 * gr + 1)
    n_train = outer[0] * outer[1] - inner[0] * inner[1]
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    def box_sum(a: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        pad = ((size[0] // 2, size[0] // 2), (size[1] // 2, size[1] // 2))
        ap = np.pad(a, pad, mode="edge")
        c = np.cumsum(np.cumsum(ap, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        return c[size[0] :, size[1] :] - c[: -size[0], size[1] :] - c[size[0] :, : -size[1]] + c[: -size[0], : -size[1]]

    train_mean = (box_sum(p, outer) - box_sum(p, inner)) / n_train
    return p > alpha * train_mean
