"""LiDAR scan patterns: where each beam of a scan points, and when it is fired.

A :class:`ScanPattern` returns, for scan number ``k``, the unit beam directions in the sensor frame (``x`` forward, ``z`` up) and the
firing time of each beam within the scan period (used for motion distortion). Implementations (``scan_patterns`` registry):

* ``spinning``     channels × azimuth samples over a vertical and horizontal field of view (spinning or solid-state grid).
* ``rosette``      Livox-style non-repetitive pattern: two counter-rotating Risley prisms steer the beam along a rosette whose phase
  continues from scan to scan, so successive scans progressively fill the field of view.
* ``mems_raster``  a resonant (sinusoidal) fast axis and a linear slow axis, in boustrophedon lines.
* ``lissajous``    a MEMS mirror driven at two frequencies; a ratio of small coprime integers repeats every scan.

Fidelity: ideal beam directions (no mirror non-linearity or jitter beyond what the sensor adds). The rosette follows the published
Risley-prism construction with a free amplitude split and frequency ratio, not a particular product's calibration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["ScanPattern", "SpinningPattern", "RosettePattern", "MemsRasterPattern", "LissajousPattern", "PATTERNS", "pattern_from_spec"]


def _dirs(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


class ScanPattern:
    """Interface. ``n_rays`` beams per scan, ``static`` if every scan is identical, ``grid_shape`` ``(rows, cols)`` when the beams form a grid."""

    name = "pattern"
    static = True
    grid_shape: Optional[Tuple[int, int]] = None
    n_rays = 0

    def directions(self, scan: int = 0) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def time_offsets(self, period: float) -> np.ndarray:
        """Firing time of each beam within a scan of duration ``period`` (increasing, in ``[0, period)``)."""
        return np.arange(self.n_rays) / max(self.n_rays, 1) * period

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class SpinningPattern(ScanPattern):
    name = "spinning"

    def __init__(self, channels: int = 16, samples: int = 360, v_fov: Tuple[float, float] = (-15.0, 15.0), h_fov: float = 360.0):
        if channels < 1 or samples < 1 or not 0 < h_fov <= 360 or v_fov[1] < v_fov[0]:
            raise ValueError("need channels, samples >= 1, 0 < h_fov <= 360 and v_fov = (low, high)")
        self.channels, self.samples, self.v_fov, self.h_fov = int(channels), int(samples), tuple(float(x) for x in v_fov), float(h_fov)
        el = np.deg2rad(np.linspace(v_fov[0], v_fov[1], channels)) if channels > 1 else np.array([np.deg2rad(sum(v_fov) / 2)])
        full = abs(h_fov - 360.0) < 1e-6
        az = np.deg2rad(np.linspace(-h_fov / 2, h_fov / 2, samples, endpoint=not full))
        E, A = np.meshgrid(el, az, indexing="ij")
        self._d = _dirs(A, E).reshape(-1, 3)
        self.n_rays, self.grid_shape = channels * samples, (channels, samples)
        self._t = np.tile(np.arange(samples) / samples, channels)  # all channels fire together as the head turns

    def directions(self, scan: int = 0) -> np.ndarray:
        return self._d

    def time_offsets(self, period: float) -> np.ndarray:
        return self._t * period

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="spinning", channels=self.channels, samples=self.samples, v_fov=list(self.v_fov), h_fov=self.h_fov)


class RosettePattern(ScanPattern):
    """Two Risley prisms: beam offset ``A1 e^{i w1 t} + A2 e^{i w2 t}`` (angles). ``fov_deg`` is the full circular field of view,
    ``split`` the share of the deflection carried by the first prism, ``ratio`` the (irrational) frequency ratio ``w2 / w1``.
    The phase continues between scans (``scan`` selects the segment), so consecutive scans differ."""

    name = "rosette"
    static = False

    def __init__(
        self,
        points_per_scan: int = 4000,
        fov_deg: float = 38.4,
        point_rate_hz: float = 100_000.0,
        base_hz: float = 55.0,
        ratio: float = -0.6180339887,
        split: float = 0.5,
    ):
        if points_per_scan < 8 or fov_deg <= 0 or point_rate_hz <= 0 or not 0.0 < split < 1.0 or ratio == 0:
            raise ValueError("need points_per_scan >= 8, fov_deg > 0, point_rate_hz > 0, 0 < split < 1 and ratio != 0")
        self.n_rays, self.fov = int(points_per_scan), math.radians(fov_deg)
        self.rate, self.base, self.ratio, self.split = float(point_rate_hz), float(base_hz), float(ratio), float(split)

    def directions(self, scan: int = 0) -> np.ndarray:
        t = (scan * self.n_rays + np.arange(self.n_rays)) / self.rate
        a1, a2 = self.split * self.fov / 2.0, (1.0 - self.split) * self.fov / 2.0
        ph1, ph2 = 2.0 * np.pi * self.base * t, 2.0 * np.pi * self.base * self.ratio * t
        off = a1 * np.exp(1j * ph1) + a2 * np.exp(1j * ph2)  # complex (azimuth, elevation) offset in radians
        return _dirs(off.real, off.imag)

    def time_offsets(self, period: float) -> np.ndarray:
        return np.arange(self.n_rays) / self.rate if self.n_rays / self.rate <= period else np.arange(self.n_rays) / self.n_rays * period

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            type="rosette",
            points_per_scan=self.n_rays,
            fov_deg=math.degrees(self.fov),
            point_rate_hz=self.rate,
            base_hz=self.base,
            ratio=self.ratio,
            split=self.split,
        )


class MemsRasterPattern(ScanPattern):
    """``lines`` scan lines of ``points`` samples: sinusoidal fast axis (azimuth, alternating direction) and a linear slow axis (elevation)."""

    name = "mems_raster"

    def __init__(self, h_fov: float = 60.0, v_fov: float = 20.0, lines: int = 32, points: int = 128):
        if lines < 1 or points < 2 or not 0 < h_fov < 180 or not 0 < v_fov < 180:
            raise ValueError("need lines >= 1, points >= 2 and fields of view in (0, 180) degrees")
        self.h_fov, self.v_fov, self.lines, self.points = float(h_fov), float(v_fov), int(lines), int(points)
        self.n_rays, self.grid_shape = lines * points, (lines, points)
        ph = (np.arange(points) + 0.5) / points  # position inside one line
        fast = np.sin(np.pi * (ph - 0.5))  # resonant mirror: dense near the turning points
        az = np.zeros((lines, points))
        for i in range(lines):
            az[i] = fast if i % 2 == 0 else fast[::-1]  # boustrophedon: every other line runs backwards
        el = np.repeat(np.linspace(-0.5, 0.5, lines)[:, None], points, axis=1)
        self._d = _dirs(np.deg2rad(az * h_fov / 2.0), np.deg2rad(el * v_fov)).reshape(-1, 3)

    def directions(self, scan: int = 0) -> np.ndarray:
        return self._d

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="mems_raster", h_fov=self.h_fov, v_fov=self.v_fov, lines=self.lines, points=self.points)


class LissajousPattern(ScanPattern):
    """Azimuth ``A sin(2 pi fx t)``, elevation ``B sin(2 pi fy t + phase)`` over one scan; ``fx`` and ``fy`` are cycles per scan (integers repeat)."""

    name = "lissajous"

    def __init__(self, points: int = 2000, h_fov: float = 50.0, v_fov: float = 25.0, fx: float = 23.0, fy: float = 17.0, phase_deg: float = 90.0):
        if points < 8 or h_fov <= 0 or v_fov <= 0 or fx <= 0 or fy <= 0:
            raise ValueError("need points >= 8 and positive fields of view and frequencies")
        self.n_rays, self.h_fov, self.v_fov, self.fx, self.fy, self.phase = (
            int(points),
            float(h_fov),
            float(v_fov),
            float(fx),
            float(fy),
            float(phase_deg),
        )
        t = np.arange(points) / points
        az = np.deg2rad(0.5 * h_fov) * np.sin(2 * np.pi * fx * t)
        el = np.deg2rad(0.5 * v_fov) * np.sin(2 * np.pi * fy * t + np.deg2rad(phase_deg))
        self._d = _dirs(az, el)

    def directions(self, scan: int = 0) -> np.ndarray:
        return self._d

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="lissajous", points=self.n_rays, h_fov=self.h_fov, v_fov=self.v_fov, fx=self.fx, fy=self.fy, phase_deg=self.phase)


PATTERNS = {c.name: c for c in (SpinningPattern, RosettePattern, MemsRasterPattern, LissajousPattern)}


def pattern_from_spec(spec: Any) -> ScanPattern:
    """A pattern instance from an instance, a name, or ``{"type": name, **params}``."""
    if isinstance(spec, ScanPattern):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    d = dict(spec)
    kind = d.pop("type")
    if kind not in PATTERNS:
        raise ValueError(f"unknown scan pattern {kind!r}; known: {sorted(PATTERNS)}")
    return PATTERNS[kind](**d)
