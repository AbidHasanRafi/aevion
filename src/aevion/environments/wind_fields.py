"""Spatially varying wind: a log-law boundary layer, imported (CFD or measured) gridded fields, and a parametric urban wake model.

* :class:`LogLawWind`   neutral atmospheric boundary layer with roughness length and displacement height.
* :class:`GriddedWind`  a regular 3-D (optionally time-dependent) velocity grid with trilinear interpolation, for the output of a CFD
  run, a mesoscale model or measurements (``.npz`` or ``.csv``).
* :class:`UrbanWind`    ambient log-law wind modified around the world's buildings: windward stagnation and uplift, a recirculation
  cavity sized by Fackrell's (1984) reattachment length, a recovering wake and downwash.

Fidelity: :class:`LogLawWind` is the standard neutral surface-layer profile. :class:`GriddedWind` is exact at the nodes and reproduces
linear fields, but does not enforce mass conservation between nodes. :class:`UrbanWind` is an empirical, qualitative model in the
spirit of Röckle's (1990) zonal approach: it gives the right sign and scale of building effects for flight-control and
gust-rejection studies, not a CFD-grade field. All models are deterministic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .wind import Wind

__all__ = ["LogLawWind", "GriddedWind", "UrbanWind", "fackrell_reattachment_length"]

KAPPA = 0.41  # von Karman constant


class LogLawWind(Wind):
    """``u(z) = (u* / kappa) ln((z - d) / z0)`` blowing towards ``direction_deg`` (counter-clockwise from +x / east).

    Parameterised like :class:`~aevion.environments.wind.WindShear`: the speed ``speed_ref`` is given at ``height_ref``; ``u*`` follows.
    Below ``d + 2 z0`` the profile is held (the log law is invalid inside the roughness layer).
    """

    def __init__(self, speed_ref: float = 5.0, height_ref: float = 10.0, z0: float = 0.03, displacement: float = 0.0, direction_deg: float = 0.0):
        if z0 <= 0 or height_ref <= displacement + z0:
            raise ValueError("need z0 > 0 and height_ref above the displacement height plus roughness length")
        self.speed_ref, self.height_ref, self.z0, self.displacement = float(speed_ref), float(height_ref), float(z0), float(displacement)
        self.direction = math.radians(direction_deg)
        self.u_star = KAPPA * self.speed_ref / math.log((self.height_ref - self.displacement) / self.z0)

    def speed_at(self, z: np.ndarray) -> np.ndarray:
        zz = np.maximum(np.asarray(z, float) - self.displacement, 2.0 * self.z0)
        return self.u_star / KAPPA * np.log(zz / self.z0)

    def sample(self, B: Any, pos: Any, vel: Any, t: float, dt: float) -> Any:
        p = B.to_numpy(pos)
        mag = self.speed_at(p[:, 2])
        out = np.stack([mag * math.cos(self.direction), mag * math.sin(self.direction), np.zeros_like(mag)], -1)
        return B.asarray(out)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            type="LogLawWind",
            speed_ref=self.speed_ref,
            height_ref=self.height_ref,
            z0=self.z0,
            displacement=self.displacement,
            direction_deg=math.degrees(self.direction),
        )


# ------------------------------------------------------------------------------------------------ gridded
class GriddedWind(Wind):
    """Trilinear interpolation of a regular velocity grid.

    ``data`` has shape ``(nx, ny, nz, 3)`` (static) or ``(nt, nx, ny, nz, 3)`` with ``times`` (strictly increasing, seconds); node
    ``(i, j, k)`` sits at ``origin + (i, j, k) * spacing``. ``outside``: ``"nearest"`` holds the boundary value, ``"zero"`` gives calm
    air beyond the grid. ``scale`` multiplies all velocities; ``loop`` repeats a time series periodically.
    """

    def __init__(
        self,
        data: Any,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        spacing: Union[float, Sequence[float]] = 1.0,
        times: Optional[Sequence[float]] = None,
        outside: str = "nearest",
        scale: float = 1.0,
        loop: bool = False,
        source: Optional[str] = None,
    ):
        d = np.asarray(data, dtype=float)
        if times is None:
            if d.ndim != 4 or d.shape[-1] != 3:
                raise ValueError(f"static wind data must have shape (nx, ny, nz, 3), got {d.shape}")
            d = d[None]
            self.times = None
        else:
            self.times = np.asarray(times, float)
            if d.ndim != 5 or d.shape[-1] != 3 or d.shape[0] != len(self.times):
                raise ValueError(f"time-dependent wind data must have shape (nt, nx, ny, nz, 3) with nt = len(times); got {d.shape}")
            if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
                raise ValueError("times must be strictly increasing with at least two entries")
        if min(d.shape[1:4]) < 2:
            raise ValueError("each grid axis needs at least two nodes")
        if not np.all(np.isfinite(d)):
            raise ValueError("wind data contains NaN or inf")
        if outside not in ("nearest", "zero"):
            raise ValueError("outside must be 'nearest' or 'zero'")
        self.data = d
        self.origin = np.asarray(origin, float).reshape(3)
        self.spacing = np.broadcast_to(np.asarray(spacing, float), (3,)).copy()
        if np.any(self.spacing <= 0):
            raise ValueError("grid spacing must be positive")
        self.outside, self.scale, self.loop, self.source = outside, float(scale), bool(loop), source

    # ---------------------------------------------------------------------------------------- loading
    @classmethod
    def from_npz(cls, path: Union[str, Path], **kw: Any) -> "GriddedWind":
        """``.npz`` with ``u`` (see class doc), ``origin``, ``spacing`` and optionally ``t``."""
        with np.load(path, allow_pickle=False) as z:
            missing = {"u", "origin", "spacing"} - set(z.files)
            if missing:
                raise ValueError(f"{path}: missing array(s) {sorted(missing)}; need u, origin, spacing (and optionally t)")
            return cls(z["u"], z["origin"], z["spacing"], z["t"] if "t" in z.files else None, source=str(path), **kw)

    @classmethod
    def from_csv(cls, path: Union[str, Path], **kw: Any) -> "GriddedWind":
        """CSV with header ``x,y,z,u,v,w`` listing every node of a regular grid (any row order)."""
        arr = np.genfromtxt(path, delimiter=",", names=True)
        if not {"x", "y", "z", "u", "v", "w"} <= set(arr.dtype.names or ()):
            raise ValueError(f"{path}: header must contain x, y, z, u, v, w")
        axes = [np.unique(arr[k]) for k in ("x", "y", "z")]
        shape = tuple(len(a) for a in axes)
        if int(np.prod(shape)) != len(arr):
            raise ValueError(f"{path}: {len(arr)} rows do not fill a regular {shape} grid")
        step = []
        for a in axes:
            dd = np.diff(a)
            if len(dd) == 0 or not np.allclose(dd, dd[0], rtol=1e-6):
                raise ValueError(f"{path}: grid spacing is not uniform")
            step.append(float(dd[0]))
        idx = [np.searchsorted(a, arr[k]) for a, k in zip(axes, ("x", "y", "z"))]
        data = np.full(shape + (3,), np.nan)
        for c, k in enumerate(("u", "v", "w")):
            data[idx[0], idx[1], idx[2], c] = arr[k]
        if np.isnan(data).any():
            raise ValueError(f"{path}: duplicate or missing grid nodes")
        return cls(data, [a[0] for a in axes], step, source=str(path), **kw)

    # ---------------------------------------------------------------------------------------- queries
    @property
    def extent(self) -> np.ndarray:
        """``[xmin, ymin, zmin, xmax, ymax, zmax]`` covered by the grid."""
        hi = self.origin + (np.array(self.data.shape[1:4]) - 1) * self.spacing
        return np.concatenate([self.origin, hi])

    def _frame_weights(self, t: float) -> List[tuple]:
        if self.times is None:
            return [(0, 1.0)]
        if self.loop:
            t = self.times[0] + (t - self.times[0]) % (self.times[-1] - self.times[0])
        if t <= self.times[0]:
            return [(0, 1.0)]
        if t >= self.times[-1]:
            return [(len(self.times) - 1, 1.0)]
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        a = (t - self.times[k]) / (self.times[k + 1] - self.times[k])
        return [(k, 1.0 - a), (k + 1, a)]

    def interpolate(self, pos: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Velocity ``(N, 3)`` at world positions ``pos`` ``(N, 3)`` and time ``t``."""
        p = np.asarray(pos, float).reshape(-1, 3)
        f = (p - self.origin) / self.spacing
        dims = np.array(self.data.shape[1:4]) - 1
        inside = np.all((f >= 0.0) & (f <= dims), axis=1)
        fc = np.clip(f, 0.0, dims - 1e-9)
        i0 = np.floor(fc).astype(int)
        w = fc - i0
        out = np.zeros((len(p), 3))
        for frame, fw in self._frame_weights(t):
            g = self.data[frame]
            acc = np.zeros((len(p), 3))
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * (w[:, 2] if dz else 1 - w[:, 2])
                        acc += wt[:, None] * g[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
            out += fw * acc
        out *= self.scale
        if self.outside == "zero":
            out[~inside] = 0.0
        return out

    def divergence(self, frame: int = 0) -> np.ndarray:
        """Central-difference divergence of one frame on the interior nodes [1/s]; near zero for incompressible CFD data."""
        g = self.data[frame]
        dx, dy, dz = self.spacing
        return (
            (g[2:, 1:-1, 1:-1, 0] - g[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
            + (g[1:-1, 2:, 1:-1, 1] - g[1:-1, :-2, 1:-1, 1]) / (2 * dy)
            + (g[1:-1, 1:-1, 2:, 2] - g[1:-1, 1:-1, :-2, 2]) / (2 * dz)
        )

    def sample(self, B: Any, pos: Any, vel: Any, t: float, dt: float) -> Any:
        return B.asarray(self.interpolate(B.to_numpy(pos), t))

    def to_dict(self) -> Dict[str, Any]:
        if not self.source:
            raise ValueError("a GriddedWind built from arrays has no source file and cannot be serialised; save it with save_npz() first")
        return dict(type="GriddedWind", file=self.source, outside=self.outside, scale=self.scale, loop=self.loop)

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Write the field to ``path`` and remember it as the ``source`` (so the wind can be serialised)."""
        arrays: Dict[str, Any] = dict(u=self.data if self.times is not None else self.data[0], origin=self.origin, spacing=self.spacing)
        if self.times is not None:
            arrays["t"] = self.times
        p = Path(path)
        np.savez_compressed(p, **arrays)
        self.source = str(p)
        return p


# ------------------------------------------------------------------------------------------------ urban
def fackrell_reattachment_length(height: float, width: float, length: float) -> float:
    """Length of the recirculation cavity behind a rectangular building [m] (Fackrell 1984):
    ``L_R / H = 1.8 (W/H) / [ (L/H)^0.3 (1 + 0.24 W/H) ]`` with W across and L along the wind."""
    h = max(float(height), 1e-6)
    return h * 1.8 * (width / h) / (((length / h) ** 0.3) * (1.0 + 0.24 * width / h))


_K_REATTACH = -math.log(0.8)  # f(1) = 0 for f(r) = 1 - 1.25 exp(-k r^2)


class UrbanWind(Wind):
    """Log-law wind modified by the buildings of the world (``buildings=None`` takes the world's boxes when the world is bound).

    For each box of height ``H`` the wind-aligned frame has ``xs`` downwind of the lee face and ``ys`` across the wake axis. The
    horizontal wind is multiplied by ``1 - mask * (1 - f)`` where ``f(r) = 1 - 1.25 exp(-0.223 r^2)``, ``r = xs / L_R``: reversed flow
    (``f = -0.25``) at the lee face, zero at the reattachment length ``L_R`` (Fackrell), recovering to the free stream beyond. Upwind,
    a stagnation zone slows the flow and deflects it upward; in the cavity the flow descends (downwash). The masks fall off smoothly
    across and above the building so the field is continuous (important for the controller).
    """

    def __init__(
        self,
        speed_ref: float = 6.0,
        height_ref: float = 10.0,
        z0: float = 0.5,
        displacement: float = 0.0,
        direction_deg: float = 0.0,
        buildings: Optional[Sequence[Sequence[float]]] = None,
        strength: float = 1.0,
        min_height: float = 2.0,
    ):
        self.ambient = LogLawWind(speed_ref, height_ref, z0, displacement, direction_deg)
        self.strength, self.min_height = float(strength), float(min_height)
        self._given = None if buildings is None else np.asarray(buildings, float).reshape(-1, 6)
        self._boxes = self._given
        self._geom: Optional[np.ndarray] = None
        self._prepare()

    # ------------------------------------------------------------------------------------- geometry
    def bind_world(self, world: Any) -> None:
        if self._given is None:
            self._boxes = np.asarray(world.boxes, float).reshape(-1, 6)
            self._prepare()

    def _prepare(self) -> None:
        """Per-building wake parameters in the wind-aligned frame: [s_lee, c_mid, H, W, L, L_R]."""
        boxes = self._boxes if self._boxes is not None else np.zeros((0, 6))
        th = self.ambient.direction
        ex, ey = math.cos(th), math.sin(th)
        rows = []
        for b in boxes:
            if b[5] - max(b[2], 0.0) < self.min_height:
                continue
            xs, ys = np.array([b[0], b[3], b[0], b[3]]), np.array([b[1], b[1], b[4], b[4]])
            s, c = xs * ex + ys * ey, -xs * ey + ys * ex
            height = float(b[5])
            width, length = float(c.max() - c.min()), float(s.max() - s.min())
            rows.append([s.max(), 0.5 * (c.max() + c.min()), height, width, length, fackrell_reattachment_length(height, width, length)])
        self._geom = np.asarray(rows, float).reshape(-1, 6)

    @property
    def n_buildings(self) -> int:
        return 0 if self._geom is None else len(self._geom)

    def cavity_lengths(self) -> np.ndarray:
        """Reattachment length of every building for the current wind direction [m]."""
        return np.array([]) if self._geom is None else self._geom[:, 5].copy()

    # ------------------------------------------------------------------------------------------ field
    def field(self, pos: np.ndarray) -> np.ndarray:
        p = np.asarray(pos, float).reshape(-1, 3)
        base = np.stack([self.ambient.speed_at(p[:, 2]), np.zeros(len(p)), np.zeros(len(p))], -1)  # wind-aligned: x along the wind
        th = self.ambient.direction
        ex, ey = math.cos(th), math.sin(th)
        s, c = p[:, 0] * ex + p[:, 1] * ey, -p[:, 0] * ey + p[:, 1] * ex
        ratio = np.ones(len(p))
        wz = np.zeros(len(p))
        g = self._geom
        if g is not None and len(g) and self.strength > 0:
            lee, cmid, hgt, wid, _, lr = (g[:, i][None, :] for i in range(6))
            xs, ys, z = s[:, None] - lee, c[:, None] - cmid, p[:, 2][:, None]
            u_h = self.ambient.speed_at(hgt)  # (1, K)
            down = xs >= 0.0
            r = np.where(down, xs / lr, 0.0)
            half = 0.5 * wid + 0.12 * np.maximum(xs, 0.0)  # the wake widens slowly
            lat = np.exp(-(np.maximum(np.abs(ys) - half, 0.0) ** 2) / (2.0 * (0.3 * wid) ** 2))
            ver = np.where(z <= hgt, 1.0, np.exp(-((z - hgt) ** 2) / (2.0 * (0.35 * hgt) ** 2)))
            f = 1.0 - 1.25 * np.exp(-_K_REATTACH * r * r)
            wake = down * lat * ver
            # windward side: stagnation (slow-down) and uplift, over about one building height
            up = ~down
            xu = np.where(up, -xs, 0.0)
            lat_u = np.exp(-(np.maximum(np.abs(ys) - 0.5 * wid, 0.0) ** 2) / (2.0 * (0.3 * wid) ** 2))
            stag = up * lat_u * ver * np.exp(-xu / (0.6 * hgt))
            fac = 1.0 - self.strength * (wake * (1.0 - f) + 0.35 * stag)
            ratio = np.prod(fac, axis=1)
            cavity = wake * np.where(r < 1.0, np.sin(np.pi * np.clip(r, 0, 1)), 0.0)  # downwash peaks mid-cavity
            wz = np.sum(self.strength * (-0.2 * u_h * cavity + 0.25 * u_h * stag), axis=1)
        local = np.stack([base[:, 0] * ratio, np.zeros(len(p)), wz], -1)
        return np.stack([local[:, 0] * ex, local[:, 0] * ey, local[:, 2]], -1)

    def sample(self, B: Any, pos: Any, vel: Any, t: float, dt: float) -> Any:
        return B.asarray(self.field(B.to_numpy(pos)))

    def to_dict(self) -> Dict[str, Any]:
        a = self.ambient
        d: Dict[str, Any] = dict(
            type="UrbanWind",
            speed_ref=a.speed_ref,
            height_ref=a.height_ref,
            z0=a.z0,
            displacement=a.displacement,
            direction_deg=math.degrees(a.direction),
            strength=self.strength,
            min_height=self.min_height,
        )
        if self._given is not None:
            d["buildings"] = self._given.tolist()
        return d
