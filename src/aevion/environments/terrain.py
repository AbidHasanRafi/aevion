"""Terrain from elevation files: ESRI ASCII grids, SRTM ``.hgt`` tiles, NumPy arrays and (optionally) GeoTIFF.

All loaders return a :class:`Terrain` on a regular metric grid in the simulation's world frame, ready for
:meth:`aevion.environments.world.World.set_heightmap` (``heights[iy, ix]`` at ``origin + (ix, iy) * resolution``).

Geographic data is resampled to a metric grid around a :class:`~aevion.environments.geodesy.LocalFrame` with local radii of curvature
(equirectangular about the origin latitude; error below 0.1 % over a 10 km tile at mid-latitudes). With ``relative=True`` heights are
measured from the frame's altitude. No geoid correction is applied.

Loaders are written for untrusted files: dimensions are capped, headers validated and voids filled by diffusion. GeoTIFF needs the
optional ``tifffile`` package, imported only when a ``.tif`` is opened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

from .geodesy import LocalFrame, meridian_radius, prime_vertical_radius

__all__ = ["Terrain", "load_terrain", "fill_voids", "resample_to_metric"]

MAX_CELLS = 100_000_000
SRTM_VOID = -32768


class TerrainError(ValueError):
    """Unreadable or unsupported elevation data."""


@dataclass
class Terrain:
    heights: np.ndarray  # (ny, nx) metres
    origin: Tuple[float, float]  # world (x, y) of heights[0, 0]
    resolution: float  # metres per cell

    @property
    def shape(self) -> Tuple[int, int]:
        return self.heights.shape  # type: ignore[return-value]

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        ny, nx = self.heights.shape
        return (self.origin[0], self.origin[1], self.origin[0] + (nx - 1) * self.resolution, self.origin[1] + (ny - 1) * self.resolution)

    def decimated(self, factor: int) -> "Terrain":
        f = max(int(factor), 1)
        return Terrain(self.heights[::f, ::f].copy(), self.origin, self.resolution * f)

    def apply(self, world: Any) -> Any:
        world.set_heightmap(self.heights, self.origin, self.resolution)
        x0, y0, x1, y1 = self.extent
        b = np.array(world.bounds, float)
        b[[0, 1]], b[[3, 4]] = [min(b[0], x0), min(b[1], y0)], [max(b[3], x1), max(b[4], y1)]
        b[5] = max(b[5], float(self.heights.max()) + 50.0)
        world.bounds = b
        return world


def fill_voids(h: np.ndarray, mask: np.ndarray, iterations: int = 400) -> np.ndarray:
    """Fill ``mask`` cells by repeatedly averaging their valid neighbours (nearest-valid seeding, then diffusion)."""
    h = np.array(h, float)
    if not mask.any():
        return h
    if mask.all():
        raise TerrainError("elevation data has no valid cells")
    # seed every void with a nearby valid value by repeated dilation, then smooth the filled cells by diffusion
    valid = ~mask
    filled = valid.copy()
    for _ in range(max(h.shape)):
        if filled.all():
            break
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb_f, nb_h = np.roll(filled, shift, axis), np.roll(h, shift, axis)
            take = ~filled & nb_f
            h[take], filled[take] = nb_h[take], True
    for _ in range(iterations):
        p = np.pad(h, 1, mode="edge")
        avg = 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
        new = np.where(mask, avg, h)
        if np.max(np.abs(new - h)) < 1e-4:
            h = new
            break
        h = new
    return h


def _bilinear(grid: np.ndarray, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    ny, nx = grid.shape
    fy, fx = np.clip(fy, 0, ny - 1 - 1e-9), np.clip(fx, 0, nx - 1 - 1e-9)
    iy, ix = np.floor(fy).astype(int), np.floor(fx).astype(int)
    ty, tx = fy - iy, fx - ix
    return (grid[iy, ix] * (1 - tx) + grid[iy, ix + 1] * tx) * (1 - ty) + (grid[iy + 1, ix] * (1 - tx) + grid[iy + 1, ix + 1] * tx) * ty


def resample_to_metric(
    heights: np.ndarray,
    lat0: float,
    lon0: float,
    dlat: float,
    dlon: float,
    frame: LocalFrame,
    resolution: float,
    extent_m: Optional[float] = None,
    relative: bool = True,
) -> Terrain:
    """Resample a geographic grid (``heights[iy, ix]`` at ``lat0 + iy*dlat``, ``lon0 + ix*dlon``, south-up) to a metric grid about ``frame``."""
    ny, nx = heights.shape
    lat_c = lat0 + 0.5 * dlat * (ny - 1)
    m_lat = np.radians(1.0) * float(meridian_radius(lat_c))
    m_lon = np.radians(1.0) * float(prime_vertical_radius(lat_c)) * np.cos(np.radians(lat_c))
    # grid corners in metres relative to the frame origin (equirectangular about the frame's latitude)
    x_of = lambda lon: (lon - frame.lon) * np.radians(1.0) * float(prime_vertical_radius(frame.lat)) * np.cos(np.radians(frame.lat))  # noqa: E731
    y_of = lambda lat: (lat - frame.lat) * np.radians(1.0) * float(meridian_radius(frame.lat))  # noqa: E731
    x0, x1 = x_of(lon0), x_of(lon0 + dlon * (nx - 1))
    y0, y1 = y_of(lat0), y_of(lat0 + dlat * (ny - 1))
    if extent_m is not None:
        half = extent_m / 2.0
        x0, x1, y0, y1 = max(x0, -half), min(x1, half), max(y0, -half), min(y1, half)
    if x1 <= x0 or y1 <= y0:
        raise TerrainError("the elevation data does not cover the requested area around the frame origin")
    xs, ys = np.arange(x0, x1 + 1e-9, resolution), np.arange(y0, y1 + 1e-9, resolution)
    if len(xs) * len(ys) > MAX_CELLS:
        raise TerrainError("requested terrain is too large; raise the resolution or reduce the extent")
    X, Y = np.meshgrid(xs, ys)
    lon = frame.lon + X / (np.radians(1.0) * float(prime_vertical_radius(frame.lat)) * np.cos(np.radians(frame.lat)))
    lat = frame.lat + Y / (np.radians(1.0) * float(meridian_radius(frame.lat)))
    fy, fx = (lat - lat0) / dlat, (lon - lon0) / dlon
    out = _bilinear(heights, fy, fx)
    del m_lat, m_lon
    if relative:
        out = out - frame.alt
    return Terrain(out, (float(xs[0]), float(ys[0])), float(resolution))


# --------------------------------------------------------------------------------------------- readers
def _read_asc(path: Path) -> Tuple[np.ndarray, dict]:
    with open(path, encoding="ascii", errors="replace") as fh:
        head = {}
        for _ in range(6):
            line = fh.readline().split()
            if len(line) != 2:
                raise TerrainError(f"{path}: malformed ESRI ASCII header")
            head[line[0].lower()] = float(line[1])
        nrows, ncols = int(head.get("nrows", 0)), int(head.get("ncols", 0))
        if nrows < 2 or ncols < 2 or nrows * ncols > MAX_CELLS:
            raise TerrainError(f"{path}: grid of {nrows} x {ncols} cells is empty or too large")
        data = np.loadtxt(fh, dtype=float, ndmin=2)
    if data.shape != (nrows, ncols):
        raise TerrainError(f"{path}: expected {nrows} x {ncols} values, found {data.shape}")
    nodata = head.get("nodata_value", -9999.0)
    mask = data == nodata
    return np.flipud(data), dict(head, mask=np.flipud(mask))


def load_terrain(
    path: Union[str, Path],
    frame: Optional[LocalFrame] = None,
    resolution: Optional[float] = None,
    extent_m: Optional[float] = None,
    relative: bool = True,
    geographic: Optional[bool] = None,
) -> Terrain:
    """Read an elevation file as a metric :class:`Terrain`.

    ``frame`` is required for geographic data (SRTM ``.hgt``, ``.asc``/``.tif`` in degrees); ``resolution`` (m) defaults to the native cell
    size (metric data) or ~30 m (geographic); ``extent_m`` crops a square of that side centred on the frame origin.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".npy", ".npz"):
        if ext == ".npy":
            h = np.load(p, allow_pickle=False)
            origin, res = (0.0, 0.0), float(resolution or 1.0)
        else:
            with np.load(p, allow_pickle=False) as z:
                if "heights" not in z.files:
                    raise TerrainError(f"{p}: .npz must contain 'heights' (and optionally 'origin', 'resolution')")
                h = z["heights"]
                origin = (float(z["origin"][0]), float(z["origin"][1])) if "origin" in z.files else (0.0, 0.0)
                res = float(z["resolution"]) if "resolution" in z.files else float(resolution or 1.0)
        if h.ndim != 2 or min(h.shape) < 2 or h.size > MAX_CELLS or not np.all(np.isfinite(h)):
            raise TerrainError(f"{p}: heights must be a finite 2-D array of at least 2 x 2 cells")
        return Terrain(np.asarray(h, float), (origin[0], origin[1]), res)
    if ext == ".hgt":
        raw = p.read_bytes()
        n = round((len(raw) / 2) ** 0.5)
        if n * n * 2 != len(raw) or n < 2 or n * n > MAX_CELLS:
            raise TerrainError(f"{p}: not a square 16-bit SRTM tile")
        h = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(float)
        m = re.match(r"([NS])(\d{2})([EW])(\d{3})", p.stem.upper())
        if m is None:
            raise TerrainError(f"{p}: SRTM tile names look like N47E008.hgt (the name gives the south-west corner)")
        lat0 = (1 if m.group(1) == "N" else -1) * int(m.group(2))
        lon0 = (1 if m.group(3) == "E" else -1) * int(m.group(4))
        if frame is None:
            raise TerrainError("SRTM tiles are geographic: pass frame=LocalFrame(lat, lon, alt)")
        h = fill_voids(np.flipud(h), np.flipud(h == SRTM_VOID))  # file rows run north to south
        step = 1.0 / (n - 1)
        return resample_to_metric(h, lat0, lon0, step, step, frame, resolution or 30.0, extent_m, relative)
    if ext in (".asc", ".txt"):
        h, head = _read_asc(p)
        cell = head.get("cellsize", 0.0)
        if cell <= 0:
            raise TerrainError(f"{p}: cellsize must be positive")
        h = fill_voids(h, head["mask"])
        x0 = head.get("xllcenter", head.get("xllcorner", 0.0) + 0.5 * cell)
        y0 = head.get("yllcenter", head.get("yllcorner", 0.0) + 0.5 * cell)
        geo = geographic if geographic is not None else (abs(x0) <= 180 and abs(y0) <= 90 and cell < 0.1)
        if geo:
            if frame is None:
                raise TerrainError("this ASCII grid is in degrees: pass frame=LocalFrame(lat, lon, alt)")
            return resample_to_metric(h, y0, x0, cell, cell, frame, resolution or 30.0, extent_m, relative)
        t = Terrain(h - (frame.alt if (frame is not None and relative) else 0.0), (x0, y0), cell)
        return t if resolution is None or resolution == cell else _resample_metric(t, resolution)
    if ext in (".tif", ".tiff"):
        return _load_geotiff(p, frame, resolution, extent_m, relative, geographic)
    raise TerrainError(f"unsupported terrain format {ext!r}; use .asc, .hgt, .npy, .npz or .tif")


def _resample_metric(t: Terrain, resolution: float) -> Terrain:
    ny, nx = t.heights.shape
    xs = np.arange(0.0, (nx - 1) * t.resolution + 1e-9, resolution)
    ys = np.arange(0.0, (ny - 1) * t.resolution + 1e-9, resolution)
    X, Y = np.meshgrid(xs / t.resolution, ys / t.resolution)
    return Terrain(_bilinear(t.heights, Y, X), t.origin, float(resolution))


def _load_geotiff(
    p: Path, frame: Optional[LocalFrame], resolution: Optional[float], extent_m: Optional[float], relative: bool, geographic: Optional[bool]
) -> Terrain:
    try:
        import tifffile  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on the optional package
        raise ImportError("reading GeoTIFF needs the optional 'tifffile' package: pip install tifffile (or convert the DEM to .asc / .npz)") from e
    with tifffile.TiffFile(str(p)) as tf:  # pragma: no cover
        page = tf.pages[0]
        data = page.asarray().astype(float)
        scale = page.tags.get("ModelPixelScaleTag")
        tie = page.tags.get("ModelTiepointTag")
        nodata = page.tags.get("GDAL_NODATA")
    if data.ndim != 2 or scale is None or tie is None:  # pragma: no cover
        raise TerrainError(f"{p}: need a single-band GeoTIFF with pixel scale and tiepoint tags")
    sx, sy = float(scale.value[0]), float(scale.value[1])  # pragma: no cover
    x0, y_top = float(tie.value[3]), float(tie.value[4])  # pragma: no cover
    mask = np.zeros(data.shape, bool)  # pragma: no cover
    if nodata is not None:  # pragma: no cover
        mask = data == float(str(nodata.value).strip("\x00"))
    data = fill_voids(np.flipud(data), np.flipud(mask))  # pragma: no cover
    geo = geographic if geographic is not None else (abs(x0) <= 180 and abs(y_top) <= 90 and sx < 0.1)  # pragma: no cover
    y0 = y_top - sy * (data.shape[0] - 1)  # pragma: no cover
    if geo:  # pragma: no cover
        if frame is None:
            raise TerrainError("this GeoTIFF is in degrees: pass frame=LocalFrame(lat, lon, alt)")
        return resample_to_metric(data, y0, x0, sy, sx, frame, resolution or 30.0, extent_m, relative)
    t = Terrain(data - (frame.alt if (frame is not None and relative) else 0.0), (x0, y0), sx)  # pragma: no cover
    return t if resolution is None or resolution == sx else _resample_metric(t, resolution)  # pragma: no cover
