"""Buildings from map data: GeoJSON footprints and Overpass-API JSON (OpenStreetMap), extruded to meshes or boxes.

Only JSON is parsed (no XML), so there is no entity-expansion attack surface; feature counts and sizes are capped.

Heights come from the first available of ``height`` (metres, ``"12"`` or ``"12 m"``), ``building:levels`` x ``level_height`` (default
3 m) or ``default_height``. ``min_height`` / ``building:min_level`` lift the base (building parts on stilts). Footprints with holes are
extruded as solid (courtyards are not carved out); multipolygon relations use their outer rings.

Coordinates are converted with a :class:`~aevion.environments.geodesy.LocalFrame` (local ENU about the world origin, ellipsoidal
height ignored), so the result lines up with terrain loaded about the same frame.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .geodesy import LocalFrame
from .mesh import MeshError, TriangleMesh, extrude_polygon

__all__ = ["Footprint", "load_footprints", "buildings_to_meshes", "buildings_to_boxes", "MAX_FEATURES"]

MAX_FEATURES = 200_000
MAX_FILE_BYTES = 256 * 1024 * 1024
_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(m|ft|')?\s*$")


class Footprint:
    """One building outline in local metres plus its height range."""

    __slots__ = ("base", "height", "name", "polygon")

    def __init__(self, polygon: np.ndarray, height: float, base: float = 0.0, name: str = ""):
        self.polygon, self.height, self.base, self.name = polygon, float(height), float(base), name

    def __repr__(self) -> str:
        return f"Footprint({len(self.polygon)} vertices, height={self.height:.1f} m)"


def _parse_length(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if np.isfinite(v) else None
    m = _NUM.match(str(v))
    if m is None:
        return None
    x = float(m.group(1))
    return x * 0.3048 if m.group(2) in ("ft", "'") else x


def _heights(tags: Dict[str, Any], default_height: float, level_height: float) -> Tuple[float, float]:
    top = _parse_length(tags.get("height"))
    if top is None:
        levels = _parse_length(tags.get("building:levels"))
        top = levels * level_height if levels is not None and levels > 0 else default_height
    base = _parse_length(tags.get("min_height"))
    if base is None:
        lv = _parse_length(tags.get("building:min_level"))
        base = lv * level_height if lv is not None and lv > 0 else 0.0
    return max(top, base + 0.5), max(base, 0.0)


def _to_local(frame: LocalFrame, ring: Iterable[Tuple[float, float]]) -> np.ndarray:
    ll = np.asarray(list(ring), float)  # (lon, lat) pairs as in GeoJSON
    if ll.ndim != 2 or ll.shape[1] < 2 or not np.all(np.isfinite(ll[:, :2])) or np.any(np.abs(ll[:, 1]) > 90) or np.any(np.abs(ll[:, 0]) > 540):
        raise MeshError("invalid coordinates in a building outline")
    enu = frame.to_enu(np.column_stack([ll[:, 1], ll[:, 0], np.full(len(ll), frame.alt)]))
    return enu[:, :2]


def _load_json(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(source, dict):
        return source
    p = Path(source)
    if p.stat().st_size > MAX_FILE_BYTES:
        raise MeshError(f"{p}: file too large")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MeshError(f"{p}: not valid JSON ({e})") from e


def load_footprints(
    source: Union[str, Path, Dict[str, Any]], frame: LocalFrame, default_height: float = 10.0, level_height: float = 3.0, min_area: float = 4.0
) -> List[Footprint]:
    """Read GeoJSON (``FeatureCollection`` of Polygon/MultiPolygon) or Overpass JSON (``elements`` with ``geometry``) into footprints."""
    doc = _load_json(source)
    out: List[Footprint] = []

    def add(ring: Any, tags: Dict[str, Any], name: str) -> None:
        if len(out) >= MAX_FEATURES:
            raise MeshError(f"more than {MAX_FEATURES} buildings; crop the map data")
        try:
            poly = _to_local(frame, ring)
        except MeshError:
            return
        if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        if len(poly) < 3:
            return
        x, y = poly[:, 0], poly[:, 1]
        if abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) < min_area:
            return
        top, base = _heights(tags, default_height, level_height)
        out.append(Footprint(poly, top, base, name))

    if doc.get("type") == "FeatureCollection" or doc.get("type") == "Feature":
        feats = doc["features"] if doc.get("type") == "FeatureCollection" else [doc]
        for f in feats:
            g, props = (f.get("geometry") or {}), (f.get("properties") or {})
            name = str(props.get("name", ""))
            if g.get("type") == "Polygon":
                add(g["coordinates"][0], props, name)
            elif g.get("type") == "MultiPolygon":
                for part in g["coordinates"]:
                    add(part[0], props, name)
    elif "elements" in doc:  # Overpass `out geom;`
        for el in doc["elements"]:
            tags = el.get("tags") or {}
            if "building" not in tags and "building:part" not in tags:
                continue
            if el.get("type") == "way" and "geometry" in el:
                add([(p["lon"], p["lat"]) for p in el["geometry"]], tags, str(tags.get("name", "")))
            elif el.get("type") == "relation":
                for m in el.get("members", []):
                    if m.get("role") == "outer" and "geometry" in m:
                        add([(p["lon"], p["lat"]) for p in m["geometry"]], tags, str(tags.get("name", "")))
    else:
        raise MeshError("unrecognised map data: expected GeoJSON or Overpass JSON")
    return out


def buildings_to_meshes(
    footprints: List[Footprint], merge: bool = True, color: Tuple[float, float, float] = (0.78, 0.76, 0.72), segments_per_batch: int = 5000
) -> List[TriangleMesh]:
    """Extrude footprints into prisms (outlines that cannot be triangulated, e.g. self-intersecting ones, are skipped)."""
    meshes, skipped = [], 0
    for fp in footprints:
        try:
            meshes.append(extrude_polygon(fp.polygon, fp.height - fp.base, fp.base, name=fp.name or "building", color=color))
        except MeshError:
            skipped += 1
    if not meshes:
        return []
    if not merge:
        return meshes
    out = [
        TriangleMesh.merge(meshes[i : i + segments_per_batch], name=f"buildings_{i // segments_per_batch}")
        for i in range(0, len(meshes), segments_per_batch)
    ]
    return [m.with_color(color) for m in out]


def buildings_to_boxes(footprints: List[Footprint]) -> np.ndarray:
    """Axis-aligned bounding boxes ``(K, 6)`` of the footprints: cheap for physics, coarse for L-shaped or rotated buildings."""
    rows = [[*fp.polygon.min(0), fp.base, *fp.polygon.max(0), fp.height] for fp in footprints]
    return np.asarray(rows, float).reshape(-1, 6)
