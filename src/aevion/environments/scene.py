"""Build a :class:`~aevion.environments.world.World` from a declarative scene description (the ``scene`` world type).

.. code-block:: yaml

    world:
      type: scene
      params:
        origin: {lat: 47.3769, lon: 8.5417, alt: 408.0}   # where the world's (0, 0, 0) is on Earth
        utc: "2026-06-21T10:00:00Z"                        # simulation t = 0 (sun, GNSS geometry)
        weather: {preset: rain}
        terrain: {file: dem/N47E008.hgt, resolution: 30, extent_m: 4000}
        buildings: {file: zurich.geojson, default_height: 12, as: mesh}
        meshes:
          - {file: tower.glb, position: [100, 50, 0], rotation_deg: [0, 0, 90], scale: 1.0}
          - {type: box, center: [0, 0, 5], size: [10, 10, 10]}
        boxes: [{center: [20, 0, 3], size: [4, 4, 6]}]
        spheres: [{center: [0, 30, 10], radius: 3}]
        cylinders: [{xy: [-20, 0], radius: 1.0, z1: 12}]

Relative file names resolve against ``base_dir``; with ``allowed_root`` set (untrusted runs) every file must lie inside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .geodesy import LocalFrame
from .mesh import MeshError, TriangleMesh
from .osm import buildings_to_boxes, buildings_to_meshes, load_footprints
from .terrain import load_terrain
from .weather import Weather
from .world import World

__all__ = ["build_scene"]


def _resolve(file: Any, base_dir: Optional[Path], allowed_root: Optional[Path]) -> Path:
    if not isinstance(file, (str, Path)) or not str(file):
        raise ValueError("'file' must be a path")
    p = Path(file)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    p = p.resolve()
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        if p != root and root not in p.parents:
            raise PermissionError(f"file {str(file)!r} is outside the permitted directory")
    if not p.is_file():
        raise FileNotFoundError(f"scene file not found: {p}")
    return p


def _color(c: Any, default: Sequence[float]) -> Sequence[float]:
    return default if c is None else tuple(float(x) for x in c)


def _primitive_mesh(spec: Dict[str, Any]) -> TriangleMesh:
    kind = spec["type"]
    color = _color(spec.get("color"), (0.72, 0.72, 0.75))
    if kind == "box":
        return TriangleMesh.box(spec.get("center", (0, 0, 0)), spec.get("size", (1, 1, 1)), color=color)
    if kind == "cylinder":
        return TriangleMesh.cylinder(
            spec.get("center", (0, 0, 0)), float(spec.get("radius", 1.0)), float(spec.get("height", 1.0)), int(spec.get("segments", 24)), color=color
        )
    if kind == "sphere":
        return TriangleMesh.sphere(spec.get("center", (0, 0, 0)), float(spec.get("radius", 1.0)), color=color)
    if kind == "extrude":
        return TriangleMesh.extrude(spec["polygon"], float(spec["height"]), float(spec.get("base_z", 0.0)), color=color)
    raise MeshError(f"unknown mesh primitive {kind!r}; use box, cylinder, sphere or extrude")


def build_scene(
    seed: int = 0,
    origin: Optional[Dict[str, float]] = None,
    utc: Optional[str] = None,
    weather: Optional[Dict[str, Any]] = None,
    terrain: Optional[Dict[str, Any]] = None,
    buildings: Optional[Dict[str, Any]] = None,
    meshes: Sequence[Dict[str, Any]] = (),
    boxes: Sequence[Dict[str, Any]] = (),
    spheres: Sequence[Dict[str, Any]] = (),
    cylinders: Sequence[Dict[str, Any]] = (),
    bounds: Optional[Sequence[float]] = None,
    name: str = "scene",
    base_dir: Optional[Path] = None,
    allowed_root: Optional[Path] = None,
) -> World:
    """See the module docstring. ``seed`` is accepted for registry uniformity (scenes are deterministic)."""
    world = World(name=name)
    frame = LocalFrame.from_dict(origin) if origin else None
    world.origin, world.utc = frame, utc
    if utc is not None:
        from .solar import julian_day

        julian_day(utc)  # validate early: raises on a malformed timestamp
    if weather:
        Weather.from_dict(weather).apply(world)

    if terrain:
        spec = dict(terrain)
        path = _resolve(spec.pop("file"), base_dir, allowed_root)
        t = load_terrain(path, frame=frame, **{k: spec[k] for k in ("resolution", "extent_m", "relative", "geographic") if k in spec})
        t.apply(world)

    if buildings:
        spec = dict(buildings)
        if frame is None:
            raise ValueError("buildings need an 'origin' (map data is in latitude / longitude)")
        path = _resolve(spec.pop("file"), base_dir, allowed_root)
        fps = load_footprints(
            path, frame, float(spec.get("default_height", 10.0)), float(spec.get("level_height", 3.0)), float(spec.get("min_area", 4.0))
        )
        mode = spec.get("as", "mesh")
        if mode == "box":
            for b in buildings_to_boxes(fps):
                world.add_box_corners(b[:3], b[3:], color=spec.get("color"))
        elif mode == "mesh":
            for m in buildings_to_meshes(fps, merge=True, color=tuple(_color(spec.get("color"), (0.78, 0.76, 0.72)))):  # type: ignore[arg-type]
                world.add_mesh(m)
        else:
            raise ValueError("buildings.as must be 'mesh' or 'box'")

    for spec_in in meshes:
        spec = dict(spec_in)
        if "file" in spec:
            m = TriangleMesh.load(_resolve(spec["file"], base_dir, allowed_root), up=spec.get("up"))
        else:
            m = _primitive_mesh(spec)
        if any(k in spec for k in ("scale", "rotation_deg", "position")):
            m = m.transformed(spec.get("scale", 1.0), spec.get("rotation_deg", (0, 0, 0)), spec.get("position", (0, 0, 0)))
        if spec.get("color") is not None:
            m = m.with_color(_color(spec["color"], m.color))
        world.add_mesh(m)

    for b in boxes:
        world.add_box(b["center"], b["size"], b.get("color")) if "center" in b else world.add_box_corners(b["lo"], b["hi"], b.get("color"))
    for s in spheres:
        world.add_sphere(s["center"], float(s["radius"]), s.get("color"))
    for c in cylinders:
        world.add_cylinder(c["xy"], float(c["radius"]), float(c.get("z0", 0.0)), float(c.get("z1", 10.0)), c.get("color"))

    if bounds is not None:
        world.bounds = np.asarray(bounds, float)
    else:
        pts = [np.array(world.bounds[:3]), np.array(world.bounds[3:])] if world.heightmap is not None else []  # terrain already sized the bounds
        if len(world.boxes):
            pts += [world.boxes[:, :3].min(0), world.boxes[:, 3:].max(0)]
        if len(world.spheres):
            pts += [(world.spheres[:, :3] - world.spheres[:, 3:]).min(0), (world.spheres[:, :3] + world.spheres[:, 3:]).max(0)]
        if len(world.cylinders):
            cyl = world.cylinders
            pts += [np.array([*(cyl[:, :2] - cyl[:, 4:]).min(0), cyl[:, 2].min()]), np.array([*(cyl[:, :2] + cyl[:, 4:]).max(0), cyl[:, 3].max()])]
        for m in world.meshes:
            pts += [m.bounds[0], m.bounds[1]]
        if pts:  # grow the default bounds to cover the content with a margin
            allp = np.stack(pts)
            b = np.concatenate([allp.min(0) - [10, 10, 0], allp.max(0) + [10, 10, 30]])
            b[2] = min(0.0, b[2])
            world.bounds = b
    return world
