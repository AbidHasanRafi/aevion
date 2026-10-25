"""Terrain files, map buildings, declarative scenes and signed-distance fields."""

import json
import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments import geodesy as G
from aevion.environments.mesh import MeshError, TriangleMesh
from aevion.environments.osm import buildings_to_boxes, buildings_to_meshes, load_footprints
from aevion.environments.scene import build_scene
from aevion.environments.sdf import SignedDistanceField, box_sdf, cylinder_sdf, esdf_from_occupancy, euclidean_distance_transform, sphere_sdf
from aevion.environments.terrain import Terrain, TerrainError, fill_voids, load_terrain
from aevion.environments.world import World

B = NumpyBackend()
FRAME = G.LocalFrame(47.0 + 0.5, 8.0 + 0.5, 400.0)  # centre of tile N47E008


# ---------------------------------------------------------------------------------------------- terrain
def write_asc(path, data, cell=10.0, xll=1000.0, yll=2000.0, nodata=-9999, corner=True):
    nrows, ncols = data.shape
    head = f"ncols {ncols}\nnrows {nrows}\n{'xllcorner' if corner else 'xllcenter'} {xll}\n{'yllcorner' if corner else 'yllcenter'} {yll}\ncellsize {cell}\nNODATA_value {nodata}\n"
    rows = "\n".join(" ".join(str(v) for v in r) for r in data)  # first row is the NORTH edge
    path.write_text(head + rows + "\n")
    return path


def test_ascii_grid_orientation_origin_and_resampling(tmp_path):
    ny, nx = 6, 8
    iy, ix = np.mgrid[0:ny, 0:nx]
    south_up = 2.0 * ix + 5.0 * iy  # height increases to the east and the north
    p = write_asc(tmp_path / "d.asc", np.flipud(south_up), cell=10.0, xll=1000.0, yll=2000.0)
    t = load_terrain(p)
    np.testing.assert_allclose(t.heights, south_up)  # rows are flipped to south-up
    assert t.origin == (1005.0, 2005.0) and t.resolution == 10.0  # xllcorner -> centre of the first cell
    assert load_terrain(write_asc(tmp_path / "c.asc", np.flipud(south_up), corner=False)).origin == (1000.0, 2000.0)
    coarse = load_terrain(p, resolution=20.0)
    assert coarse.resolution == 20.0 and coarse.heights[1, 1] == pytest.approx(south_up[2, 2])
    rel = load_terrain(p, frame=G.LocalFrame(0, 0, 100.0))
    np.testing.assert_allclose(rel.heights, south_up - 100.0)


def test_ascii_voids_are_filled_and_bad_files_rejected(tmp_path):
    iy, ix = np.mgrid[0:8, 0:8]
    data = (3.0 * ix + 2.0 * iy).astype(float)
    holed = data.copy()
    holed[3:5, 3:5] = -9999
    t = load_terrain(write_asc(tmp_path / "h.asc", np.flipud(holed)))
    assert np.all(np.isfinite(t.heights)) and abs(t.heights[3, 3] - data[3, 3]) < 1.0  # plane recovered inside the hole
    (tmp_path / "bad.asc").write_text("ncols 2\nnrows 2\n")
    with pytest.raises(TerrainError):
        load_terrain(tmp_path / "bad.asc")
    (tmp_path / "short.asc").write_text("ncols 3\nnrows 3\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n1 2 3\n")
    with pytest.raises(TerrainError):
        load_terrain(tmp_path / "short.asc")
    with pytest.raises(TerrainError, match="no valid"):
        fill_voids(np.zeros((3, 3)), np.ones((3, 3), bool))
    with pytest.raises(TerrainError, match="unsupported"):
        load_terrain(tmp_path / "x.dem")


def test_srtm_tile_is_resampled_to_a_metric_grid_around_the_frame(tmp_path):
    n = 61  # a coarse tile: 1 degree / 60 = 1.85 km spacing
    lat = np.linspace(47.0, 48.0, n)[:, None] * np.ones((1, n))
    lon = np.linspace(8.0, 9.0, n)[None, :] * np.ones((n, 1))
    plane = 100.0 + 800.0 * (lat - 47.0) + 400.0 * (lon - 8.0)  # metres
    raw = np.flipud(plane).astype(">i2")  # SRTM rows run north -> south
    raw[10, 10] = -32768  # a void
    p = tmp_path / "N47E008.hgt"
    p.write_bytes(raw.tobytes())
    t = load_terrain(p, frame=FRAME, resolution=500.0, extent_m=20000.0)
    assert isinstance(t, Terrain) and t.resolution == 500.0 and np.all(np.isfinite(t.heights))
    ny, nx = t.heights.shape
    x0, y0 = t.origin
    centre = t.heights[round(-y0 / 500.0), round(-x0 / 500.0)]
    assert centre == pytest.approx(100.0 + 800.0 * 0.5 + 400.0 * 0.5 - 400.0, abs=1.5)  # plane value at the origin minus the frame altitude
    dh_dy = (t.heights[-1, nx // 2] - t.heights[0, nx // 2]) / ((ny - 1) * 500.0)  # northward slope: 800 m per degree of latitude
    assert dh_dy == pytest.approx(800.0 / (111_200.0), rel=0.01)
    with pytest.raises(TerrainError, match="frame"):
        load_terrain(p)
    (tmp_path / "wrong.hgt").write_bytes(raw.tobytes())
    with pytest.raises(TerrainError, match="N47E008"):
        load_terrain(tmp_path / "wrong.hgt", frame=FRAME)
    (tmp_path / "N47E009.hgt").write_bytes(b"\0" * 10)
    with pytest.raises(TerrainError, match="square"):
        load_terrain(tmp_path / "N47E009.hgt", frame=FRAME)
    far = G.LocalFrame(10.0, 10.0, 0.0)
    with pytest.raises(TerrainError, match="cover"):
        load_terrain(p, frame=far, extent_m=1000.0)


def test_numpy_terrain_and_applying_it_to_a_world(tmp_path):
    h = np.add.outer(np.arange(6.0), np.arange(9.0)) * 2.0
    np.savez(tmp_path / "t.npz", heights=h, origin=[-20.0, -10.0], resolution=5.0)
    t = load_terrain(tmp_path / "t.npz")
    w = t.apply(World())
    assert w.heightmap.shape == (6, 9) and w.hm_res == 5.0 and w.bounds[0] <= -100 and w.bounds[5] >= h.max() + 50
    assert float(w.ground_height(B, np.array([-20.0]), np.array([-10.0]))[0]) == pytest.approx(0.0)
    np.save(tmp_path / "n.npy", h)
    assert load_terrain(tmp_path / "n.npy", resolution=2.0).resolution == 2.0
    np.savez(tmp_path / "bad.npz", other=h)
    with pytest.raises(TerrainError):
        load_terrain(tmp_path / "bad.npz")
    np.save(tmp_path / "nan.npy", np.full((3, 3), np.nan))
    with pytest.raises(TerrainError):
        load_terrain(tmp_path / "nan.npy")
    assert t.decimated(2).resolution == 10.0 and t.extent == (-20.0, -10.0, 20.0, 15.0)


# ------------------------------------------------------------------------------------------------- OSM
def square_deg(frame, side_m, offset_m=(0.0, 0.0)):
    """Closed lon/lat ring of a square building ``side_m`` wide whose south-west corner is ``offset_m`` (east, north) from the frame."""
    corners = [(0, 0), (side_m, 0), (side_m, side_m), (0, side_m), (0, 0)]
    out = []
    for e, n in corners:
        lat, lon, _ = frame.to_lla([offset_m[0] + e, offset_m[1] + n, 0.0])
        out.append([lon, lat])
    return out


def feature(ring, **props):
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_geojson_footprints_heights_units_and_filters():
    f = FRAME
    doc = {
        "type": "FeatureCollection",
        "features": [
            feature(square_deg(f, 20, (50, 50)), height="12 m", name="A"),
            feature(square_deg(f, 20, (100, 50)), **{"building:levels": "4"}),
            feature(square_deg(f, 20, (150, 50)), height="30 ft"),
            feature(square_deg(f, 20, (200, 50))),
            feature(square_deg(f, 20, (250, 50)), height="20", min_height="8"),
            feature(square_deg(f, 1, (300, 50)), height="9"),  # 1 m^2: below min_area
            feature([[0, 0], [0, 0], [0, 0], [0, 0]], height="9"),  # degenerate
            {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [8.5, 47.5]}},
        ],
    }
    fps = load_footprints(doc, f, default_height=10.0)
    assert [round(p.height, 2) for p in fps] == [12.0, 12.0, 9.14, 10.0, 20.0] and fps[0].name == "A" and fps[4].base == 8.0
    assert abs(
        0.5 * np.sum(fps[0].polygon[:, 0] * np.roll(fps[0].polygon[:, 1], -1) - np.roll(fps[0].polygon[:, 0], -1) * fps[0].polygon[:, 1])
    ) == pytest.approx(400.0, rel=1e-3)
    assert fps[1].polygon.min(0)[0] == pytest.approx(100.0, abs=0.05)  # geodetic -> local metres


def test_overpass_json_ways_and_relations():
    f = FRAME
    ring = square_deg(f, 30, (10, 10))
    way = {"type": "way", "tags": {"building": "yes", "height": "15"}, "geometry": [{"lat": la, "lon": lo} for lo, la in ring]}
    rel = {
        "type": "relation",
        "tags": {"building": "yes", "building:levels": "2"},
        "members": [
            {"role": "outer", "geometry": [{"lat": la, "lon": lo} for lo, la in square_deg(f, 25, (100, 10))]},
            {"role": "inner", "geometry": [{"lat": la, "lon": lo} for lo, la in square_deg(f, 5, (110, 20))]},
        ],
    }
    road = {"type": "way", "tags": {"highway": "residential"}, "geometry": [{"lat": 47.5, "lon": 8.5}, {"lat": 47.51, "lon": 8.5}]}
    fps = load_footprints({"elements": [way, rel, road]}, f)
    assert [round(p.height, 1) for p in fps] == [15.0, 6.0]
    with pytest.raises(MeshError, match="unrecognised"):
        load_footprints({"hello": 1}, f)


def test_building_meshes_and_boxes(tmp_path):
    f = FRAME
    doc = {
        "type": "FeatureCollection",
        "features": [feature(square_deg(f, 20, (50, 50)), height="12"), feature(square_deg(f, 10, (100, 50)), height="30")],
    }
    fps = load_footprints(doc, f)
    meshes = buildings_to_meshes(fps)
    assert len(meshes) == 1 and meshes[0].is_watertight() and meshes[0].volume == pytest.approx(20 * 20 * 12 + 10 * 10 * 30, rel=2e-3)
    assert len(buildings_to_meshes(fps, merge=False)) == 2
    boxes = buildings_to_boxes(fps)
    assert boxes.shape == (2, 6) and boxes[0, 5] == 12.0 and boxes[1, 3] - boxes[1, 0] == pytest.approx(10.0, rel=1e-3)
    bow = {
        "type": "FeatureCollection",
        "features": [feature([[8.5, 47.5], [8.5004, 47.5004], [8.5004, 47.5], [8.5, 47.5004], [8.5, 47.5]], height="9")],
    }
    assert buildings_to_meshes(load_footprints(bow, f)) == []  # a self-intersecting outline is skipped, not fatal
    p = tmp_path / "b.json"
    p.write_text(json.dumps(doc))
    assert len(load_footprints(p, f)) == 2
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(MeshError, match="JSON"):
        load_footprints(tmp_path / "bad.json", f)


# ----------------------------------------------------------------------------------------------- scenes
def test_scene_builds_every_component_and_sizes_the_bounds(tmp_path):
    f = FRAME
    (tmp_path / "b.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature(square_deg(f, 20, (50, 0)), height="12")]}))
    np.savez(tmp_path / "t.npz", heights=np.zeros((5, 5)) + 3.0, origin=[-100.0, -100.0], resolution=50.0)
    TriangleMesh.box((0, 0, 0), (2, 2, 2)).transformed(translation=(0, 0, 1))
    w = build_scene(
        origin={"lat": f.lat, "lon": f.lon, "alt": 400.0},
        utc="2026-06-21T10:00:00Z",
        weather={"preset": "rain"},
        terrain={"file": "t.npz"},
        buildings={"file": "b.geojson"},
        meshes=[
            {"type": "box", "center": [0, 40, 5], "size": [4, 4, 10], "color": [1, 0, 0]},
            {"type": "cylinder", "center": [30, 30, 0], "radius": 2, "height": 8, "position": [0, 0, 0]},
        ],
        boxes=[{"center": [20, 0, 3], "size": [4, 4, 6]}, {"lo": [-30, -30, 0], "hi": [-20, -20, 4]}],
        spheres=[{"center": [0, -40, 10], "radius": 3}],
        cylinders=[{"xy": [-20, 20], "radius": 1.0, "z1": 12}],
        base_dir=tmp_path,
    )
    assert len(w.meshes) == 3 and len(w.boxes) == 2 and len(w.spheres) == 1 and len(w.cylinders) == 1 and w.heightmap is not None
    assert w.weather.rain_mm_h > 0 and w.origin.lat == f.lat and w.utc == "2026-06-21T10:00:00Z"
    assert np.all(w.bounds[:3] <= [-100, -100, 0]) and np.all(w.bounds[3:] >= [70, 100, 12])
    w2 = World.from_dict(json.loads(json.dumps(w.to_dict())))
    assert len(w2.meshes) == 3 and w2.origin.to_dict() == w.origin.to_dict() and w2.utc == w.utc and w2.weather.to_dict() == w.weather.to_dict()


def test_scene_file_access_is_confined_and_validated(tmp_path):
    inside = tmp_path / "ok"
    inside.mkdir()
    (inside / "t.npz").write_bytes(b"")
    with pytest.raises(PermissionError):
        build_scene(terrain={"file": str(tmp_path / "t.npz")}, allowed_root=inside)
    with pytest.raises(PermissionError):
        build_scene(meshes=[{"file": "../secret.obj"}], base_dir=inside, allowed_root=inside)
    with pytest.raises(FileNotFoundError):
        build_scene(meshes=[{"file": "nope.obj"}], base_dir=tmp_path)
    with pytest.raises(ValueError, match="origin"):
        build_scene(buildings={"file": "x"}, base_dir=tmp_path)
    with pytest.raises(ValueError):
        build_scene(utc="not a time")
    with pytest.raises(MeshError):
        build_scene(meshes=[{"type": "torus"}])


def test_scene_is_a_registered_world_type_and_runs_in_the_engine():
    from aevion.engine import Simulation

    cfg = {
        "duration": 1.0,
        "world": {
            "type": "scene",
            "params": {
                "weather": {"preset": "hot"},
                "meshes": [{"type": "box", "center": [10, 0, 2], "size": [2, 8, 4]}],
                "spheres": [{"center": [0, 10, 3], "radius": 1.0}],
            },
        },
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2]}},
    }
    sim = Simulation(cfg)
    arch = sim.run()
    assert arch.n_steps > 10 and arch.world.n_mesh_triangles == 12
    assert arch.meta["world"]["meshes"] and arch.meta["world"]["weather"]["temperature_c"] == 38.0
    from aevion.registry import registries

    assert "scene" in registries.worlds.names()


# -------------------------------------------------------------------------------------------------- SDF
def test_primitive_sdfs_are_exact():
    p = np.array([[3.0, 0, 0], [0, 0, 0], [1.0, 1.0, 0]])
    np.testing.assert_allclose(sphere_sdf(p, [0, 0, 0], 1.0), [2.0, -1.0, math.sqrt(2) - 1])
    np.testing.assert_allclose(box_sdf(np.array([[3.0, 0, 0], [0, 0, 0], [2.0, 2.0, 0]]), [-1, -1, -1], [1, 1, 1]), [2.0, -1.0, math.sqrt(2)])
    c = cylinder_sdf(np.array([[3.0, 0, 5], [0, 0, 5], [3.0, 0, 11]]), (0, 0), 0.0, 10.0, 1.0)
    np.testing.assert_allclose(c, [2.0, -1.0, math.hypot(2.0, 1.0)])


def test_sdf_from_world_matches_analytic_distances_and_ground():
    w = World()
    w.add_box((0, 0, 5), (4, 4, 10))
    w.add_sphere((20, 0, 8), 3.0)
    w.add_cylinder((-20, 0), 2.0, 0.0, 12.0)
    w.add_mesh(TriangleMesh.box((0, 25, 5), (4, 4, 10)))
    f = SignedDistanceField.from_world(w, bounds=[-30, -10, 0, 30, 35, 20], spacing=1.0)
    pts = np.array([[8.0, 0, 5], [20, 6, 8], [-20, 0, 18], [0, 33, 5], [0, 0, 5], [12, 12, 15]])
    ground = pts[:, 2]
    want = np.minimum.reduce(
        [
            box_sdf(pts, [-2, -2, 0], [2, 2, 10]),
            sphere_sdf(pts, [20, 0, 8], 3.0),
            cylinder_sdf(pts, (-20, 0), 0, 12, 2.0),
            box_sdf(pts, [-2, 23, 0], [2, 27, 10]),
            ground,
        ]
    )
    np.testing.assert_allclose(f.distance(pts), want, atol=1e-6)  # grid points: exact
    assert f.distance(np.array([[0.0, 0, 5.0]]))[0] == pytest.approx(-2.0)  # inside the building
    assert f.collides(np.array([[2.4, 0, 5.0]]), radius=0.5)[0] and not f.collides(np.array([[5.0, 0, 5.0]]), radius=0.5)[0]


def test_sdf_with_terrain_uses_vertical_clearance():
    w = World()
    w.set_heightmap(np.full((40, 40), 3.0), (-20.0, -20.0), 1.0)
    f = SignedDistanceField.from_world(w, bounds=[-10, -10, 0, 10, 10, 10], spacing=1.0)
    assert f.distance(np.array([[0.0, 0.0, 7.0]]))[0] == pytest.approx(4.0)
    free = SignedDistanceField.from_world(World(), bounds=[-10, -10, 4, 10, 10, 10], spacing=1.0, ground=False)
    assert np.all(np.isinf(free.values))  # no obstacles and no ground: nothing to be near


def test_sdf_interpolation_and_gradient_are_exact_for_linear_fields():
    fn = lambda p: 2.0 * p[..., 0] - 3.0 * p[..., 1] + 0.5 * p[..., 2] + 1.0
    f = SignedDistanceField.from_function(fn, [0, 0, 0, 10, 10, 10], 0.5)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 9.9, (200, 3))
    np.testing.assert_allclose(f.distance(p), fn(p), atol=1e-9)
    np.testing.assert_allclose(f.gradient(p), np.tile([2.0, -3.0, 0.5], (200, 1)), atol=1e-9)
    assert f.distance(np.array([[-5.0, 5.0, 5.0]]))[0] == pytest.approx(fn(np.array([0.0, 5.0, 5.0])))  # clamped outside the grid
    assert f.distance(p.reshape(20, 10, 3)).shape == (20, 10)


def test_sdf_gradient_of_a_sphere_points_away_with_unit_length():
    f = SignedDistanceField.from_function(lambda p: sphere_sdf(p, [5, 5, 5], 2.0), [0, 0, 0, 10, 10, 10], 0.25)
    p = np.array([[8.0, 5, 5], [5, 1.5, 5], [5, 5, 9.0]])
    g = f.gradient(p)
    np.testing.assert_allclose(np.linalg.norm(g, axis=1), 1.0, atol=0.03)
    np.testing.assert_allclose(g[0], [1, 0, 0], atol=0.06)  # one-sided cell gradient: error ~ spacing / radius
    np.testing.assert_allclose(g[1], [0, -1, 0], atol=0.06)


def test_sdf_validation():
    with pytest.raises(ValueError):
        SignedDistanceField(np.zeros((1, 5, 5)), (0, 0, 0), 1.0)
    with pytest.raises(ValueError):
        SignedDistanceField(np.zeros((3, 3, 3)), (0, 0, 0), 0.0)
    with pytest.raises(ValueError, match="too large"):
        SignedDistanceField.from_function(lambda p: p[..., 0], [0, 0, 0, 1e4, 1e4, 1e4], 1.0)


def test_euclidean_distance_transform_is_exact_isotropic_and_anisotropic():
    rng = np.random.default_rng(1)
    m = rng.random((12, 10, 9)) < 0.04
    m[0, 0, 0] = True
    pts = np.argwhere(m)
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in m.shape], indexing="ij"), -1).reshape(-1, 3)
    for sp in ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.5, 1.0, 2.0)):
        brute = np.min(np.linalg.norm((idx[:, None] - pts[None]) * np.array(sp), axis=-1), axis=1).reshape(m.shape)
        np.testing.assert_allclose(euclidean_distance_transform(m, sp), brute, atol=1e-9)
    assert np.all(np.isinf(euclidean_distance_transform(np.zeros((3, 3, 3), bool))))
    with pytest.raises(ValueError):
        euclidean_distance_transform(np.zeros((3, 3), bool))


def test_signed_esdf_from_occupancy():
    occ = np.zeros((20, 20, 20), bool)
    occ[8:12, 8:12, 8:12] = True  # a 4-cell cube
    e = esdf_from_occupancy(occ, spacing=0.5)
    assert e[4, 10, 10] == pytest.approx(0.5 * (8 - 4) - 0.25)  # four cells from the cube face, in metres, minus half a cell
    assert e[9, 9, 9] < e[8, 9, 9] < 0  # deeper inside is more negative
    assert np.all(np.isinf(esdf_from_occupancy(np.zeros((3, 3, 3), bool))))
    assert np.all(esdf_from_occupancy(np.ones((3, 3, 3), bool)) == -np.inf)
    sdf = SignedDistanceField.from_occupancy(occ, (0, 0, 0), 0.5)
    assert sdf.collides(np.array([[5.0, 5.0, 5.0]]))[0] and not sdf.collides(np.array([[1.0, 1.0, 1.0]]), 0.2)[0]
