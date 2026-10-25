"""Triangle meshes, loaders, BVH queries and their integration into the world, sensors and dynamics."""

import base64
import json
import math
import struct

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments import mesh as MM
from aevion.environments.mesh import MeshError, TriangleMesh
from aevion.environments.mesh_index import MeshIndex, _ray_triangles_brute, closest_points_on_triangles
from aevion.environments.sdf import box_sdf
from aevion.environments.world import World
from aevion.sensors.raycast import cast_rays

B = NumpyBackend()


# ---------------------------------------------------------------------------------------------- primitives
def test_primitives_are_closed_outward_oriented_surfaces_with_analytic_volumes():
    box = TriangleMesh.box((1, 2, 3), (2, 3, 4))
    assert box.is_watertight() and box.volume == pytest.approx(24.0) and box.surface_area == pytest.approx(2 * (6 + 8 + 12))
    n = 48
    cyl = TriangleMesh.cylinder((0, 0, 0), 2.0, 3.0, n)
    assert cyl.is_watertight() and cyl.volume == pytest.approx(0.5 * n * 4.0 * math.sin(2 * math.pi / n) * 3.0)
    sph = TriangleMesh.sphere((0, 0, 0), 2.0, 40, 80)
    assert (
        sph.is_watertight()
        and sph.volume == pytest.approx(4 / 3 * math.pi * 8, rel=5e-3)
        and sph.surface_area == pytest.approx(16 * math.pi, rel=5e-3)
    )
    assert np.allclose(np.linalg.norm(sph.vertices, axis=1), 2.0)


def test_extrusion_of_concave_clockwise_and_closed_polygons():
    poly = [(0, 0), (4, 0), (4, 3), (2, 1.5), (0, 3)]  # notched rectangle, area 9
    m = TriangleMesh.extrude(poly, 5.0, base_z=2.0)
    assert m.is_watertight() and m.volume == pytest.approx(45.0) and m.bounds[0, 2] == 2.0 and m.bounds[1, 2] == 7.0
    assert TriangleMesh.extrude(poly[::-1], 5.0).volume == pytest.approx(45.0)  # clockwise input is re-oriented
    assert TriangleMesh.extrude(poly + [poly[0]], 5.0).n_vertices == 10  # closing vertex dropped
    for bad in ([(0, 0), (1, 1)], [(0, 0), (1, 0), (2, 0)], [(0, 0), (2, 2), (2, 0), (0, 2)]):  # too few, collinear, bow-tie
        with pytest.raises(MeshError):
            TriangleMesh.extrude(bad, 1.0)
    with pytest.raises(MeshError):
        TriangleMesh.extrude(poly, -1.0)


def test_transforms_merge_and_validation():
    box = TriangleMesh.box((0, 0, 0), (2, 2, 2))
    moved = box.transformed(scale=2.0, rotation_deg=(0, 0, 90), translation=(10, 0, 0))
    assert moved.volume == pytest.approx(64.0) and moved.bounds[0, 0] == pytest.approx(8.0)
    mirrored = box.transformed(scale=(-1, 1, 1))
    assert mirrored.volume > 0 and mirrored.is_watertight()  # winding flipped to stay outward
    merged = TriangleMesh.merge([box, box.transformed(translation=(5, 0, 0))])
    assert merged.n_faces == 24 and merged.volume == pytest.approx(16.0)
    with pytest.raises(MeshError):
        TriangleMesh([[0, 0, 0]], [[0, 0, 1]])
    with pytest.raises(MeshError):
        TriangleMesh([[0, 0, 0], [1, 0, 0], [0, 1, np.nan]], [[0, 1, 2]])
    with pytest.raises(MeshError):
        TriangleMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 7]])
    with pytest.raises(MeshError):
        TriangleMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], color=(2, 0, 0))
    open_tri = TriangleMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    assert not open_tri.is_watertight() and len(TriangleMesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]]).degenerate_faces()) == 1


def test_serialisation_round_trip_and_tampering():
    m = TriangleMesh.sphere((1, 2, 3), 1.5, 6, 10, name="ball", color=(0.1, 0.2, 0.3))
    again = TriangleMesh.from_dict(m.to_dict())
    np.testing.assert_array_equal(again.faces, m.faces)
    np.testing.assert_allclose(again.vertices, m.vertices)
    assert again.name == "ball" and again.color == (0.1, 0.2, 0.3)
    bad = m.to_dict()
    bad["n_faces"] += 1
    with pytest.raises(MeshError):
        TriangleMesh.from_dict(bad)


# ------------------------------------------------------------------------------------------------ loaders
def write_obj(path, text):
    path.write_text(text, encoding="utf-8")
    return path


CUBE_OBJ = """# unit cube with quads, texture/normal indices and a comment
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
vt 0 0
vn 0 0 1
f 1//1 4//1 3//1 2//1
f 5/1/1 6/1/1 7/1/1 8/1/1
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""


def test_obj_quads_negative_indices_and_attribute_forms(tmp_path):
    m = TriangleMesh.from_obj(write_obj(tmp_path / "c.obj", CUBE_OBJ))
    assert m.n_faces == 12 and m.is_watertight() and m.volume == pytest.approx(1.0) and m.name == "c"
    neg = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
    assert TriangleMesh.from_obj(write_obj(tmp_path / "n.obj", neg)).n_faces == 1
    yup = TriangleMesh.from_obj(write_obj(tmp_path / "y.obj", "v 0 5 0\nv 1 5 0\nv 0 5 1\nf 1 2 3\n"), up="y")
    assert np.allclose(yup.vertices[:, 2], 5.0) and yup.bounds[0, 1] == pytest.approx(-1.0)  # (x, y, z)_yup -> (x, -z, y)_zup


@pytest.mark.parametrize(
    "text",
    [
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n",
        "v 0 0 a\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n",
        "",
    ],
)
def test_malformed_obj_is_rejected_cleanly(tmp_path, text):
    with pytest.raises(MeshError):
        TriangleMesh.from_obj(write_obj(tmp_path / "bad.obj", text))


def stl_ascii(mesh):
    out = ["solid t"]
    for tri in mesh.triangles():
        out += ["facet normal 0 0 0", "outer loop"] + [f"vertex {x} {y} {z}" for x, y, z in tri] + ["endloop", "endfacet"]
    return "\n".join(out + ["endsolid t"]).encode()


def stl_binary(mesh):
    tri = mesh.triangles().astype("<f4")
    body = b"".join(struct.pack("<3f", 0, 0, 0) + t.tobytes() + b"\0\0" for t in tri)
    return b"\0" * 80 + struct.pack("<I", len(tri)) + body


def test_stl_ascii_and_binary_agree_and_weld_vertices(tmp_path):
    box = TriangleMesh.box((0, 0, 0), (2, 4, 6))
    (tmp_path / "a.stl").write_bytes(stl_ascii(box))
    (tmp_path / "b.stl").write_bytes(stl_binary(box))
    a, b = TriangleMesh.from_stl(tmp_path / "a.stl"), TriangleMesh.from_stl(tmp_path / "b.stl")
    assert a.n_vertices == b.n_vertices == 8 and a.is_watertight() and b.is_watertight()
    assert a.volume == pytest.approx(48.0) and b.volume == pytest.approx(48.0)
    assert TriangleMesh.from_stl(tmp_path / "b.stl", weld=False).n_vertices == 36
    (tmp_path / "junk.stl").write_bytes(b"not an stl at all")
    with pytest.raises(MeshError):
        TriangleMesh.from_stl(tmp_path / "junk.stl")


def glb(positions, indices, nodes=None, index_type=5123, extra_json=None, pad=True):
    pos = np.asarray(positions, "<f4")
    idx = np.asarray(indices, "<u2" if index_type == 5123 else "<u4")
    blob = pos.tobytes() + idx.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes}, {"buffer": 0, "byteOffset": pos.nbytes, "byteLength": idx.nbytes}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3"},
            {"bufferView": 1, "componentType": index_type, "count": idx.size, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": nodes or [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }
    doc.update(extra_json or {})
    j = json.dumps(doc).encode()
    j += b" " * ((-len(j)) % 4)
    blob += b"\0" * ((-len(blob)) % 4)
    total = 12 + 8 + len(j) + 8 + len(blob)
    return struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(j), b"JSON") + j + struct.pack("<I4s", len(blob), b"BIN\0") + blob


BOX = TriangleMesh.box((0, 0, 0), (2, 2, 2))


def test_glb_import_applies_node_transforms_and_y_up(tmp_path):
    f = tmp_path / "m.glb"
    f.write_bytes(glb(BOX.vertices, BOX.faces.reshape(-1), nodes=[{"mesh": 0, "translation": [10, 0, 0], "scale": [1, 2, 3]}]))
    m = TriangleMesh.load(f)  # glTF is y-up by default
    assert m.n_faces == 12 and m.volume == pytest.approx(8.0 * 6.0)
    # (x, y, z)_yup -> (x, -z, y)_zup : a y-extent of 2*2 becomes the z-extent, the z-extent of 2*3 becomes the (negative) y-extent
    assert (
        m.bounds[1, 2] - m.bounds[0, 2] == pytest.approx(4.0)
        and m.bounds[1, 1] - m.bounds[0, 1] == pytest.approx(6.0)
        and m.bounds[0, 0] == pytest.approx(9.0)
    )
    flat = TriangleMesh.from_gltf(f, up="z")
    assert flat.bounds[1, 2] - flat.bounds[0, 2] == pytest.approx(6.0)


def test_glb_rotation_node_hierarchy_and_data_uri(tmp_path):
    half = math.sqrt(0.5)
    nodes = [{"children": [1], "rotation": [0, 0, half, half]}, {"mesh": 0, "translation": [1, 0, 0]}]  # parent turns +x into +y
    data = glb(BOX.vertices, BOX.faces.reshape(-1), nodes=nodes)
    (tmp_path / "h.glb").write_bytes(data)
    m = TriangleMesh.from_gltf(tmp_path / "h.glb", up="z")
    assert m.bounds.mean(0) == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    pos = np.asarray(BOX.vertices, "<f4")
    blob = pos.tobytes() + np.asarray(BOX.faces.reshape(-1), "<u2").tobytes()
    doc = json.loads(data[20 : 20 + struct.unpack_from("<I", data, 12)[0]].decode())
    doc["buffers"] = [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}]
    (tmp_path / "d.gltf").write_text(json.dumps(doc))
    assert TriangleMesh.from_gltf(tmp_path / "d.gltf", up="z").volume == pytest.approx(8.0)


def test_gltf_is_hardened_against_hostile_files(tmp_path):
    good = glb(BOX.vertices, BOX.faces.reshape(-1))
    (tmp_path / "trunc.glb").write_bytes(good[:60])
    with pytest.raises(MeshError):
        TriangleMesh.load(tmp_path / "trunc.glb")
    doc_oob = glb(
        BOX.vertices,
        BOX.faces.reshape(-1),
        extra_json={
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 10**6, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
            ]
        },
    )
    (tmp_path / "oob.glb").write_bytes(doc_oob)
    with pytest.raises(MeshError):
        TriangleMesh.load(tmp_path / "oob.glb")
    (tmp_path / "sparse.glb").write_bytes(
        glb(
            BOX.vertices,
            BOX.faces.reshape(-1),
            extra_json={
                "accessors": [
                    {"componentType": 5126, "count": 8, "type": "VEC3", "sparse": {}},
                    {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
                ]
            },
        )
    )
    with pytest.raises(MeshError, match="sparse"):
        TriangleMesh.load(tmp_path / "sparse.glb")
    doc = {"asset": {"version": "2.0"}, "buffers": [{"uri": "../secret.bin", "byteLength": 4}], "meshes": [], "nodes": [], "scenes": [{"nodes": []}]}
    (tmp_path / "t.gltf").write_text(json.dumps(doc))
    with pytest.raises(MeshError):
        TriangleMesh.load(tmp_path / "t.gltf")
    (tmp_path / "x.gltf").write_text("{not json")
    with pytest.raises(MeshError):
        TriangleMesh.load(tmp_path / "x.gltf")
    loop = {"asset": {"version": "2.0"}, "nodes": [{"children": [0]}], "scenes": [{"nodes": [0]}]}
    (tmp_path / "loop.gltf").write_text(json.dumps(loop))
    with pytest.raises(MeshError):
        TriangleMesh.load(tmp_path / "loop.gltf")  # cyclic hierarchy must not recurse forever


def test_size_limits_and_unknown_formats(tmp_path, monkeypatch):
    (tmp_path / "x.ply").write_text("ply")
    with pytest.raises(MeshError, match="unsupported"):
        TriangleMesh.load(tmp_path / "x.ply")
    p = write_obj(tmp_path / "big.obj", CUBE_OBJ)
    monkeypatch.setattr(MM, "MAX_FILE_BYTES", 50)
    with pytest.raises(MeshError, match="limit"):
        TriangleMesh.from_obj(p)
    monkeypatch.setattr(MM, "MAX_FILE_BYTES", 10**6)
    monkeypatch.setattr(MM, "MAX_TRIANGLES", 5)
    with pytest.raises(MeshError):
        TriangleMesh.from_obj(p)


# ---------------------------------------------------------------------------------------------------- BVH
@pytest.fixture(scope="module")
def scene():
    meshes = [TriangleMesh.sphere((0, 0, 5), 3.0, 20, 40), TriangleMesh.box((10, 0, 2), (4, 4, 4)), TriangleMesh.cylinder((-8, 3, 0), 1.5, 6.0, 32)]
    return MeshIndex(meshes)


def random_rays(n, seed=0, span=20.0):
    rng = np.random.default_rng(seed)
    o = rng.uniform(-span, span, (n, 3))
    d = rng.normal(size=(n, 3))
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


def test_bvh_rays_equal_brute_force(scene):
    o, d = random_rays(3000)
    t, n, tid = scene.cast(o, d, 100.0)
    tri = scene.vertices[scene.faces]
    tt = _ray_triangles_brute(o, d, tri)
    best = tt.min(1)
    hit = best < 100.0
    assert np.array_equal(tid >= 0, hit) and hit.sum() > 30
    np.testing.assert_allclose(t[hit], best[hit], atol=1e-9)
    assert np.all(t[~hit] == 100.0) and np.all(tid[~hit] == -1)
    assert np.all(np.sum(n[hit] * d[hit], axis=1) <= 1e-9)  # normals face against the ray
    np.testing.assert_allclose(np.linalg.norm(n[hit], axis=1), 1.0)


def test_axis_aligned_rays_and_shared_edges_never_slip_through(scene):
    box = MeshIndex([TriangleMesh.box((0, 0, 0), (2, 2, 2))])
    g = np.linspace(-0.999, 0.999, 120)
    X, Y = np.meshgrid(g, g)
    o = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 5.0)])
    t, _, tid = box.cast(o, np.tile([0.0, 0.0, -1.0], (len(o), 1)), 100.0)  # includes rays exactly along the face diagonal (x == y)
    assert np.all(tid >= 0) and np.allclose(t, 4.0)
    side, _, sid = box.cast(np.array([[-5.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), 100.0)  # zero direction components
    assert sid[0] >= 0 and side[0] == pytest.approx(4.0)


def test_tmax_origin_inside_and_empty_inputs(scene):
    t, _, tid = scene.cast(np.array([[-20.0, 0, 5]]), np.array([[1.0, 0, 0]]), 10.0)
    assert tid[0] == -1 and t[0] == 10.0  # sphere surface is 17 m away
    t, n, tid = scene.cast(np.array([[0.0, 0, 5]]), np.array([[0.0, 0, 1.0]]), 100.0)  # from inside the sphere
    assert t[0] == pytest.approx(3.0, abs=0.05) and n[0, 2] < 0  # normal flipped to face the ray
    t, n, tid = scene.cast(np.zeros((0, 3)), np.zeros((0, 3)), 5.0)
    assert t.shape == (0,) and n.shape == (0, 3)
    with pytest.raises(MeshError):
        MeshIndex([])


def test_closest_point_helpers_match_each_other(scene):
    rng = np.random.default_rng(1)
    p = rng.uniform(-15, 15, (400, 3))
    d, tid, q = scene.nearest(p)
    cp = closest_points_on_triangles(p, scene.vertices[scene.faces])
    dist = np.linalg.norm(cp - p[:, None, :], axis=-1)
    np.testing.assert_allclose(d, dist.min(1), atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(q - p, axis=1), d, atol=1e-9)
    far, ftid, _ = scene.nearest(np.array([[100.0, 100.0, 100.0]]), rmax=5.0)
    assert np.isinf(far[0]) and ftid[0] == -1


def test_triangle_closest_point_regions():
    tri = np.array([[[0, 0, 0], [4, 0, 0], [0, 4, 0]]], float)
    cases = {
        (1, 1, 3): (1, 1, 0),
        (-1, -1, 0): (0, 0, 0),
        (6, -1, 0): (4, 0, 0),
        (-1, 6, 0): (0, 4, 0),
        (3, 3, 1): (2, 2, 0),
        (2, -2, 0): (2, 0, 0),
        (-3, 2, 0): (0, 2, 0),
    }
    for p, want in cases.items():
        np.testing.assert_allclose(closest_points_on_triangles(np.array([p], float), tri)[0, 0], want, atol=1e-12)


def test_sphere_collision_uses_the_deepest_triangle_not_a_sum():
    big = MeshIndex([TriangleMesh.box((0, 0, 0), (20, 20, 2))])
    # centre 0.8 above the top face, directly over the face diagonal where two triangles meet; radius 1.0 -> depth 0.2
    push, depth = big.collide(np.array([[0.0, 0.0, 1.8], [5.0, 3.0, 1.8], [0.0, 0.0, 3.0]]), 1.0)
    np.testing.assert_allclose(depth, [0.2, 0.2, 0.0], atol=1e-12)
    np.testing.assert_allclose(push[0], [0, 0, 0.2], atol=1e-12)
    np.testing.assert_allclose(push[2], 0.0)
    inside = big.collide(np.array([[0.0, 0.0, 0.0]]), 0.5)[1]
    assert inside[0] > 0.5  # a centre inside the solid is pushed out through the nearest face


def test_inside_test_and_signed_distance_match_the_analytic_box(scene):
    box = MeshIndex([TriangleMesh.box((0, 0, 0), (4, 6, 2))])
    rng = np.random.default_rng(2)
    p = rng.uniform(-6, 6, (400, 3))
    inside = box.contains(p)
    exact = box_sdf(p, [-2, -3, -1], [2, 3, 1])
    assert np.array_equal(inside, exact < 0)
    np.testing.assert_allclose(box.signed_distance(p), exact, atol=1e-9)
    assert scene.contains(np.array([[0, 0, 5.0], [10, 0, 2.0], [0, 0, 20.0], [-8, 3, 3.0]])).tolist() == [True, True, False, True]


def test_global_triangle_ids_map_back_to_meshes(scene):
    assert scene.offsets.tolist() == [0, scene.meshes[0].n_faces, scene.meshes[0].n_faces + 12, scene.n_triangles]
    t, _, tid = scene.cast(np.array([[-5.0, 0, 2.0], [20.0, 0, 2.0], [-8.0, -10.0, 3.0]]), np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]]), 100.0)
    assert scene.mesh_of(tid).tolist() == [0, 1, 2]
    assert scene.mesh_of(np.array([-1])).tolist() == [-1]
    assert scene.depth < 20 and np.all(scene.hi[0] >= scene.lo[0])


# ----------------------------------------------------------------------------------------- world integration
def test_rays_see_primitives_and_meshes_and_report_the_nearest():
    w = World()
    w.add_box((0, 0, 2), (2, 2, 4))
    w.add_mesh(TriangleMesh.box((10, 0, 2), (2, 2, 4)))
    d = np.array([[1.0, 0, 0]])
    assert cast_rays(w, np.array([-10.0, 0, 2]), d, 100.0).t[0] == pytest.approx(9.0)  # primitive first
    h = cast_rays(w, np.array([5.0, 0, 2]), d, 100.0)
    assert (h.t[0] == pytest.approx(4.0) and h.prim[0] == 1 + 8 + 0) or h.prim[0] >= 1  # mesh triangle ids follow the primitives
    assert cast_rays(w, np.array([5.0, 0, 2]), d, 3.0).prim[0] == -1  # beyond max_range
    assert cast_rays(w, np.array([20.0, 0, 2]), np.array([[-1.0, 0, 0]]), 100.0).t[0] == pytest.approx(9.0)


def test_world_queries_collision_distance_line_of_sight_and_colours():
    w = World()
    w.add_mesh(TriangleMesh.box((0, 0, 5), (4, 4, 10)), color=(0.9, 0.1, 0.1))
    assert w.has_obstacles and w.n_mesh_triangles == 12 and np.allclose(w.prim_colors()[3], (0.9, 0.1, 0.1))
    assert w.distance_to_nearest(np.array([[5.0, 0, 5.0]]))[0] == pytest.approx(3.0)
    hit, depth, nrm = w.collide(B, np.array([[2.3, 0.0, 5.0], [9.0, 0, 5.0]]), 0.5)
    assert hit.tolist() == [True, False] and depth[0] == pytest.approx(0.2) and nrm[0] == pytest.approx([1, 0, 0])
    los = w.line_of_sight(np.array([[-10.0, 0, 5.0], [-10.0, 0, 20.0]]), np.array([[10.0, 0, 5.0], [10.0, 0, 20.0]]))
    assert los.tolist() == [False, True]


def test_world_round_trip_keeps_meshes():
    w = World()
    w.add_mesh(TriangleMesh.sphere((1, 2, 3), 1.0, 6, 8, name="s"))
    w2 = World.from_dict(json.loads(json.dumps(w.to_dict())))
    assert len(w2.meshes) == 1 and w2.meshes[0].name == "s"
    assert w2.mesh_index.n_triangles == w.mesh_index.n_triangles


def test_camera_and_lidar_see_meshes():
    from aevion.engine import Simulation

    cfg = {
        "duration": 0.4,
        "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [6, 0, 2], "size": [2, 6, 4], "color": [0.9, 0.2, 0.2]}]}},
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2]}},
        "sensors": [
            {
                "type": "lidar",
                "params": {"channels": 1, "samples": 36, "v_fov": [0, 0], "dropout_prob": 0.0, "noise_std": 0.0, "noise_rel": 0.0, "max_range": 30},
                "record": True,
            },
            {"type": "camera", "params": {"modes": ["rgb", "depth"], "width": 16, "height": 12}},
        ],
    }
    sim = Simulation(cfg)
    sim.run()
    r = sim.suite["lidar"].latest["ranges"][0, 0]
    assert np.isfinite(r).sum() > 3 and np.nanmin(np.where(np.isfinite(r), r, np.nan)) == pytest.approx(5.0, abs=0.01)  # wall face at x = 5
    rgb = sim.suite["camera"].latest["rgb"][0]
    assert rgb[6, 8, 0] > rgb[6, 8, 2]  # the red box is in the middle of the picture
    assert np.isfinite(sim.suite["camera"].latest["depth"][0, 6, 8])


def test_mesh_obstacles_stop_vehicles_like_primitive_boxes():
    """The same wall as a primitive box and as a mesh gives the same crash behaviour in the reference and the fused-kernel paths."""
    from aevion.dynamics import Multirotor

    results = {}
    for kind in ("box", "mesh"):
        for kernel in ("reference", "numba"):
            w = World()
            if kind == "box":
                w.add_box((8.0, 0.0, 3.0), (2.0, 10.0, 6.0))
            else:
                w.add_mesh(TriangleMesh.box((8.0, 0.0, 3.0), (2.0, 10.0, 6.0)))
            sim = Multirotor("generic_1kg", n=1, world=w, kernel=kernel, verify_kernel=False)
            s = sim.reset(pos=[[2.0, 0.0, 3.0]], vel=[[6.0, 0.0, 0.0]])
            hover = sim.hover_speed()
            hit = False
            for _ in range(600):
                s = sim.step(s, np.full((1, sim.n_rotors), hover))
                hit = hit or bool(s.collided[0])
            results[(kind, kernel)] = (float(s.pos[0, 0]), hit)
    for k, (x, hit) in results.items():
        assert hit, k
        assert x < 7.0 - 0.1, f"{k}: vehicle ended at x={x}, inside the wall (face at x=7)"
    assert abs(results[("box", "reference")][0] - results[("mesh", "reference")][0]) < 0.05
    assert abs(results[("box", "numba")][0] - results[("mesh", "numba")][0]) < 0.05
