"""Vision datasets - PNG / .flo files, COCO and YOLO labels, boxes from segmentation and optical-flow ground truth."""

import json
import math
import struct
import zlib

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.datasets.vision import (
    CocoWriter,
    VisionRecorder,
    boxes_from_segmentation,
    camera_pose,
    depth_to_points,
    flow_to_rgb,
    object_table,
    optical_flow,
    project_points,
    read_flo,
    read_png,
    write_flo,
    write_png,
    write_yolo_labels,
    write_yolo_yaml,
)
from aevion.engine import Simulation
from aevion.engine.hooks import HookError
from aevion.environments.world import World
from aevion.sensors import Camera


def bound_camera(world, **kw):
    sim = type("S", (), {})()
    sim.B, sim.n, sim.world, sim.dt = NumpyBackend(), 1, world, 0.01
    sim.hover_speed = lambda: 400.0
    kw.setdefault("modes", ("depth", "seg"))
    kw.setdefault("depth_noise", (0.0, 0.0))
    kw.setdefault("mount", (0.0, 0.0, 0.0))
    cam = Camera(**kw)
    cam.bind(sim, 1)
    return cam


def state(pos, quat=(1.0, 0.0, 0.0, 0.0)):
    st = type("St", (), {})()
    st.pos, st.vel, st.omega, st.t = np.array([pos], float), np.zeros((1, 3)), np.zeros((1, 3)), 0.0
    st.quat = np.array([quat], float)
    return st


def yaw_quat(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


# ------------------------------------------------------------------------------------------------ file formats
@pytest.mark.parametrize("shape,dtype", [((7, 9), np.uint8), ((7, 9, 3), np.uint8), ((7, 9, 4), np.uint8), ((7, 9), np.uint16)])
def test_png_round_trips_and_matplotlib_reads_it_the_same(tmp_path, shape, dtype):
    rng = np.random.default_rng(0)
    img = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
    p = write_png(tmp_path / "a.png", img)
    assert np.array_equal(read_png(p), img)
    plt_img = pytest.importorskip("matplotlib.image").imread(str(p))
    scale = float(np.iinfo(dtype).max)
    assert np.allclose(np.asarray(plt_img), img / scale, atol=1.5 / scale)


def _filtered_png(img, ftype):
    """Encode a grey 8-bit image with a given PNG scan-line filter (to test the decoder against files other writers produce)."""
    h, w = img.shape
    rows = []
    prev = np.zeros(w, np.int32)
    for i in range(h):
        cur = img[i].astype(np.int32)
        if ftype == 0:
            f = cur
        elif ftype == 1:
            f = cur - np.concatenate([[0], cur[:-1]])
        elif ftype == 2:
            f = cur - prev
        elif ftype == 3:
            left = np.concatenate([[0], cur[:-1]])
            f = cur - ((left + prev) >> 1)
        else:
            left = np.concatenate([[0], cur[:-1]])
            ul = np.concatenate([[0], prev[:-1]])
            pa, pb, pc = np.abs(prev - ul), np.abs(left - ul), np.abs(left + prev - 2 * ul)
            pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, ul))
            f = cur - pred
        rows.append(bytes([ftype]) + (f & 255).astype(np.uint8).tobytes())
        prev = cur

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("ftype", [0, 1, 2, 3, 4])
def test_the_decoder_handles_every_scanline_filter(tmp_path, ftype):
    img = np.random.default_rng(ftype).integers(0, 256, (12, 15), dtype=np.uint8)
    (tmp_path / "f.png").write_bytes(_filtered_png(img, ftype))
    assert np.array_equal(read_png(tmp_path / "f.png"), img)


def test_png_errors_are_reported(tmp_path):
    with pytest.raises(ValueError, match="uint8"):
        write_png(tmp_path / "x.png", np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError, match="3 or 4"):
        write_png(tmp_path / "x.png", np.zeros((3, 3, 2), np.uint8))
    p = write_png(tmp_path / "ok.png", np.zeros((4, 4), np.uint8))
    raw = bytearray(p.read_bytes())
    raw[40] ^= 0xFF
    (tmp_path / "bad.png").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="corrupt"):
        read_png(tmp_path / "bad.png")
    (tmp_path / "n.png").write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="not a PNG"):
        read_png(tmp_path / "n.png")


def test_flo_round_trip_and_colour_wheel(tmp_path):
    f = np.random.default_rng(1).normal(size=(6, 8, 2)).astype(np.float32)
    p = write_flo(tmp_path / "x.flo", f)
    assert p.read_bytes()[:4] == b"PIEH" and np.array_equal(read_flo(p), f)
    with pytest.raises(ValueError):
        write_flo(tmp_path / "y.flo", np.zeros((3, 3)))
    (tmp_path / "z.flo").write_bytes(b"nope")
    with pytest.raises(ValueError):
        read_flo(tmp_path / "z.flo")
    rgb = flow_to_rgb(np.zeros((4, 4, 2)))
    assert rgb.dtype == np.uint8 and np.all(rgb == 255)  # no motion: white
    a, b = flow_to_rgb(np.tile([[[3.0, 0.0]]], (2, 2, 1)), 3.0), flow_to_rgb(np.tile([[[-3.0, 0.0]]], (2, 2, 1)), 3.0)
    assert not np.array_equal(a, b)  # opposite directions get different hues


# ------------------------------------------------------------------------------------------------ labels
def test_coco_writer_validates_and_serialises(tmp_path):
    w = CocoWriter(["box", "sphere"])
    i = w.add_image("images/000000.png", 64, 48)
    a = w.add_annotation(i, "sphere", 10, 5, 30, 25, area=300.0)
    assert (i, a) == (1, 1) and w.annotations[0]["category_id"] == 2 and w.annotations[0]["bbox"] == [10.0, 5.0, 20.0, 20.0]
    p = w.save(tmp_path / "ann.json")
    d = json.loads(p.read_text())
    assert set(d) == {"info", "images", "annotations", "categories"} and d["categories"][1] == {"id": 2, "name": "sphere"}
    with pytest.raises(ValueError, match="unknown category"):
        w.add_annotation(i, "tree", 0, 0, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        w.add_annotation(i, "box", 5, 5, 5, 9)
    w.add_annotation(i, "box", 60, 40, 70, 49)
    with pytest.raises(ValueError, match="outside"):
        w.save(tmp_path / "bad.json")
    orphan = CocoWriter(["box"])
    orphan.add_annotation(7, "box", 0, 0, 1, 1)
    with pytest.raises(ValueError, match="missing image"):
        orphan.save(tmp_path / "o.json")


def test_yolo_labels_are_normalised_boxes(tmp_path):
    p = write_yolo_labels(tmp_path / "l.txt", [(0, 10, 4, 30, 20), (2, 0, 0, 64, 48)], 64, 48)
    rows = [list(map(float, ln.split())) for ln in p.read_text().splitlines()]
    assert rows[0][0] == 0 and rows[0][1:] == pytest.approx([20 / 64, 12 / 48, 20 / 64, 16 / 48], abs=1e-6)
    assert rows[1][1:] == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert write_yolo_labels(tmp_path / "e.txt", [], 64, 48).read_text() == ""
    y = write_yolo_yaml(tmp_path / "data.yaml", ["box", "sphere"]).read_text()
    assert "nc: 2" in y and "0: box" in y and "1: sphere" in y


# ------------------------------------------------------------------------------------------------ geometry
def box_world():
    w = World()
    w.add_box_corners([10.0, -1.0, 0.5], [11.0, 1.0, 2.5], (0.8, 0.2, 0.2))  # a 1 x 2 x 2 box, its face 10 m ahead of the camera
    return w


def test_segmentation_boxes_match_the_analytic_projection_of_a_fronto_parallel_box():
    cam = bound_camera(box_world(), width=128, height=96, fov_deg=90.0)
    out = cam.measure(state((0.0, 0.0, 1.5)))
    seg = out["seg"][0]
    (b,) = boxes_from_segmentation(seg, min_pixels=1)
    f = cam.lens.f
    # face at x = 10: y in [-1, 1] -> u in cx -/+ f/10, z in [0.5, 2.5] -> v: camera at z = 1.5, so (z - 1.5) in [-1, 1]
    assert b["x0"] == pytest.approx(cam.lens.cx - f * 1.0 / 10.0, abs=1.01) and b["x1"] == pytest.approx(cam.lens.cx + f * 1.0 / 10.0, abs=1.01)
    assert b["y0"] == pytest.approx(cam.lens.cy - f * 1.0 / 10.0, abs=1.01) and b["y1"] == pytest.approx(cam.lens.cy + f * 1.0 / 10.0, abs=1.01)
    assert b["pixels"] == pytest.approx((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), abs=2)  # a filled rectangle
    owner, objs = object_table(cam.sim.world)
    assert len(objs) == 1 and objs[0]["category"] == "box" and boxes_from_segmentation(seg, owner)[0]["object"] == 0
    assert boxes_from_segmentation(seg, min_pixels=10**6) == []


def test_projection_of_a_known_point_and_pose():
    cam = bound_camera(box_world(), width=64, height=48)
    o, R = camera_pose(cam, [1.0, 2.0, 3.0], yaw_quat(math.pi / 2))
    assert np.allclose(o, [1.0, 2.0, 3.0]) and np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)  # facing +y after a 90 degree yaw
    p_world = np.array([1.0, 12.0, 3.0])  # ten metres straight ahead
    uv, inside = project_points((p_world - o) @ R, cam.lens)
    assert inside and np.allclose(uv, [cam.lens.cx, cam.lens.cy], atol=1e-9)
    behind, ins = project_points(np.array([-5.0, 0.0, 0.0]), cam.lens)
    assert not ins
    tilted = bound_camera(box_world(), pitch_deg=30.0)
    _, Rt = camera_pose(tilted, [0, 0, 0], (1, 0, 0, 0))
    assert Rt @ [1, 0, 0] == pytest.approx([math.cos(math.radians(30)), 0, -math.sin(math.radians(30))])  # positive pitch looks down


def plane_depth(cam, z):
    return np.full((cam.lens.H, cam.lens.W), float(z))


def test_flow_of_a_fronto_parallel_plane_matches_the_closed_forms():
    cam = bound_camera(box_world(), width=64, height=48, fov_deg=90.0)
    lens, Z = cam.lens, 8.0
    depth = plane_depth(cam, Z)
    u, v = lens.pixel_grid()
    p0 = camera_pose(cam, [0, 0, 0], (1, 0, 0, 0))
    # moving forward by d: the image expands about the principal point by Z / (Z - d)
    d = 1.0
    r = optical_flow(depth, lens, p0, camera_pose(cam, [d, 0, 0], (1, 0, 0, 0)))
    s = Z / (Z - d)
    ok = r["valid"]
    assert ok.mean() > 0.4
    fu, fv = r["flow"][..., 0].reshape(-1), r["flow"][..., 1].reshape(-1)
    m = ok.reshape(-1)
    assert np.allclose(fu[m], (u - lens.cx)[m] * (s - 1), atol=1e-9) and np.allclose(fv[m], (v - lens.cy)[m] * (s - 1), atol=1e-9)
    # moving left by t shifts everything right by f t / Z
    t = 0.4
    r2 = optical_flow(depth, lens, p0, camera_pose(cam, [0, t, 0], (1, 0, 0, 0)))
    assert np.allclose(r2["flow"][r2["valid"]][:, 0], lens.f * t / Z) and np.allclose(r2["flow"][r2["valid"]][:, 1], 0.0, atol=1e-9)
    # no motion: no flow, everything valid
    r0 = optical_flow(depth, lens, p0, p0)
    assert r0["valid"].all() and np.abs(r0["flow"]).max() < 1e-9


def test_rotation_flow_is_independent_of_depth_and_equals_the_homography():
    cam = bound_camera(box_world(), width=64, height=48, fov_deg=80.0)
    lens = cam.lens
    rng = np.random.default_rng(2)
    depth = rng.uniform(3, 30, (lens.H, lens.W))  # arbitrary depths: a pure rotation cannot depend on them
    ang = math.radians(4.0)
    p0 = camera_pose(cam, [0, 0, 0], (1, 0, 0, 0))
    p1 = camera_pose(cam, [0, 0, 0], yaw_quat(ang))
    r = optical_flow(depth, lens, p0, p1)
    r_flat = optical_flow(np.full_like(depth, 5.0), lens, p0, p1)
    both = r["valid"] & r_flat["valid"]
    assert np.allclose(r["flow"][both], r_flat["flow"][both], atol=1e-9)
    # independent check in the OpenCV convention: x_2 ~ K R^T K^-1 x_1
    C = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.0]])  # camera frame (forward, left, up) -> (right, down, forward)
    Rrel = p0[1].T @ p1[1]  # frame-2 axes expressed in frame 1
    Rcv = C @ Rrel @ C.T
    K = lens.intrinsics
    H = K @ Rcv.T @ np.linalg.inv(K)
    u, v = lens.pixel_grid()
    x = H @ np.stack([u, v, np.ones_like(u)])
    exp = (x[:2] / x[2]).T - np.stack([u, v], 1)
    got = r["flow"].reshape(-1, 2)
    m = r["valid"].reshape(-1)
    assert np.allclose(got[m], exp[m], atol=1e-9)


def test_flow_from_rendered_frames_predicts_the_same_surface_in_the_next_frame():
    w = World()
    w.add_box_corners([20.0, -40.0, 0.0], [21.0, 40.0, 40.0], (0.5, 0.5, 0.5))  # far wall
    w.add_box_corners([7.0, -1.5, 0.0], [8.0, -0.2, 3.0], (0.8, 0.2, 0.2))  # near pillar left of the axis
    w.add_box_corners([11.0, 0.5, 0.0], [12.0, 3.0, 4.0], (0.2, 0.2, 0.8))  # mid pillar to the right
    cam = bound_camera(w, width=96, height=72, fov_deg=90.0)
    a, b = state((0.0, 0.0, 1.5)), state((0.8, 0.5, 1.5), yaw_quat(0.05))
    oa, ob = cam.measure(a), cam.measure(b)
    pa, pb = camera_pose(cam, a.pos[0], a.quat[0]), camera_pose(cam, b.pos[0], b.quat[0])
    r = optical_flow(oa["depth"][0], cam.lens, pa, pb, ob["depth"][0])
    ok = r["valid"] & ~r["occluded"] & (oa["seg"][0] >= 0)
    u, v = cam.lens.pixel_grid()
    uv = np.stack([u, v], -1).reshape(cam.lens.H, cam.lens.W, 2) + r["flow"]
    iu, iv = np.clip(np.floor(uv[..., 0]).astype(int), 0, cam.lens.W - 1), np.clip(np.floor(uv[..., 1]).astype(int), 0, cam.lens.H - 1)
    same = (ob["seg"][0][iv, iu] == oa["seg"][0])[ok]
    assert ok.sum() > 1500 and same.mean() > 0.97  # the flow lands every pixel on the surface it came from (edges aside)
    assert r["occluded"].sum() > 0  # the sideways step hides part of the wall behind the pillars
    behind = r["occluded"] & (oa["seg"][0] == 0)
    assert not (r["occluded"] & ~r["valid"]).any() and behind.sum() >= 0


def test_unprojecting_depth_gives_the_points_the_camera_saw():
    cam = bound_camera(box_world(), width=64, height=48)
    out = cam.measure(state((0.0, 0.0, 1.5)))
    pts, ok = depth_to_points(out["depth"][0], cam.lens)
    hit = ok & (out["seg"][0] == 0)
    assert hit.any() and np.allclose(pts[hit][:, 0], 10.0, atol=1e-6)  # the box face is the plane x = 10 in the camera frame
    fish = bound_camera(box_world(), lens="equidistant", fov_deg=170.0, width=64, height=48)
    f_out = fish.measure(state((0.0, 0.0, 1.5)))
    fp, fok = depth_to_points(f_out["depth"][0], fish.lens)
    assert np.allclose(
        np.linalg.norm(fp[fok & np.isfinite(f_out["depth"][0])], axis=-1), f_out["depth"][0][fok & np.isfinite(f_out["depth"][0])]
    )  # range, not z-depth


# ------------------------------------------------------------------------------------------------ the recorder in a run
def test_the_recorder_writes_a_labelled_dataset_during_a_flight(tmp_path):
    cfg = {
        "duration": 3.0,
        "seed": 1,
        "world": {
            "type": "scene",
            "params": {
                "boxes": [{"center": [14, 0, 2], "size": [2, 3, 4]}, {"center": [20, -3, 3], "size": [2, 2, 6]}],
                "spheres": [{"center": [10, 2.5, 2.0], "radius": 1.0}],
                "bounds": [-5, -15, 0, 40, 15, 20],
            },
        },
        "trajectory": {"type": "polyline", "params": {"points": [[0, 0, 2], [6, 0, 2]], "speed": 2.0}},
        "sensors": [{"type": "camera", "params": {"rate": 10.0, "width": 64, "height": 48, "modes": ["depth", "rgb", "seg"], "far": 40.0}}],
    }
    rec = VisionRecorder(tmp_path / "ds", camera="camera", every=1, max_frames=100)
    Simulation(cfg, hooks=[rec]).run()
    n = rec.frames
    assert 25 <= n <= 31
    ds = tmp_path / "ds"
    assert (
        len(list((ds / "images").glob("*.png"))) == n
        and len(list((ds / "depth16").glob("*.png"))) == n
        and len(list((ds / "labels").glob("*.txt"))) == n
    )
    assert len(list((ds / "flow").glob("*.flo"))) == n - 1  # flow to the next frame: none for the last
    coco = json.loads((ds / "annotations.json").read_text())
    assert len(coco["images"]) == n and len(coco["annotations"]) >= n and {c["name"] for c in coco["categories"]} == {"box", "sphere"}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        assert x >= 0 and y >= 0 and x + w <= 64 and y + h <= 48 and a["area"] > 0
    first = [ln.split() for ln in (ds / "labels" / "000000.txt").read_text().splitlines()]
    assert first and all(0.0 <= float(v) <= 1.0 for row in first for v in row[1:])
    assert "nc: 2" in (ds / "data.yaml").read_text()
    img = read_png(ds / "images" / "000000.png")
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8
    depth16 = read_png(ds / "depth16" / "000000.png")
    assert depth16.dtype == np.uint16 and depth16.max() > 5000  # millimetres: something is several metres away
    flo = read_flo(ds / "flow" / "000005.flo")
    assert flo.shape == (48, 64, 2) and np.isfinite(flo).mean() > 0.3  # NaN marks pixels with no depth (sky) or leaving the image
    seg = read_png(ds / "seg" / "000000.png")
    assert seg.max() >= 1  # object ids are stored +1 (0 = background)


def test_the_recorder_checks_that_the_camera_can_provide_what_it_needs(tmp_path):
    base = {"duration": 0.5, "sensors": [{"type": "camera", "params": {"modes": ["depth"]}}]}
    with pytest.raises(HookError, match="modes"):
        Simulation(base, hooks=[VisionRecorder(tmp_path / "a")]).run()
    with pytest.raises(HookError, match="no sensor named"):
        Simulation({"duration": 0.5}, hooks=[VisionRecorder(tmp_path / "b")]).run()
