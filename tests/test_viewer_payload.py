"""Viewer payload building: channels, layers, derived events, comparison runs and the decimation helpers."""

import base64
import json

import numpy as np
import pytest
from viewer_support import NR_CONFIG, sim_archive

from aevion.engine import Simulation
from aevion.viewer.payload import FORMAT, FrameSet, build_payload, decimate_indices, derive_events, json_safe
from aevion.viewer.protocol import decode_bundle


@pytest.fixture(scope="module")
def nr():
    return Simulation(NR_CONFIG).run()


@pytest.fixture(scope="module")
def plain():
    return sim_archive(world={"kind": "forest", "n_trees": 5, "half_extent": 8}, sensors=["imu", "lidar"], record_sensors=["lidar"])


def f32(payload, key):
    return np.frombuffer(base64.b64decode(payload[key]), dtype=np.float32)


# --------------------------------------------------------------------------------------- helpers
def test_decimate_indices_keeps_endpoints_and_bounds_length():
    idx = decimate_indices(1000, 50)
    assert idx[0] == 0 and idx[-1] == 999 and len(idx) <= 50 and np.all(np.diff(idx) > 0)
    assert np.array_equal(decimate_indices(10, 50), np.arange(10))
    assert np.array_equal(decimate_indices(2, 1), [0, 1])


def test_json_safe_turns_numpy_and_nan_into_valid_json():
    out = json_safe({"a": np.float32(1.5), "b": np.array([1, np.nan]), "c": (np.int64(3), float("inf")), "d": np.bool_(True)})
    assert out == {"a": 1.5, "b": [1.0, None], "c": [3, None], "d": True}
    json.dumps(out, allow_nan=False)


# ------------------------------------------------------------------------------------- payload
def test_payload_declares_format_shapes_and_time_axis(plain):
    p = build_payload(plain, max_frames=40)
    assert p["format"] == FORMAT
    m = p["meta"]
    assert m["T"] == len(f32(p, "t")) <= 40 and m["t0"] == 0.0 and m["duration"] == pytest.approx(f32(p, "t")[-1], rel=1e-5)
    for name, shape in m["shapes"].items():
        assert f32(p, name).size == int(np.prod(shape)), name
        assert shape[0] == m["T"]
    assert {"pos", "quat", "vel", "euler", "rotor", "ref", "err"} <= set(m["shapes"])
    json.dumps(p, allow_nan=False)


def test_payload_values_match_the_archive(plain):
    p = build_payload(plain, max_frames=10_000)
    np.testing.assert_allclose(f32(p, "pos").reshape(plain["pos"].shape), plain["pos"].astype(np.float32), atol=1e-6)
    err = np.linalg.norm(plain["pos"] - plain["ref_pos"], axis=-1)
    np.testing.assert_allclose(f32(p, "err").reshape(err.shape), err, atol=1e-5)


def test_extras_can_be_disabled_or_selected(nr):
    assert build_payload(nr, max_frames=20, extras=None)["extras"] == {}
    only = build_payload(nr, max_frames=20, extras=["link_sinr_db"])["extras"]
    assert list(only) == ["x:link_sinr_db"] and only["x:link_sinr_db"]["shape"][0] <= 20


def test_extras_budget_drops_big_channels_and_says_so(nr):
    p = build_payload(nr, max_frames=50, extras_budget_mb=0.0005)
    assert p["meta"]["dropped_channels"]
    assert len(p["extras"]) + len(p["meta"]["dropped_channels"]) == len(FrameSet(nr).extra)


def test_world_lidar_and_cameras(plain):
    p = build_payload(plain, max_frames=20)
    assert p["world"]["cylinders"] and len(p["world"]["bounds"]) == 6
    lid = p["lidar"]
    assert lid["cfg"]["channels"] and lid["frames"] > 0
    assert f32(lid, "ranges").size == lid["frames"] * plain.n_vehicles * lid["cfg"]["channels"] * lid["cfg"]["samples"]


def test_camera_frustum_data_comes_from_sensor_config():
    cfg = {
        "duration": 1.0,
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2]}},
        "sensors": [{"type": "camera", "fov_deg": 70, "width": 32, "height": 24, "far": 25.0}],
    }
    a = Simulation(cfg).run()
    cams = build_payload(a, max_frames=10)["camera"]
    assert cams and cams[0]["fov_deg"] == 70 and cams[0]["far"] == 25.0


# ----------------------------------------------------------------------------------- wireless
def test_nr_cellular_runs_export_cells_rem_and_beams(nr):
    """Regression: v1 read meta['wireless']['bs'], which nr_cellular does not have."""
    p = build_payload(nr, max_frames=60)
    w = p["wireless"]
    assert w["kind"] == "cells" and len(w["cells"]) == 6 and w["outage_db"] == -6.0
    assert "x:link_sinr_db" in p["extras"] and "x:link_serving" in p["extras"]
    rem = p["rem"]
    assert rem["slices"] and rem["slices"][0]["vmax"] > rem["slices"][0]["vmin"]
    s = rem["slices"][0]
    assert len(base64.b64decode(s["u8"])) == len(s["xs"]) * len(s["ys"])
    kinds = {lobe["kind"] for lobe in p["beams"]["lobes"]}
    assert kinds == {"cell", "uav"}
    lobe = p["beams"]["lobes"][0]
    assert f32(lobe, "gain_db").size == lobe["n_az"] * lobe["n_el"]


def test_rem_and_beams_can_be_switched_off(nr):
    p = build_payload(nr, max_frames=20, rem=False, beams=False)
    assert "rem" not in p and "beams" not in p and p["wireless"]


def test_runs_without_a_network_have_no_wireless_layer(plain):
    assert build_payload(plain, max_frames=10)["wireless"] is None


def test_radio_events_are_derived_from_the_link_channels(nr):
    fs = FrameSet(nr)
    events, dropped = derive_events(nr, fs)
    assert dropped == 0
    ho = [e for e in events if e["type"] == "handover"]
    s = nr["link_serving"]
    assert len(ho) == int(np.count_nonzero(s[1:] != s[:-1]))
    assert all(0 <= e["t"] <= fs.time[-1] and e["from"] != e["to"] for e in ho)
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)


def test_event_limit_thins_handovers_but_keeps_other_events(nr):
    class Fake:
        meta = {"events": [{"type": "collision", "t": float(nr.t[0]) + 1.0, "vehicle": 0}]}
        t = nr.t

        def __contains__(self, k):
            return k == "link_serving"

        def __getitem__(self, k):
            n = len(nr.t)
            return np.tile(np.arange(n)[:, None], (1, 1)).astype(float)  # serving cell changes at every step

    fs = FrameSet.__new__(FrameSet)
    fs.time = np.asarray(nr.t) - nr.t[0]
    events, dropped = derive_events(Fake(), fs, limit=20)
    assert len(events) == 20 and dropped > 0
    assert any(e["type"] == "collision" for e in events)


# ---------------------------------------------------------------------------------- frame sets
def test_frameset_window_and_bundle(nr):
    fs = FrameSet(nr)
    head, arr = decode_bundle(fs.bundle(["pos", "x:link_sinr_db", "nope"], start=2.0, stop=4.0, max_frames=30))
    assert head["frames"] == len(arr["t"]) <= 30
    assert set(arr) == {"t", "pos", "x:link_sinr_db"}
    t = arr["t"] + head["t0"]
    assert t[0] >= 2.0 - 0.05 and t[-1] <= 4.0 + 0.05
    assert arr["pos"].shape == (len(t), nr.n_vehicles, 3)


def test_frameset_window_clamps_out_of_range_requests(nr):
    fs = FrameSet(nr)
    idx = fs.window(-5.0, 1e9, 10)
    assert idx[0] == 0 and idx[-1] == fs.T - 1
    assert len(fs.window(1e9, 2e9, 10)) >= 1


def test_chunk_returns_full_rate_frames(nr):
    head, arr = decode_bundle(FrameSet(nr).chunk(10, 20))
    assert head["frames"] == 10 and arr["pos"].shape[0] == 10


# ---------------------------------------------------------------------------------- comparison
def test_comparison_runs_are_embedded_as_light_ghost_payloads(plain):
    other = sim_archive(wind={"type": "ConstantWind", "velocity": [2, 0, 0]})
    p = build_payload(plain, max_frames=30, runs=[("windy", other)], run_max_frames=25)
    r = p["runs"][0]
    assert r["label"] == "windy" and r["meta"]["T"] <= 25 and "pos" in r["channels"] and "t" in r["channels"]
    assert r["meta"]["shapes"]["pos"] == [r["meta"]["T"], other.n_vehicles, 3]


# ------------------------------------------------------------------------------ environment layers
ENV_CFG = {
    "duration": 2.0,
    "trajectory": {"type": "hover", "params": {"pos": [0, 0, 5.0]}},
    "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [12, 0, 4], "size": [4, 4, 8], "color": [0.8, 0.3, 0.3]}]}},
    "environment": {
        "origin": {"lat": 47.37, "lon": 8.54, "alt": 408.0},
        "utc": "2026-06-21T10:00:00Z",
        "weather": {"preset": "fog"},
        "actors": [{"type": "circular", "params": {"center": [0, 0, 6.0], "path_radius": 8.0, "omega": 0.5}}],
        "geofences": [{"name": "area", "kind": "inclusion", "zone": {"type": "cylinder", "center": [0, 0], "radius": 40.0}}],
    },
}


def test_payload_carries_meshes_actors_geofences_and_the_environment():
    arch = Simulation(ENV_CFG).run()
    p = build_payload(arch, max_frames=60)
    mesh = p["world"]["meshes"][0]
    assert (
        mesh["n_faces"] == 12
        and mesh["color"] == [0.8, 0.3, 0.3]
        and np.frombuffer(base64.b64decode(mesh["faces"]), dtype=np.uint32).max() < mesh["n_vertices"]
    )
    assert np.frombuffer(base64.b64decode(mesh["vertices"]), dtype=np.float32).size == mesh["n_vertices"] * 3
    a = p["actors"]
    assert a["n"] == 1 and a["frames"] == len(f32(p, "t")) and f32(a, "pos").size == a["frames"] * 3 and a["shapes"][0]["shape"] == "sphere"
    assert p["geofences"][0]["zone"]["type"] == "cylinder"
    assert p["meta"]["environment"]["utc"] == "2026-06-21T10:00:00Z" and p["meta"]["environment"]["weather"]["visibility_m"] == 200.0
    json.dumps(p, allow_nan=False)


def test_mesh_budget_leaves_out_what_would_stall_the_viewer(monkeypatch):
    import aevion.viewer.payload as P

    arch = Simulation(ENV_CFG).run()
    monkeypatch.setattr(P, "MAX_MESH_TRIANGLES", 5)
    w = build_payload(arch, max_frames=20)["world"]
    assert w["meshes"] == [] and w["meshes_skipped"] == 1


def test_runs_without_environment_features_have_no_extra_layers(plain):
    p = build_payload(plain, max_frames=20)
    assert p["actors"] is None and p["geofences"] is None and p["meta"]["environment"] is None and "meshes" not in p["world"]
