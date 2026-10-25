"""The local viewer server (HTTP API, static files, security checks) and the live WebSocket stream."""

import http.client
import json
import socket
import threading
import time

import numpy as np
import pytest
from viewer_support import NR_CONFIG, sim_archive

from aevion.engine import Simulation
from aevion.viewer import ws as W
from aevion.viewer.live import LiveHub, LiveStreamHook, viewer_row
from aevion.viewer.protocol import decode_bundle
from aevion.viewer.server import ViewerServer


@pytest.fixture(scope="module")
def archive():
    return sim_archive(duration=3.0)


@pytest.fixture(scope="module")
def server(archive):
    with ViewerServer(archive) as srv:
        yield srv


def get(srv, path, headers=None, method="GET"):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, dict(r.getheaders()), body


# ------------------------------------------------------------------------------------------- http
def test_index_injects_boot_options_and_security_headers(server):
    status, h, body = get(server, "/")
    assert status == 200 and b'"mode": "server"' in body and b"/*AEVION_BOOT*/" not in body
    assert h["X-Content-Type-Options"] == "nosniff" and h["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"] and "default-src 'self'" in h["Content-Security-Policy"]


def test_meta_lists_channels_and_layers(server, archive):
    status, h, body = get(server, "/api/meta")
    m = json.loads(body)
    assert status == 200 and m["format"] == "aevion.view/2" and "pos" in m["channels"] and m["meta"]["T"] == archive.n_steps
    assert m["t1"] == pytest.approx(archive.t[-1] - archive.t[0])


def test_frames_endpoint_returns_a_bounded_window(server):
    status, h, body = get(server, "/api/frames?names=pos,soc&start=0.5&stop=1.5&max=20")
    head, arr = decode_bundle(body)
    assert status == 200 and h["Content-Type"] == "application/octet-stream"
    assert set(arr) == {"t", "pos", "soc"} and len(arr["t"]) <= 20


@pytest.mark.parametrize(
    "query", ["names=nope", "names=pos&start=abc", "names=pos&max=nan", "names=" + ",".join(["pos"] * 500), "names=pos&stop=inf"]
)
def test_bad_frame_requests_are_rejected_not_crashed(server, query):
    status, _, body = get(server, "/api/frames?" + query)
    assert status == 400 and "error" in json.loads(body)


def test_huge_max_is_capped(server):
    status, _, body = get(server, "/api/frames?names=pos&max=1e12")
    assert status == 200 and len(decode_bundle(body)[1]["t"]) <= 20000


@pytest.mark.parametrize(
    "path",
    [
        "/src/../../../pyproject.toml",
        "/src/%2e%2e/%2e%2e/pyproject.toml",
        "/..%5c..%5cpyproject.toml",
        "/vendor/three/../../../x",
        "/index.py",
        "/src/",
        "/css/%00.css",
    ],
)
def test_path_traversal_and_unknown_paths_are_not_served(server, path):
    status, _, body = get(server, path)
    assert status in (400, 404) and b"[project]" not in body


def test_static_files_are_typed_and_other_files_are_not_served(server):
    status, h, body = get(server, "/src/app.js")
    assert status == 200 and h["Content-Type"].startswith("text/javascript") and b"export async function boot" in body
    assert get(server, "/css/aevion.css")[1]["Content-Type"].startswith("text/css")
    assert get(server, "/vendor/three/LICENSE")[0] == 404  # not in the type allow-list
    assert get(server, "/vendor/three/MANIFEST.json")[0] == 200


def test_foreign_host_header_is_refused(server):
    assert get(server, "/", headers={"Host": "evil.example"})[0] == 403
    assert get(server, "/api/meta", headers={"Host": "127.0.0.1.evil.example"})[0] == 403


def test_server_is_read_only(server):
    for m in ("POST", "PUT", "DELETE"):
        assert get(server, "/api/meta", method=m)[0] == 405


def test_inverted_window_is_clamped_not_an_error(server):
    assert get(server, "/api/frames?names=pos&start=1&stop=0")[0] == 200


def test_head_requests_have_no_body(server):
    status, h, body = get(server, "/api/health", method="HEAD")
    assert status == 200 and body == b""


def test_binds_to_loopback_by_default(server):
    assert server.httpd.server_address[0] == "127.0.0.1" and server.url.startswith("http://localhost:")


def test_server_needs_data():
    with pytest.raises(ValueError):
        ViewerServer()


def test_archive_server_has_no_websocket(server):
    with pytest.raises(W.WebSocketClosed):
        W.connect("127.0.0.1", server.port, "/ws")


# ----------------------------------------------------------------------------------- websocket
def test_websocket_frame_codec_roundtrip_through_a_socket_pair():
    a, b = socket.socketpair()
    client, srv = W.WebSocket(a, client=True), W.WebSocket(b, client=False, max_size=1 << 20)
    for payload in ("hi", "x" * 200, "y" * 70000):
        client.send_text(payload)
        assert srv.recv_text() == payload
    srv.send_binary(b"\x00\x01\x02" * 50)
    assert client.recv() == (W.OP_BIN, b"\x00\x01\x02" * 50)
    client.close()
    with pytest.raises(W.WebSocketClosed):
        srv.recv()


def test_websocket_enforces_frame_size_limit():
    a, b = socket.socketpair()
    big = W.WebSocket(a, client=True)
    small = W.WebSocket(b, client=False, max_size=100)
    big.send_text("z" * 1000)
    with pytest.raises(W.WebSocketClosed):
        small.recv()


def test_accept_key_matches_the_rfc_example():
    assert W.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


# ---------------------------------------------------------------------------------------- live
def run_live(hub, config, **kw):
    return Simulation(config, hooks=[LiveStreamHook(hub, speed=None, **kw)]).run()


def collect(port, timeout=20):
    ws = W.connect("127.0.0.1", port, "/ws", origin=f"http://127.0.0.1:{port}")
    ws.sock.settimeout(timeout)
    msgs = []
    while True:
        op, data = ws.recv()
        msgs.append((op, data))
        if op == 1 and json.loads(data)["type"] == "end":
            ws.close()
            return msgs


def test_live_stream_sends_meta_then_frames_then_end():
    hub = LiveHub()
    with ViewerServer(hub=hub) as srv:
        out = []
        th = threading.Thread(target=lambda: out.extend(collect(srv.port)))
        th.start()
        while hub.n_clients == 0:
            time.sleep(0.02)
        arch = run_live(hub, NR_CONFIG)
        th.join(20)
    kinds = [(op, json.loads(d)["type"]) if op == 1 else (op, "frames") for op, d in out]
    assert kinds[0] == (1, "meta") and kinds[-1] == (1, "end") and (2, "frames") in kinds
    meta = json.loads(out[0][1])
    assert meta["format"] == "aevion.view/2" and meta["meta"]["n"] == 1 and meta["wireless"]["kind"] == "cells" and meta["widths"]["pos"] == 3
    frames = [decode_bundle(d) for op, d in out if op == 2]
    times = np.concatenate([a["t"] + h["t0"] for h, a in frames])
    assert np.all(np.diff(times) > 0) and times[0] == 0.0 and times[-1] == pytest.approx(arch.t[-1] - arch.t[0], abs=0.2)
    assert all(a["pos"].shape[1:] == (1, 3) for _, a in frames)
    assert "x:link_sinr_db" in frames[0][1]
    assert json.loads(out[-1][1])["frames"] == sum(len(a["t"]) for _, a in frames)


def test_late_joiner_is_caught_up():
    hub = LiveHub()
    with ViewerServer(hub=hub) as srv:
        run_live(hub, {**NR_CONFIG, "duration": 3.0})
        late = collect(srv.port)
    assert json.loads(late[0][1])["type"] == "meta" and json.loads(late[-1][1])["type"] == "end"
    assert sum(1 for op, _ in late if op == 2) >= 1


def test_live_radio_events_agree_with_the_archive_derivation():
    from aevion.viewer.payload import FrameSet, derive_events

    hub, got = LiveHub(), []
    publish = hub.publish_event
    hub.publish_event = lambda e: (got.append(e), publish(e))[1]
    arch = run_live(hub, {**NR_CONFIG, "duration": 8.0})
    want = [e for e in derive_events(arch, FrameSet(arch))[0] if e["type"] == "handover"]
    live = [e for e in got if e["type"] == "handover"]
    assert live and want
    assert live[0]["t"] == pytest.approx(want[0]["t"], abs=0.2)


def test_websocket_refuses_foreign_origins():
    hub = LiveHub()
    with ViewerServer(hub=hub) as srv:
        with pytest.raises(W.WebSocketClosed):
            W.connect("127.0.0.1", srv.port, "/ws", origin="http://evil.example")


def test_hub_retention_cap_keeps_control_messages():
    hub = LiveHub(retain_mb=0.0001)
    hub.publish_meta({"format": "x"})
    hub.publish_frames(b"\0" * 1000)
    hub.publish_event({"type": "handover", "t": 1})
    hub.finish()
    assert hub.dropped_retention and [t for t, _ in hub._retained] == [True, True, True]


def test_viewer_row_maps_channels_and_skips_non_vehicle_arrays():
    row = {
        "pos": np.zeros((2, 3)),
        "ref_pos": np.ones((2, 3)),
        "link_sinr_db": np.array([1.0, 2.0]),
        "t": 0.0,
        "scalar": np.float64(1.0),
        "sensors/x": np.zeros((2,)),
    }
    out = viewer_row(row, 2)
    assert set(out) == {"pos", "ref", "err", "x:link_sinr_db"} and out["err"].shape == (2,)


def test_stream_hook_validates_speed():
    with pytest.raises(ValueError):
        LiveStreamHook(LiveHub(), speed=0)
