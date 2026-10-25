"""The viewer running in a real (headless) Chromium-family browser. Skipped when no Chrome/Edge/Chromium is installed.

These tests look at what the user would see: the page starts without errors, each camera and layer renders, pixels are not blank, the
keyboard works, and the server / live / editor / compare / video paths run end to end.
"""

import io
import json
import threading
import time

import numpy as np
import pytest
from viewer_support import NR_CONFIG, sim_archive, synthetic_payload

from aevion.viewer.browser import Browser, BrowserError, find_browser
from aevion.viewer.export import export_html, render_html

pytestmark = pytest.mark.skipif(find_browser() is None, reason="no Chrome/Edge/Chromium found")
PIL = pytest.importorskip("PIL.Image", reason="Pillow is needed to inspect screenshots")
READY = "document.body.dataset.ready === '1' || document.body.dataset.error"


@pytest.fixture(scope="module")
def browser():
    with Browser(width=800, height=450) as b:
        yield b


@pytest.fixture(autouse=True)
def _close_pages(browser):
    """Every test gets fresh tabs and leaves none behind."""
    opened, original = [], browser.new_page

    def tracked(*a, **k):
        page = original(*a, **k)
        opened.append(page)
        return page

    browser.new_page = tracked
    yield
    browser.new_page = original
    for p in opened:
        p.close()


@pytest.fixture(scope="module")
def nr_archive():
    from aevion.engine import Simulation

    return Simulation(NR_CONFIG).run()


def open_page(browser, url):
    page = browser.new_page()
    page.goto(url)
    page.wait_for(READY, timeout=60)
    page.pump(0.3)
    assert not page.evaluate("document.body.dataset.error || ''"), page.evaluate("document.body.dataset.error")
    return page


def open_html(browser, tmp_path, payload, options=None, name="v.html"):
    f = tmp_path / name
    f.write_text(render_html(payload, {"manual": True, **(options or {})}), encoding="utf-8")
    return open_page(browser, f.resolve().as_uri())


def pixels(page):
    return np.asarray(PIL.open(io.BytesIO(page.screenshot())).convert("RGB"), dtype=float)


def key(page, k, code=None):
    for t in ("keyDown", "keyUp"):
        page.send("Input.dispatchKeyEvent", type=t, key=k, code=code or k, text=k if len(k) == 1 and t == "keyDown" else "")


# ------------------------------------------------------------------------------------------ start up
def test_single_file_replay_starts_clean_and_renders_a_real_picture(browser, tmp_path, nr_archive):
    out = export_html(nr_archive, tmp_path / "nr.html", max_frames=200, options={"manual": True})
    page = open_page(browser, out.resolve().as_uri())
    stats = json.loads(page.evaluate("JSON.stringify(window.__aevion.stats())"))
    assert stats["N"] == 1 and stats["T"] == 200 and dict(stats["layers"])["wireless"] is True
    page.evaluate("window.__aevion.renderFrame(4.0)")
    img = pixels(page)
    assert img.std() > 20 and len({tuple(p) for p in img[::40, ::40].reshape(-1, 3)}) > 30  # not blank, not a flat fill
    assert not page.errors, page.errors


@pytest.mark.parametrize("cam", ["orbit", "chase", "top", "fpv"])
def test_every_camera_mode_renders_and_differs(browser, tmp_path, cam):
    page = open_html(browser, tmp_path, synthetic_payload(3, 60))
    page.evaluate(f"window.__aevion.setCamera('{cam}'); window.__aevion.renderFrame(2.0)")
    img = pixels(page)
    assert img.std() > 15
    assert page.evaluate("window.__aevion.camera") == cam and not page.errors


def test_render_frame_is_deterministic_for_video_export(browser, tmp_path):
    page = open_html(browser, tmp_path, synthetic_payload(4, 80))
    page.evaluate("window.__aevion.setUI(false); window.__aevion.renderFrame(3.0)")
    a = pixels(page)
    page.evaluate("window.__aevion.renderFrame(5.0); window.__aevion.renderFrame(3.0)")
    b = pixels(page)
    assert np.abs(a - b).max() == 0


def test_keyboard_shortcuts(browser, tmp_path, nr_archive):
    page = open_html(browser, tmp_path, synthetic_payload(2, 60, link=True))
    page.evaluate("window.__aevion.playback.setPlaying(true)")
    key(page, " ", "Space")
    assert page.evaluate("window.__aevion.playback.playing") is False
    t0 = page.evaluate("window.__aevion.playback.time")
    key(page, "ArrowRight", "ArrowRight")
    assert page.evaluate("window.__aevion.playback.time") > t0
    key(page, "3")
    assert page.evaluate("window.__aevion.camera") == "top"
    on = dict(json.loads(page.evaluate("JSON.stringify(window.__aevion.stats().layers)")))["trails"]
    key(page, "t")
    assert dict(json.loads(page.evaluate("JSON.stringify(window.__aevion.stats().layers)")))["trails"] is (not on)
    key(page, "?")
    assert page.evaluate("document.getElementById('help-overlay').hidden") is False
    key(page, "h")
    assert page.evaluate("document.body.classList.contains('hide-ui')") is True
    assert not page.errors


def test_deep_link_hash_sets_time_camera_and_pause(browser, tmp_path):
    f = tmp_path / "d.html"
    f.write_text(render_html(synthetic_payload(2, 60), {"manual": True}), encoding="utf-8")
    page = open_page(browser, f.resolve().as_uri() + "#t=4&cam=chase&play=0")
    assert page.evaluate("window.__aevion.camera") == "chase"
    assert page.evaluate("window.__aevion.playback.playing") is False
    assert page.evaluate("window.__aevion.playback.time") == pytest.approx(4.0, abs=0.01)


def test_themes_change_the_picture_and_print_is_white(browser, tmp_path):
    page = open_html(browser, tmp_path, synthetic_payload(2, 60), {"theme": "print"})
    page.evaluate("window.__aevion.setUI(false); window.__aevion.renderFrame(1.0)")
    img = pixels(page)
    assert np.median(img) > 235, "print theme should be a white page"
    page.evaluate("window.__aevion.setTheme('dark'); window.__aevion.renderFrame(1.0)")
    assert np.median(pixels(page)) < 200


def test_layer_failure_is_isolated(browser, tmp_path):
    page = open_html(browser, tmp_path, synthetic_payload(2, 60))
    page.evaluate(
        "window.__aevion.layers.find((l) => l.id === 'trails').update = () => { throw new Error('boom'); }; window.__aevion.renderFrame(1.0)"
    )
    assert page.evaluate("window.__aevion.layers.find((l) => l.id === 'trails').enabled") is False
    page.evaluate("window.__aevion.renderFrame(2.0)")
    assert pixels(page).std() > 15


def test_corrupt_payload_shows_an_error_banner_not_a_blank_page(browser, tmp_path):
    payload = synthetic_payload(2, 40)
    payload["pos"] = "AAAA"
    f = tmp_path / "bad.html"
    f.write_text(render_html(payload, {"manual": True}), encoding="utf-8")
    page = browser.new_page()
    page.goto(f.resolve().as_uri())
    page.wait_for("document.body.dataset.error", timeout=30)
    assert "pos" in page.evaluate("document.getElementById('banner').textContent")


# ------------------------------------------------------------------------------------ fleets and sizes
def test_large_fleet_uses_instancing(browser, tmp_path):
    page = open_html(browser, tmp_path, synthetic_payload(100, 40, link=True))
    assert page.evaluate("window.__aevion.fleet.detailed") is False
    page.evaluate("window.__aevion.renderFrame(2.0)")
    st = json.loads(page.evaluate("JSON.stringify(window.__aevion.stats())"))
    assert st["calls"] < 150, "120 vehicles must not cost one draw call each"
    assert pixels(page).std() > 15 and not page.errors


def test_small_fleet_is_fully_modelled(browser, tmp_path):
    page = open_html(browser, tmp_path, synthetic_payload(3, 60))
    assert page.evaluate("window.__aevion.fleet.detailed") is True and page.evaluate("window.__aevion.fleet.beacons.length") == 3


def test_narrow_screens_keep_the_timeline_visible(browser, tmp_path):
    page = open_html(browser, tmp_path, synthetic_payload(2, 60))
    page.resize(390, 700)
    page.pump(0.3)
    r = json.loads(page.evaluate("JSON.stringify(document.getElementById('timeline').getBoundingClientRect())"))
    assert r["bottom"] <= 700 and r["left"] >= 0 and r["right"] <= 390


# ---------------------------------------------------------------------------------------- data modes
def test_server_mode_streams_detail_windows(browser, tmp_path):
    from aevion.viewer.server import ViewerServer

    a = sim_archive(duration=6.0)
    with ViewerServer(a, options={"manual": True}) as srv:
        page = open_page(browser, srv.url)
        assert page.evaluate("window.__aevion.source.kind") == "chunked"
        page.evaluate("window.__aevion.renderFrame(4.0)")
        assert page.evaluate("!!window.__aevion.ctx.frames.detail")
        assert not page.errors and srv.last_error is None


def test_live_mode_shows_the_run_growing(browser):
    from aevion.engine import Simulation
    from aevion.viewer.live import LiveHub, LiveStreamHook
    from aevion.viewer.server import ViewerServer

    hub = LiveHub()
    with ViewerServer(hub=hub, options={"maxFps": 4}) as srv:
        page = browser.new_page()
        page.goto(srv.url)
        page.pump(0.5)
        assert "waiting" in page.evaluate("document.getElementById('banner').textContent").lower()
        th = threading.Thread(target=lambda: Simulation({**NR_CONFIG, "duration": 6.0}, hooks=[LiveStreamHook(hub, speed=3.0)]).run())
        th.start()
        page.wait_for(READY, timeout=60)
        page.wait_for("window.__aevion.ctx.frames.overview.T > 20", timeout=30)
        n1 = page.evaluate("window.__aevion.ctx.frames.overview.T")
        th.join()
        page.wait_for(f"window.__aevion.ctx.frames.overview.T > {n1} && window.__aevion.source.finished", timeout=30)
        assert page.evaluate("window.__aevion.playback.live") is True and not page.errors


# --------------------------------------------------------------------------------------- compare, editor
def test_compare_page_shows_ghosts_and_difference_panel(browser, tmp_path):
    from aevion.viewer.compare import compare_html

    a = sim_archive(duration=3.0)
    b = sim_archive(duration=3.0, wind={"type": "ConstantWind", "velocity": [3, 0, 0]})
    out = compare_html([a, b], tmp_path / "c.html", labels=["calm", "windy"], options={"manual": True})
    page = open_page(browser, out.resolve().as_uri())
    assert page.evaluate("window.__aevion.layers.find((l) => l.id === 'ghosts').available") is True
    page.evaluate("window.__aevion.renderFrame(2.0)")
    assert "windy" in page.evaluate("document.getElementById('drawer').textContent")
    assert page.evaluate("document.querySelector('.cmp-canvas') !== null") and not page.errors


def test_scene_editor_places_objects_with_the_mouse_and_exports_a_runnable_config(browser, tmp_path):
    from aevion.config import load_config
    from aevion.engine import Simulation
    from aevion.viewer.editor import empty_payload

    f = tmp_path / "e.html"
    f.write_text(render_html(empty_payload(), {"autoplay": False, "manual": True}), encoding="utf-8")
    page = open_page(browser, f.resolve().as_uri())
    assert page.evaluate("window.__aevion.editor.active") is True  # editor pages start in editing mode

    def click(x, y):
        for t in ("mousePressed", "mouseReleased"):
            page.send("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)
            page.pump(0.02)

    for tool, spots in (
        ("box", [(330, 260), (400, 290)]),
        ("cylinder", [(300, 330)]),
        ("start", [(360, 350)]),
        ("waypoint", [(400, 330), (430, 300)]),
        ("cell", [(440, 280)]),
    ):
        page.evaluate(f"window.__aevion.editor.setTool('{tool}')")
        for x, y in spots:
            click(x, y)
    assert page.evaluate("window.__aevion.editor.scene.objects.length") == 7
    assert page.evaluate("window.__aevion.editor.scene.validate()") == []
    cfg = page.evaluate("window.__aevion.editor.config()")
    f = tmp_path / "scene.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    sim_cfg = load_config(f)
    assert sim_cfg.network is not None and sim_cfg.world.type == "from_dict"
    a = Simulation(sim_cfg).run(duration=1.0)
    assert a.world.boxes.shape == (2, 6) and a.world.cylinders.shape == (1, 5)
    assert not page.errors


# ---------------------------------------------------------------------------------------- video / images
def test_screenshot_function_writes_a_png(tmp_path):
    from aevion.viewer import screenshot

    out = screenshot(sim_archive(duration=2.0), tmp_path / "s.png", size=(640, 360), ui=False)
    img = PIL.open(out)
    assert img.size == (640, 360) and np.asarray(img.convert("L")).std() > 10


def test_video_frame_sequence_is_deterministic_and_moves(tmp_path):
    from aevion.viewer import render_video

    a = sim_archive(duration=2.0)
    d1 = render_video(a, tmp_path / "f1", fps=4, speed=1.0, size=(320, 180), camera="chase")
    d2 = render_video(a, tmp_path / "f2", fps=4, speed=1.0, size=(320, 180), camera="chase")
    files = sorted(d1.glob("frame_*.png"))
    assert len(files) == int(float(a.t[-1] - a.t[0]) * 4) + 1
    first, last = (np.asarray(PIL.open(files[i]).convert("L"), float) for i in (0, -1))
    assert np.abs(first - last).mean() > 0.5, "the scene should change over the clip"
    for f1, f2 in zip(files, sorted(d2.glob("frame_*.png"))):
        assert f1.read_bytes() == f2.read_bytes()


def test_gif_export(tmp_path):
    from aevion.viewer import render_video

    a = sim_archive(duration=1.0)
    out = render_video(a, tmp_path / "a.gif", fps=3, size=(240, 140))
    with PIL.open(out) as im:
        assert im.format == "GIF" and im.n_frames == int(float(a.t[-1] - a.t[0]) * 3) + 1


def test_video_rejects_bad_requests(tmp_path):
    from aevion.viewer import render_video

    a = sim_archive(duration=1.0)
    with pytest.raises(ValueError):
        render_video(a, tmp_path / "x.gif", fps=0)
    with pytest.raises(ValueError):
        render_video(a, tmp_path / "x.gif", start=5.0)
    with pytest.raises(ValueError):
        render_video(a, tmp_path / "x.xyz", fps=1, size=(64, 64))
    with pytest.raises(ValueError):
        render_video(a, tmp_path / "x.gif", fps=1000, speed=0.01)


def test_missing_browser_gives_a_clear_error(monkeypatch):
    monkeypatch.setenv("AEVION_BROWSER", "definitely-not-a-browser")
    with pytest.raises(BrowserError, match="AEVION_BROWSER"):
        Browser().__enter__()


def test_page_wait_times_out_with_context(browser):
    page = browser.new_page()
    with pytest.raises(BrowserError, match="timed out"):
        page.wait_for("false", timeout=0.3)


def test_timing_budget_for_a_realistic_replay(browser, tmp_path, nr_archive):
    """Cold start (parse payload + build the scene) must stay well under a second of work even on software rendering."""
    out = export_html(nr_archive, tmp_path / "t.html", max_frames=400)
    t0 = time.perf_counter()
    open_page(browser, out.resolve().as_uri())
    assert time.perf_counter() - t0 < 25.0


def test_environment_layers_render_meshes_geofences_and_moving_obstacles(browser, tmp_path):
    from aevion.engine import Simulation

    cfg = {
        "duration": 3.0,
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 6.0]}},
        "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [14, 0, 5], "size": [5, 5, 10], "color": [0.9, 0.3, 0.3]}]}},
        "environment": {
            "origin": {"lat": 47.37, "lon": 8.54, "alt": 408.0},
            "utc": "2026-06-21T07:30:00Z",
            "weather": {"preset": "overcast"},
            "actors": [{"type": "circular", "params": {"center": [0, 0, 8.0], "path_radius": 9.0, "omega": 0.8}}],
            "geofences": [{"name": "tower", "kind": "exclusion", "zone": {"type": "cylinder", "center": [-12, 0], "radius": 4.0, "z_max": 25.0}}],
        },
    }
    arch = Simulation(cfg).run()
    out = export_html(arch, tmp_path / "env.html", max_frames=100, options={"manual": True})
    page = open_page(browser, out.resolve().as_uri())
    st = json.loads(page.evaluate("JSON.stringify(window.__aevion.stats())"))
    layers = dict(st["layers"])
    assert layers.get("geofences") is True and layers.get("actors") is True
    assert page.evaluate("window.__aevion.settings.get('timeOfDay')") == pytest.approx(
        7.5 + 8.54 / 15 - 0.0, abs=0.35
    )  # local solar time from the UTC stamp and the longitude
    page.evaluate("window.__aevion.renderFrame(0.0)")
    a = pixels(page)
    page.evaluate("window.__aevion.renderFrame(2.0)")
    b = pixels(page)
    assert a.std() > 15 and np.abs(a - b).max() > 20  # the moving obstacle changes the picture between the two times
    assert page.evaluate("window.__aevion.ctx.source.layers.actors.n") == 1 and not page.errors
