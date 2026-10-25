"""Lens models, rolling shutter, motion blur, exposure, sun and weather, thermal and event cameras."""

import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments.actors import LinearActor
from aevion.environments.geodesy import LocalFrame
from aevion.environments.weather import Weather
from aevion.environments.world import World
from aevion.sensors import Camera, EventCamera
from aevion.sensors.camera_models import FisheyeLens, PinholeLens, make_lens

B = NumpyBackend()


class Sim:
    def __init__(self, world, n):
        self.B, self.n, self.world = B, n, world


def st(pos=(0, 0, 2.0), vel=(0, 0, 0), omega=(0, 0, 0), t=0.0, n=1):
    s = type("S", (), {})()
    s.pos, s.vel, s.omega = (np.tile(np.asarray(x, float), (n, 1)) for x in (pos, vel, omega))
    s.quat, s.t = np.tile([1.0, 0, 0, 0], (n, 1)), t
    return s


def cam(world, cls=Camera, n=1, **kw):
    kw.setdefault("mount", (0.0, 0.0, 0.0))
    kw.setdefault("depth_noise", (0.0, 0.0))
    kw.setdefault("pixel_noise", 0.0)
    c = cls(**kw)
    c.bind(Sim(world, n), 1)
    return c


def pole_world(x=10.0, y=0.0, color=(1.0, 1.0, 1.0)):
    w = World()
    w.add_cylinder((x, y), 0.15, 0.0, 30.0, color)
    return w


# ------------------------------------------------------------------------------------------------- lenses
ALL_LENSES = [
    PinholeLens(64, 48, 90.0),
    PinholeLens(64, 48, 70.0, distortion=(-0.25, 0.08, 0.001, -0.002, 0.0)),
    PinholeLens(64, 48, 60.0, distortion=(0.15, 0.0, 0.0, 0.0, 0.0)),
    FisheyeLens(64, 48, 180.0, "equidistant"),
    FisheyeLens(64, 48, 200.0, "equidistant"),
    FisheyeLens(64, 48, 180.0, "equisolid"),
    FisheyeLens(64, 48, 180.0, "stereographic"),
    FisheyeLens(64, 48, 120.0, "orthographic"),
]


@pytest.mark.parametrize(
    "lens", ALL_LENSES, ids=lambda l: f"{l.kind}-{getattr(l, 'model', '')}-{l.fov:g}-{'d' if getattr(l, 'distorted', False) else ''}"
)
def test_unproject_and_project_are_inverse(lens):
    rng = np.random.default_rng(0)
    u, v = rng.uniform(2, lens.W - 2, 400), rng.uniform(2, lens.H - 2, 400)
    d, ok = lens.unproject(u, v)
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)
    pix = lens.project(d[ok])
    np.testing.assert_allclose(pix, np.stack([u, v], -1)[ok], atol=1e-5)  # sub-micropixel round trip
    centre, _ = lens.unproject(np.array([lens.cx]), np.array([lens.cy]))
    np.testing.assert_allclose(centre[0], [1, 0, 0], atol=1e-12)


def test_pinhole_geometry_and_field_of_view():
    l = PinholeLens(64, 48, 90.0)
    assert l.f == pytest.approx(32.0)  # f = (W / 2) / tan(fov / 2)
    d, _ = l.unproject(np.array([0.0, 64.0]), np.array([24.0, 24.0]))
    np.testing.assert_allclose(np.degrees(np.arctan2(d[:, 1], d[:, 0])), [45.0, -45.0], atol=1e-9)  # image left = body +y (left)
    np.testing.assert_allclose(l.project(np.array([[1.0, 0.0, 0.0]])), [[32.0, 24.0]])
    np.testing.assert_allclose(l.intrinsics, [[32, 0, 32], [0, 32, 24], [0, 0, 1]])
    with pytest.raises(ValueError):
        PinholeLens(64, 48, 180.0)


def test_distortion_follows_the_brown_conrady_formula():
    k1 = -0.3
    l = PinholeLens(100, 100, 90.0, distortion=(k1, 0, 0, 0, 0))
    d = np.array([[1.0, -0.4, 0.0]]) / np.linalg.norm([1.0, -0.4, 0.0])  # a point at x_n = 0.4, y_n = 0
    pix = l.project(d)[0]
    xn = 0.4
    assert (pix[0] - 50.0) / l.f == pytest.approx(xn * (1 + k1 * xn**2), rel=1e-12)  # barrel distortion pulls the point inwards
    assert pix[0] < PinholeLens(100, 100, 90.0).project(d)[0][0]
    pin = PinholeLens(100, 100, 90.0, distortion=(0.0, 0.0, 0.01, 0.0, 0.0))
    assert (
        pin.project(np.array([[1.0, -0.3, -0.2]]))[0][1] != PinholeLens(100, 100, 90.0).project(np.array([[1.0, -0.3, -0.2]]))[0][1]
    )  # tangential terms shift both axes


@pytest.mark.parametrize(
    "model,radius",
    [
        ("equidistant", lambda t: t),
        ("equisolid", lambda t: 2 * math.sin(t / 2)),
        ("stereographic", lambda t: 2 * math.tan(t / 2)),
        ("orthographic", math.sin),
    ],
)
def test_fisheye_projection_functions(model, radius):
    l = FisheyeLens(200, 100, 120.0, model)
    for theta_deg in (10.0, 30.0, 55.0):
        th = math.radians(theta_deg)
        d = np.array([[math.cos(th), -math.sin(th), 0.0]])  # theta to the left
        pix = l.project(d)[0]
        assert pix[0] - l.cx == pytest.approx(-radius(th) * l.f, rel=1e-9) or pix[0] - l.cx == pytest.approx(radius(th) * l.f, rel=1e-9)
    edge, ok = l.unproject(np.array([0.0]), np.array([50.0]))
    assert ok[0] and np.degrees(np.arccos(edge[0, 0])) == pytest.approx(60.0, abs=1e-9)  # the image circle's edge is half the field of view


def test_fisheye_image_circle_and_wide_angles():
    l = FisheyeLens(64, 48, 180.0, "equidistant")
    _, ok = l.unproject()
    assert ok.reshape(48, 64)[24, 32] and not ok.reshape(48, 64)[0, 0] and not ok.reshape(48, 64)[47, 63]  # corners lie outside the circle
    wide = FisheyeLens(64, 48, 200.0)
    d, _ = wide.unproject(np.array([0.0]), np.array([24.0]))
    assert d[0, 0] < 0  # beyond 180 degrees: rays reach behind the camera
    for bad in (
        lambda: FisheyeLens(64, 48, 200.0, "orthographic"),
        lambda: FisheyeLens(64, 48, 90.0, "fisheyeish"),
        lambda: PinholeLens(1, 48),
        lambda: make_lens("lensless", 64, 48, 90.0),
    ):
        with pytest.raises(ValueError):
            bad()
    assert (
        isinstance(make_lens("equisolid", 64, 48, 150.0), FisheyeLens) and make_lens({"type": "pinhole", "fov_deg": 60.0}, 64, 48, 90.0).fov == 60.0
    )
    assert make_lens(l, 1, 1, 1) is l


# ------------------------------------------------------------------------------------- camera rendering
def column_of_pole(c, state):
    seg = c.measure(state)["seg"][0]
    cols = np.nonzero((seg >= 0).any(axis=0))[0]
    return cols


def test_pole_appears_at_the_pixel_predicted_by_the_lens():
    for lens, kw in (
        ("pinhole", {}),
        ("pinhole", {"distortion": (-0.25, 0.05, 0.0, 0.0, 0.0)}),
        ("equidistant", {"fov_deg": 170.0}),
        ("stereographic", {"fov_deg": 150.0}),
    ):
        w = pole_world(10.0, 4.0)  # 22 degrees to the left of the optical axis
        c = cam(w, lens=lens, modes=("seg",), width=160, height=48, **{"fov_deg": 100.0, **kw})
        cols = column_of_pole(c, st())
        u, _ = c.lens.project(np.array([[10.0, 4.0, 0.0]]))[0]
        assert len(cols) > 0 and abs(cols.mean() + 0.5 - u) < 1.2, (lens, cols.mean(), u)  # within a pixel of the analytic projection


def test_fisheye_sees_both_side_walls_at_once_and_a_pinhole_does_not():
    w = World()
    w.add_box((0, 8, 2), (30, 0.5, 20))
    w.add_box((0, -8, 2), (30, 0.5, 20))
    fish = cam(w, lens="equidistant", fov_deg=180.0, modes=("depth",), width=96, height=96, far=40.0)
    d = fish.measure(st(pos=(0, 0, 2.0)))["depth"][0]
    assert np.isfinite(d[48, 1]) and np.isfinite(d[48, 94])  # left and right edges of the image circle look sideways
    assert d[48, 1] == pytest.approx(7.75, abs=0.2) and d[48, 94] == pytest.approx(7.75, abs=0.2)  # 8 m minus half a wall thickness
    pin = cam(w, lens="pinhole", fov_deg=90.0, modes=("depth",), width=96, height=96, far=40.0).measure(st(pos=(0, 0, 2.0)))["depth"][0]
    assert (
        pin[48, 0] == pytest.approx(7.75, abs=0.2) and pin[48, 48] > 14.0
    )  # a pinhole's edge ray meets the wall at 45 degrees: z-depth 7.75 m, but it reports planar depth, not range


def test_default_pinhole_depth_is_the_planar_distance():
    w = World()
    w.add_box((12.0, 0, 2.0), (1.0, 40.0, 20.0))
    c = cam(w, modes=("depth",), width=32, height=24, far=60.0)
    d = c.measure(st())["depth"][0]
    np.testing.assert_allclose(
        d[:12], 11.5, atol=1e-9
    )  # z-depth to a perpendicular wall is the same for every pixel (upper half: the lower rays meet the ground first)
    assert np.isfinite(d).all() and c.to_dict()["lens"] == {"type": "pinhole", "fov_deg": 90.0, "distortion": None}


def test_rolling_shutter_skews_a_vertical_pole_while_a_global_shutter_does_not():
    w = pole_world(10.0, 0.0)
    v = 20.0  # the camera translates sideways past the pole
    kw = dict(modes=("seg",), width=200, height=60, fov_deg=60.0, far=40.0)
    g = cam(w, **kw)
    cols_g = [np.nonzero(g.measure(st(vel=(0, v, 0)))["seg"][0][r] >= 0)[0].mean() for r in (3, 30, 56)]
    assert np.ptp(cols_g) < 0.6  # global shutter: the pole is perfectly vertical
    ro = 0.03
    r = cam(w, rolling_shutter_s=ro, **kw)
    seg = r.measure(st(vel=(0, v, 0)))["seg"][0]
    cols = {row: np.nonzero(seg[row] >= 0)[0].mean() for row in (3, 30, 56)}
    shift = cols[56] - cols[3]
    predicted = (
        r.lens.f * v * ro * (53.0 / 60.0) / 10.0
    )  # f * (v * dt_rows) / Z pixels between row 3 and row 56 (camera moves left: pole drifts right in the image)
    assert abs(shift) == pytest.approx(abs(predicted), rel=0.2)
    assert np.sign(cols[30] - cols[3]) == np.sign(shift) and abs(cols[30] - cols[3]) < abs(shift)  # the skew grows steadily down the image


def test_rotation_during_readout_also_skews_the_image():
    w = pole_world(10.0, 0.0)
    r = cam(w, rolling_shutter_s=0.03, modes=("seg",), width=200, height=60, fov_deg=60.0, far=40.0)
    seg = r.measure(st(omega=(0, 0, 1.0)))["seg"][0]  # yawing at 1 rad/s
    top, bottom = np.nonzero(seg[3] >= 0)[0].mean(), np.nonzero(seg[56] >= 0)[0].mean()
    assert abs(bottom - top) == pytest.approx(r.lens.f * 1.0 * 0.03 * 53.0 / 60.0, rel=0.2)  # f * omega * dt


def bright_edge_world():
    w = World()
    w.add_box((12.0, 0.0, 3.0), (1.0, 10.0, 12.0), (1.0, 1.0, 1.0))  # a white block in front of a dark far wall
    w.add_box((30.0, 0.0, 3.0), (1.0, 80.0, 40.0), (0.05, 0.05, 0.05))
    return w


def test_motion_blur_averages_the_exposure():
    kw = dict(modes=("rgb",), width=96, height=24, fov_deg=60.0, far=60.0, shadows=False)
    sharp = cam(bright_edge_world(), **kw).measure(st(vel=(0, 30.0, 0)))["rgb"][0, 12, :, 0].astype(float)
    blur = cam(bright_edge_world(), exposure_s=0.02, blur_samples=8, **kw).measure(st(vel=(0, 30.0, 0)))["rgb"][0, 12, :, 0].astype(float)
    mid = lambda row: np.sum((row > row.min() + 0.15 * np.ptp(row)) & (row < row.max() - 0.15 * np.ptp(row)))
    assert mid(blur) > mid(sharp) + 3  # more pixels in the transition zone: the edge is smeared over the 0.6 m of motion
    assert abs(blur.mean() - sharp.mean()) < 0.1 * sharp.mean()  # but no light is created or lost


def test_exposure_and_vignetting():
    w = World()
    w.add_box((12.0, 0, 3.0), (1.0, 60.0, 30.0), (0.3, 0.3, 0.3))
    kw = dict(modes=("rgb",), width=64, height=48, fov_deg=100.0, shadows=False, far=60.0)
    base = cam(w, **kw).measure(st())["rgb"][0].astype(float)
    bright = cam(w, exposure_ev=1.0, **kw).measure(st())["rgb"][0].astype(float)
    np.testing.assert_allclose(bright[24, 32], 2.0 * base[24, 32], rtol=0.03)
    vig = cam(w, vignetting=0.8, **kw)
    img = vig.measure(st())["rgb"][0].astype(float)
    corner_gain = img[0, 0].mean() / base[0, 0].mean()
    expected = 0.2 + 0.8 * vig._off_axis.reshape(48, 64)[0, 0] ** 4
    assert corner_gain == pytest.approx(expected, rel=0.05) and img[24, 32].mean() == pytest.approx(base[24, 32].mean(), rel=0.05)


def test_sun_follows_the_clock_and_the_night_is_dark():
    def world_at(utc):
        w = World()
        w.add_box((14.0, 0, 3.0), (1.0, 60.0, 30.0), (0.7, 0.7, 0.7))
        w.origin, w.utc = LocalFrame(47.37, 8.54, 400.0), utc
        return w

    kw = dict(modes=("rgb",), width=32, height=24, fov_deg=80.0, far=80.0)
    noon = cam(world_at("2026-06-21T10:30:00Z"), **kw).measure(st())["rgb"].astype(float).mean()
    night = cam(world_at("2026-06-21T23:30:00Z"), **kw).measure(st())["rgb"].astype(float).mean()
    assert noon > 3.0 * night > 0.0
    plain = cam(World(), **kw)
    assert plain._environment(0.0)["day"] == 1.0 and np.allclose(
        plain._environment(0.0)["sun"], Camera.SUN / np.linalg.norm(Camera.SUN)
    )  # no origin/UTC: the fixed legacy sun
    sun = cam(world_at("2026-06-21T10:30:00Z"), **kw)._environment(0.0)["sun"]
    assert sun[2] > 0.7 and np.linalg.norm(sun) == pytest.approx(1.0)  # high summer sun
    assert cam(world_at("2026-06-21T10:30:00Z"), use_environment=False, **kw)._environment(0.0)["day"] == 1.0


def test_fog_and_cloud_wash_out_distant_objects():
    def world_with(weather=None):
        w = World()
        w.add_box((55.0, 0, 3.0), (1.0, 80.0, 30.0), (0.1, 0.1, 0.1))
        if weather:
            Weather.preset(weather).apply(w)
        return w

    kw = dict(modes=("rgb",), width=32, height=24, fov_deg=60.0, far=100.0, shadows=False, fog_distance=1e9)
    clear = cam(world_with(), **kw).measure(st())["rgb"][0, 12, 16].astype(float).mean()
    fog = cam(world_with("fog"), **kw).measure(st())["rgb"][0, 12, 16].astype(float).mean()
    assert fog > clear + 60  # the dark wall dissolves into the bright haze at 200 m visibility


def test_thermal_camera_ground_actors_sky_and_fog():
    w = World()
    Weather(temperature_c=15.0, visibility_m=50_000.0).apply(w)
    w.origin, w.utc = LocalFrame(47.37, 8.54, 400.0), "2026-06-21T10:30:00Z"
    hot = LinearActor(start=(15.0, 0.0, 2.0), velocity=(0, 0, 0), radius=1.5)
    hot.temperature_c = 60.0
    w.add_actor(hot)
    kw = dict(modes=("thermal",), width=48, height=48, fov_deg=90.0, far=60.0, thermal_noise_k=0.0)
    w.advance(0.0)
    c = cam(w, **kw)
    c.pitch = 0.0
    img = c.measure(st(pos=(0, 0, 2.0)))["thermal"][0]
    assert img[24, 24] == pytest.approx(60.0, abs=0.5)  # the hot body at 15 m through clear air
    assert img[2, 24] < 15.0 - 10.0  # the sky above the horizon is colder than the air
    down = cam(w, **{**kw, "pitch_deg": 60.0}).measure(st(pos=(0, 0, 2.0)))["thermal"][0]
    assert np.nanmax(down) > 15.0 + 8.0  # sunlit ground is warmer than the air
    night = World()
    Weather(temperature_c=15.0).apply(night)
    night.origin, night.utc = LocalFrame(47.37, 8.54, 400.0), "2026-06-21T23:30:00Z"
    cold = cam(night, **{**kw, "pitch_deg": 60.0}).measure(st(pos=(0, 0, 2.0)))["thermal"][0]
    assert np.nanmax(cold) == pytest.approx(15.0, abs=0.01)  # at night the ground is at air temperature
    fogw = World()
    Weather.preset("fog").apply(fogw)
    hot2 = LinearActor(start=(15.0, 0.0, 2.0), velocity=(0, 0, 0), radius=1.5)
    hot2.temperature_c = 60.0
    fogw.add_actor(hot2)
    fogw.advance(0.0)
    foggy = cam(fogw, **kw).measure(st(pos=(0, 0, 2.0)))["thermal"][0][24, 24]
    assert 6.0 < foggy < 60.0 - 5.0  # fog attenuates the signal towards the ambient temperature
    noisy = cam(w, **{**kw, "thermal_noise_k": 0.1}, n=1).measure(st(pos=(0, 0, 2.0)))["thermal"][0]
    clean = cam(w, **kw).measure(st(pos=(0, 0, 2.0)))["thermal"][0]
    assert np.nanstd(noisy - clean) == pytest.approx(0.1, rel=0.1)  # NETD noise


def test_camera_validation():
    for bad in (dict(modes=("lidar",)), dict(blur_samples=0), dict(vignetting=2.0), dict(rolling_shutter_s=-1.0)):
        with pytest.raises(ValueError):
            Camera(**bad)


# ----------------------------------------------------------------------------------------------- events
def event_cam(world, **kw):
    kw.setdefault("background_rate_hz", 0.0)
    kw.setdefault("threshold_noise", 0.0)
    return cam(world, EventCamera, modes=None, width=64, height=32, fov_deg=60.0, shadows=False, far=60.0, **kw)


def test_static_scene_emits_no_events_and_the_first_call_only_initialises():
    c = event_cam(bright_edge_world(), rate=500.0)
    first = c.measure(st(t=0.0))
    assert first["n_events"][0] == 0 and first["events"].shape == (1, 0, 4)
    again = c.measure(st(t=0.002))
    assert again["n_events"][0] == 0 and np.all(again["frame"] == 0)


def test_a_moving_edge_produces_ordered_events_with_the_right_polarity():
    c = event_cam(bright_edge_world(), rate=500.0)
    c.measure(st(t=0.0))
    out = c.measure(st(pos=(0, 0.5, 2.0), t=0.002))  # the camera slides 50 cm to the left: both edges of the block sweep right across several pixels
    ev = out["events"][0][: int(out["n_events"][0])]
    assert len(ev) > 20 and np.all(ev[:, 2] >= 0.0) and np.all(ev[:, 2] <= 0.002 + 1e-12) and np.all(np.diff(ev[:, 2]) >= 0)
    pol = {int(p) for p in ev[:, 3]}
    assert pol == {-1, 1}  # both polarities: a leading and a trailing edge
    assert out["frame"].shape == (1, 32, 64) and np.abs(out["frame"]).sum() >= len(ev) * 0.5


def test_event_count_scales_inversely_with_the_contrast_threshold():
    counts = []
    for thr in (0.1, 0.4):
        c = event_cam(bright_edge_world(), contrast_threshold=thr, refractory_s=0.0)
        c.measure(st(t=0.0))
        counts.append(int(c.measure(st(pos=(0, 0.3, 2.0), t=0.002))["n_events"][0]))
    assert counts[0] > 2.0 * counts[1] > 0


def test_background_activity_and_event_cap():
    quiet = event_cam(bright_edge_world(), background_rate_hz=0.0)
    noisy = event_cam(bright_edge_world(), background_rate_hz=20.0)
    for c in (quiet, noisy):
        c.measure(st(t=0.0))
    nq = quiet.measure(st(t=1.0))["n_events"][0]
    nn = noisy.measure(st(t=1.0))["n_events"][0]
    assert nq == 0 and nn == pytest.approx(20.0 * 64 * 32 * 1.0, rel=0.1)  # Poisson background: rate x pixels x time
    capped = event_cam(bright_edge_world(), max_events=50, background_rate_hz=0.0)
    capped.measure(st(t=0.0))
    assert capped.measure(st(pos=(0, 1.0, 2.0), t=0.002))["n_events"][0] == 50


def test_event_camera_validation_registry_and_engine():
    for bad in (dict(contrast_threshold=0.0), dict(max_events=0), dict(refractory_s=-1.0)):
        with pytest.raises(ValueError):
            EventCamera(**bad)
    from aevion.registry import registries

    assert "event_camera" in registries.sensors and "dvs" in registries.sensors
    from aevion.engine import Simulation

    cfg = {
        "duration": 1.5,
        "world": {"type": "scene", "params": {"boxes": [{"center": [10, 0, 3], "size": [1, 6, 8], "color": [1, 1, 1]}]}},
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 3], "radius": 1.5, "omega": 2.5}},
        "sensors": [{"type": "event_camera", "params": {"rate": 200.0, "width": 32, "height": 24}, "record": True}],
    }
    sim = Simulation(cfg)
    arch = sim.run()
    frames = arch["sensors/event"]
    assert frames.shape[2:] == (24, 32) and np.abs(frames).sum() > 0
