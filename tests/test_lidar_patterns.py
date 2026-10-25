"""LiDAR scan patterns, motion distortion and FMCW Doppler."""

import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments.actors import LinearActor
from aevion.environments.world import World
from aevion.sensors import Lidar
from aevion.sensors.scan_patterns import PATTERNS, LissajousPattern, MemsRasterPattern, RosettePattern, SpinningPattern, pattern_from_spec

B = NumpyBackend()


class Sim:
    def __init__(self, world, n):
        self.B, self.n, self.world = B, n, world


def st(pos=(0, 0, 3.0), vel=(0, 0, 0), omega=(0, 0, 0), t=0.0, n=1):
    s = type("S", (), {})()
    s.pos, s.vel, s.omega = (np.tile(np.asarray(x, float), (n, 1)) for x in (pos, vel, omega))
    s.quat, s.t = np.tile([1.0, 0, 0, 0], (n, 1)), t
    return s


def lidar(world=None, n=1, **kw):
    kw.setdefault("noise_std", 0.0)
    kw.setdefault("noise_rel", 0.0)
    kw.setdefault("dropout_prob", 0.0)
    kw.setdefault("mount", (0.0, 0.0, 0.0))
    l = Lidar(**kw)
    l.bind(Sim(world or wall(), n), 1)
    return l


def wall(x=20.0):
    w = World()
    w.add_box((x, 0, 3.0), (0.5, 80.0, 40.0))
    return w


def angle_from_axis(d):
    return np.degrees(np.arccos(np.clip(d[:, 0], -1, 1)))


# ---------------------------------------------------------------------------------------------- patterns
def test_every_pattern_returns_unit_vectors_inside_its_field_of_view():
    for pat in (SpinningPattern(8, 90, (-10, 10), 120.0), RosettePattern(2000), MemsRasterPattern(), LissajousPattern()):
        d = pat.directions(0)
        assert d.shape == (pat.n_rays, 3) and np.allclose(np.linalg.norm(d, axis=1), 1.0)
        t = pat.time_offsets(0.1)
        assert t.shape == (pat.n_rays,) and t.min() >= 0 and t.max() < 0.1 + 1e-12
        if pat.grid_shape is None:
            assert np.all(np.diff(t) >= 0)  # free-form patterns fire in order; grids fire all channels of a column together
    r = RosettePattern(3000, fov_deg=38.4)
    assert angle_from_axis(np.concatenate([r.directions(k) for k in range(5)])).max() <= 19.2 + 0.7  # the circular FoV (chord vs arc tolerance)
    mems = MemsRasterPattern(h_fov=60.0, v_fov=20.0)
    az, el = np.degrees(np.arctan2(mems.directions()[:, 1], mems.directions()[:, 0])), np.degrees(np.arcsin(mems.directions()[:, 2]))
    assert abs(az).max() <= 30.0 and abs(el).max() <= 10.0 + 1e-9
    liss = LissajousPattern(h_fov=50.0, v_fov=25.0)
    azl, ell = np.degrees(np.arctan2(liss.directions()[:, 1], liss.directions()[:, 0])), np.degrees(np.arcsin(liss.directions()[:, 2]))
    assert abs(azl).max() == pytest.approx(25.0, abs=0.5) and abs(ell).max() == pytest.approx(12.5, abs=0.5)


def test_spinning_pattern_is_the_original_grid():
    old = Lidar(channels=4, samples=8, v_fov=(-10.0, 10.0), h_fov=360.0)
    new = SpinningPattern(4, 8, (-10.0, 10.0), 360.0)
    np.testing.assert_allclose(old.dirs_body, new.directions())
    assert old.pattern is None and new.grid_shape == (4, 8) and new.static
    via = Lidar(pattern={"type": "spinning", "channels": 4, "samples": 8, "v_fov": [-10, 10]})
    np.testing.assert_allclose(via.dirs_body, new.directions())


def test_rosette_is_non_repetitive_and_fills_the_field_of_view_progressively():
    r = RosettePattern(1500, fov_deg=40.0, point_rate_hz=100_000.0)
    assert not r.static and r.grid_shape is None
    assert not np.allclose(r.directions(0), r.directions(1))
    assert np.allclose(r.directions(3), r.directions(3))  # a scan is a pure function of its number

    def coverage(scans):
        d = np.concatenate([r.directions(k) for k in range(scans)])
        az, el = np.degrees(np.arctan2(d[:, 1], d[:, 0])), np.degrees(np.arcsin(d[:, 2]))
        cells = set(zip(np.floor(az / 1.0).astype(int), np.floor(el / 1.0).astype(int)))
        return len(cells)

    c1, c4, c16 = coverage(1), coverage(4), coverage(16)
    assert c1 < c4 < c16 and c16 > 3 * c1  # successive scans fill in new cells
    a = RosettePattern(500, ratio=-0.6180339887)  # irrational ratio: no two scans start at the same point
    assert len({tuple(np.round(a.directions(k)[0], 9)) for k in range(20)}) == 20


def test_mems_raster_is_boustrophedon_with_resonant_spacing():
    m = MemsRasterPattern(h_fov=60.0, v_fov=20.0, lines=6, points=64)
    az = np.degrees(np.arctan2(m.directions()[:, 1], m.directions()[:, 0])).reshape(6, 64)
    assert az[0, 0] < 0 < az[0, -1] and az[1, 0] > 0 > az[1, -1]  # alternate lines run in opposite directions
    step = np.abs(np.diff(az[0]))
    assert step[0] < 0.5 * step[len(step) // 2]  # dense near the turning points, sparse in the middle (sinusoidal mirror)
    el = np.degrees(np.arcsin(m.directions()[:, 2])).reshape(6, 64)
    np.testing.assert_allclose(el[:, 0], np.linspace(-10, 10, 6), atol=0.05)  # linear slow axis
    assert m.grid_shape == (6, 64) and m.static


def test_lissajous_figure_closes_each_scan_for_integer_frequencies():
    p = LissajousPattern(points=4000, fx=5.0, fy=4.0, phase_deg=90.0)
    d = p.directions()
    assert np.linalg.norm(d[-1] - d[0]) < 0.01  # the curve returns to its start after one scan
    assert len(PATTERNS) == 4 and set(PATTERNS) == {"spinning", "rosette", "mems_raster", "lissajous"}


def test_pattern_specs_validation_and_serialisation():
    for p in (SpinningPattern(3, 5), RosettePattern(100), MemsRasterPattern(lines=3, points=4), LissajousPattern(points=50)):
        again = pattern_from_spec(p.to_dict())
        np.testing.assert_allclose(again.directions(2), p.directions(2))
    assert isinstance(pattern_from_spec("mems_raster"), MemsRasterPattern) and pattern_from_spec(p) is p
    with pytest.raises(ValueError, match="unknown scan pattern"):
        pattern_from_spec("fan")
    for bad in (
        lambda: SpinningPattern(0, 5),
        lambda: RosettePattern(2),
        lambda: RosettePattern(100, split=1.5),
        lambda: MemsRasterPattern(points=1),
        lambda: LissajousPattern(fx=0),
    ):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------------------------------------- lidar
def test_default_lidar_output_is_unchanged():
    l = lidar(channels=4, samples=16, v_fov=(-5.0, 5.0), h_fov=60.0, rate=10.0)
    out = l.measure(st())
    assert out["ranges"].shape == (1, 4, 16) and out["points"].shape == (1, 64, 3) and "radial_velocity" not in out
    assert np.isfinite(out["ranges"]).sum() > 0 and np.nanmin(np.where(np.isfinite(out["ranges"]), out["ranges"], np.nan)) == pytest.approx(
        19.75 / (math.cos(math.radians(2.0)) * math.cos(math.radians(5.0 / 3.0))), abs=1e-6
    )  # nearest beam: 2 deg in azimuth, 1.67 deg in elevation


def test_rosette_lidar_scans_differ_and_see_the_wall():
    l = lidar(pattern={"type": "rosette", "points_per_scan": 800, "fov_deg": 30.0}, rate=10.0, max_range=60.0)
    a = l.measure(st(t=0.0))
    b = l.measure(st(t=0.1))
    assert a["ranges"].shape == (1, 800) and a["points"].shape == (1, 800, 3)
    assert np.isfinite(a["ranges"]).mean() > 0.9 and not np.allclose(a["ranges"], b["ranges"], equal_nan=True)
    d0 = np.linalg.norm(a["points"][0], axis=1)
    assert np.nanmin(d0) == pytest.approx(19.75, abs=0.05)  # perpendicular wall at x = 19.75
    assert not np.array_equal(np.round(a["points"][0][:20], 6), np.round(b["points"][0][:20], 6))  # a different part of the rosette


def test_motion_distortion_shifts_ranges_within_one_scan():
    kw = dict(channels=1, samples=40, v_fov=(0.0, 0.0), h_fov=10.0, rate=10.0, max_range=60.0)
    still = lidar(**kw, motion_distortion=True).measure(st(vel=(0, 0, 0)))["ranges"][0, 0]
    moving_off = lidar(**kw, motion_distortion=False).measure(st(vel=(20.0, 0, 0)))["ranges"][0, 0]
    moving = lidar(**kw, motion_distortion=True).measure(st(vel=(20.0, 0, 0)))["ranges"][0, 0]
    np.testing.assert_allclose(moving_off, still, atol=1e-9)  # without distortion the scan is a snapshot
    drop = moving - still
    assert drop[0] == pytest.approx(0.0, abs=1e-6) and drop[-1] < -1.8  # 20 m/s closing: 2 m over the 0.1 s scan
    assert np.all(np.diff(drop) < 1e-9)  # monotonically shorter as the scan progresses
    assert drop[20] == pytest.approx(-20.0 * 0.5 * 0.1, abs=0.1)  # half-way through the scan: 1 m


def test_rotation_distortion_bends_a_straight_wall():
    kw = dict(channels=1, samples=60, v_fov=(0.0, 0.0), h_fov=40.0, rate=10.0, max_range=60.0)
    flat = lidar(**kw, motion_distortion=True).measure(st())["points"][0]
    spun = lidar(**kw, motion_distortion=True).measure(st(omega=(0, 0, 1.0)))["points"][0]  # yawing 1 rad/s: 0.1 rad over a scan
    assert np.allclose(flat[:, 0], 19.75, atol=1e-6)  # a perpendicular wall is straight without rotation
    # the points are reported in the scan-start sensor frame with the *nominal* beam direction, so the range is what changes
    r_flat, r_spun = np.linalg.norm(flat, axis=1), np.linalg.norm(spun, axis=1)
    assert np.max(np.abs(r_spun - r_flat)) > 0.2 and abs(r_spun[0] - r_flat[0]) < 1e-9


def test_fmcw_doppler_measures_the_radial_velocity_of_each_return():
    kw = dict(channels=1, samples=21, v_fov=(0.0, 0.0), h_fov=60.0, rate=10.0, max_range=60.0, doppler=True, doppler_noise=0.0)
    l = lidar(**kw)
    out = l.measure(st(vel=(5.0, 0, 0)))
    vr = out["radial_velocity"][0, 0]
    az = np.radians(np.linspace(-30, 30, 21))
    hit = np.isfinite(out["ranges"][0, 0])
    np.testing.assert_allclose(vr[hit], 5.0 * np.cos(az[hit]), atol=1e-9)  # closing speed along each beam
    sky = lidar(World(), **{**kw, "v_fov": (45.0, 45.0)}).measure(st())  # beams that hit nothing report NaN
    assert np.all(np.isnan(sky["radial_velocity"]))
    w = World()
    w.add_actor(LinearActor(start=(25.0, 0, 3.0), velocity=(-6.0, 0, 0), radius=2.0))
    w.advance(0.0)
    mover = lidar(w, **{**kw, "samples": 3, "h_fov": 4.0}).measure(st(vel=(0, 0, 0)))
    assert mover["radial_velocity"][0, 0, 1] == pytest.approx(6.0, abs=1e-6)


def test_lidar_patterns_in_the_engine():
    from aevion.engine import Simulation

    cfg = {
        "duration": 0.6,
        "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [15, 0, 3], "size": [1, 20, 10]}]}},
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 3.0]}},
        "sensors": [
            {
                "type": "lidar",
                "params": {"pattern": {"type": "mems_raster", "lines": 8, "points": 32}, "return_points": False, "max_range": 40.0},
                "record": True,
            }
        ],
    }
    sim = Simulation(cfg)
    arch = sim.run()
    rec = arch["sensors/lidar"]
    assert rec.shape[2:] == (8, 32) and np.isfinite(rec[0, 0]).mean() > 0.9
    assert sim.suite["lidar"].to_dict()["pattern"]["type"] == "mems_raster"
    from aevion.registry import registries

    assert set(registries.scan_patterns.names()) == set(PATTERNS)
