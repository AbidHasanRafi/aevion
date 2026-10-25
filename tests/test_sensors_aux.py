"""Pitot, optical flow, ultrasonic and UWB sensors."""

import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments.geodesy import LocalFrame
from aevion.environments.mesh import TriangleMesh
from aevion.environments.weather import Weather
from aevion.environments.world import World
from aevion.sensors import OpticalFlow, PitotTube, Ultrasonic, UwbRanging, multilaterate

B = NumpyBackend()


class Sim:
    def __init__(self, world, n):
        self.B, self.n, self.world = B, n, world


def st(pos=(0, 0, 10.0), vel=(0, 0, 0), quat=(1, 0, 0, 0), omega=(0, 0, 0), wind=(0, 0, 0), t=0.0, n=1):
    s = type("S", (), {})()
    s.pos, s.vel, s.omega, s.wind = (np.tile(np.asarray(x, float), (n, 1)) for x in (pos, vel, omega, wind))
    s.quat, s.t = np.tile(np.asarray(quat, float), (n, 1)), t
    return s


def bind(sensor, world=None, n=1, seed=1):
    sensor.bind(Sim(world or World(), n), seed)
    return sensor


def yaw_quat(deg):
    return (math.cos(math.radians(deg) / 2), 0.0, 0.0, math.sin(math.radians(deg) / 2))


def pitch_quat(deg):
    return (math.cos(math.radians(deg) / 2), 0.0, math.sin(math.radians(deg) / 2), 0.0)


# ------------------------------------------------------------------------------------------------- pitot
def test_pitot_reads_half_rho_v_squared_and_indicated_airspeed():
    p = bind(PitotTube(ideal=True))
    r = p.measure(st(pos=(0, 0, 0.0), vel=(20.0, 0, 0)))
    rho = float(World().atmosphere.density(B, np.array([0.0]))[0])
    assert r["diff_pressure"][0] == pytest.approx(0.5 * rho * 400.0, rel=1e-9) and r["diff_pressure"][0] == pytest.approx(245.0, rel=1e-3)
    assert r["ias"][0] == pytest.approx(20.0, rel=2e-4) and r["tas"][0] == pytest.approx(20.0, rel=2e-4)
    assert p.measure(st(pos=(0, 0, 0.0), vel=(-20.0, 0, 0)))["diff_pressure"][0] < 0  # moving backwards reads negative


def test_pitot_uses_relative_air_velocity_and_the_vehicle_attitude():
    p = bind(PitotTube(ideal=True))
    base = p.measure(st(pos=(0, 0, 0.0), vel=(20.0, 0, 0)))["ias"][0]
    assert p.measure(st(pos=(0, 0, 0.0), vel=(20.0, 0, 0), wind=(10.0, 0, 0)))["ias"][0] == pytest.approx(base / 2, rel=2e-4)  # tail wind halves it
    assert p.measure(st(pos=(0, 0, 0.0), vel=(20.0, 0, 0), wind=(20.0, 0, 0)))["ias"][0] == pytest.approx(0.0, abs=1e-3)  # drifting with the wind
    assert p.measure(st(pos=(0, 0, 0.0), vel=(0.0, 20.0, 0), quat=yaw_quat(90)))["ias"][0] == pytest.approx(base, rel=1e-6)  # nose into the motion
    assert p.measure(st(pos=(0, 0, 0.0), vel=(0.0, 20.0, 0), quat=yaw_quat(0)))["ias"][0] == pytest.approx(0.0, abs=0.5)  # sideways: no pressure


def test_pitot_acceptance_cone():
    p = bind(PitotTube(ideal=True, acceptance_deg=20.0, cutoff_deg=45.0))
    v = 20.0
    full = p.measure(st(pos=(0, 0, 0.0), vel=(v, 0, 0)))["diff_pressure"][0]
    for ang, want in ((10.0, math.cos(math.radians(10)) ** 2), (20.0, math.cos(math.radians(20)) ** 2)):
        dp = p.measure(st(pos=(0, 0, 0.0), vel=(v * math.cos(math.radians(ang)), v * math.sin(math.radians(ang)), 0)))["diff_pressure"][0]
        assert dp / full == pytest.approx(want, rel=1e-6)
    mid = p.measure(st(pos=(0, 0, 0.0), vel=(v * math.cos(math.radians(32.5)), v * math.sin(math.radians(32.5)), 0)))["diff_pressure"][0] / full
    assert mid == pytest.approx(math.cos(math.radians(32.5)) ** 2 * 0.5, rel=1e-6)  # halfway through the ramp
    assert p.measure(st(pos=(0, 0, 0.0), vel=(v * math.cos(math.radians(60)), v * math.sin(math.radians(60)), 0)))["diff_pressure"][
        0
    ] == pytest.approx(0.0, abs=1e-9)


def test_tas_exceeds_ias_with_altitude_and_heat():
    p = bind(PitotTube(ideal=True))
    low = p.measure(st(pos=(0, 0, 0.0), vel=(30.0, 0, 0)))
    high = p.measure(st(pos=(0, 0, 3000.0), vel=(30.0, 0, 0)))
    rho0, rho3 = (float(World().atmosphere.density(B, np.array([z]))[0]) for z in (0.0, 3000.0))
    assert high["ias"][0] < low["ias"][0] and high["tas"][0] == pytest.approx(30.0, rel=1e-3)  # TAS recovers the true speed
    assert high["ias"][0] == pytest.approx(30.0 * math.sqrt(rho3 / 1.225), rel=1e-3) and rho3 < 0.75 * rho0
    w = World()
    Weather.preset("hot").apply(w)
    hot = bind(PitotTube(ideal=True), w).measure(st(pos=(0, 0, 0.0), vel=(30.0, 0, 0)))
    assert hot["ias"][0] < low["ias"][0]  # thinner hot air: lower dynamic pressure for the same speed


def test_pitot_lag_noise_drift_and_blockage():
    lag = bind(PitotTube(rate=100.0, lag_s=0.2, noise_pa=0.0, drift_pa=0.0))
    s = st(pos=(0, 0, 0.0), vel=(0, 0, 0))
    lag.measure(s)  # initialise at rest
    final = 0.5 * 1.225 * 400.0
    out = []
    for _ in range(60):  # 0.6 s after a step to 20 m/s
        out.append(lag.measure(st(pos=(0, 0, 0.0), vel=(20.0, 0, 0)))["diff_pressure"][0])
    t63 = (np.argmax(np.array(out) > 0.632 * final) + 1) / 100.0
    assert t63 == pytest.approx(0.2, abs=0.03)  # first-order lag: 63 % at one time constant
    noisy = bind(PitotTube(noise_pa=2.0, drift_pa=0.0, lag_s=0.0), n=2000)
    dp = noisy.measure(st(pos=(0, 0, 0.0), vel=(0, 0, 0), n=2000))["diff_pressure"]
    assert np.std(dp) == pytest.approx(2.0, rel=0.08) and abs(np.mean(dp)) < 0.2
    iced = bind(PitotTube(noise_pa=0.0, drift_pa=0.0, lag_s=0.0))
    iced.blockage = 1.0
    assert abs(iced.measure(st(pos=(0, 0, 0.0), vel=(20.0, 0, 0)))["diff_pressure"][0]) < 1e-9


def test_pitot_validation():
    for bad in (dict(axis=(0, 0, 0)), dict(acceptance_deg=50.0, cutoff_deg=45.0), dict(lag_s=-1.0), dict(noise_pa=-1.0)):
        with pytest.raises(ValueError):
            PitotTube(**bad)


# --------------------------------------------------------------------------------------------- optical flow
def flow_sensor(**kw):
    return bind(OpticalFlow(noise=0.0, noise_rel=0.0, gyro_noise=0.0, **kw))


def test_flow_of_a_translating_vehicle_is_velocity_over_height():
    f = flow_sensor()
    r = f.measure(st(pos=(0, 0, 5.0), vel=(2.0, 0, 0)))
    assert r["range"][0] == pytest.approx(5.0) and r["valid"][0]
    np.testing.assert_allclose(r["flow_comp"][0], [0.0, 2.0 / 5.0], atol=1e-12)  # forward motion -> +flow_y = v / h
    np.testing.assert_allclose(f.measure(st(pos=(0, 0, 5.0), vel=(0, 3.0, 0)))["flow_comp"][0], [-3.0 / 5.0, 0.0], atol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(5):  # velocity is recovered from the compensated flow and the range
        v = rng.uniform(-3, 3, 2)
        h = rng.uniform(1, 10)
        r = f.measure(st(pos=(0, 0, h), vel=(v[0], v[1], 0)))
        np.testing.assert_allclose([r["flow_comp"][0][1] * r["range"][0], -r["flow_comp"][0][0] * r["range"][0]], v, atol=1e-9)


def test_flow_of_a_rotating_vehicle_is_removed_by_gyro_compensation():
    f = flow_sensor()
    r = f.measure(st(pos=(0, 0, 5.0), vel=(0, 0, 0), omega=(0.5, -0.3, 0.2)))
    np.testing.assert_allclose(r["flow"][0], [-0.5, 0.3], atol=1e-12)  # raw flow is minus the rotation about x and y
    np.testing.assert_allclose(r["flow_comp"][0], 0.0, atol=1e-12)


def test_flow_range_follows_tilt_and_obstacles():
    f = flow_sensor()
    assert f.measure(st(pos=(0, 0, 4.0), quat=pitch_quat(30.0)))["range"][0] == pytest.approx(4.0 / math.cos(math.radians(30.0)), rel=1e-9)
    w = World()
    w.add_box((0, 0, 1.5), (6, 6, 3.0))  # a roof 3 m high
    assert bind(OpticalFlow(noise=0.0), w).measure(st(pos=(0, 0, 5.0)))["range"][0] == pytest.approx(2.0)


def test_flow_validity_and_quality():
    f = flow_sensor()
    good = f.measure(st(pos=(0, 0, 3.0), vel=(0.5, 0, 0)))
    assert good["quality"][0] > 150
    assert not f.measure(st(pos=(0, 0, 40.0), vel=(0, 0, 0)))["valid"][0]  # above max_range
    assert not f.measure(st(pos=(0, 0, 0.05), vel=(0, 0, 0)))["valid"][0]  # below min_range
    fast = f.measure(st(pos=(0, 0, 1.0), vel=(20.0, 0, 0)))  # flow 20 rad/s: blurred beyond use
    assert not fast["valid"][0] and fast["quality"][0] == 0.0 and np.all(fast["flow"][0] == 0.0)
    assert f.measure(st(pos=(0, 0, 12.0), vel=(0.5, 0, 0)))["quality"][0] < good["quality"][0]  # the sensor degrades towards its limit
    night = World()
    night.origin, night.utc = LocalFrame(47.37, 8.54, 400.0), "2026-06-21T23:30:00Z"
    assert not bind(OpticalFlow(noise=0.0), night).measure(st(pos=(0, 0, 3.0), t=0.0))["valid"][0]
    assert bind(OpticalFlow(noise=0.0, illuminated=True), night).measure(st(pos=(0, 0, 3.0)))["valid"][0]
    day = World()
    day.origin, day.utc = LocalFrame(47.37, 8.54, 400.0), "2026-06-21T10:00:00Z"
    assert bind(OpticalFlow(noise=0.0), day).measure(st(pos=(0, 0, 3.0)))["valid"][0]
    for bad in (dict(min_range=0.0), dict(max_range=0.1), dict(max_flow=-1.0)):
        with pytest.raises(ValueError):
            OpticalFlow(**bad)


def test_flow_noise_statistics():
    f = bind(OpticalFlow(noise=0.02, noise_rel=0.0, gyro_noise=0.0), n=3000)
    r = f.measure(st(pos=(0, 0, 5.0), vel=(1.0, 0, 0), n=3000))
    assert np.std(r["flow"][:, 1]) == pytest.approx(0.02, rel=0.08) and np.mean(r["flow"][:, 1]) == pytest.approx(0.2, abs=0.002)


# ----------------------------------------------------------------------------------------------- ultrasonic
def wall_world(x=3.0, tilt_deg=0.0):
    w = World()
    m = TriangleMesh.box((0, 0, 0), (0.2, 40.0, 40.0)).transformed(rotation_deg=(0, 0, tilt_deg), translation=(x, 0, 5.0))
    return w.add_mesh(m)


def quiet_us(world, **kw):
    kw.setdefault("temperature_compensated", True)  # geometry only; the speed-of-sound bias has its own test
    return bind(Ultrasonic(noise_std=0.0, noise_rel=0.0, resolution=0.0, **kw), world)


def test_ultrasonic_measures_the_distance_to_a_wall_and_rejects_oblique_surfaces():
    us = quiet_us(wall_world(3.0))
    r = us.measure(st(pos=(0, 0, 5.0)))
    assert r["range"][0] == pytest.approx(2.9, abs=1e-9) and r["valid"][0]  # near face of the 0.2 m slab at x = 2.9
    steep = quiet_us(wall_world(3.0, tilt_deg=60.0)).measure(st(pos=(0, 0, 5.0)))
    assert not steep["valid"][0] and steep["range"][0] == 5.0  # specular: the echo never returns; reports max range
    ok = quiet_us(wall_world(3.0, tilt_deg=25.0)).measure(st(pos=(0, 0, 5.0)))
    assert ok["valid"][0]
    assert not quiet_us(World()).measure(st(pos=(0, 0, 50.0)))["valid"][0]  # nothing in range


def test_ultrasonic_cone_sees_an_off_axis_obstacle_that_a_beam_would_miss():
    w = wall_world(5.5)
    ring_y = -2.0 * math.tan(math.radians(15.0))  # a pole placed on the cone's edge, 2 m ahead
    w.add_cylinder((2.0, ring_y), 0.15, 0.0, 12.0)
    r = quiet_us(w, max_range=8.0).measure(st(pos=(0, 0, 5.0)))
    assert 1.8 < r["range"][0] < 2.15  # the pole (about 2 m away) beats the wall at 5.4 m
    narrow = bind(
        Ultrasonic(noise_std=0.0, noise_rel=0.0, resolution=0.0, half_angle_deg=3.0, temperature_compensated=True, max_range=8.0), w
    ).measure(st(pos=(0, 0, 5.0)))
    assert narrow["range"][0] == pytest.approx(5.4, abs=1e-6)  # a narrow beam misses the pole


def test_ultrasonic_speed_of_sound_bias_resolution_and_noise():
    w = wall_world(3.0)
    hot = World()
    Weather.preset("hot").apply(hot)
    hot.add_mesh(w.meshes[0])
    c_true = math.sqrt(1.4 * 287.05287 * (273.15 + 38.0))
    biased = quiet_us(hot, temperature_compensated=False).measure(st(pos=(0, 0, 5.0)))["range"][0]
    assert biased == pytest.approx(2.9 * 343.0 / c_true, rel=2e-4) and biased < 2.9
    assert quiet_us(hot, temperature_compensated=True).measure(st(pos=(0, 0, 5.0)))["range"][0] == pytest.approx(2.9)
    coarse = bind(Ultrasonic(noise_std=0.0, noise_rel=0.0, resolution=0.05, temperature_compensated=True), w).measure(st(pos=(0, 0, 5.0)))["range"][0]
    assert coarse == pytest.approx(2.9) and (coarse / 0.05) == pytest.approx(round(coarse / 0.05))
    many = bind(Ultrasonic(noise_std=0.01, noise_rel=0.0, resolution=0.0, temperature_compensated=True), w, n=2000).measure(
        st(pos=(0, 0, 5.0), n=2000)
    )["range"]
    assert np.std(many) == pytest.approx(0.01, rel=0.08)
    for bad in (dict(direction=(0, 0, 0)), dict(half_angle_deg=95.0), dict(max_range=0.1), dict(rays=2)):
        with pytest.raises(ValueError):
            Ultrasonic(**bad)


# -------------------------------------------------------------------------------------------------- UWB
ANCHORS = [(0, 0, 2.0), (20, 0, 2.0), (20, 20, 2.0), (0, 20, 2.0)]


def clean_uwb(world=None, n=1, **kw):
    return bind(UwbRanging(anchors=ANCHORS, noise_std=0.0, antenna_delay_std=0.0, outlier_probability=0.0, **kw), world, n)


def test_uwb_ranges_are_euclidean_distances_and_respect_the_operating_range():
    u = clean_uwb(max_range=30.0)
    p = np.array([5.0, 7.0, 1.0])
    r = u.measure(st(pos=p))
    np.testing.assert_allclose(r["ranges"][0], np.linalg.norm(p - np.array(ANCHORS), axis=1), atol=1e-12)
    assert r["valid"][0].all() and not r["nlos"].any()
    far = u.measure(st(pos=(100.0, 100.0, 1.0)))
    assert not far["valid"].any() and np.all(far["ranges"] == 0.0)


def test_uwb_noise_and_fixed_antenna_delays():
    u = bind(UwbRanging(anchors=ANCHORS, noise_std=0.05, antenna_delay_std=0.04, outlier_probability=0.0), n=3000)
    r = u.measure(st(pos=(5, 7, 1.0), n=3000))["ranges"]
    truth = np.linalg.norm(np.array([5, 7, 1.0]) - np.array(ANCHORS), axis=1)
    np.testing.assert_allclose(np.std(r, axis=0), 0.05, rtol=0.07)
    bias = r.mean(0) - truth
    np.testing.assert_allclose(bias, u.delay[0], atol=0.004)  # the same calibration error on every sample
    assert np.std(u.delay) > 0.0


def test_uwb_nlos_anchors_read_long_and_outliers_are_positive():
    w = World()
    w.add_mesh(TriangleMesh.box((10, 10, 5), (4, 4, 10)))  # a pillar in the middle of the anchor square
    u = bind(
        UwbRanging(
            anchors=[(0, 0, 2.0), (20, 20, 2.0), (20, 0, 2.0), (0, 20, 2.0)],
            noise_std=0.0,
            antenna_delay_std=0.0,
            outlier_probability=0.0,
            nlos_excess_mean=0.6,
            nlos_noise_std=0.0,
        ),
        w,
        n=4000,
    )
    r = u.measure(st(pos=(5.0, 5.0, 2.0), n=4000))
    truth = np.linalg.norm(np.array([5, 5, 2.0]) - np.array(u.anchors), axis=1)
    assert r["nlos"][0].tolist() == [False, True, False, False]  # only the anchor behind the pillar is blocked
    excess = r["ranges"][:, 1] - truth[1]
    assert np.all(excess >= 0) and np.mean(excess) == pytest.approx(0.6, rel=0.08)
    assert np.allclose(r["ranges"][:, 0], truth[0]) and np.allclose(r["ranges"][:, 2], truth[2])
    out = bind(UwbRanging(anchors=ANCHORS, noise_std=0.0, antenna_delay_std=0.0, outlier_probability=0.1, outlier_max=3.0), n=4000)
    e = out.measure(st(pos=(5, 7, 1.0), n=4000))["ranges"] - np.linalg.norm(np.array([5, 7, 1.0]) - np.array(ANCHORS), axis=1)
    assert np.mean(e > 1e-9) == pytest.approx(0.1, abs=0.015) and e.max() <= 3.0 + 1e-9 and e.min() >= -1e-9
    assert not bind(UwbRanging(anchors=ANCHORS, use_world_los=False), w).measure(st(pos=(5.0, 5.0, 2.0)))["nlos"].any()


def test_multilateration_recovers_positions_and_reports_degenerate_cases():
    anchors = np.array(ANCHORS)
    anchors[1, 2], anchors[2, 2] = 6.0, 9.0  # non-coplanar: unambiguous in 3-D
    pts = np.array([[5.0, 7.0, 1.5], [15.0, 3.0, 4.0], [10.0, 10.0, 0.5]])
    r = np.linalg.norm(pts[:, None, :] - anchors[None], axis=-1)
    x, used = multilaterate(r, anchors, x0=pts + 1.0)  # Gauss-Newton needs a starting point on the right side of the anchors
    np.testing.assert_allclose(x, pts, atol=1e-7)
    assert used.tolist() == [4, 4, 4]
    valid = np.ones_like(r, bool)
    valid[0, :2] = False  # only two anchors left for the first point
    x2, used2 = multilaterate(r, anchors, valid=valid, x0=pts + 1.0)
    assert np.all(np.isnan(x2[0])) and np.allclose(x2[1:], pts[1:], atol=1e-7) and used2[0] == 2
    rng = np.random.default_rng(0)
    noisy = r[None].repeat(1500, 0)[:, 0, :] + rng.normal(0, 0.05, (1500, 4))
    xs, _ = multilaterate(noisy, anchors, x0=pts[0])
    err = np.linalg.norm(xs - pts[0], axis=1)
    assert 0.03 < np.median(err) < 0.2  # centimetre ranging gives decimetre positioning with this geometry
    heavy = noisy.copy()
    heavy[:, 1] += 3.0
    wt = np.ones_like(heavy)
    wt[:, 1] = 0.0  # excluding the biased anchor restores accuracy
    assert (
        np.median(np.linalg.norm(multilaterate(heavy, anchors, weights=wt, x0=pts[0])[0] - pts[0], axis=1))
        < 0.2
        < np.median(np.linalg.norm(multilaterate(heavy, anchors, x0=pts[0])[0] - pts[0], axis=1))
    )


def test_uwb_validation_and_registry():
    for bad in (dict(anchors=[]), dict(anchors=[(0, 0)]), dict(noise_std=-1.0), dict(outlier_probability=2.0), dict(max_range=0.0)):
        with pytest.raises(ValueError):
            UwbRanging(**bad)
    from aevion.registry import registries

    for name in ("pitot", "optical_flow", "flow", "ultrasonic", "uwb"):
        assert name in registries.sensors
    assert isinstance(registries.sensors.create("uwb", anchors=ANCHORS), UwbRanging)


def test_sensors_run_in_the_engine():
    from aevion.engine import Simulation

    cfg = {
        "duration": 2.0,
        "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [4.5, 0, 2], "size": [1, 8, 4]}]}},
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2.0]}},
        "sensors": ["pitot", "optical_flow", "ultrasonic", {"type": "uwb", "params": {"anchors": ANCHORS}}],
    }
    sim = Simulation(cfg)
    sim.run()
    assert abs(float(sim.suite["pitot"].latest["ias"][0])) < 2.0
    assert sim.suite["flow"].latest["valid"][0] and sim.suite["flow"].latest["range"][0] == pytest.approx(2.0, abs=0.1)
    assert sim.suite["ultrasonic"].latest["range"][0] == pytest.approx(
        4.0 * 343.0 / 340.3, abs=0.05
    )  # the wall face at x = 4, scaled by the firmware speed-of-sound assumption
    assert sim.suite["uwb"].latest["ranges"].shape == (1, 4)
