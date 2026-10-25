import math

import numpy as np
import pytest

from aevion import Multirotor, World
from aevion.core.constants import G0
from aevion.sensors import GPS, IMU, Barometer, Camera, Lidar, Magnetometer, RangeFinder, SensorSuite
from aevion.sensors.raycast import cast_rays, ray_aabb_brute


def hover_run(sensor, steps=2000, n=64, **kw):
    sim = Multirotor("generic_1kg", n=n, dt=0.005, **kw)
    sensor.bind(sim, seed=3)
    s = sim.reset(pos=[0, 0, 3.0])
    out = []
    for _ in range(steps):
        s = sim.step(s, sim.hover_command())
        if sensor.update(s):
            out.append(sensor.latest)
    return sim, s, out


def test_sensor_rate_and_latency():
    imu = IMU(rate=50.0, latency=0.1)
    sim = Multirotor("generic_1kg", n=1, dt=0.005)
    imu.bind(sim)
    s = sim.reset()
    stamps = []
    for _ in range(400):
        s = sim.step(s, sim.hover_command())
        if imu.update(s):
            stamps.append((s.t, imu.stamp))
    assert abs(len(stamps) - 98) <= 4  # 2 s at 50 Hz minus 0.1 s latency
    assert all(abs((t - st) - 0.1) < 0.006 for t, st in stamps)


def test_imu_ideal_hover_reads_gravity():
    _, _, out = hover_run(IMU(ideal=True), steps=50, n=2)
    np.testing.assert_allclose(out[-1]["accel"][0], [0, 0, G0], atol=0.01)
    np.testing.assert_allclose(out[-1]["gyro"][0], 0.0, atol=1e-9)


def test_imu_noise_density_matches_config():
    imu = IMU(rate=200.0, accel_bias_init=0.0, gyro_bias_init=0.0, accel_bias_walk=0.0, gyro_bias_walk=0.0, scale_factor_std=0.0)
    _, _, out = hover_run(imu, steps=6000, n=32)
    a = np.stack([o["accel"] for o in out])[20:]
    assert a[..., 0].std() == pytest.approx(2.0e-3 / math.sqrt(1 / 200.0), rel=0.08)
    g = np.stack([o["gyro"] for o in out])[20:]
    assert g[..., 1].std() == pytest.approx(2.4e-4 / math.sqrt(1 / 200.0), rel=0.08)


def test_imu_bias_random_walk_grows():
    imu = IMU(rate=100.0, accel_bias_init=0.0, accel_bias_walk=1e-2, accel_noise_density=0.0)
    _, _, out = hover_run(imu, steps=4000, n=200)  # 20 s
    a = np.stack([o["accel"] for o in out])[..., 0]
    assert a[-1].var() == pytest.approx((1e-2) ** 2 * 20.0, rel=0.35)


def test_gps_is_noisy_but_unbiased_and_delayed():
    gps = GPS(rate=10.0, latency=0.1, sigma_h=0.8, tau=5.0)
    _, s, out = hover_run(gps, steps=4000, n=100)
    err = np.stack([o["pos"] for o in out]) - np.array([0, 0, 3.0])
    assert abs(err[..., :2].mean()) < 0.25
    assert 0.4 < err[..., 0].std() < 1.5
    assert out[-1]["t"] <= s.t - 0.09


def test_barometer_altitude():
    _, _, out = hover_run(Barometer(ideal=True), steps=50, n=1)
    assert out[-1]["altitude"][0] == pytest.approx(3.0, abs=0.05)
    _, _, out = hover_run(Barometer(noise_pa=6.0, drift_pa=0.0), steps=2000, n=100)
    alt = np.stack([o["altitude"] for o in out])
    assert alt.mean() == pytest.approx(3.0, abs=0.1)
    assert 0.3 < alt.std() < 1.3  # 6 Pa is about 0.5 m


def test_magnetometer_field_and_heading():
    sim = Multirotor("generic_1kg", n=1)
    mag = Magnetometer(ideal=True)
    mag.bind(sim)
    mag.update(sim.reset(yaw=0.0))
    m0 = mag.latest["field"][0]
    assert np.linalg.norm(m0) == pytest.approx(50.0, rel=1e-9)
    assert m0[1] == pytest.approx(50 * math.cos(math.radians(60)), rel=1e-9)  # facing east: north is to the left (+y)
    mag.bind(sim)  # fresh sampling schedule
    mag.update(sim.reset(yaw=math.pi / 2))  # facing north: field is straight ahead (+x)
    m1 = mag.latest["field"][0]
    assert m1[0] == pytest.approx(50 * math.cos(math.radians(60)), rel=1e-9) and abs(m1[1]) < 1e-9


def test_raycast_matches_reference_for_boxes():
    w = World()
    rng = np.random.default_rng(0)
    for _ in range(12):
        c = rng.uniform(-10, 10, 3)
        c[2] = abs(c[2]) + 1
        w.add_box(c, rng.uniform(0.5, 3, 3))
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.array([0.0, 0.0, 2.5])
    h = cast_rays(w, o, d, 100.0)
    for i in range(len(d)):
        ref = min((ray_aabb_brute(o, d[i], b[:3], b[3:]) for b in w.boxes), default=np.inf)
        if d[i, 2] < 0:
            ref = min(ref, -o[2] / d[i, 2])
        assert h.t[i] == pytest.approx(min(ref, 100.0), abs=1e-6)


def test_raycast_analytic_primitives():
    w = World()
    w.add_sphere((10, 0, 2), 2.0)
    w.add_cylinder((0, 10), 1.0, 0.0, 5.0)
    w.add_box((-10, 0, 2), (2, 2, 4))
    o = np.array([0.0, 0.0, 2.0])
    dirs = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    h = cast_rays(w, o, dirs, 100.0)
    np.testing.assert_allclose(h.t, [8.0, 9.0, 9.0, 100.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(h.normal[0], [-1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(h.normal[4], [0, 0, 1], atol=1e-9)
    assert list(h.prim) == [1, 2, 0, -1, -2]


def test_raycast_heightmap_slope():
    w = World()
    x = np.arange(0, 41.0)
    w.set_heightmap(np.tile(0.5 * x, (41, 1)), origin=(0, 0), resolution=1.0)  # z = 0.5 x
    h = cast_rays(w, np.array([2.0, 5.0, 10.0]), np.array([[0.0, 0.0, -1.0]]), 50.0)
    assert h.t[0] == pytest.approx(9.0, abs=1e-3)  # terrain height under x=2 is 1.0
    assert h.normal[0][0] < 0 and h.normal[0][2] > 0.8


def test_lidar_ranges_and_noise():
    w = World()
    w.add_box((10, 0, 1), (2, 40, 6))  # wall face at x=9
    sim = Multirotor("generic_1kg", n=1, world=w)
    lid = Lidar(channels=1, samples=3, v_fov=(0, 0), h_fov=2.0, noise_std=0.0, noise_rel=0.0, dropout_prob=0.0, mount=(0, 0, 0))
    lid.bind(sim)
    lid.update(sim.reset(pos=[0, 0, 1.0]))
    assert lid.latest["ranges"][0, 0, 1] == pytest.approx(9.0, abs=1e-6)
    noisy = Lidar(channels=1, samples=3, v_fov=(0, 0), h_fov=2.0, noise_std=0.05, noise_rel=0.0, dropout_prob=0.0, mount=(0, 0, 0))
    sim2 = Multirotor("generic_1kg", n=500, world=w)
    noisy.bind(sim2)
    noisy.update(sim2.reset(pos=[0, 0, 1.0]))
    assert noisy.latest["ranges"][:, 0, 1].std() == pytest.approx(0.05, rel=0.15)


def test_depth_camera_planar_wall():
    w = World()
    w.add_box((6, 0, 1), (2, 40, 6))  # wall face at x=5
    sim = Multirotor("generic_1kg", n=1, world=w)
    cam = Camera(width=32, height=24, modes=("depth", "rgb", "seg"), depth_noise=(0, 0), pixel_noise=0.0, mount=(0, 0, 0))
    cam.bind(sim)
    cam.update(sim.reset(pos=[0, 0, 1.0]))
    assert np.allclose(cam.latest["depth"][0][10:14, 14:18], 5.0, atol=1e-6)  # z-depth to a fronto-parallel wall is constant
    assert cam.latest["rgb"].shape == (1, 24, 32, 3) and cam.latest["rgb"].dtype == np.uint8
    assert (cam.latest["seg"][0] >= 0).any()


def test_rangefinder_height_above_ground():
    sim = Multirotor("generic_1kg", n=1)
    rf = RangeFinder(noise_std=0.0, noise_rel=0.0)
    rf.bind(sim)
    rf.update(sim.reset(pos=[0, 0, 2.5]))
    assert rf.latest["range"][0] == pytest.approx(2.5, abs=1e-9) and rf.latest["valid"][0]


def test_sensor_suite_dropout():
    sim = Multirotor("generic_1kg", n=1)
    suite = SensorSuite(IMU(rate=100, dropout=0.5), GPS(rate=10)).bind(sim, 1)
    s = sim.reset()
    n_imu = 0
    for _ in range(1000):
        s = sim.step(s, sim.hover_command())
        suite.update(s)
        n_imu += suite["imu"].fresh
    assert 0.35 < n_imu / 500 < 0.65
