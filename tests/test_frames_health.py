import math

import numpy as np
import pytest

from aevion.frames import (
    BODY,
    WORLD,
    ClockModel,
    FrameError,
    Transform,
    TransformTree,
    matrix_to_quat,
    quat_from_rpy,
    quat_mul,
    quat_slerp,
    quat_to_matrix,
    quat_to_rpy,
)


def rand_transform(rng):
    q = rng.normal(size=4)
    return Transform(rng.normal(size=3) * 3, q)


def test_quaternion_matrix_roundtrip_for_random_rotations():
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        q2 = matrix_to_quat(R)
        assert np.allclose(q2, q, atol=1e-9) or np.allclose(q2, -q, atol=1e-9)


def test_matrix_to_quat_handles_all_shepperd_branches():
    for axis in np.eye(3):
        R = quat_to_matrix(np.concatenate([[math.cos(1.5)], axis * math.sin(1.5)]))  # 172 degrees: trace < 0
        q = matrix_to_quat(R)
        np.testing.assert_allclose(quat_to_matrix(q), R, atol=1e-12)


def test_rpy_roundtrip_and_composition():
    rpy = np.array([0.3, -0.4, 1.2])
    np.testing.assert_allclose(quat_to_rpy(quat_from_rpy(*rpy)), rpy, atol=1e-12)
    a, b = quat_from_rpy(0.2, 0, 0), quat_from_rpy(0, 0.3, 0)
    np.testing.assert_allclose(quat_to_matrix(quat_mul(a, b)), quat_to_matrix(a) @ quat_to_matrix(b), atol=1e-12)


def test_transform_inverse_compose_apply():
    rng = np.random.default_rng(1)
    for _ in range(50):
        T, U = rand_transform(rng), rand_transform(rng)
        assert (T @ T.inverse()).isclose(Transform.identity(), atol=1e-9)
        assert (T.inverse() @ T).isclose(Transform.identity(), atol=1e-9)
        p = rng.normal(size=(5, 3))
        np.testing.assert_allclose((T @ U).apply(p), T.apply(U.apply(p)), atol=1e-12)
        np.testing.assert_allclose(T.inverse().apply(T.apply(p)), p, atol=1e-12)
        np.testing.assert_allclose(Transform.from_matrix(T.matrix()).matrix(), T.matrix(), atol=1e-12)


def test_slerp_endpoints_and_midpoint():
    a, b = quat_from_rpy(0, 0, 0), quat_from_rpy(0, 0, math.pi / 2)
    np.testing.assert_allclose(quat_slerp(a, b, 0.0), a, atol=1e-12)
    np.testing.assert_allclose(quat_slerp(a, b, 1.0), b, atol=1e-12)
    np.testing.assert_allclose(quat_to_rpy(quat_slerp(a, b, 0.5)), [0, 0, math.pi / 4], atol=1e-12)
    assert np.isclose(np.linalg.norm(quat_slerp(a, -b, 0.3)), 1.0)  # shortest path through sign flip


def test_transform_tree_lookup_chain_and_dynamic_frames():
    tree = TransformTree()
    tree.add(BODY, WORLD, provider=lambda t, v: Transform([t, v, 0.0], quat_from_rpy(0, 0, math.pi / 2)))
    tree.add("camera", BODY, Transform.from_rpy([0, 0, 0], [1.0, 0, 0]))
    tree.add("lens", "camera", Transform.from_rpy([0, 0, 0], [0, 0, 0.5]))
    # camera origin in the world at t=2, vehicle 3: body at (2,3,0) facing +y, camera 1 m forward -> (2, 4, 0)
    np.testing.assert_allclose(tree.lookup(WORLD, "camera", t=2.0, vehicle=3).t, [2, 4, 0], atol=1e-12)
    np.testing.assert_allclose(tree.transform_points([0, 0, 0], WORLD, "lens", t=2.0, vehicle=3), [2, 4, 0.5], atol=1e-12)
    # sibling lookups and identity
    tree.add("imu", BODY, Transform([0, 0.2, 0]))
    T = tree.lookup("camera", "imu")
    np.testing.assert_allclose(T.t, [-1.0, 0.2, 0.0], atol=1e-12)
    assert tree.lookup("camera", "camera").isclose(Transform.identity())
    assert tree.is_static("camera") and not tree.is_static(BODY)
    assert "camera [static]" in tree.describe() and "body [dynamic]" in tree.describe()


def test_transform_tree_errors():
    tree = TransformTree()
    tree.add("a", WORLD, Transform())
    with pytest.raises(FrameError, match="unknown parent"):
        tree.add("b", "ghost", Transform())
    with pytest.raises(FrameError, match="cycle"):
        tree.add("a", "a", Transform())
    tree.add("b", "a", Transform())
    with pytest.raises(FrameError, match="cycle"):
        tree.add("a", "b", Transform())
    with pytest.raises(FrameError, match="unknown frame"):
        tree.lookup("a", "nope")
    with pytest.raises(FrameError, match="exactly one"):
        tree.add("c", "a")
    with pytest.raises(FrameError, match="root"):
        tree.add(WORLD, "a", Transform())


def test_tree_static_serialisation_is_order_independent():
    tree = TransformTree()
    tree.add("a", WORLD, Transform.from_rpy([10, 20, 30], [1, 2, 3], degrees=True))
    tree.add("b", "a", Transform([0, 0, 1]))
    d = tree.static_dict()
    rev = dict(reversed(list(d.items())))
    again = TransformTree.from_static_dict(rev)
    assert again.lookup(WORLD, "b").isclose(tree.lookup(WORLD, "b"))


def test_clock_model():
    c = ClockModel(offset_s=0.5, drift_ppm=100.0)
    assert c.stamp(10.0) == pytest.approx(10.0 * (1 + 1e-4) + 0.5)
    assert c.true_time(c.stamp(7.3)) == pytest.approx(7.3)
    assert c.error_at(10.0) == pytest.approx(0.5 + 1e-3)
    j = ClockModel(jitter_std_s=0.01, seed=3)
    s = np.array([j.stamp(1.0) for _ in range(2000)])
    assert s.std() == pytest.approx(0.01, rel=0.1)
    j.reset()
    assert j.stamp(1.0) == s[0]


def test_sensor_extrinsics_rotate_the_scan_pattern():
    from aevion.sensors import Lidar

    ext = Transform.from_rpy([0, 0, 90], [0.1, 0, 0.2], degrees=True)  # lidar yawed 90 deg and offset
    lid = Lidar(channels=1, samples=4, v_fov=(0, 0), h_fov=360.0, extrinsics=ext)
    base = Lidar(channels=1, samples=4, v_fov=(0, 0), h_fov=360.0)
    np.testing.assert_allclose(lid.dirs_body, base.dirs_body @ ext.rotation.T, atol=1e-12)
    np.testing.assert_allclose(lid.mount, [0.1, 0, 0.2])
    assert lid.to_dict()["extrinsics"]["translation"] == [0.1, 0.0, 0.2]


def test_sensor_clock_stamps_differ_from_true_time():
    from aevion import Multirotor
    from aevion.sensors import IMU

    sim = Multirotor("generic_1kg", n=1)
    imu = IMU(rate=100.0, clock={"offset_s": 0.25})
    imu.bind(sim)
    s = sim.reset()
    for _ in range(10):
        s = sim.step(s, sim.hover_command())
        imu.update(s)
    assert imu.stamp_reported == pytest.approx(imu.stamp + 0.25)


# ------------------------------------------------------------------------------------------ health
def _state(sim=None, **over):
    from aevion import Multirotor

    sim = sim or Multirotor("generic_1kg", n=3)
    s = sim.reset(pos=[0, 0, 2.0])
    for k, v in over.items():
        getattr(s, k)[...] = v
    return sim, s


def _ctx(sim):
    from aevion.health import HealthContext

    return HealthContext(sim.B, sim.spec, sim.dt)


def test_finite_check_identifies_vehicle_and_field():
    from aevion.health import FiniteCheck

    sim, s = _state()
    assert FiniteCheck().check(s, _ctx(sim)) == []
    s.vel[2, 1] = np.nan
    v = FiniteCheck().check(s, _ctx(sim))
    assert len(v) == 1 and v[0].vehicle == 2 and v[0].quantity == "vel" and v[0].hard
    s.vel[2, 1] = np.inf
    assert FiniteCheck().check(s, _ctx(sim))[0].quantity == "vel"


def test_physical_limit_checks():
    from aevion.health import AltitudeCheck, QuaternionNormCheck, RateCheck, RotorSpeedCheck, SpeedCheck

    sim, s = _state()
    ctx = _ctx(sim)
    s.vel[1] = [300, 0, 0]
    assert SpeedCheck(200).check(s, ctx)[0].vehicle == 1
    s.omega[0] = [0, 0, 900]
    assert RateCheck(500).check(s, ctx)[0].vehicle == 0
    s.pos[2, 2] = 50000
    assert AltitudeCheck(20000).check(s, ctx)[0].vehicle == 2
    s.quat[0] *= 1.001
    assert QuaternionNormCheck(1e-6).check(s, ctx)[0].vehicle == 0
    s.rotor[1, 0] = 5000
    assert RotorSpeedCheck().check(s, ctx)[0].vehicle == 1


def test_soft_checks_warn_once_and_never_raise():
    from aevion.health import BatteryCheck, HealthMonitor

    sim, s = _state()
    s.soc[1] = 0.0
    mon = HealthMonitor([BatteryCheck()], policy="raise")
    with pytest.warns(UserWarning, match="battery depleted"):
        found = mon.check(s, _ctx(sim), 1, 0.1)
    assert found and not found[0].hard
    assert mon.check(s, _ctx(sim), 2, 0.2) == []  # reported once per vehicle


def test_monitor_policies_and_last_good_state():
    from aevion.errors import SimulationDiverged
    from aevion.health import FiniteCheck, HealthMonitor

    sim, s = _state()
    ctx = _ctx(sim)
    mon = HealthMonitor([FiniteCheck()], policy="raise")
    mon.check(s, ctx, 1, 0.005)
    good_pos = mon.last_good.pos.copy()
    s.pos[0, 0] = np.nan
    with pytest.raises(SimulationDiverged) as e:
        mon.check(s, ctx, 2, 0.01)
    assert e.value.step == 2 and e.value.vehicle == 0 and e.value.quantity == "pos"
    np.testing.assert_array_equal(e.value.last_good.pos, good_pos)  # snapshot is the pre-failure state
    assert "step 2" in str(e.value)
    mon2 = HealthMonitor([FiniteCheck()], policy="stop")
    mon2.check(s, ctx, 3, 0.015)
    assert mon2.should_stop
    mon3 = HealthMonitor([FiniteCheck()], policy="warn")
    with pytest.warns(UserWarning, match="diverged"):
        mon3.check(s, ctx, 4, 0.02)
    assert HealthMonitor(every=5).check(s, ctx, 3, 0.0) == []  # skipped between check steps
    with pytest.raises(ValueError):
        HealthMonitor(policy="explode")


def test_engine_stops_cleanly_on_divergence_or_raises():
    from aevion.config import SimulationConfig
    from aevion.engine import Simulation
    from aevion.errors import SimulationDiverged

    cfg = {"duration": 3.0, "health": {"on_diverge": "stop", "max_speed": 0.5}, "trajectory": {"type": "circle", "radius": 2.0, "omega": 1.5}}
    sim = Simulation(SimulationConfig.from_dict(cfg))
    arch = sim.run()
    assert sim.stop_reason and "speed" in sim.stop_reason and arch.t[-1] < 3.0
    cfg["health"]["on_diverge"] = "raise"
    with pytest.raises(SimulationDiverged) as e:
        Simulation(SimulationConfig.from_dict(cfg)).run()
    assert e.value.last_good is not None
