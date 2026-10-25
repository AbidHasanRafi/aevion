"""The error-state Kalman filter - kinematics, Jacobians, statistical consistency, delayed fusion, observability."""

import math

import numpy as np
import pytest
from eskf_support import M_REF, G, monte_carlo

from aevion.core import rotation as rot
from aevion.core.backend import NumpyBackend
from aevion.estimation.consistency import consistency_report, nees_interval
from aevion.estimation.eskf import ESKF, STATE_DIM
from aevion.estimation.observability import direction_observability, observability_matrix, observable_rank, unobservable_subspace

B = NumpyBackend()


def fresh(n=1, **kw):
    f = ESKF(B, n, **kw)
    f.reset()
    return f


# ------------------------------------------------------------------------------------------------ kinematics
def test_strapdown_propagation_matches_the_closed_form_of_a_banked_turn():
    """Constant body rate about z and constant lateral specific force: v, p and the heading have closed forms."""
    f = fresh()
    fx, om, T, dt = 1.0, 0.5, 5.0, 0.01
    for _ in range(int(T / dt)):
        f.predict(np.array([[0.0, 0.0, om]]), np.array([[fx, 0.0, G]]), dt)
    t = T
    v = np.array([fx / om * math.sin(om * t), fx / om * (1 - math.cos(om * t)), 0.0])
    p = np.array([fx / om**2 * (1 - math.cos(om * t)), fx / om**2 * (om * t - math.sin(om * t)), 0.0])
    assert np.allclose(f.v[0], v, atol=2e-4) and np.allclose(f.p[0], p, atol=2e-3)
    assert rot.to_euler(B, f.q)[0, 2] == pytest.approx(om * t, abs=1e-9)


def test_error_transition_matches_finite_differences_of_the_nominal_propagation():
    """F must be the Jacobian of (nominal ⊕ error) -> next error, for the body-frame attitude error used here."""
    rng = np.random.default_rng(1)
    dt = 0.005
    gyro, accel = np.array([[0.3, -0.2, 0.4]]), np.array([[1.0, -2.0, 9.5]])
    base = ESKF(B, 1)
    base.reset([0, 0, 5.0], [1.0, 0.5, 0.1], rot.from_euler(B, np.array([0.2]), np.array([-0.3]), np.array([0.8])))
    base.ba, base.bg = np.array([[0.02, -0.01, 0.03]]), np.array([[0.004, 0.002, -0.003]])
    R = rot.to_matrix(B, base.q)
    F = base.transition(R, accel - base.ba, gyro - base.bg, dt)[0]
    eps = 1e-6
    dx = rng.normal(size=STATE_DIM) * eps
    pert = ESKF(B, 1)
    pert.reset(base.p + dx[0:3], base.v + dx[3:6], rot.qmul(B, base.q, rot.from_rotvec(B, dx[None, 6:9])))
    pert.ba, pert.bg = base.ba + dx[9:12], base.bg + dx[12:15]
    base.predict(gyro, accel, dt)
    pert.predict(gyro, accel, dt)
    got = base.error_to(pert.p, pert.v, pert.q, pert.ba, pert.bg)[0]
    want = F @ dx
    assert np.linalg.norm(got - want) < 2e-3 * np.linalg.norm(dx) + 1e-12  # the discrete map equals F to O(dt^2)
    assert np.linalg.norm(got - dx) > 1e-3 * np.linalg.norm(dx)  # ... and the test is not trivially satisfied by F = I


def test_attitude_from_accelerometer_recovers_roll_and_pitch():
    f = fresh()
    q = rot.from_euler(B, np.array([0.3]), np.array([-0.2]), np.array([0.0]))
    spec = rot.rotate_inv(B, q, np.array([[0.0, 0.0, G]]))  # gravity reaction seen in the body
    f.attitude_from_accel(spec)
    assert rot.angle_between(B, f.q, q)[0] < 1e-9


# ------------------------------------------------------------------------------------------------ consistency
@pytest.fixture(scope="module")
def mc_all():
    return monte_carlo(n=64, duration=10.0, sensors=("gnss", "baro", "mag", "range", "flow"), seed=3)


def test_monte_carlo_nees_and_nis_are_consistent_with_every_aiding_sensor(mc_all):
    """The reported covariance equals the actual error: NEES ~ chi2(9), and each NIS ~ chi2(dim of the measurement)."""
    r, f = mc_all, mc_all["filter"]
    late = r["t"] > 2.0
    rep = consistency_report(r["nees"][late], 9)
    assert 0.85 < rep["ratio"] < 1.25, rep  # the ESKF is a first-order filter: a few percent over-confidence is expected, not tens
    for part in ("nees_p", "nees_v", "nees_th"):
        assert 0.8 < r[part][late].mean() / 3 < 1.3, part
    dims = dict(gnss=6, baro=1, mag=1, range=1, flow=2)
    for kind, dof in dims.items():
        mean = float(np.concatenate(f.nis_log[kind]).mean())
        assert 0.85 * dof < mean < 1.2 * dof, (kind, mean)
    assert f.rejected.get("gnss", 0) <= 2 and f.rejected.get("flow", 0) <= 5  # the 0.9999 gate rarely fires on honest data


def test_pose_updates_are_consistent_too():
    r = monte_carlo(n=48, duration=8.0, sensors=("gnss", "pose"), seed=5)
    f = r["filter"]
    late = r["t"] > 2.0
    assert 0.8 < consistency_report(r["nees"][late], 9)["ratio"] < 1.3
    assert 0.85 * 6 < float(np.concatenate(f.nis_log["pose"]).mean()) < 1.2 * 6


def test_the_acceptance_interval_shrinks_with_the_number_of_runs():
    lo1, hi1 = nees_interval(9, 1)
    lo100, hi100 = nees_interval(9, 100)
    assert lo1 < lo100 < 9.0 < hi100 < hi1 and hi100 - lo100 < 0.3 * (hi1 - lo1)
    assert nees_interval(9, 1)[0] == pytest.approx(2.700, abs=1e-3) and nees_interval(9, 1)[1] == pytest.approx(19.023, abs=1e-3)  # tables


def test_covariance_stays_symmetric_and_positive_definite_through_a_long_run():
    r = monte_carlo(n=8, duration=15.0, sensors=("gnss", "baro", "mag"), seed=2)
    P = r["filter"].P
    assert np.allclose(P, P.transpose(0, 2, 1), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(P) > 0.0)
    assert np.all(r["filter"].std()[:, 0:3] < 1.0)  # position uncertainty shrank below its 1 m prior


# ------------------------------------------------------------------------------------------------ delayed measurements
def test_delayed_measurements_give_the_same_estimate_as_on_time_fusion():
    """A fix stamped t_m but delivered 0.2 s late is fused at t_m (rewind and replay) - identical to receiving it on time."""
    on_time = monte_carlo(n=6, duration=4.0, sensors=("gnss", "baro", "mag"), seed=9, flush=True)
    late = monte_carlo(n=6, duration=4.0, sensors=("gnss", "baro", "mag"), seed=9, delay_s=0.2, flush=True)
    a, b = on_time["filter"], late["filter"]
    assert np.allclose(a.p, b.p, atol=1e-9) and np.allclose(a.v, b.v, atol=1e-9)
    assert np.allclose(a.q, b.q, atol=1e-9) and np.allclose(a.P, b.P, atol=1e-9)
    assert b.rejected.get("too_old", 0) == 0


def test_ignoring_the_delay_is_measurably_worse():
    """Fusing a 0.3 s old fix as if it were current biases a moving vehicle - the replay is what removes that."""
    n = 40
    kw = dict(n=n, duration=8.0, sensors=("gnss",), seed=4, delay_s=0.3, flush=True)
    good = monte_carlo(**kw)
    # same data, stamps dropped (fused "now"): build by wrapping the filter so t is ignored
    f2 = ESKF(B, n, max_delay_s=0.5)
    orig = f2.update_gnss
    f2.update_gnss = lambda *a, **k: orig(*a, **{**k, "t": None})
    naive = monte_carlo(filt=f2, **kw)
    e_good = float(np.mean(np.linalg.norm(good["filter"].p - good["truth"].p, axis=1)))
    e_naive = float(np.mean(np.linalg.norm(naive["filter"].p - naive["truth"].p, axis=1)))
    assert e_naive > 1.1 * e_good


def test_a_measurement_older_than_the_history_is_refused_not_misapplied():
    f = ESKF(B, 1, max_delay_s=0.1)
    f.reset()
    for k in range(50):
        f.predict(np.zeros((1, 3)), np.array([[0.0, 0.0, G]]), 0.01, 0.01 * (k + 1))
    before = f.p.copy()
    f.update_position(np.array([[100.0, 0.0, 0.0]]), 1.0, t=0.05)
    assert f.rejected["too_old"] == 1 and np.array_equal(f.p, before)


# ------------------------------------------------------------------------------------------------ gating and the individual updates
def test_gross_outliers_are_gated_out():
    f = fresh()
    f.reset([0, 0, 10.0], [0, 0, 0.0], None)
    f.predict(np.zeros((1, 3)), np.array([[0.0, 0.0, G]]), 0.01, 0.01)
    f.update_gnss(np.array([[500.0, 0.0, 10.0]]), np.zeros((1, 3)), 1.0, 0.15, t=0.01)
    assert f.rejected["gnss"] == 1 and np.allclose(f.p[0], [0, 0, 10.0], atol=1e-2)
    assert f.nis["gnss"][0] > 1e3
    f.update_gnss(np.array([[1.0, 0.0, 10.0]]), np.zeros((1, 3)), 1.0, 0.15, t=0.01)  # an honest fix is accepted
    assert f.rejected["gnss"] == 1 and f.p[0, 0] > 0.1


def test_gating_can_be_switched_off():
    f = fresh(gate_prob=None)
    f.update_position(np.array([[500.0, 0.0, 0.0]]), 1.0)
    assert (f.p[0, 0] > 400.0 and "position" not in f.rejected) or f.rejected["position"] == 0


def test_range_updates_correct_altitude_and_depend_on_tilt():
    f = fresh()
    f.reset([0, 0, 12.0], [0, 0, 0.0], rot.from_euler(B, np.array([0.3]), np.array([0.0]), np.array([0.0])))
    true_z = 10.0
    for _ in range(40):
        f.update_range(np.array([true_z / math.cos(0.3)]), np.array([True]), 0.05)  # slant range through a 0.3 rad roll
    assert abs(f.p[0, 2] - true_z) < 0.1
    flat = fresh()
    flat.reset([0, 0, 12.0], [0, 0, 0.0], None)
    flat.update_range(np.array([true_z]), np.array([True]), 0.05)
    assert flat.p[0, 2] < 12.0 and abs(flat.p[0, 2] - true_z) < 0.5  # one update already pulls most of the way
    f2 = fresh()
    f2.reset([0, 0, 12.0], [0, 0, 0.0], rot.from_euler(B, np.array([1.45]), np.array([0.0]), np.array([0.0])))
    f2.update_range(np.array([3.0]), np.array([True]), 0.05)  # beam nearly horizontal: ignored
    assert f2.p[0, 2] == pytest.approx(12.0, abs=1e-9)
    f3 = fresh()
    f3.reset([0, 0, 12.0], [0, 0, 0.0], None)
    f3.update_range(np.array([3.0]), np.array([False]), 0.05)  # invalid reading: ignored
    assert f3.p[0, 2] == pytest.approx(12.0, abs=1e-9)


def test_magnetometer_pulls_heading_to_the_reference_even_with_a_steep_field():
    f = fresh()
    f.mag_ref = np.tile(M_REF, (1, 1))
    true_q = rot.from_euler(B, np.array([0.15]), np.array([-0.1]), np.array([0.9]))
    f.reset([0, 0, 5.0], [0, 0, 0.0], rot.from_euler(B, np.array([0.15]), np.array([-0.1]), np.array([0.9 - 0.25])))  # yaw 14 deg off
    f.mag_ref = np.tile(M_REF, (1, 1))
    f.P[:, 6:9, 6:9] = np.diag([0.02**2, 0.02**2, 0.4**2])  # tilt is known from the accelerometer, heading is not
    m_b = rot.rotate_inv(B, true_q, np.tile(M_REF, (1, 1)))
    for _ in range(60):
        f.update_mag_heading(m_b, 2.0)
    err = rot.angle_between(B, f.q, true_q)[0]
    assert err < math.radians(3.0) < 0.25 / 4  # 14 degrees of heading error removed (the rest is tilt/heading ambiguity of a steep field)
    assert abs(rot.to_euler(B, f.q)[0, 0] - 0.15) < 0.04 and abs(rot.to_euler(B, f.q)[0, 1] + 0.1) < 0.04  # tilt barely moved by a heading update


def test_magnetometer_learns_its_reference_from_the_first_reading():
    f = fresh()
    q = rot.from_euler(B, np.array([0.0]), np.array([0.0]), np.array([0.4]))
    f.reset([0, 0, 0.0], [0, 0, 0.0], q)
    m_b = rot.rotate_inv(B, q, np.tile(M_REF, (1, 1)))
    f.update_mag_heading(m_b, 3.0)
    assert np.allclose(f.mag_ref[0], M_REF) and f.nis == {}  # the first reading defines the reference and is not an update
    f.update_mag_heading(m_b, 3.0)
    assert f.nis["mag"][0] < 1e-6


def test_flow_updates_remove_a_horizontal_velocity_error():
    f = fresh()
    q = rot.from_euler(B, np.array([0.1]), np.array([-0.05]), np.array([0.7]))
    v_true = np.array([[2.0, -1.0, 0.0]])
    f.reset([0, 0, 8.0], [0.0, 0.0, 0.0], q)
    f.P[:, 3:6, 3:6] = np.eye(3) * 4.0  # a wide velocity prior so the first updates are accepted
    rho = 8.0 / rot.to_matrix(B, q)[0, 2, 2]
    vb = rot.rotate_inv(B, q, v_true)
    flow = np.array([[-vb[0, 1], vb[0, 0]]]) / rho
    for _ in range(50):
        f.update_flow(flow, np.array([rho]), None, 0.01)
    assert np.allclose(f.v[0, 0:2], v_true[0, 0:2], atol=0.3)


def test_pose_update_pulls_position_and_attitude():
    f = fresh(init_pos_std=5.0, init_att_std_deg=30.0)
    q_true = rot.from_euler(B, np.array([0.1]), np.array([0.2]), np.array([-0.5]))
    for _ in range(30):
        f.update_pose(np.array([[3.0, -2.0, 1.0]]), q_true, 0.05, 1.0)
    assert np.allclose(f.p[0], [3.0, -2.0, 1.0], atol=0.05) and rot.angle_between(B, f.q, q_true)[0] < math.radians(0.3)


def test_unusable_fixes_do_not_disturb_the_filter():
    f = fresh()
    f.reset([0, 0, 5.0], [0, 0, 0.0], None)
    f.update_gnss(np.array([[np.nan, 0.0, 5.0]]), np.zeros((1, 3)), 1.0, 0.15)
    assert np.all(np.isfinite(f.p)) and np.all(np.isfinite(f.P)) and f.rejected["gnss"] == 1


# ------------------------------------------------------------------------------------------------ observability
def _linear_system(with_mag, maneuver, seconds=6.0, dt=0.1):
    """Linearised ESKF along a path: hover (constant specific force) or a manoeuvre. Returns ``Phi_k`` and ``H_k`` lists."""
    f = ESKF(B, 1)
    f.reset([0, 0, 5.0], [0, 0, 0.0], rot.from_euler(B, np.array([0.05]), np.array([0.03]), np.array([0.4])))
    Hg = np.zeros((6, STATE_DIM))
    Hg[0:3, 0:3] = np.eye(3)
    Hg[3:6, 3:6] = np.eye(3)

    def H_now():
        H = Hg
        if with_mag:
            hxy2 = M_REF[0] ** 2 + M_REF[1] ** 2
            hw = np.array([-M_REF[2] * M_REF[0] / hxy2, -M_REF[2] * M_REF[1] / hxy2, 1.0])
            Hm = np.zeros((1, STATE_DIM))
            Hm[0, 6:9] = hw @ rot.to_matrix(B, f.q)[0]
            H = np.vstack([Hg, Hm])
        return H

    Phis, Hs, t = [], [H_now()], 0.0
    for _ in range(int(seconds / dt)):
        if maneuver:
            gyro = np.array([[0.4 * math.sin(0.9 * t), 0.3 * math.cos(0.7 * t), 0.2]])
            accel = np.array([[1.5 * math.sin(1.1 * t), 1.0 * math.cos(0.8 * t), G]])
        else:
            gyro, accel = np.zeros((1, 3)), np.array([[0.0, 0.0, G]])
        Phis.append(f.transition(rot.to_matrix(B, f.q), accel - f.ba, gyro - f.bg, dt)[0])
        f.predict(gyro, accel, dt, t + dt)
        t += dt
        Hs.append(H_now())
    return Phis, Hs


SCALE = np.array([1.0] * 6 + [0.05] * 3 + [0.05] * 3 + [0.005] * 3)  # typical size of each error state: m, m/s, rad, m/s^2, rad/s
Q_START = rot.from_euler(B, np.array([0.05]), np.array([0.03]), np.array([0.4]))


def _up_in_body():
    d = np.zeros(STATE_DIM)
    d[6:9] = rot.rotate_inv(B, Q_START, np.array([[0.0, 0.0, 1.0]]))[0]  # a rotation about the world vertical = heading error
    return d


def test_heading_is_unobservable_in_hover_without_a_magnetometer_and_observable_with_one():
    O_nomag = observability_matrix(*_linear_system(with_mag=False, maneuver=False))
    O_mag = observability_matrix(*_linear_system(with_mag=True, maneuver=False))
    assert observable_rank(O_nomag, SCALE) < observable_rank(O_mag, SCALE)
    assert unobservable_subspace(O_nomag, SCALE).shape[1] >= 1
    assert direction_observability(O_nomag, _up_in_body(), SCALE) < 0.05
    assert direction_observability(O_mag, _up_in_body(), SCALE) > 0.9


def test_horizontal_acceleration_makes_heading_observable_without_a_magnetometer():
    """The classic INS result: with position/velocity aiding, heading is observable only while the vehicle accelerates."""
    O_hover = observability_matrix(*_linear_system(with_mag=False, maneuver=False))
    O_move = observability_matrix(*_linear_system(with_mag=False, maneuver=True))
    assert direction_observability(O_hover, _up_in_body(), SCALE) < 0.05
    assert direction_observability(O_move, _up_in_body(), SCALE) > 0.9
    assert observable_rank(O_move, SCALE) > observable_rank(O_hover, SCALE)


def test_position_and_velocity_are_observable_from_gnss():
    O = observability_matrix(*_linear_system(with_mag=False, maneuver=False))
    for i in range(6):
        d = np.zeros(STATE_DIM)
        d[i] = 1.0
        assert direction_observability(O, d, SCALE) > 0.99


# ------------------------------------------------------------------------------------------------ backends
def test_the_filter_runs_on_the_torch_backend_and_agrees_with_numpy():
    torch = pytest.importorskip("torch")
    from aevion.core.backend import TorchBackend

    Bt = TorchBackend(dtype=torch.float64)
    fn, ft = ESKF(B, 3), ESKF(Bt, 3)
    q0 = rot.from_euler(B, np.array([0.1, 0.0, -0.1]), np.array([0.0, 0.05, 0.0]), np.array([0.3, 0.2, 0.1]))
    fn.reset([0, 0, 5.0], [1.0, 0.0, 0.0], q0)
    ft.reset(Bt.asarray(fn.p), Bt.asarray(fn.v), Bt.asarray(q0))
    gyro, accel = np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.2, G])
    for k in range(50):
        fn.predict(np.tile(gyro, (3, 1)), np.tile(accel, (3, 1)), 0.01, 0.01 * (k + 1))
        ft.predict(Bt.asarray(np.tile(gyro, (3, 1))), Bt.asarray(np.tile(accel, (3, 1))), 0.01, 0.01 * (k + 1))
        if k % 10 == 0:
            fn.update_gnss(np.array([[0.1, 0, 5.0]] * 3), np.array([[1.0, 0, 0]] * 3), 1.0, 0.15, t=0.01 * (k + 1))
            ft.update_gnss(Bt.asarray(np.array([[0.1, 0, 5.0]] * 3)), Bt.asarray(np.array([[1.0, 0, 0]] * 3)), 1.0, 0.15, t=0.01 * (k + 1))
    assert np.allclose(fn.p, Bt.to_numpy(ft.p), atol=1e-8) and np.allclose(fn.P, Bt.to_numpy(ft.P), atol=1e-8)


# ------------------------------------------------------------------------------------------------ in the engine
def _engine_run(estimator, sensors, faults=(), duration=14.0, seed=1):
    from aevion.engine import Simulation

    cfg = {
        "duration": duration,
        "seed": seed,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 6.0], "radius": 4.0, "omega": 0.6}},
        "sensors": sensors,
        "estimator": estimator,
        "faults": list(faults),
    }
    return Simulation(cfg).run()


def test_the_eskf_flies_the_vehicle_better_than_the_complementary_filter_and_reports_its_uncertainty():
    sensors = ["imu", "mag", "baro", {"type": "gps", "params": {"latency": 0.15}}]
    a = _engine_run({"type": "eskf", "handover_s": 3.0}, sensors)
    b = _engine_run({"type": "mahony_kf", "handover_s": 3.0}, sensors)
    late = a.t > 4.0
    att = lambda x: np.degrees(rot.angle_between(B, x["est_quat"][:, 0], x["quat"][:, 0]))[late].mean()
    pos = lambda x: np.linalg.norm(x["est_pos"][:, 0] - x["pos"][:, 0], axis=1)[late].mean()
    assert att(a) < 3.0 and att(a) < 0.5 * att(b)  # attitude: a full filter vs tilt-from-GPS-acceleration heuristics
    assert pos(a) < 1.2 and pos(a) <= pos(b) * 1.05
    track = np.linalg.norm(a["pos"][:, 0] - a["ref_pos"][:, 0], axis=1)[late].mean()
    assert track < 1.0  # control on the estimate stays tight
    std = a["est_pos_std"][:, 0]
    assert std.shape[1] == 3 and np.all(std[-1] < std[0]) and np.all(std[-1] < 0.5)  # the recorded 1-sigma shrinks as GNSS accumulates


def test_latency_is_compensated_by_fusing_at_the_sample_time():
    fast = _engine_run({"type": "eskf", "handover_s": 3.0}, ["imu", "mag", "baro", {"type": "gps", "params": {"latency": 0.0}}], seed=3)
    slow = _engine_run({"type": "eskf", "handover_s": 3.0}, ["imu", "mag", "baro", {"type": "gps", "params": {"latency": 0.3}}], seed=3)
    for a in (fast, slow):
        late = a.t > 4.0
        assert np.linalg.norm(a["est_pos"][:, 0] - a["pos"][:, 0], axis=1)[late].mean() < 1.5
    e = lambda a: np.linalg.norm(a["est_pos"][:, 0] - a["pos"][:, 0], axis=1)[a.t > 4.0].mean()
    assert e(slow) < 1.5 * e(fast) + 0.2  # a 0.3 s old fix of a 2.4 m/s vehicle would otherwise bias the estimate by ~0.7 m


def test_flow_and_range_hold_the_estimate_through_a_gnss_outage():
    outage = [{"type": "gps_outage", "t_start": 8.0}]
    base = ["imu", "mag", "baro", {"type": "gps", "params": {"latency": 0.1}}]
    without = _engine_run({"type": "eskf", "handover_s": 3.0}, base, outage, duration=18.0, seed=2)
    with_flow = _engine_run(
        {"type": "eskf", "handover_s": 3.0}, base + [{"type": "rangefinder"}, {"type": "optical_flow", "name": "flow"}], outage, duration=18.0, seed=2
    )
    last = -1
    s0, s1 = without["est_pos_std"][last, 0, 0], with_flow["est_pos_std"][last, 0, 0]
    assert s0 > 3.0 * s1 and s1 < 0.5  # without velocity aiding the position uncertainty explodes; with flow it stays bounded
    e0 = np.linalg.norm(without["est_pos"][last, 0] - without["pos"][last, 0])
    e1 = np.linalg.norm(with_flow["est_pos"][last, 0] - with_flow["pos"][last, 0])
    assert np.isfinite(e0) and e1 < 2.5  # ... and the estimate itself stays close


def test_a_nan_burst_from_the_gnss_receiver_does_not_poison_the_estimate():
    nan = [{"type": "sensor_nan", "t_start": 5.0, "params": {"sensor": "gps", "field": "pos", "probability": 0.4}}]
    a = _engine_run({"type": "eskf", "handover_s": 3.0}, ["imu", "mag", "baro", "gps"], nan, duration=10.0)
    assert np.all(np.isfinite(a["est_pos"])) and np.all(np.isfinite(a["est_pos_std"]))


def test_the_estimator_is_registered_and_validates_in_a_configuration():
    from aevion.config import SimulationConfig
    from aevion.registry import registries

    assert "eskf" in registries.estimators.names()
    cfg = SimulationConfig.from_dict(
        {"duration": 1.0, "sensors": ["imu"], "estimator": {"type": "error_state_kf", "params": {"use_flow": False, "max_delay_s": 0.8}}}
    )
    assert cfg.estimator.params["max_delay_s"] == 0.8
