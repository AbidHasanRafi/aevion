"""MPC, MPPI, INDI, L1 adaptive and sliding-mode control - algorithms on ideal plants, behaviour in the engine."""

import numpy as np
import pytest

from aevion.control import reference as R
from aevion.control.advanced import (
    BatchedBoxQP,
    DisturbanceObserver,
    MPCController,
    MPPIController,
)
from aevion.control.controllers import GeometricController
from aevion.control.reference import Reference
from aevion.core import rotation as rot
from aevion.core.backend import NumpyBackend
from aevion.dynamics import Multirotor
from aevion.engine import Simulation
from aevion.environments.sdf import SignedDistanceField, sphere_sdf
from aevion.planning.qp import solve_qp

B = NumpyBackend()


class St:
    pass


def ideal(ctrl, ref_fn, duration, dt=0.005, disturbance=(0.0, 0.0, 0.0), p0=(0.0, 0.0, 0.0), v0=(0.0, 0.0, 0.0)):
    """Closed loop of a controller's acceleration law with a perfect double integrator (isolates the algorithm from the airframe)."""
    ctrl.reset(1)
    p, v = np.array([p0], float), np.array([v0], float)
    d = np.array([disturbance], float)
    log = []
    for k in range(int(duration / dt)):
        t = k * dt
        ref = ref_fn(t)
        st = St()
        st.pos, st.vel = p.copy(), v.copy()
        a = np.asarray(ctrl.acceleration_command(st, ref, dt, st.pos - ref.pos, st.vel - ref.vel))
        if hasattr(ctrl, "obs") and ctrl.obs is not None and ctrl._acc_hold is not None:
            ctrl.obs.command_applied(ctrl._acc_hold)  # what wrench() does inside the engine
        v = v + (a + d) * dt
        p = p + v * dt
        log.append((t, p[0].copy(), v[0].copy(), a[0].copy(), ref.pos[0].copy()))
    return log


def hold(pos):
    return lambda t: Reference.hover(B, 1, pos=pos)


def sim1():
    return Multirotor("generic_1kg", n=1)


# ------------------------------------------------------------------------------------------------ solver and observer
def test_the_batched_qp_matches_the_general_solver():
    rng = np.random.default_rng(0)
    N = 12
    L = np.tril(np.full((N, N), 0.1))
    H = 2.0 * (L.T @ L + 0.05 * np.eye(N))
    A = np.vstack([np.eye(N), L])
    qp = BatchedBoxQP(H, A, rho=2.0)
    q = rng.normal(size=(5, N)) * 2.0
    lo = np.hstack([np.full((5, N), -3.0), np.full((5, N), -2.0)])
    hi = -lo
    x, y, res = qp.solve(q, lo, hi, None, None, iters=400)
    assert res < 1e-5
    for i in range(5):
        ref = solve_qp(H, q[i], A, lo[i], hi[i], eps_abs=1e-9, eps_rel=1e-9, max_iter=20000)
        assert np.allclose(x[i], ref.x, atol=2e-3)
    fewer, _, _ = qp.solve(q, lo, hi, x, y, iters=3)  # a warm start is already (almost) at the solution
    assert np.abs(fewer - x).max() < 1e-3


def test_the_disturbance_observer_finds_a_constant_acceleration_without_bias():
    obs = DisturbanceObserver()
    obs.reset(2)
    d = np.array([[1.0, -0.5, 0.3], [0.0, 0.0, 0.0]])
    p, v = np.zeros((2, 3)), np.zeros((2, 3))
    u = np.array([[0.2, 0.0, -0.1], [0.5, 0.5, 0.5]])
    dt = 0.01
    for _ in range(600):
        obs.command_applied(u)
        v = v + (u + d) * dt
        p = p + v * dt + 0.5 * (u + d) * dt * dt - 0.5 * (u + d) * dt * dt  # (kept consistent with the explicit-Euler plant)
        est = obs.update(p, v, dt)
    assert np.allclose(est, d, atol=0.05)
    obs.reset(1)
    assert not obs.started and np.all(obs.disturbance == 0.0)


# ------------------------------------------------------------------------------------------------ MPC
def test_mpc_step_response_respects_the_acceleration_and_speed_limits_and_does_not_overshoot():
    c = MPCController(sim1(), observer=False, v_max=4.0)
    log = ideal(c, hold((20.0, 0.0, 0.0)), 14.0)
    A = np.array([l[3] for l in log])
    V = np.array([l[2] for l in log])
    P = np.array([l[1] for l in log])
    assert np.abs(A[:, 0]).max() <= c.a_xy * 1.001 and np.abs(V[:, 0]).max() <= 4.0 * 1.03  # constraints hold (ADMM tolerance)
    assert abs(P[-1, 0] - 20.0) < 0.05 and P[:, 0].max() < 20.0 * 1.03  # arrives, with negligible overshoot
    assert A[:, 0].min() > -c.a_xy * 1.001  # it brakes as hard as it accelerates, never harder
    assert c.solves > 50


def test_mpc_is_offset_free_where_a_pd_law_is_not():
    d = (1.0, -0.6, 0.0)
    mpc = ideal(MPCController(sim1()), hold((2.0, 1.0, 0.0)), 25.0, disturbance=d)
    pd = ideal(GeometricController(sim1()), hold((2.0, 1.0, 0.0)), 25.0, disturbance=d)
    e_mpc = np.linalg.norm(mpc[-1][1] - mpc[-1][4])
    e_pd = np.linalg.norm(pd[-1][1] - pd[-1][4])
    assert e_mpc < 5e-3 and e_pd > 0.02  # the PD law settles on d / wn^2; the observer removes it
    c = MPCController(sim1())
    ideal(c, hold((0, 0, 0)), 8.0, disturbance=d)
    assert np.allclose(c.disturbance_estimate[0], d, atol=0.1)
    assert MPCController(sim1(), observer=False).disturbance_estimate is None


def test_a_preview_of_the_true_trajectory_tracks_better_than_extrapolating_the_reference():
    tr = R.Lissajous.figure8(center=(0, 0, 6.0), size=4.0, period=10.0)
    fn = lambda t: tr.reference(B, t, 1)
    plain = MPCController(sim1(), observer=False)
    exact = MPCController(sim1(), observer=False)
    exact.attach_trajectory(tr)
    e = lambda log: np.mean([np.linalg.norm(l[1] - l[4]) for l in log if l[0] > 2.0])
    assert e(ideal(exact, fn, 14.0, p0=(0, 0, 6.0))) < 0.85 * e(ideal(plain, fn, 14.0, p0=(0, 0, 6.0)))
    stateful = MPCController(sim1())
    from aevion.planning.behavior import BehaviorTreeMission, Hover

    stateful.attach_trajectory(BehaviorTreeMission(Hover(1.0)))  # missions are stateful: never sampled ahead of time
    assert stateful._traj is None


# ------------------------------------------------------------------------------------------------ MPPI
def test_mppi_tracks_a_moving_reference_and_holds_against_a_disturbance():
    c = MPPIController(sim1(), seed=3)
    tr = R.Lissajous.figure8(center=(0, 0, 6.0), size=4.0, period=12.0)
    c.attach_trajectory(tr)
    log = ideal(c, lambda t: tr.reference(B, t, 1), 12.0, p0=(0, 0, 6.0), disturbance=(0.8, 0.0, 0.0))
    assert np.mean([np.linalg.norm(l[1] - l[4]) for l in log if l[0] > 2.0]) < 0.35  # sampling noise, but offset-free thanks to the observer
    A = np.array([l[3] for l in log])
    assert np.all(np.abs(A[:, :2]) <= c.a_xy + 1e-9) and np.all(A[:, 2] <= c.a_z[1] + 1e-9)


def test_mppi_flies_around_an_obstacle_that_a_pure_tracker_would_hit():
    sdf = SignedDistanceField.from_function(lambda p: sphere_sdf(p, (5.0, 0.0, 5.0), 1.5), (-3, -8, -3, 14, 8, 12), 0.2)
    ref = Reference.hover(B, 1, pos=(10.0, 0.0, 5.0))
    free = MPPIController(sim1(), seed=1, v_max=3.0)
    aware = MPPIController(sim1(), seed=1, sdf=sdf, clearance=1.2, obstacle_weight=300.0, v_max=3.0)
    d_free = min(sdf.distance(l[1][None])[0] for l in ideal(free, lambda t: ref, 14.0, p0=(0.0, 0.0, 5.0)))
    log = ideal(aware, lambda t: ref, 14.0, p0=(0.0, 0.0, 5.0))
    d_aware = min(sdf.distance(l[1][None])[0] for l in log)
    assert d_free < 0.0  # flying straight to the goal goes through the sphere
    assert d_aware > 0.1  # the sampled rollouts feel the field and steer around it
    assert np.linalg.norm(log[-1][1] - [10.0, 0.0, 5.0]) < 0.6


def test_mppi_runs_on_the_torch_backend_and_respects_the_same_limits():
    torch = pytest.importorskip("torch")
    from aevion.core.backend import TorchBackend

    Bt = TorchBackend(dtype=torch.float32)
    sim = type("S", (), {})()
    real = sim1()
    sim.B, sim.spec, sim.n, sim.world = Bt, real.spec, 2, real.world
    c = MPPIController(sim, seed=2, samples=64)
    c.reset(2)
    st = type("St", (), {})()
    st.pos, st.vel = Bt.asarray(np.array([[0.0, 0, 0], [1.0, 0, 0]])), Bt.asarray(np.zeros((2, 3)))
    ref = Reference.hover(Bt, 2, pos=(3.0, 0.0, 0.0))
    a = c.acceleration_command(st, ref, 0.005, st.pos - ref.pos, st.vel - ref.vel)
    a = Bt.to_numpy(a)
    assert a.shape == (2, 3) and np.all(np.isfinite(a)) and np.all(np.abs(a[:, :2]) <= c.a_xy + 1e-6)
    assert a[0, 0] > 0.5 and a[1, 0] > 0.5  # both vehicles are asked to accelerate towards the goal


# ------------------------------------------------------------------------------------------------ INDI, L1, SMC (engine)
def _flight(ctrl, wind=None, faults=(), duration=14.0, trajectory=None, seed=1):
    cfg = {
        "duration": duration,
        "seed": seed,
        "trajectory": trajectory or {"type": "figure8", "params": {"center": [0, 0, 6.0], "size": 4.0, "period": 14.0}},
        "wind": {"type": "constant", "velocity": list(wind)} if wind else {"type": "none"},
        "controller": ctrl,
        "faults": list(faults),
    }
    sim = Simulation(cfg)
    a = sim.run()
    err = np.linalg.norm(a["pos"][:, 0] - a["ref_pos"][:, 0], axis=1)
    return a, sim, float(err[a.t > 2.0].mean())


def test_indi_cancels_an_unmodelled_torque_that_leaves_the_pd_attitude_loop_with_an_offset():
    com = [{"type": "com_shift", "t_start": 2.0, "params": {"offset": [0.04, 0.0, 0.0]}}]  # 4 cm: a constant torque of T * 0.04
    hover = {"type": "hover", "params": {"pos": [0, 0, 5.0]}}
    a_geo, _, _ = _flight("geometric", faults=com, duration=8.0, trajectory=hover)
    a_indi, _, _ = _flight("indi", faults=com, duration=8.0, trajectory=hover)
    drift = lambda a: float(np.linalg.norm(a["pos"][-1, 0, :2] - a["ref_pos"][-1, 0, :2]))
    assert drift(a_geo) > 0.02 and drift(a_indi) < 0.2 * drift(a_geo)


def test_indi_recovers_from_a_tilt_as_fast_as_the_nominal_loop_without_any_disturbance():
    hover = {"type": "hover", "params": {"pos": [0, 0, 5.0]}}
    out = {}
    for name in ("geometric", "indi"):
        cfg = {"duration": 2.5, "trajectory": hover, "vehicle": {"initial": {"euler": [0.25, -0.2, 0.0]}}, "controller": name}
        a = Simulation(cfg).run()
        tilt = np.degrees(rot.angle_between(B, a["quat"][:, 0], np.tile([1.0, 0, 0, 0], (len(a.t), 1))))
        out[name] = float(a.t[np.nonzero(tilt > 1.0)[0][-1]])
        assert abs(a["pos"][-1, 0, 2] - 5.0) < 0.05
    assert out["indi"] < 1.3 * out["geometric"] + 0.1


def test_l1_adaptation_removes_the_effect_of_wind_and_a_payload():
    pay = [{"type": "payload_change", "t_start": 3.0, "params": {"mass": 0.4}}]
    _, _, e_geo = _flight("geometric", wind=(5.0, 2.0, 0.0), faults=pay)
    a, sim, e_l1 = _flight("l1", wind=(5.0, 2.0, 0.0), faults=pay)
    assert e_geo > 0.05 and e_l1 < 0.25 * e_geo
    assert np.all(np.isfinite(np.asarray(sim.ctrl.uncertainty_estimate)))


def test_sliding_mode_is_robust_while_its_gain_exceeds_the_disturbance_and_degrades_when_it_does_not():
    strong = (9.0, 4.0, 0.0)  # a disturbance acceleration of a few m/s^2
    _, _, e_geo = _flight("geometric", wind=strong)
    _, _, e_ok = _flight({"type": "smc", "params": {"k_switch": 8.0}}, wind=strong)
    _, _, e_low = _flight({"type": "smc", "params": {"k_switch": 2.0}}, wind=strong)
    assert e_ok < e_geo and e_ok < 0.3  # inside the boundary-layer bound, better than the PD law
    assert e_low > 3.0 * e_ok  # K below the disturbance bound: the sliding condition fails and tracking degrades


def test_mpc_in_the_engine_is_offset_free_with_wind_and_a_payload():
    pay = [{"type": "payload_change", "t_start": 3.0, "params": {"mass": 0.4}}]
    _, _, e_geo = _flight("geometric", wind=(5.0, 2.0, 0.0), faults=pay)
    a, sim, e_mpc = _flight("mpc", wind=(5.0, 2.0, 0.0), faults=pay)
    assert e_mpc < 0.3 * e_geo and sim.ctrl.solves > 100
    assert np.all(np.isfinite(a["cmd"]))


def test_predictive_controllers_handle_several_vehicles():
    cfg = {
        "duration": 8.0,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 5.0], "radius": 3.0, "omega": 0.3}},
        "vehicle": {"count": 3, "formation": {"type": "line", "spacing": 2.0}},
        "controller": {"type": "mppi", "params": {"samples": 256, "seed": 2}},
    }
    a = Simulation(cfg).run()
    assert a["pos"].shape[1] == 3 and np.all(np.isfinite(a["pos"]))
    err = np.linalg.norm(a["pos"] - a["ref_pos"], axis=2)
    assert err[a.t > 3.0].mean() < 0.3  # 256 samples are enough; a handful of samples is visibly noisy


# ------------------------------------------------------------------------------------------------ registry, validation, reproducibility
def test_the_controllers_are_registered_validated_and_reproducible():
    from aevion.config import SimulationConfig
    from aevion.registry import registries
    from aevion.scenario import verify

    assert {"mpc", "mppi", "indi", "l1", "smc"} <= set(registries.controllers.names())
    assert "sliding_mode" in registries.controllers and "linear_mpc" in registries.controllers
    cfg = SimulationConfig.from_dict({"duration": 1.0, "controller": {"type": "mpc", "params": {"horizon": 25, "wn_att": [20, 20, 10]}}})
    assert cfg.controller.params["horizon"] == 25
    with pytest.raises(ValueError, match="horizon"):
        Simulation({"duration": 1.0, "controller": {"type": "mpc", "params": {"horizon": 1}}})
    cfg2 = {
        "duration": 5.0,
        "seed": 4,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.5}},
        "controller": {"type": "mppi", "params": {"samples": 64}},
    }
    a, b = Simulation(cfg2).run(), Simulation(cfg2).run()
    assert np.array_equal(a["pos"], b["pos"])  # the sampling controller is seeded: same configuration, same flight
    assert verify(a)["ok"]


def test_the_geometric_controller_refactor_changed_nothing():
    """The overridable laws default to the original ones: hover thrust and zero torque at the reference."""
    sim = sim1()
    g = GeometricController(sim)
    g.reset(1)
    st = sim.reset(pos=np.array([[0.0, 0.0, 2.0]]))
    thrust, tau = g.wrench(st, Reference.hover(B, 1, pos=(0.0, 0.0, 2.0)), 0.005)
    assert thrust[0] == pytest.approx(sim.spec.mass * 9.80665, rel=1e-9) and np.allclose(tau, 0.0, atol=1e-12)
