"""Lift-plus-cruise and tilt-rotor VTOL aircraft - rotor mechanics, blended control, forward and back transitions."""

import itertools
import math

import numpy as np
import pytest

from aevion.aircraft import QuadPlane, QuadPlaneController, TiltRotorController, TiltRotorPlane

G = 9.80665
H0 = 100.0


def simulate(model, ctl, profile, duration, hold=H0, controls=None):
    T0 = model.hover_thrust()
    u0 = [0, 0, 0, 0, T0, T0, T0, T0] + ([0.0] if model.n_controls == 9 else [])
    st = model.reset(pos=(0.0, 0.0, H0), airspeed=0.0, controls=u0 if controls is None else controls)
    ctl.reset(1)
    rows = []
    for k in range(int(duration / model.dt)):
        t = k * model.dt
        u = ctl.update(st, 0.0, hold, profile(t))
        st = model.step(st, u[0])
        e = model.euler(st)[0]
        rows.append(
            dict(
                t=t,
                z=st.pos[0, 2],
                Va=st.airspeed[0],
                alpha=st.alpha[0],
                pitch=-e[1],
                roll=e[0],
                rotor=float(st.controls[0, 4:8].sum()),
                pusher=float(st.controls[0, 3]),
                tilt=float(st.controls[0, 8]) if model.n_controls == 9 else 0.0,
                x=st.pos[0, 0],
                y=st.pos[0, 1],
            )
        )
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return cols, st


DT = 0.01
PROFILE = lambda t: 0.0 if t < 3.0 else (30.0 if t < 32.0 else 0.0)  # hover, accelerate to cruise, decelerate back


@pytest.fixture(scope="module")
def qp_run():
    m = QuadPlane(n=1, dt=DT)
    return m, simulate(m, QuadPlaneController(m, accel=2.0), PROFILE, 52.0)[0]


@pytest.fixture(scope="module")
def tilt_run():
    m = TiltRotorPlane(n=1, dt=DT)
    return m, simulate(m, TiltRotorController(m, accel=2.0), PROFILE, 52.0)[0]


def at(cols, t, key):
    return float(cols[key][min(int(t / DT), len(cols[key]) - 1)])


# ------------------------------------------------------------------------------------------------ rotor mechanics
def test_rotor_forces_and_moments_follow_the_arm_geometry():
    m = QuadPlane(n=1)
    T = np.array([[10.0, 10.0, 10.0, 10.0]])
    F, M = m._rotor_wrench(T)
    assert np.allclose(F[0], [0, 0, 40.0]) and abs(M[0, 0]) < 1e-12 and abs(M[0, 1]) < 1e-12 and abs(M[0, 2]) < 1e-12  # symmetric layout: pure thrust
    T = np.array([[12.0, 10.0, 10.0, 10.0]])  # one rotor stronger: rotor 0 sits at (x, y) = (0.55, -0.85)
    F, M = m._rotor_wrench(T)
    p = m.pos_r[0]
    assert M[0, 0] == pytest.approx(p[1] * 2.0 + 0.0) and M[0, 1] == pytest.approx(-p[0] * 2.0)  # tau = r x F for the 2 N excess
    assert M[0, 2] == pytest.approx(-m.spin[0] * m.qp.kq_over_kt * 2.0)  # reaction torque of the differential
    tr = TiltRotorPlane(n=1)
    F, M = tr._rotor_wrench(np.array([[10.0] * 4]), math.pi / 2)  # tilted to horizontal: the thrust points forward, nothing lifts
    assert np.allclose(F[0], [40.0, 0.0, 0.0], atol=1e-9)


def test_the_allocation_reproduces_the_requested_collective_and_moments():
    c = QuadPlaneController(QuadPlane(n=1))
    want = np.array([[90.0, 3.0, -2.0, 0.4]])  # total thrust, roll, pitch and yaw moments
    T = want @ c.Apinv.T
    assert np.allclose(c.A @ T[0], want[0], atol=1e-9) and np.all(T > 0.0)  # an exact solution for a symmetric layout


# ------------------------------------------------------------------------------------------------ hover
def test_hover_holds_altitude_and_attitude_with_the_rotors_carrying_the_weight(qp_run):
    m, c = qp_run
    sel = c["t"] < 3.0
    assert (
        np.abs(c["z"][sel] - H0).max() < 0.02 and np.degrees(np.abs(c["pitch"][sel]).max()) < 0.5 and np.degrees(np.abs(c["roll"][sel]).max()) < 0.5
    )
    assert at(c, 2.9, "rotor") == pytest.approx(m.p.mass * G, rel=0.01)  # rotor thrust = weight


def test_hover_in_a_steady_wind_keeps_altitude_and_stays_upright():
    m = QuadPlane(n=1, dt=DT, wind=np.array([5.0, 2.0, 0.0]))
    c, _ = simulate(m, QuadPlaneController(m), lambda t: 0.0, 15.0)
    assert (
        np.abs(c["z"] - H0).max() < 1.0 and np.degrees(np.abs(c["roll"]).max()) < 10.0 and np.degrees(np.abs(c["pitch"]).max()) < 10.0
    )  # (the sudden 5 m/s relative wind at the start is the largest disturbance)


# ------------------------------------------------------------------------------------------------ lift-plus-cruise transitions
def test_the_forward_transition_reaches_cruise_holding_altitude_and_attitude(qp_run):
    m, c = qp_run
    fwd = (c["t"] > 2.0) & (c["t"] < 31.0)
    assert at(c, 30.0, "Va") == pytest.approx(30.0, abs=0.3)
    assert np.abs(c["z"][fwd] - H0).max() < 0.3  # altitude held within 30 cm through the whole transition
    assert np.degrees(np.abs(c["pitch"][fwd]).max()) < 3.0 and np.degrees(np.abs(c["roll"][fwd]).max()) < 0.5
    flying = fwd & (c["Va"] > 8.0)
    assert np.abs(c["alpha"][flying]).max() < 0.25 * m.p.alpha0  # nowhere near the stall
    assert at(c, 30.0, "pitch") == pytest.approx(math.radians(1.64), abs=math.radians(0.15))  # the wing's lift-balance attitude


def test_the_rotors_hand_the_weight_to_the_wing_as_airspeed_builds(qp_run):
    m, c = qp_run
    W = m.p.mass * G
    r = [at(c, t, "rotor") / W for t in (2.5, 8.0, 12.0, 15.0, 25.0)]
    assert all(a > b for a, b in itertools.pairwise(r))  # monotonically less rotor thrust
    assert r[0] > 0.95 and r[-1] < 0.05  # rotors carry the weight at the start, the wing at the end
    # the wing's share of the weight equals its lift: W - rotor thrust is the lift (vertical balance), which grows like Va^2
    lift = W - c["rotor"]
    sel = (c["t"] > 6.0) & (c["t"] < 20.0)
    assert np.corrcoef(c["Va"][sel] ** 2, lift[sel])[0, 1] > 0.98
    assert at(c, 2.0, "pusher") < 0.01 < at(c, 25.0, "pusher")  # the pusher takes over the acceleration


def test_the_back_transition_returns_to_hover_without_climbing_or_sinking(qp_run):
    m, c = qp_run
    back = c["t"] > 32.0
    assert np.abs(c["z"][back] - H0).max() < 0.3 and np.degrees(np.abs(c["pitch"][back]).max()) < 3.0
    assert at(c, 51.9, "Va") < 5.5 and at(c, 51.9, "rotor") == pytest.approx(m.p.mass * G, rel=0.03)  # slow again, rotors carrying the weight
    assert np.all(np.diff(c["Va"][(c["t"] > 34.0) & (c["t"] < 46.0)]) < 1e-6)  # speed falls monotonically


def test_control_authority_moves_from_the_rotors_to_the_surfaces_with_airspeed():
    m = QuadPlane(n=1)
    ctl = QuadPlaneController(m)
    for Va, expect_surfaces in ((2.0, False), (30.0, True)):
        st = m.reset(pos=(0, 0, H0), airspeed=Va, euler=(0.15, 0.0, 0.0), controls=[0, 0, 0, 0.3] + [m.hover_thrust() if Va < 10 else 0.0] * 4)
        ctl.reset(1)
        u = ctl.update(st, 0.0, H0, Va)[0]
        roll_from_surfaces, roll_from_rotors = abs(u[1]), abs((u[4] + u[2]) - (u[5] + u[3]))  # aileron; left/right rotor thrust difference
        assert (roll_from_surfaces > 1e-3) == expect_surfaces and (roll_from_rotors > 0.5) == (not expect_surfaces)
    assert ctl._lambda(np.array([0.0, 8.0, 14.0, 20.0, 40.0])).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


# ------------------------------------------------------------------------------------------------ tilt-rotor
def test_the_tilt_rotor_accelerates_with_a_tilting_thrust_vector_and_has_no_pusher(tilt_run):
    m, c = tilt_run
    assert np.all(c["pusher"] == 0.0)
    assert at(c, 30.0, "Va") == pytest.approx(30.0, abs=0.7)
    fwd = (c["t"] > 2.0) & (c["t"] < 31.0)
    assert np.abs(c["z"][fwd] - H0).max() < 1.0 and np.degrees(np.abs(c["pitch"][fwd]).max()) < 3.0
    tilt = np.degrees(c["tilt"])
    ramp = (c["t"] > 4.0) & (c["t"] < 16.0)
    assert np.all(np.diff(tilt[ramp]) > -0.5) and at(c, 3.0, "tilt") < math.radians(10.0)  # tilts forward as speed builds
    assert at(c, 28.0, "tilt") > math.radians(85.0)  # horizontal in cruise: rotors act as propellers
    assert at(c, 28.0, "rotor") < 0.25 * m.p.mass * G  # only enough thrust to balance the drag


def test_the_tilt_rotor_returns_to_hover_by_tilting_back(tilt_run):
    m, c = tilt_run
    back = c["t"] > 34.0
    assert np.abs(c["z"][back] - H0).max() < 1.0 and at(c, 51.9, "Va") < 5.0
    assert at(c, 51.9, "tilt") < math.radians(10.0) and at(c, 51.9, "rotor") == pytest.approx(m.p.mass * G, rel=0.03)
    assert at(c, 36.0, "tilt") < at(c, 30.0, "tilt")  # it tilts back (even a little aft) to brake


# ------------------------------------------------------------------------------------------------ model housekeeping
def test_the_idle_rotors_add_parasite_drag_and_nothing_else():
    from aevion.aircraft import FixedWing

    qp = QuadPlane(n=1)
    fw = FixedWing(qp.qp.aero, 1, qp.dt)
    st = qp.reset(pos=(0, 0, H0), airspeed=30.0, controls=[0.0, 0.0, 0.0, 0.4, 0, 0, 0, 0])
    F_q, M_q, info = qp.forces_moments(st.vel, st.quat, st.omega, st.controls, np.zeros((1, 3)))
    F_f, M_f, _ = fw.forces_moments(st.vel, st.quat, st.omega, st.controls[:, :4], np.zeros((1, 3)))
    drag = 0.5 * qp.p.rho * 30.0**2 * qp.p.S * qp.qp.parasite_cd_rotors_off
    assert F_f[0, 0] - F_q[0, 0] == pytest.approx(drag, rel=1e-6) and np.allclose(F_q[0, 1:], F_f[0, 1:], atol=1e-9) and np.allclose(M_q, M_f)


def test_batches_of_vtol_aircraft_are_independent():
    m = QuadPlane(n=2, dt=DT)
    ctl = QuadPlaneController(m)
    T0 = m.hover_thrust()
    st = m.reset(pos=[(0, 0, 100.0), (0, 0, 200.0)], airspeed=0.0, controls=[0, 0, 0, 0, T0, T0, T0, T0])
    ctl.reset(2)
    for _ in range(600):
        u = ctl.update(st, np.array([0.0, 0.3]), np.array([100.0, 215.0]), np.array([0.0, 4.0]))
        st = m.step(st, u)
    assert (
        abs(st.pos[0, 2] - 100.0) < 0.2 and st.pos[1, 2] > 203.0 and st.airspeed[1] > 2.5 and st.airspeed[0] <= 0.5 + 1e-9
    )  # one hovers, one climbs and flies
