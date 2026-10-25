"""The fixed-wing model, trim, linearised modes and the autopilot - against flight-dynamics theory."""

import math

import numpy as np
import pytest

from aevion.aircraft import AEROSONDE, FixedWing
from aevion.aircraft.autopilot import Autopilot, PathFollower, wrap_pi
from aevion.aircraft.trim import linearize, modes, state_from_trim, trim

G = 9.80665
VA = 35.0


def model(n=1, dt=0.01, **kw):
    return FixedWing(AEROSONDE.replace(**kw) if kw else None, n, dt)


# ------------------------------------------------------------------------------------------------ aerodynamics
def test_the_lift_curve_is_linear_then_stalls_into_flat_plate_lift():
    fw = model()
    a = np.linspace(-math.pi / 2, math.pi / 2, 4001)
    cl = fw.C_L(a)
    p = AEROSONDE
    lin = p.C_L_0 + p.C_L_alpha * a
    assert np.allclose(cl[np.abs(a) < 0.2], lin[np.abs(a) < 0.2], atol=1e-3)  # attached flow: the linear model
    assert fw.sigma(np.array([p.alpha0]))[0] == pytest.approx(0.5, abs=1e-6)  # the blend is half-way at the stall angle
    big = np.abs(a) > 1.0
    assert np.allclose(cl[big], 2.0 * np.sign(a[big]) * np.sin(a[big]) ** 2 * np.cos(a[big]), atol=1e-3)  # flat-plate lift far beyond stall
    assert cl.max() > 1.0 and cl[np.argmin(np.abs(a - 1.4))] < cl[np.argmin(np.abs(a - p.alpha0))]  # lift collapses after the stall
    assert fw.C_D(np.array([0.0]))[0] == pytest.approx(p.C_D_p + p.C_L_0**2 / (math.pi * p.e * p.AR))  # induced drag of the zero-alpha lift


def test_the_inertia_tensor_in_flu_axes_keeps_its_principal_moments_and_the_cross_product_sign():
    J = AEROSONDE.inertia
    assert np.allclose(np.linalg.eigvalsh(J), np.linalg.eigvalsh(np.array([[0.8244, 0, -0.1204], [0, 1.135, 0], [-0.1204, 0, 1.759]])))
    assert J[0, 2] == pytest.approx(AEROSONDE.Jxz) and np.allclose(J, J.T)


def test_at_rest_in_the_air_a_symmetric_wing_produces_only_lift_drag_and_pitching_moment():
    fw = model()
    st = fw.reset(pos=(0, 0, 300.0), airspeed=VA)
    F, M, info = fw.forces_moments(st.vel, st.quat, st.omega, st.controls, np.zeros((1, 3)))
    assert abs(F[0, 1]) < 1e-9 and abs(M[0, 0]) < 1e-9 and abs(M[0, 2]) < 1e-9  # no side force, roll or yaw moment in symmetric flight
    q = 0.5 * AEROSONDE.rho * VA**2 * AEROSONDE.S
    assert info["lift"][0] == pytest.approx(q * AEROSONDE.C_L_0) and info["drag"][0] == pytest.approx(q * fw.C_D(np.array([0.0]))[0])
    assert F[0, 2] > 0.0  # lift acts upward (FLU +z) at zero angle of attack


# ------------------------------------------------------------------------------------------------ trim and equilibrium
def test_straight_level_trim_is_an_equilibrium_that_holds_for_a_minute():
    fw = model()
    t = trim(fw, VA)
    assert t.converged and t.residual < 1e-9
    st = state_from_trim(fw, t, pos=(0, 0, 300.0))
    from aevion.core import rotation as rot
    from aevion.core.backend import NumpyBackend

    F, M, info = fw.forces_moments(st.vel, st.quat, st.omega, st.controls, np.zeros((1, 3)))
    F_world = rot.rotate(NumpyBackend(), st.quat, F)[0]
    assert np.allclose(F_world, [0.0, 0.0, AEROSONDE.mass * G], atol=1e-4) and np.allclose(
        M, 0.0, atol=1e-6
    )  # lift carries the weight, thrust balances drag
    assert (
        info["thrust"][0] == pytest.approx(info["drag"][0], rel=0.02) and info["thrust"][0] > 10.0
    )  # the propeller thrust balances the drag (about 18 N here)
    for _ in range(6000):
        st = fw.step(st, t.controls)
    assert abs(st.pos[0, 2] - 300.0) < 0.01 and abs(np.linalg.norm(st.vel[0]) - VA) < 1e-3 and np.linalg.norm(st.omega) < 1e-6
    assert t.pitch_up == pytest.approx(t.alpha, abs=1e-12)  # level flight: nose-up pitch equals the angle of attack


def test_climb_and_descent_trims_need_more_and_less_throttle():
    fw = model()
    level, climb, dive = trim(fw, VA), trim(fw, VA, gamma=math.radians(8.0)), trim(fw, VA, gamma=math.radians(-6.0))
    assert all(x.converged for x in (level, climb, dive))
    assert dive.controls[3] < level.controls[3] < climb.controls[3]
    # the power needed to climb: extra thrust of about m g sin(gamma)
    extra = 0.5 * AEROSONDE.rho * AEROSONDE.S_prop * AEROSONDE.C_prop * AEROSONDE.k_motor**2 * (climb.controls[3] ** 2 - level.controls[3] ** 2)
    assert extra == pytest.approx(AEROSONDE.mass * G * math.sin(math.radians(8.0)), rel=0.1)
    st = state_from_trim(fw, climb, pos=(0, 0, 100.0))
    z0 = st.pos[0, 2]
    for _ in range(1000):
        st = fw.step(st, climb.controls)
    assert (st.pos[0, 2] - z0) / (VA * 10.0) == pytest.approx(math.sin(math.radians(8.0)), rel=0.01)  # it really climbs at 8 degrees


def test_a_coordinated_turn_trims_to_the_bank_angle_of_the_load_factor():
    fw = model()
    R = 150.0
    t = trim(fw, VA, radius=R)
    assert t.converged
    bank = math.atan2(VA**2, G * R)  # tan(bank) = V^2 / (g R)
    assert abs(t.phi) == pytest.approx(bank, rel=0.03) and t.phi < 0.0  # a left turn (positive radius) is left wing down: negative roll
    st = state_from_trim(fw, t, pos=(0, 0, 400.0))
    x0 = st.pos[0, :2].copy()
    track = []
    for _ in range(int(2 * math.pi * R / VA / 0.01)):  # one full circle
        st = fw.step(st, t.controls)
        track.append(st.pos[0, :2].copy())
    track = np.array(track)
    assert np.linalg.norm(track[-1] - x0) < 3.0  # it comes back to where it started
    r = np.linalg.norm(track - track.mean(axis=0), axis=1)
    assert r.mean() == pytest.approx(R, rel=0.01) and r.std() < 0.5  # on a circle of the commanded radius
    assert abs(st.pos[0, 2] - 400.0) < 0.5
    right = trim(fw, VA, radius=-R)
    assert right.phi == pytest.approx(-t.phi, rel=1e-6) and right.controls[1] == pytest.approx(-t.controls[1], rel=1e-6)  # mirror image


# ------------------------------------------------------------------------------------------------ modes
def test_linearised_modes_match_the_classical_approximations():
    fw = model()
    t = trim(fw, VA)
    lin = linearize(fw, t)
    m = modes(lin)
    assert np.linalg.norm(lin["f0"]) < 1e-9  # the linearisation point is an equilibrium
    p = AEROSONDE
    q = 0.5 * p.rho * VA**2 * p.S
    # short period: sqrt of the pitch stiffness  -q S c C_m_alpha / Jy
    w_sp = math.sqrt(-q * p.c * p.C_m_alpha / p.Jy)
    assert m["short_period"]["wn"] == pytest.approx(w_sp, rel=0.15) and 0.2 < m["short_period"]["zeta"] < 0.9
    # roll subsidence: the roll damping derivative L_p = q S b^2 C_lp / (2 Va Jx)
    assert m["roll"]["eigenvalue"] == pytest.approx(q * p.b**2 * p.C_l_p / (2 * VA * p.Jx), rel=0.1)
    # dutch roll: of the order of sqrt(N_beta), stable and lightly damped; spiral: a very slow real root
    assert m["dutch_roll"]["wn"] == pytest.approx(math.sqrt(q * p.b * p.C_n_beta / p.Jz), rel=0.3) and 0.05 < m["dutch_roll"]["zeta"] < 0.6
    assert abs(m["spiral"]["eigenvalue"]) < 0.2
    # the fixed-throttle propeller damps the speed so strongly that the phugoid is overdamped; every root is stable
    assert m["phugoid"]["overdamped"] and np.all(np.linalg.eigvals(lin["A_lon"]).real < 0.0)


def test_the_phugoid_of_a_drag_free_glider_follows_lanchester():
    ideal = model(C_D_p=0.0, C_D_0=0.0, C_D_alpha=0.0, C_D_delta_e=0.0, C_D_q=0.0, e=1e9, S_prop=1e-12)
    t = trim(ideal, VA)
    assert t.residual < 1e-6
    lin = linearize(ideal, t)
    ph = modes(lin)["phugoid"]
    assert (
        ph["period"] == pytest.approx(math.pi * math.sqrt(2.0) * VA / G, rel=0.12) and ph["zeta"] < 0.02
    )  # T = pi sqrt(2) V / g (first-order theory), undamped
    # and the total energy g h + V^2 / 2 is conserved along a perturbed flight
    st = state_from_trim(ideal, t, pos=(0, 0, 500.0))
    st.vel = st.vel * 1.06
    e0 = ideal.total_energy(st)[0]
    drift = 0.0
    for _ in range(6000):
        st = ideal.step(st, t.controls)
        drift = max(drift, abs(ideal.total_energy(st)[0] - e0) / e0)
    assert drift < 2e-3
    assert st.pos[0, 2] != pytest.approx(500.0, abs=0.5)  # it did exchange altitude for speed


def test_steady_roll_rate_follows_the_aileron_derivatives():
    clean = model(Jxz=0.0, C_l_beta=0.0, C_n_p=0.0, C_n_delta_a=0.0, C_l_r=0.0, C_Y_delta_a=0.0, tau_surface=1e-3)
    t = trim(clean, VA)
    st = state_from_trim(clean, t, pos=(0, 0, 500.0))
    da = 0.04
    for _ in range(40):  # 0.4 s: long enough for the roll subsidence, short enough for the bank angle to stay small
        st = clean.step(st, [t.controls[0], da, 0.0, t.controls[3]])
    p = AEROSONDE
    p_ss = -p.C_l_delta_a * da * 2.0 * VA / (p.b * p.C_l_p)
    assert st.omega[0, 0] == pytest.approx(p_ss, rel=0.03)  # FLU roll rate equals the FRD one


# ------------------------------------------------------------------------------------------------ the model's housekeeping
def test_wind_changes_ground_speed_but_not_the_flight_relative_to_the_air():
    fw = FixedWing(None, 1, 0.01, wind=np.array([6.0, 3.0, 0.0]))
    t = trim(fw, VA)
    st = state_from_trim(fw, t, pos=(0, 0, 300.0))
    st.vel = st.vel + np.array([6.0, 3.0, 0.0])  # the air mass moves: so does the aircraft
    for _ in range(3000):
        st = fw.step(st, t.controls)
    assert abs(st.pos[0, 2] - 300.0) < 0.05 and st.airspeed[0] == pytest.approx(VA, abs=1e-3)
    assert st.pos[0, 0] == pytest.approx(30.0 * (VA + 6.0), rel=0.02) or abs(st.pos[0, 0]) > 900.0  # it was carried downwind
    assert fw.course(st)[0] != pytest.approx(0.0, abs=1e-3)  # the ground track is crabbed by the crosswind component


def test_batches_are_independent_quaternions_stay_unit_and_the_ground_stops_the_aircraft():
    fw = model(n=3)
    st = fw.reset(pos=[(0, 0, 400.0), (0, 0, 400.0), (0, 0, 8.0)], airspeed=VA, flight_path=-0.15)  # gliding downward: the third reaches the ground
    cmd = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.05, 0.0, 0.4], [0.0, 0.0, 0.0, 0.0]])
    for _ in range(1500):
        st = fw.step(st, cmd)
    assert np.allclose(np.linalg.norm(st.quat, axis=1), 1.0, atol=1e-9)
    assert abs(fw.euler(st)[1, 0]) > 0.1 and abs(fw.euler(st)[0, 0]) < 0.1  # vehicle 1 rolled (aileron), vehicle 0 did not
    assert st.crashed is not None and st.crashed[2] and not st.crashed[0]
    assert np.all(st.vel[2] == 0.0) and st.pos[2, 2] == 0.0 and np.all(st.omega[2] == 0.0)  # the grounded one is frozen
    s2 = fw.step(st, cmd)
    assert np.array_equal(s2.pos[2], st.pos[2])


# ------------------------------------------------------------------------------------------------ autopilot
def fly(ap, fw, t, steps, course, altitude, airspeed, h0=200.0):
    st = state_from_trim(fw, t, pos=(0.0, 0.0, h0))
    ap.reset(1)
    rows = []
    for _ in range(steps):
        st = fw.step(st, ap.update(st, course, altitude, airspeed)[0])
        rows.append((st.t, st.pos[0, 2], math.degrees(fw.course(st)[0]), st.airspeed[0], st.alpha[0], -fw.euler(st)[0, 1], fw.euler(st)[0, 0]))
    return np.array(rows), st


@pytest.fixture(scope="module")
def setup():
    fw = model()
    t = trim(fw, VA)
    return fw, t, Autopilot(fw, t)


def test_the_gains_place_the_closed_loop_poles_where_they_were_designed(setup):
    fw, t, _ = setup
    ap = Autopilot(fw, t, w_roll=30.0, zeta=0.8)  # fast enough for the second-order pole-placement design
    c, g = ap.coefficients, ap.g
    assert c["b_phi"] * g.kp_phi == pytest.approx(30.0**2) and c["a_phi1"] + c["b_phi"] * g.kd_phi == pytest.approx(2 * 0.8 * 30.0)
    assert c["a_th2"] + c["a_th3"] * g.kp_theta == pytest.approx(ap.w_pitch**2) and c["a_th1"] + c["a_th3"] * g.kd_theta == pytest.approx(
        2 * 0.8 * ap.w_pitch
    )
    assert ap.w_pitch > math.sqrt(c["a_th2"])  # faster than the airframe's own short-period oscillation
    slow = Autopilot(fw, t, w_roll=6.0)  # slower than the roll subsidence: a first-order lag loop with no rate feedback
    assert slow.g.kd_phi == 0.0 and slow.g.kp_phi * slow.coefficients["b_phi"] / slow.coefficients["a_phi1"] == pytest.approx(6.0)


def test_altitude_hold_climbs_30_m_without_stalling_or_losing_airspeed(setup):
    fw, t, ap = setup
    L, _ = fly(ap, fw, t, 6000, 0.0, 230.0, VA)
    assert L[:, 1].max() - 230.0 < 0.15 * 30.0  # overshoot below 15 %
    assert abs(L[2000, 1] - 230.0) < 0.5 and abs(L[-1, 1] - 230.0) < 0.2  # settled within 20 s
    assert np.abs(L[:, 3] - VA).max() < 2.0 and L[:, 4].max() < AEROSONDE.alpha0 * 0.5  # airspeed kept; far from the stall angle
    assert abs(L[:, 6]).max() < math.radians(2.0)  # wings level throughout


def test_course_hold_turns_to_a_new_heading_without_losing_altitude(setup):
    fw, t, ap = setup
    L, _ = fly(ap, fw, t, 3500, math.radians(45.0), 200.0, VA)
    assert L[:, 2].max() - 45.0 < 3.0 and abs(L[1500, 2] - 45.0) < 1.0 and abs(L[-1, 2] - 45.0) < 0.2
    assert np.abs(L[:, 1] - 200.0).max() < 3.0 and np.degrees(np.abs(L[:, 6]).max()) <= 40.5  # bank limit respected, altitude held
    left, _ = fly(ap, fw, t, 1500, math.radians(-60.0), 200.0, VA)
    assert left[-1, 2] < -30.0  # and the other way


def test_airspeed_hold_uses_the_throttle(setup):
    fw, t, ap = setup
    L, st = fly(ap, fw, t, 4000, 0.0, 200.0, 30.0)
    assert abs(L[2000, 3] - 30.0) < 0.3 and abs(L[-1, 3] - 30.0) < 0.1
    assert np.abs(L[:, 1] - 200.0).max() < 4.0 and st.controls[0, 3] < t.controls[3]  # slower flight needs less throttle


def test_the_autopilot_flies_in_a_steady_wind(setup):
    fw, t, ap = setup
    windy = FixedWing(None, 1, 0.01, wind=np.array([6.0, 3.0, 0.0]))
    st = state_from_trim(windy, t, pos=(0.0, 0.0, 200.0))
    st.vel = st.vel + np.array([6.0, 3.0, 0.0])
    ap.reset(1)
    rows = []
    for _ in range(4000):
        st = windy.step(st, ap.update(st, math.radians(-90.0), 180.0, VA)[0])
        rows.append((st.pos[0, 2], math.degrees(windy.course(st)[0])))
    rows = np.array(rows)
    assert abs(rows[-1, 1] - (-90.0)) < 1.0 and abs(rows[-1, 0] - 180.0) < 1.0  # the GROUND course and altitude are held despite the wind


def test_a_path_follower_flies_a_rectangle_and_returns_to_its_start(setup):
    fw, t, ap = setup
    W = [(0, 0, 200), (700, 0, 200), (700, 500, 220), (0, 500, 200), (0, 0, 200), (700, 0, 200)]
    pf = PathFollower(W, airspeed=VA, turn_radius=170.0)
    st = state_from_trim(fw, t, pos=(0.0, -25.0, 200.0))
    ap.reset(1)
    xte, alt = [], []
    for k in range(int(130.0 / 0.01)):
        chi, h, va = pf.command(st.pos[0])
        st = fw.step(st, ap.update(st, chi, h, va)[0])
        a_, b_ = pf._segment()
        d_start, d_end = np.linalg.norm(st.pos[0, :2] - a_[:2]), np.linalg.norm(st.pos[0, :2] - b_[:2])
        if k % 50 == 0 and st.t > 15.0 and d_start > 330.0 and d_end > 100.0:  # on the straight part, after the turn has been absorbed
            xte.append(pf.cross_track_error(st.pos[0]))
        alt.append(st.pos[0, 2])
        if pf.i >= 4 and pf.finished:
            break
    assert pf.i >= 4  # it went through the corners in order
    assert len(xte) > 10 and np.abs(xte).max() < 10.0  # on the legs the cross-track error is a few metres
    assert max(alt) < 245.0 and min(alt) > 175.0


def test_path_following_geometry_and_validation():
    pf = PathFollower([(0, 0, 100), (1000, 0, 100)], airspeed=30.0)
    left, _, _ = pf.command(np.array([500.0, 200.0, 100.0]))  # 200 m to the left of an eastbound path: steer right (negative course)
    right, _, _ = pf.command(np.array([500.0, -200.0, 100.0]))
    on, h, va = pf.command(np.array([500.0, 0.0, 100.0]))
    assert left < 0.0 < right and on == pytest.approx(0.0, abs=1e-9) and h == pytest.approx(100.0) and va == 30.0
    assert abs(left) < math.radians(60.0)  # never more than the approach angle
    pf.command(np.array([1100.0, 0.0, 100.0]))
    assert pf.finished
    with pytest.raises(ValueError):
        PathFollower([(0, 0, 0)])
    assert wrap_pi(3 * math.pi) == pytest.approx(-math.pi, abs=1e-9) or wrap_pi(3 * math.pi) == pytest.approx(math.pi, abs=1e-9)
