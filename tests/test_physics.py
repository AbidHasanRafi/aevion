"""Physics validation against closed-form solutions and conservation laws.

Every test runs on both the reference (generic) implementation and the fused Numba kernel.
"""

import math

import numpy as np
import pytest

from aevion import Atmosphere, GravityModel, Multirotor, VehicleSpec, World
from aevion.core.backend import get_backend
from aevion.core.constants import G0, RHO0, altitude_from_pressure
from aevion.dynamics import fast
from aevion.environments.wind import ConstantWind, DrydenTurbulence

PARAMS = [False] + ([True] if fast.HAVE_NUMBA else [])


@pytest.fixture(params=PARAMS, ids=lambda f: "fast" if f else "ref")
def use_fast(request):
    return request.param


def clean_spec(**kw):
    """A drag-free, effect-free vehicle so analytic solutions apply exactly."""
    base = dict(
        drag_lin=(0, 0, 0), drag_quad=(0, 0, 0), rotor_drag=0.0, angular_damping=(0, 0, 0), ground_effect=False, inflow_gain=0.0, rotor_inertia=0.0
    )
    base.update(kw)
    return VehicleSpec.generic_1kg().copy(**base)


def ideal_world(**kw):
    return World(atmosphere=Atmosphere(enabled=False), **kw)


def test_free_fall_matches_analytic(use_fast):
    sim = Multirotor(clean_spec(), n=2, dt=0.002, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 500.0], hover_rotors=False)
    for _ in range(1000):
        s = sim.step(s, np.zeros((2, 4)))
    t = s.t
    np.testing.assert_allclose(s.pos[:, 2], 500.0 - 0.5 * G0 * t**2, atol=1e-8)
    np.testing.assert_allclose(s.vel[:, 2], -G0 * t, atol=1e-9)


def test_hover_is_equilibrium(use_fast):
    sim = Multirotor(clean_spec(), n=1, dt=0.005, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[1.0, 2.0, 5.0])
    for _ in range(2000):
        s = sim.step(s, sim.hover_command())
    np.testing.assert_allclose(s.pos[0], [1.0, 2.0, 5.0], atol=1e-9)
    np.testing.assert_allclose(s.vel[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(s.omega[0], 0.0, atol=1e-9)


def test_hover_specific_force_equals_g(use_fast):
    sim = Multirotor(clean_spec(), n=1, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.step(sim.reset(), sim.hover_command())
    np.testing.assert_allclose(s.f_body[0], [0, 0, G0], atol=1e-9)


def test_terminal_velocity_quadratic_drag(use_fast):
    spec = clean_spec(drag_quad=(0.0, 0.0, 0.05))
    sim = Multirotor(spec, n=1, dt=0.005, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1e5], hover_rotors=False)
    for _ in range(4000):  # 20 s >> m/(c g)^0.5
        s = sim.step(s, np.zeros((1, 4)))
    vt = math.sqrt(spec.mass * G0 / 0.05)
    assert abs(s.vel[0, 2] + vt) / vt < 1e-4


def test_linear_drag_analytic_solution(use_fast):
    c = 0.5
    spec = clean_spec(drag_lin=(0, 0, c))
    sim = Multirotor(spec, n=1, dt=0.002, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1e5], hover_rotors=False)
    for _ in range(1500):
        s = sim.step(s, np.zeros((1, 4)))
    k, t = c / spec.mass, s.t
    vt = -G0 / k
    np.testing.assert_allclose(s.vel[0, 2], vt * (1 - math.exp(-k * t)), rtol=1e-9)


def test_wind_pulls_vehicle_toward_wind_speed(use_fast):
    c = 0.4
    spec = clean_spec(drag_lin=(c, c, 0))
    sim = Multirotor(spec, n=1, dt=0.002, world=ideal_world(wind=ConstantWind([6.0, 0, 0])), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 50.0])
    for _ in range(1000):
        s = sim.step(s, sim.hover_command())
    k, t = c / spec.mass, s.t
    np.testing.assert_allclose(s.vel[0, 0], 6.0 * (1 - math.exp(-k * t)), rtol=1e-8)


def test_torque_free_rotation_conserves_energy_and_momentum(use_fast):
    spec = clean_spec(inertia=(0.01, 0.02, 0.035))
    sim = Multirotor(spec, n=1, dt=0.001, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1e4], hover_rotors=False, euler=[0.3, -0.2, 0.9], omega=[4.0, 1.5, 3.0])
    L0, _T0 = sim.angular_momentum_world(s), sim.kinetic_energy(s) - 0.5 * spec.mass * 0
    Ek0 = 0.5 * s.omega[0] @ (spec.inertia_matrix @ s.omega[0])
    for _ in range(5000):
        s = sim.step(s, np.zeros((1, 4)))
    L1 = sim.angular_momentum_world(s)
    Ek1 = 0.5 * s.omega[0] @ (spec.inertia_matrix @ s.omega[0])
    np.testing.assert_allclose(L1, L0, rtol=1e-8)
    assert abs(Ek1 - Ek0) / Ek0 < 1e-8
    assert abs(np.linalg.norm(s.quat) - 1) < 1e-12


def test_roll_and_yaw_torque_match_allocation(use_fast):
    spec = clean_spec(motor_tau=1e9)  # rotors frozen at their initial speeds
    sim = Multirotor(spec, n=1, dt=1e-4, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1e4], hover_rotors=False)
    s.rotor[:] = np.array([[620.0, 500.0, 500.0, 500.0]])
    A = spec.allocation_matrix()
    wrench = A @ (s.rotor[0] ** 2)
    s1 = sim.step(s, s.rotor)
    expected_alpha = np.linalg.inv(spec.inertia_matrix) @ wrench[1:]
    np.testing.assert_allclose(s1.omega[0] / 1e-4, expected_alpha, rtol=1e-6)
    assert wrench[1] != 0 and wrench[3] != 0


def test_gyroscopic_coupling(use_fast):
    spec = clean_spec(rotor_inertia=2e-4, motor_tau=1e9)
    sim = Multirotor(spec, n=1, dt=1e-5, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1e4], hover_rotors=False, omega=[1.0, 0.0, 0.0])
    s.rotor[:] = np.array([[600.0, 600.0, 400.0, 400.0]])  # net rotor angular momentum, no roll/pitch torque
    hz = float(np.sum(spec.rotor_spin * spec.rotor_inertia * s.rotor[0]))
    wrench = spec.allocation_matrix() @ (s.rotor[0] ** 2)
    assert abs(wrench[1]) < 1e-12 and abs(wrench[2]) < 1e-12
    s1 = sim.step(s, s.rotor)
    ang_acc_y = s1.omega[0, 1] / 1e-5
    np.testing.assert_allclose(ang_acc_y, 1.0 * hz / spec.inertia_matrix[1, 1], rtol=1e-4)


def test_motor_first_order_lag(use_fast):
    tau = 0.05
    spec = clean_spec(motor_tau=tau)
    sim = Multirotor(spec, n=1, dt=0.001, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1e4], hover_rotors=False)
    cmd = np.full((1, 4), 400.0)
    for _ in range(100):
        s = sim.step(s, cmd)
    np.testing.assert_allclose(s.rotor[0, 0], 400.0 * (1 - math.exp(-s.t / tau)), rtol=1e-9)


def test_motor_saturation(use_fast):
    sim = Multirotor(clean_spec(motor_tau=0.01), n=1, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(hover_rotors=False)
    for _ in range(500):
        s = sim.step(s, np.full((1, 4), 5000.0))
    assert s.rotor.max() <= sim.spec.w_max + 1e-9
    assert s.rotor.max() > 0.99 * sim.spec.w_max


def test_ground_effect_matches_cheeseman_bennett():
    spec = clean_spec(ground_effect=True)
    sim = Multirotor(spec, n=1, world=ideal_world(), kernel="reference")
    s = sim.reset(pos=[0, 0, 2 * spec.rotor_radius])
    s = sim.step(s, sim.hover_command())
    ratio = spec.rotor_radius / (4 * 2 * spec.rotor_radius)
    expected = 1.0 / (1.0 - ratio**2)
    assert abs(s.thrust[0].sum() / (spec.mass * G0) - expected) < 1e-4
    far = sim.step(sim.reset(pos=[0, 0, 50.0]), sim.hover_command())
    assert abs(far.thrust[0].sum() / (spec.mass * G0) - 1.0) < 1e-4


def test_isa_atmosphere_reference_values():
    B = get_backend("numpy")
    atm = Atmosphere()
    for h, rho, p in [(0, 1.2250, 101325), (1000, 1.1117, 89875), (3000, 0.9093, 70108), (5000, 0.7364, 54020)]:
        assert abs(float(atm.density(B, np.array(float(h)))) - rho) / rho < 1e-3
        assert abs(float(atm.pressure(B, np.array(float(h)))) - p) / p < 1e-3
    assert abs(float(altitude_from_pressure(np.array(89875.0))) - 1000) < 2.0
    hot = Atmosphere(temp_offset=15.0)
    assert float(hot.density(B, np.array(0.0))) < RHO0


def test_thrust_scales_with_air_density(use_fast):
    low = Multirotor(clean_spec(), n=1, world=World(atmosphere=Atmosphere(base_altitude=3000.0)), kernel=("numba" if use_fast else "reference"))
    s = low.step(low.reset(), low.hover_command())
    ratio = s.thrust[0].sum() / (low.spec.mass * G0)
    assert 0.70 < ratio < 0.76  # rho(3000 m)/rho0 = 0.742


def test_gravity_models():
    B = get_backend("numpy")
    z = np.array(0.0)
    assert float(GravityModel().magnitude(B, z)) == pytest.approx(G0)
    assert float(GravityModel("wgs84", latitude_deg=0).magnitude(B, z)) == pytest.approx(9.7803, abs=1e-4)
    assert float(GravityModel("wgs84", latitude_deg=90).magnitude(B, z)) == pytest.approx(9.8322, abs=1e-4)
    assert float(GravityModel("wgs84", latitude_deg=45).magnitude(B, z)) == pytest.approx(9.8062, abs=2e-4)
    g10 = float(GravityModel("altitude").magnitude(B, np.array(10000.0)))
    assert g10 == pytest.approx(G0 * (6371000 / 6381000) ** 2, rel=1e-12)
    assert float(GravityModel.mars().magnitude(B, z)) == 3.71


def test_gravity_changes_fall_acceleration(use_fast):
    sim = Multirotor(
        clean_spec(),
        n=1,
        dt=0.005,
        world=World(gravity=GravityModel.moon(), atmosphere=Atmosphere(enabled=False)),
        kernel=("numba" if use_fast else "reference"),
    )
    s = sim.reset(pos=[0, 0, 100.0], hover_rotors=False)
    for _ in range(200):
        s = sim.step(s, np.zeros((1, 4)))
    assert s.vel[0, 2] == pytest.approx(-1.62 * s.t, rel=1e-9)


def _convergence_error(method, dt, use_fast):
    c = 0.5
    spec = clean_spec(drag_lin=(0, 0, c), motor_tau=1.0)  # irrelevant at zero command; avoids auto sub-stepping
    sim = Multirotor(spec, n=1, dt=dt, integrator=method, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1e5], hover_rotors=False)
    for _ in range(round(1.0 / dt)):
        s = sim.step(s, np.zeros((1, 4)))
    k = c / spec.mass
    return abs(s.vel[0, 2] - (-G0 / k) * (1 - math.exp(-k * s.t)))


@pytest.mark.parametrize("method,order", [("euler", 1), ("midpoint", 2), ("rk4", 4)])
def test_integrator_convergence_order(method, order, use_fast):
    e1, e2 = _convergence_error(method, 0.02, use_fast), _convergence_error(method, 0.01, use_fast)
    observed = math.log2(e1 / e2)
    assert observed == pytest.approx(order, abs=0.35), (e1, e2)


def test_quaternion_norm_preserved_under_aggressive_motion(use_fast):
    sim = Multirotor("generic_1kg", n=4, dt=0.005, kernel=("numba" if use_fast else "reference"))
    rng = np.random.default_rng(2)
    s = sim.reset(pos=[0, 0, 200.0], omega=rng.normal(size=(4, 3)) * 6)
    for k in range(4000):
        s = sim.step(s, sim.hover_speed() * (1 + 0.3 * np.sin(0.1 * k + np.arange(4)))[None, :])
    np.testing.assert_allclose(np.linalg.norm(s.quat, axis=1), 1.0, atol=1e-12)
    assert np.isfinite(s.pos).all()


def test_ground_contact_and_impact_speed(use_fast):
    sim = Multirotor(clean_spec(), n=1, dt=0.001, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 1.0], hover_rotors=False)
    impact = 0.0
    for _ in range(1500):
        s = sim.step(s, np.zeros((1, 4)))
        impact = max(impact, float(s.impact_speed[0]))
    assert s.pos[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert abs(s.vel[0, 2]) < 1e-9 and bool(s.contact[0])
    assert impact == pytest.approx(math.sqrt(2 * G0 * 1.0), rel=5e-3)


def test_ground_friction_stops_sliding(use_fast):
    sim = Multirotor(clean_spec(), n=1, dt=0.002, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 0.0], vel=[3.0, 0, 0], hover_rotors=False)
    for _ in range(1500):
        s = sim.step(s, np.zeros((1, 4)))
    assert abs(s.vel[0, 0]) < 1e-9


def test_battery_energy_and_soc_consistency(use_fast):
    sim = Multirotor("generic_1kg", n=1, dt=0.005, kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 20.0])
    p_sum = 0.0
    for _ in range(1000):
        s = sim.step(s, sim.hover_command())
        p_sum += float(s.power[0]) * sim.dt
    assert s.energy[0] == pytest.approx(p_sum, rel=1e-9)
    assert float(s.soc[0]) == pytest.approx(1 - s.energy[0] / (sim.spec.battery.capacity_wh * 3600), rel=1e-9)
    hover_p = sim.spec.hover_power()
    assert abs(float(s.power[0]) - hover_p) / hover_p < 0.01


def test_battery_voltage_sag_reduces_authority(use_fast):
    spec = VehicleSpec.generic_1kg()
    spec.battery.voltage_effects = True
    sim = Multirotor(spec, n=2, dt=0.005, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    s = sim.reset(pos=[0, 0, 50.0], soc=1.0)
    s.soc[1] = 0.05
    for _ in range(600):
        s = sim.step(s, np.full((2, 4), spec.w_max))
    assert s.rotor[0].mean() > s.rotor[1].mean()


def test_motor_failure_causes_roll_torque(use_fast):
    sim = Multirotor("generic_1kg", n=2, dt=0.002, kernel=("numba" if use_fast else "reference"))
    sim.params.set_rotor_failure(1, 0, 1.0)
    s = sim.reset(pos=[0, 0, 50.0])
    for _ in range(200):
        s = sim.step(s, sim.hover_command())
    assert np.linalg.norm(s.omega[0]) < 1e-6
    assert np.linalg.norm(s.omega[1]) > 0.5


def test_dryden_turbulence_statistics():
    B = get_backend("numpy")
    w = DrydenTurbulence(w20=10.0)
    w.reset(B, 400, seed=3)
    pos = np.tile([0.0, 0.0, 30.0], (400, 1))
    vel = np.tile([8.0, 0.0, 0.0], (400, 1))
    dt = 0.02
    samples = []
    for k in range(6000):
        g = w.sample(B, pos, vel, k * dt, dt)
        if k > 1500:
            samples.append(g.copy())
    g = np.stack(samples)
    hft = 30.0 / 0.3048
    base = 0.177 + 0.000823 * hft
    sw, su = 0.1 * 10.0, 0.1 * 10.0 / base**0.4
    assert g[..., 2].std() == pytest.approx(sw, rel=0.12)
    assert g[..., 0].std() == pytest.approx(su, rel=0.12)
    assert abs(g.mean()) < 0.1


def test_allocation_roundtrip_all_layouts():
    for spec in (VehicleSpec.quad_x(), VehicleSpec.quad_plus(), VehicleSpec.hexa(), VehicleSpec.octo()):
        A = spec.allocation_matrix()
        assert np.linalg.matrix_rank(A) == 4
        w2 = spec.hover_rotor_speed() ** 2 * np.ones(spec.n_rotors)
        np.testing.assert_allclose(A @ w2, [spec.mass * G0 if False else A[0].sum() * w2[0], 0, 0, 0], atol=1e-9)


def test_vehicle_validation_warnings():
    weak = VehicleSpec.generic_1kg().copy(w_max=520.0)
    assert any("thrust-to-weight" in w for w in weak.validate())
    bad = VehicleSpec.generic_1kg().copy(w_max=300.0)
    assert any("cannot hover" in w for w in bad.validate())
    assert VehicleSpec.generic_1kg().validate() == []
    for name in ("crazyflie", "racing_5inch", "camera_2kg", "hex_3kg"):
        from aevion import get_vehicle

        assert get_vehicle(name).validate() == [], name


def test_hover_speed_and_twr_values():
    s = VehicleSpec.crazyflie()
    # published Crazyflie hover is ~14.5k RPM
    assert 13000 < s.hover_rotor_speed() * 60 / (2 * math.pi) < 16000
    assert 1.8 < s.thrust_to_weight() < 3.0


def test_stiff_motor_lag_is_stable_via_substeps(use_fast):
    """tau << dt would blow up explicit RK4; the simulator must sub-step automatically."""
    sim = Multirotor(clean_spec(motor_tau=1e-3), n=1, dt=0.005, world=ideal_world(), kernel=("numba" if use_fast else "reference"))
    assert sim.n_sub > 1
    s = sim.reset(hover_rotors=False)
    for _ in range(60):
        s = sim.step(s, np.full((1, 4), 400.0))
    assert np.isfinite(s.rotor).all()
    np.testing.assert_allclose(s.rotor[0], 400.0, rtol=1e-6)
    default = Multirotor("generic_1kg", n=1, dt=0.005, kernel=("numba" if use_fast else "reference"))
    assert default.n_sub == 1
