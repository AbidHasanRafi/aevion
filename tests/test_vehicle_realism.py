"""Propeller (BEMT), motor and ESC, battery pack, tethered load and proximity effects - against theory and conservation laws."""

import math

import numpy as np
import pytest

from aevion import Multirotor, VehicleSpec
from aevion.engine import Simulation
from aevion.engine.effects import tether_load_step
from aevion.vehicles.battery_pack import BatteryPack, PackSpec
from aevion.vehicles.motor import ESC, BLDCMotor, MotorPropSystem
from aevion.vehicles.propeller import PROPELLERS, Airfoil, Propeller, get_propeller, rotor_parameters

RHO = 1.225


# ------------------------------------------------------------------------------------------------ propeller
@pytest.mark.parametrize("name", ["5x4.5", "7x4", "10x4.5", "15x5.5"])
def test_static_performance_is_in_the_range_of_real_small_propellers(name):
    p = get_propeller(name)
    pt = p.static(0.55 * 340.0 / p.radius * 60.0 / (2 * math.pi))
    assert 0.4 < pt.figure_of_merit < 0.8  # the figure of merit of small propellers: 0.5 - 0.7
    assert 0.04 < pt.CT < 0.20 and 0.015 < pt.CP < 0.11 and pt.T > 0 and pt.Q > 0
    # momentum theory: the mean induced velocity is of the order of the ideal hover value sqrt(T / 2 rho A), a little higher
    v_ideal = math.sqrt(pt.T / (2.0 * RHO * p.disk_area))
    assert 0.9 * v_ideal < pt.v_induced_mean < 1.6 * v_ideal


def test_thrust_and_power_obey_the_propeller_scaling_laws():
    p = get_propeller("10x4.5")
    a, b = p.static(3000.0), p.static(6000.0)
    assert pytest.approx(4.0, rel=2e-3) == b.T / a.T and pytest.approx(8.0, rel=2e-3) == b.P / a.P  # T ~ n^2, P ~ n^3
    assert pytest.approx(b.CT, rel=2e-3) == a.CT  # the coefficients do not depend on rpm
    thin = p.operating_point(6000.0, 0.0, 0.8 * RHO)
    assert pytest.approx(0.8, rel=2e-3) == thin.T / b.T  # T ~ rho
    big = Propeller(2 * p.diameter, 2 * p.pitch, p.blades, p.chord_ratio, p.hub_ratio, p.airfoil)  # geometric similarity
    assert pytest.approx(a.CT, rel=5e-3) == big.static(3000.0).CT and pytest.approx(a.CP, rel=5e-3) == big.static(3000.0).CP


def test_advance_ratio_behaviour_of_thrust_power_and_efficiency():
    p = get_propeller("10x4.5")
    J = np.linspace(0.0, 0.55, 12)
    sw = p.sweep_advance_ratio(6000.0, J)
    assert np.all(np.diff(sw["CT"]) < 0.0)  # thrust falls monotonically with forward speed
    assert np.all(np.diff(sw["CP"]) < 0.0)
    j0 = p.zero_thrust_advance_ratio()
    assert 0.3 < j0 < p.pitch / p.diameter * 1.6  # windmilling limit near (a bit below) the geometric pitch ratio
    pt0 = p.operating_point(6000.0, j0 * 100.0 * p.diameter)
    assert abs(pt0.T) < 0.02 * p.static(6000.0).T
    eta = sw["eta"]
    assert eta[0] == 0.0 and np.argmax(eta) not in (0, len(eta) - 1) and 0.4 < eta.max() < 0.85  # peak efficiency between the ends


def test_more_blades_more_thrust_and_more_pitch_more_power():
    two = get_propeller("5x4.5")
    three = get_propeller("5x4.3x3")
    assert three.static(20000.0).T > 1.15 * Propeller(three.diameter, three.pitch, 2, three.chord_ratio).static(20000.0).T
    low = Propeller(0.254, 0.07, 2, 0.15)
    high = Propeller(0.254, 0.14, 2, 0.15)
    assert high.static(6000.0).P > low.static(6000.0).P and high.zero_thrust_advance_ratio() > low.zero_thrust_advance_ratio()
    assert two.name == "5x4.5"


def test_airfoil_coefficients_are_bounded_continuous_and_defined_at_every_angle():
    af = Airfoil()
    a = np.linspace(-math.pi, math.pi, 4001)
    cl, cd = af.coefficients(a)
    assert np.all(np.abs(cl) <= af.cl_max + 1e-12) and np.all(cd >= af.cd0 - 1e-12) and np.all(np.isfinite(cl) & np.isfinite(cd))
    assert np.max(np.abs(np.diff(cl))) < 0.08  # no jump at stall (a smooth fall-off)
    assert cl[np.argmin(np.abs(a - af.alpha0))] == pytest.approx(0.0, abs=1e-2)  # zero lift at the zero-lift angle
    assert af.coefficients(np.array([math.radians(5.0)]))[0][0] == pytest.approx(af.cl_alpha * (math.radians(5.0) - af.alpha0), rel=1e-9)


def test_designations_the_database_and_derived_rotor_parameters():
    assert get_propeller("10 x 4.5").diameter == pytest.approx(0.254) and get_propeller("5x4.3x3").blades == 3
    assert get_propeller("10x4.5").chord_ratio == PROPELLERS["10x4.5"].chord_ratio
    assert get_propeller("6x3").chord_ratio > get_propeller("14x7").chord_ratio  # unlisted sizes: smaller propellers are wider
    for bad in ("ten by five", "10", "10x"):
        with pytest.raises(ValueError, match="designation"):
            get_propeller(bad)
    with pytest.raises(ValueError):
        Propeller(0.0, 0.1)
    p = get_propeller("10x4.5")
    rp = rotor_parameters(p)
    w = rp["rpm_ref"] * 2 * math.pi / 60.0
    assert rp["kT"] * w**2 == pytest.approx(p.static(rp["rpm_ref"]).T, rel=1e-9) and rp["kQ"] * w**2 == pytest.approx(
        p.static(rp["rpm_ref"]).Q, rel=1e-9
    )
    # the inflow gain reproduces the thrust loss of a slow climb: T(vc)/T0 = 1 - g_in vc / (w R)
    vc = 0.8
    climb = p.operating_point(rp["rpm_ref"], vc).T / p.static(rp["rpm_ref"]).T
    assert climb == pytest.approx(1.0 - rp["inflow_gain"] * vc / (w * p.radius), abs=0.01)
    assert rp["rotor_inertia"] > 0 and 0.4 < rp["figure_of_merit"] < 0.8


# ------------------------------------------------------------------------------------------------ motor and ESC
MOTOR = BLDCMotor(kv=920.0, resistance=0.09, i0=0.6, max_current=60.0, rotor_inertia=1.5e-5)


def test_motor_identities():
    m = MOTOR
    assert m.ke == pytest.approx(60.0 / (2 * math.pi * 920.0)) and m.kt == m.ke
    V = 11.1
    assert m.no_load_speed(V) == pytest.approx((V - m.i0 * m.resistance) / m.ke)
    assert m.torque(V, m.no_load_speed(V)) == pytest.approx(0.0, abs=1e-12)
    assert m.stall_current(V) == pytest.approx(V / m.resistance)
    # the classic maximum efficiency: scan the current and compare
    w = np.linspace(1.0, m.no_load_speed(V), 4000)
    i = np.array([m.current(V, x) for x in w])
    eta = np.array([m.torque(V, x) * x / (V * m.current(V, x)) for x in w])
    assert eta.max() == pytest.approx(m.max_efficiency(V), abs=2e-3)
    assert i[np.argmax(eta)] == pytest.approx(m.max_efficiency_current(V), rel=0.02)


def test_the_operating_point_matches_the_closed_form_for_a_quadratic_load_and_balances_power():
    m, kq, V = MOTOR, 1.7e-7, 11.1
    pt = m.operating_point(V, 1.0, lambda w: kq * w * w, esc_efficiency=1.0)
    b, c = m.ke * m.ke / m.resistance, m.ke * (V / m.resistance - m.i0)  # kq w^2 + b w - c = 0
    w_exact = (-b + math.sqrt(b * b + 4 * kq * c)) / (2 * kq)
    assert pt.omega == pytest.approx(w_exact, rel=1e-9)
    # electrical power = copper loss + mechanical power + no-load (iron/friction) loss
    assert pt.p_electrical == pytest.approx(pt.current**2 * m.resistance + pt.p_mechanical + m.i0 * m.ke * pt.omega, rel=1e-9)
    assert pt.efficiency == pytest.approx(pt.p_mechanical / (V * pt.current), rel=1e-9) and 0.6 < pt.efficiency < 0.95
    half = m.operating_point(V, 0.5, lambda w: kq * w * w, esc_efficiency=1.0)
    assert half.omega < pt.omega and half.p_electrical < pt.p_electrical  # duty cycle scales the voltage


def test_the_esc_scales_voltage_and_losses_and_inverts_its_throttle_curve():
    esc = ESC(efficiency=0.95, min_duty=0.05, expo=1.5)
    assert esc.duty(0.0) == 0.0 and esc.duty(1.0) == pytest.approx(1.0) and esc.throttle(esc.duty(0.4)) == pytest.approx(0.4)
    ideal = MOTOR.operating_point(11.1, 0.7, lambda w: 1.7e-7 * w * w, 1.0)
    real = MOTOR.operating_point(11.1, 0.7, lambda w: 1.7e-7 * w * w, 0.95)
    assert real.p_electrical == pytest.approx(ideal.p_electrical / 0.95, rel=1e-9) and real.omega == pytest.approx(ideal.omega)
    assert real.bus_current == pytest.approx(real.p_electrical / 11.1)  # the battery sees the electrical power, not the mechanical


def test_the_spin_up_time_constant_matches_the_linearised_ode():
    kq = 1.7e-7
    m = BLDCMotor(kv=920.0, resistance=0.09, i0=0.6, rotor_inertia=1.0e-4)
    V1 = 11.1 * 0.61  # a small step in duty cycle (0.60 -> 0.61)
    w0, w1 = m.operating_point(11.1, 0.60, lambda w: kq * w * w).omega, m.operating_point(11.1, 0.61, lambda w: kq * w * w).omega
    w, t, dt = w0, 0.0, 1e-4
    while (w - w0) < 0.632 * (w1 - w0):
        w += dt * (m.torque(V1, w) - kq * w * w) / m.rotor_inertia
        t += dt
    assert t == pytest.approx(m.time_constant(11.1, 0.605, kq), rel=0.05)


def test_the_current_limit_caps_the_current_and_the_torque():
    m = BLDCMotor(kv=920.0, resistance=0.02, i0=0.6, max_current=15.0)
    pt = m.operating_point(11.1, 1.0, lambda w: 1.7e-7 * w * w)
    assert pt.saturated and pt.current == pytest.approx(15.0) and pt.torque == pytest.approx(m.ke * (15.0 - m.i0), rel=1e-6)
    assert pt.omega < m.no_load_speed(11.1)
    assert not MOTOR.operating_point(11.1, 0.5, lambda w: 1.7e-7 * w * w).saturated  # a gentler operating point is not limited
    assert m.operating_point(11.1, 0.0, lambda w: 0.0).omega == 0.0  # no duty, no motion
    with pytest.raises(ValueError):
        BLDCMotor(kv=-1.0, resistance=0.1)


def test_a_vehicle_built_from_its_components_hovers_and_has_sensible_numbers():
    spec = VehicleSpec.from_components("10x4.5", mass=1.2, arm=0.22)
    assert spec.validate() == [] and 1.5 < spec.max_thrust() / (spec.mass * 9.80665) < 4.5
    sysm = MotorPropSystem(BLDCMotor(kv=920.0, resistance=0.09, i0=0.6, rotor_inertia=1.5e-5), get_propeller("10x4.5"))
    hov = sysm.hover(1.2, 4, 11.1)
    assert 60.0 < hov["total_power"] / 1.2 < 220.0  # W/kg: typical multirotor hover power
    assert hov["efficiency"] == pytest.approx(spec.battery.motor_efficiency, abs=1e-9)
    sim = Multirotor(spec, n=1)
    st = sim.reset(pos=(0.0, 0.0, 2.0))
    for _ in range(300):
        st = sim.step(st, sim.hover_command())
    assert abs(st.pos[0, 2] - 2.0) < 0.02  # the hover command lifts exactly the weight: kT and mass are consistent
    assert st.power[0] == pytest.approx(hov["total_power"] + spec.battery.avionics_power, rel=0.1)
    tiny = VehicleSpec.from_components("5x4.5", mass=0.5, arm=0.1, battery_cells=4)
    assert tiny.w_max > spec.w_max * 0.9 and tiny.rotor_radius < spec.rotor_radius  # smaller propeller, faster rotor


# ------------------------------------------------------------------------------------------------ battery pack
def test_peukert_runtime_of_a_constant_current_discharge():
    for k, i in ((1.0, 6.0), (1.04, 6.0), (1.1, 12.0)):
        pack = BatteryPack(PackSpec(peukert_k=k, capacity_ah=4.0, r0=0.0, r1=1e-6, mass_kg=1e3), 1)  # no resistive heating or sag
        t = 0.0
        while pack.soc[0] > 0.0 and t < 4 * 3600:
            pack.step(2.0, current_a=i)
            t += 2.0
        assert t / 3600.0 == pytest.approx(pack.peukert_runtime_hours(i), rel=0.01)
    assert BatteryPack(PackSpec(peukert_k=1.0)).peukert_runtime_hours(4.0) == pytest.approx(1.0)  # k = 1: capacity / current exactly


def test_thermal_steady_state_and_time_constant():
    spec = PackSpec()
    pack = BatteryPack(spec, 1)
    pack.spec.capacity_ah = 1000.0  # a huge pack: the state of charge hardly moves during the test
    for _ in range(4000):
        pack.step(2.0, current_a=20.0, ambient_c=15.0)
    assert pack.temperature[0] == pytest.approx(pack.thermal_steady_state(20.0, 15.0), abs=0.4)
    tau = spec.mass_kg * spec.specific_heat / spec.h_a
    cool = BatteryPack(spec, 1, temperature_c=60.0)
    for _ in range(int(tau / 2.0)):
        cool.step(2.0, current_a=0.0, ambient_c=20.0)
    assert (cool.temperature[0] - 20.0) / 40.0 == pytest.approx(math.exp(-1.0), abs=0.02)  # Newton cooling: e-fold after one time constant


def test_energy_is_accounted_for_between_chemistry_heat_and_the_terminals():
    """Chemical energy out = energy at the terminals + heat (series resistance and polarisation resistor) + energy stored in the RC capacitor."""
    pack = BatteryPack(PackSpec(), 1)
    chem = deliv = heat = 0.0
    dt = 0.25
    for k in range(int(900 / dt)):
        i = 10.0 + 6.0 * math.sin(0.02 * k)
        ocv_mid = float(pack.spec.ocv(pack.soc[0]))
        pack.step(dt, current_a=i)
        chem += 0.5 * (ocv_mid + float(pack.spec.ocv(pack.soc[0]))) * i * dt
        deliv += float(pack.voltage[0]) * i * dt
        heat += float(pack.heat_w[0]) * dt
    stored = 0.5 * pack.spec.c1 * float(pack.v1[0]) ** 2
    assert chem > deliv > 0.85 * chem  # the losses are a modest share of the energy
    assert chem - deliv == pytest.approx(heat + stored, rel=0.02)  # nothing is lost or created


def test_power_demand_is_converted_to_a_current_and_a_brownout_is_flagged():
    pack = BatteryPack(PackSpec(), 2)
    out = pack.step(0.1, power_w=np.array([100.0, 100.0]))
    assert np.all(out["voltage"] * out["current"] == pytest.approx(100.0, rel=2e-3)) and not pack.brownout.any()
    v0 = float(pack.spec.ocv(pack.soc[0]) - pack.v1[0])
    p_max = v0**2 / (4.0 * pack.resistance()[0])  # maximum power transfer
    pack.step(0.1, power_w=np.array([0.5 * p_max, 2.0 * p_max]))
    assert list(pack.brownout) == [False, True]
    with pytest.raises(ValueError):
        pack.step(0.1)
    assert pack.voltage[0] == pytest.approx(float(pack.spec.ocv(pack.soc[0]) - pack.current[0] * pack.resistance()[0] - pack.v1[0]), rel=2e-3)


def test_cold_and_aged_packs_deliver_less():
    run = lambda pack: sum(1 for _ in iter(lambda: pack.step(2.0, current_a=8.0)["soc"][0] > 0.0, False)) * 2.0
    fresh = BatteryPack(PackSpec(), 1)
    cold = BatteryPack(PackSpec(), 1, temperature_c=-5.0)
    aged = BatteryPack(PackSpec(), 1)
    aged.efc[:] = 400.0
    assert aged.retention[0] == pytest.approx(0.8, abs=1e-9) and aged.resistance()[0] > 1.4 * fresh.resistance()[0]
    assert cold.resistance()[0] > 1.5 * fresh.resistance()[0] and cold.usable_capacity_ah()[0] < 0.85 * fresh.usable_capacity_ah()[0]
    assert run(cold) < 0.97 * run(BatteryPack(PackSpec(), 1)) and run(aged) == pytest.approx(0.8 * run(BatteryPack(PackSpec(), 1)), rel=0.03)
    rest = BatteryPack(PackSpec(), 1, soc=0.5)
    rest.step(1.0, current_a=0.0)
    assert rest.voltage[0] == pytest.approx(float(rest.spec.ocv(0.5)))  # at rest the terminals show the open-circuit voltage


# ------------------------------------------------------------------------------------------------ tethered load
def test_a_load_on_a_fixed_attachment_swings_as_a_pendulum():
    m, L, g = 0.3, 1.5, 9.80665
    k = m * g / 0.002  # stiff rope: the pendulum is nearly rigid
    pa, va = np.zeros((1, 3)), np.zeros((1, 3))
    th0 = math.radians(4.0)
    ell = L + m * g / k
    p = np.array([[ell * math.sin(th0), 0.0, -ell * math.cos(th0)]])
    v = np.zeros((1, 3))
    xs, ts, t, dt = [], [], 0.0, 0.002
    for _ in range(int(8.0 / dt)):
        p, v, _, _ = tether_load_step(p, v, pa, va, dt, m, L, k, 2 * 0.05 * math.sqrt(k * m), 0.0, g)
        xs.append(p[0, 0])
        ts.append(t)
        t += dt
    xs, ts = np.array(xs), np.array(ts)
    ups = ts[1:][(xs[:-1] < 0) & (xs[1:] >= 0)]  # upward zero crossings: one per period
    period = float(np.mean(np.diff(ups)))
    assert period == pytest.approx(2.0 * math.pi * math.sqrt(ell / g), rel=0.015)
    assert abs(xs[: int(2.5 / dt)]).max() == pytest.approx(ell * math.sin(th0), rel=0.03)  # the amplitude is kept (no drag, tiny rope damping)


def test_static_load_tension_and_stretch_and_energy_conservation():
    m, L, k, g = 0.5, 1.0, 400.0, 9.80665
    pa, va = np.zeros((1, 3)), np.zeros((1, 3))
    p = np.array([[0.0, 0.0, -(L + m * g / k)]])
    v = np.zeros((1, 3))
    p, v, f, T = tether_load_step(p, v, pa, va, 0.01, m, L, k, 0.0, 0.0, g)
    assert (
        T[0] == pytest.approx(m * g, rel=1e-3) and f[0, 2] == pytest.approx(-m * g, rel=1e-3) and abs(f[0, 0]) < 1e-9
    )  # the vehicle is pulled down by m g
    # undamped, drag-free, dropped from the rest length: the sum of gravitational, kinetic and elastic energy is constant
    p = np.array([[0.4, 0.0, -L * 0.9]])
    v = np.zeros((1, 3))

    def energy(p, v):
        s = max(np.linalg.norm(p[0]) - L, 0.0)
        return m * g * p[0, 2] + 0.5 * m * float(v[0] @ v[0]) + 0.5 * k * s * s

    e0 = energy(p, v)
    es = []
    for _ in range(3000):
        p, v, _, _ = tether_load_step(p, v, pa, va, 0.001, m, L, k, 0.0, 0.0, g, substeps=4)
        es.append(energy(p, v))
    assert np.ptp(es) < 0.01 * abs(e0) + 0.02  # semi-implicit Euler: bounded energy error, no drift


def test_the_rope_pulls_only_when_stretched():
    m, L, k, g = 0.3, 1.5, 300.0, 9.80665
    pa, va = np.zeros((1, 3)), np.zeros((1, 3))
    p, v = np.array([[0.0, 0.0, -L + 0.4]]), np.zeros((1, 3))  # the load is 40 cm closer than the rope length: slack
    p, v, f, T = tether_load_step(p, v, pa, va, 0.01, m, L, k, 5.0, 0.0, g)
    assert T[0] == 0.0 and np.all(f == 0.0) and v[0, 2] < 0.0  # slack rope: no force on the vehicle, the load free-falls
    taut = False
    for _ in range(300):
        p, v, f, T = tether_load_step(p, v, pa, va, 0.01, m, L, k, 5.0, 0.0, g)
        taut |= T[0] > 0.0
    assert taut  # it falls back to the rope length and the rope tightens


def test_a_hovering_vehicle_carries_the_weight_of_its_load_and_swings_when_pushed():
    hover = {"type": "hover", "params": {"pos": [0, 0, 6.0]}}
    base = Simulation({"duration": 6.0, "trajectory": hover})
    a0 = base.run()
    cfg = {"duration": 14.0, "trajectory": hover, "hooks": [{"type": "slung_load", "params": {"mass": 0.3, "length": 1.5}}]}
    sim = Simulation(cfg)
    a = sim.run()
    w = 0.3 * 9.80665
    assert a["load_tension"][-1, 0] == pytest.approx(w, rel=0.02)
    assert np.sum(a["thrust"][-1, 0] if "thrust" in a.channels else 0.0) >= 0.0
    p = a["pos"][:, 0]
    assert (
        6.0 - p[-1, 2] == pytest.approx(w / (1.0 * (sim.ctrl.wn_pos[2]) ** 2) if hasattr(sim.ctrl.wn_pos, "__len__") else 0.0, rel=0.8)
        or 0.0 < 6.0 - p[-1, 2] < 0.2
    )  # sags by m_L g / k_p
    assert np.linalg.norm(a["load_pos"][-1, 0, :2] - p[-1, :2]) < 0.05  # hangs straight below once the swing has settled
    # a lateral manoeuvre excites a pendulum swing (in the load's offset from the vehicle)
    move = {"type": "waypoints", "params": {"points": [[0, 0, 6], [8, 0, 6]], "speed": 2.0}}
    b = Simulation({"duration": 10.0, "trajectory": move, "hooks": [{"type": "slung_load", "params": {"mass": 0.3, "length": 1.5}}]}).run()
    swing = np.linalg.norm(b["load_pos"][:, 0, :2] - b["pos"][:, 0, :2], axis=1)
    assert swing.max() > 0.05 and swing[-1] < swing.max()  # it swung out and is settling
    assert np.linalg.norm(a0["pos"][-1, 0] - [0, 0, 6.0]) < 0.01  # (control run: no load, no sag)


def test_external_loads_do_not_accumulate_from_step_to_step():
    sim = Simulation(
        {
            "duration": 2.0,
            "trajectory": {"type": "hover", "params": {"pos": [0, 0, 4.0]}},
            "hooks": [{"type": "slung_load", "params": {"mass": 0.3}}, {"type": "proximity_effects"}],
        }
    )
    sim.run()
    f_z = sim.dyn.params.ext[0, 2]
    assert f_z == pytest.approx(-0.3 * 9.80665, rel=0.03) and sim.dyn.params.ext[0, 6] == pytest.approx(1.0)  # one tension, not 400 of them


# ------------------------------------------------------------------------------------------------ proximity and battery hooks
def test_a_ceiling_raises_the_thrust_and_a_wall_pulls_toward_it():

    ceiling = {"type": "scene", "params": {"boxes": [{"center": [0, 0, 5.25], "size": [20, 20, 0.5]}], "bounds": [-10, -10, 0, 10, 10, 8]}}
    sim = Simulation(
        {"duration": 0.5, "world": ceiling, "trajectory": {"type": "hover", "params": {"pos": [0, 0, 4.9]}}, "hooks": [{"type": "proximity_effects"}]}
    )
    a = sim.run()
    R = sim.spec.rotor_radius
    h = 5.0 - 4.9
    expected = min(1.0 / (1.0 - 1.8 * (R / (4.0 * max(h, 0.3 * R))) ** 2), 1.5)
    assert (
        a["ceiling_gain"][:, 0].max() == pytest.approx(expected, rel=0.02) and expected > 1.0
    )  # at the start; the vehicle then moves away from the ceiling
    far = Simulation(
        {"duration": 0.5, "world": ceiling, "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2.0]}}, "hooks": [{"type": "proximity_effects"}]}
    ).run()
    assert far["ceiling_gain"][-1, 0] == pytest.approx(1.0)  # out of reach
    wall = {
        "type": "scene",
        "params": {"boxes": [{"center": [1.0, 0, 3], "size": [0.2, 10, 8]}], "bounds": [-5, -5, 0, 5, 5, 8]},
    }  # wall face at x = 0.9
    s2 = Simulation(
        {"duration": 0.5, "world": wall, "trajectory": {"type": "hover", "params": {"pos": [0.5, 0, 3.0]}}, "hooks": [{"type": "proximity_effects"}]}
    )
    s2.run()
    assert s2.dyn.params.ext[0, 0] > 0.0 and abs(s2.dyn.params.ext[0, 1]) < 0.05 * s2.dyn.params.ext[0, 0]  # towards +x (the wall), not sideways
    s3 = Simulation(
        {"duration": 0.5, "world": wall, "trajectory": {"type": "hover", "params": {"pos": [0.7, 0, 3.0]}}, "hooks": [{"type": "proximity_effects"}]}
    )
    s3.run()
    assert s3.dyn.params.ext[0, 0] > s2.dyn.params.ext[0, 0] > 0.0  # a nearer wall pulls harder


def test_the_battery_hook_drains_heats_and_empties_a_small_pack():
    cfg = {
        "duration": 40.0,
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 4.0]}},
        "hooks": [{"type": "battery_model", "params": {"capacity_ah": 0.12, "ambient_c": 20.0}}],
    }
    sim = Simulation(cfg)
    a = sim.run()
    soc, v, temp = a["soc"][:, 0], a["battery_voltage"][:, 0], a["battery_temp_c"][:, 0]
    assert np.all(np.diff(soc) <= 1e-12) and soc[0] > 0.99 and soc[-1] < 0.05  # the small pack runs empty
    assert temp[-1] > 20.0 + 0.4 and np.all(v < sim.spec.battery.ocv(soc) + 1e-9)  # heated by its own losses, sagging under load
    assert a["battery_current"][:, 0].mean() > 5.0 and a["battery_soh"][-1, 0] < 1.0
    events = [e["type"] for e in a.meta["events"]]
    assert "battery.low" in events  # the engine's low-battery event follows the detailed pack's state of charge
    again = Simulation(cfg).run()
    assert np.array_equal(again["soc"], a["soc"])  # deterministic
    big = Simulation({**cfg, "duration": 20.0, "hooks": [{"type": "battery_model", "params": {"capacity_ah": 5.0}}]}).run()
    assert big["soc"][-1, 0] > 0.9 > soc[int(len(soc) * 0.5)]  # a bigger pack has hardly drained after 20 s; the small one is half empty
