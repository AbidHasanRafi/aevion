"""The fused Numba kernel must reproduce the generic reference implementation to round-off."""

import numpy as np
import pytest

from aevion import Multirotor, VehicleSpec, World
from aevion.core.constants import Atmosphere, GravityModel
from aevion.dynamics import fast
from aevion.environments.wind import ConstantWind

pytestmark = pytest.mark.skipif(not fast.HAVE_NUMBA, reason="numba not installed")


def _run(sim, steps, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    s = sim.reset(
        pos=rng.uniform([-2, -2, 1], [2, 2, 4], (sim.n, 3)),
        vel=rng.normal(size=(sim.n, 3)),
        euler=rng.normal(size=(sim.n, 3)) * 0.3,
        omega=rng.normal(size=(sim.n, 3)) * 0.5,
    )
    hover = sim.hover_speed()
    for k in range(steps):
        cmd = hover * (1 + 0.15 * np.sin(0.05 * k + np.arange(sim.n_rotors)[None, :] + np.arange(sim.n)[:, None]))
        s = sim.step(s, cmd)
    return s


def _pair(world_factory, vehicle="generic_1kg", integrator="rk4", n=5, **kw):
    a = Multirotor(vehicle, n=n, dt=0.004, integrator=integrator, world=world_factory(), kernel="reference", **kw)
    b = Multirotor(vehicle, n=n, dt=0.004, integrator=integrator, world=world_factory(), kernel="numba", **kw)
    assert not a.fast and b.fast
    return a, b


def _assert_close(sa, sb, tol=1e-8):
    for name in ("pos", "vel", "quat", "omega", "rotor", "soc", "energy", "acc", "f_body", "thrust", "power"):
        np.testing.assert_allclose(getattr(sa, name), getattr(sb, name), rtol=tol, atol=tol, err_msg=name)
    np.testing.assert_array_equal(sa.contact, sb.contact)


@pytest.mark.parametrize("integrator", ["rk4", "midpoint", "euler", "semi_implicit"])
def test_integrators(integrator):
    a, b = _pair(lambda: World(wind=ConstantWind([2.0, -1.0, 0.3])), integrator=integrator)
    _assert_close(_run(a, 300), _run(b, 300))


@pytest.mark.parametrize("vehicle", ["crazyflie", "racing_5inch", "camera_2kg", "hex_3kg", "generic_1kg"])
def test_vehicles(vehicle):
    a, b = _pair(lambda: World(), vehicle=vehicle)
    _assert_close(_run(a, 250), _run(b, 250))


def test_gravity_atmosphere_variants():
    def wf():
        return World(
            gravity=GravityModel("wgs84", latitude_deg=60.0, altitude_ref=1500.0), atmosphere=Atmosphere(base_altitude=1500.0, temp_offset=15.0)
        )

    a, b = _pair(wf)
    _assert_close(_run(a, 300), _run(b, 300))
    a, b = _pair(lambda: World(gravity=GravityModel("altitude")))
    _assert_close(_run(a, 200), _run(b, 200))


def test_terrain_and_contact():
    def wf():
        return World().set_heightmap(np.random.default_rng(3).uniform(0, 2, (40, 40)), origin=(-10, -10), resolution=0.5)

    a, b = _pair(wf)
    _assert_close(_run(a, 400), _run(b, 400), tol=1e-7)


def test_obstacle_collisions():
    def wf():
        w = World()
        w.add_box((1, 0, 2), (1.0, 1.0, 4.0))
        w.add_sphere((-1.5, 1.0, 2.0), 0.8)
        w.add_cylinder((0.0, -1.5), 0.5, 0.0, 5.0)
        return w

    a, b = _pair(wf, n=8)
    sa, sb = _run(a, 500), _run(b, 500)
    _assert_close(sa, sb, tol=1e-6)


def test_motor_failure_and_randomized_params():
    a, b = _pair(lambda: World())
    for sim in (a, b):
        sim.params.set_rotor_failure(0, 1, 1.0)
        sim.params.set_rotor_failure(1, 0, 0.4)
        sim.params.randomize(sim.B.make_rng(5), mass=0.1, inertia=0.1, kT=0.05, tau=0.2, drag=0.3, com=0.01)
    _assert_close(_run(a, 300), _run(b, 300))


def test_battery_voltage_effects():
    spec = VehicleSpec.generic_1kg()
    spec.battery.voltage_effects = True
    spec.battery.capacity_wh = 2.0
    a, b = _pair(lambda: World(), vehicle=spec)
    _assert_close(_run(a, 400), _run(b, 400))


def test_a_payload_changes_the_motion_identically_in_both_kernels():
    """Regression: the fused kernel once ignored ``payload_mass`` (the ``payload_change`` fault did nothing under it)."""
    a, b = _pair(lambda: World())
    for sim in (a, b):
        sim.params.payload_mass[:, 0] = np.array([0.0, 0.2, 0.5, -0.2, 0.1])
        sim.params.com_offset[2] = np.array([0.02, -0.01, 0.0])
    sa, sb = _run(a, 300), _run(b, 300)
    _assert_close(sa, sb)
    plain = _run(Multirotor("generic_1kg", n=5, dt=0.004, world=World(), kernel="numba"), 300)
    assert np.abs(sb.pos[2] - plain.pos[2]).max() > 0.05  # a 0.5 kg payload visibly changes the flight of vehicle 2


def test_external_loads_and_the_thrust_multiplier_agree_between_kernels():
    """The hooks that model tethers and wall / ceiling effects write ``params.ext``; both kernels must apply it identically."""
    rng = np.random.default_rng(3)
    ext = rng.normal(size=(5, 7)) * [2.0, 2.0, 2.0, 0.05, 0.05, 0.02, 0.0] + [0, 0, 0, 0, 0, 0, 1.0]
    ext[:, 6] = [1.0, 1.1, 0.9, 1.25, 1.0]

    def run(sim, steps=300, loads=True):
        r = np.random.default_rng(0)
        s = sim.reset(pos=r.uniform([-2, -2, 1], [2, 2, 4], (sim.n, 3)), vel=r.normal(size=(sim.n, 3)), euler=r.normal(size=(sim.n, 3)) * 0.3)
        if loads:
            sim.params.ext[:] = ext  # reset() clears the loads, so they are applied after it, as the hooks do every step
        hover = sim.hover_speed()
        for k in range(steps):
            s = sim.step(s, hover * (1 + 0.15 * np.sin(0.05 * k + np.arange(sim.n_rotors)[None, :] + np.arange(sim.n)[:, None])))
        return s

    a, b = _pair(lambda: World())
    sa, sb = run(a), run(b)
    _assert_close(sa, sb)
    assert np.abs(sb.pos - run(Multirotor("generic_1kg", n=5, dt=0.004, world=World(), kernel="numba"), loads=False).pos).max() > 0.2  # they act
    a.reset()
    assert np.all(a.params.ext[:, :6] == 0.0) and np.all(a.params.ext[:, 6] == 1.0)  # a new episode starts from nothing
