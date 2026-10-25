"""The pseudorange GNSS receiver."""

import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments.geodesy import LocalFrame
from aevion.environments.mesh import TriangleMesh
from aevion.environments.world import World
from aevion.sensors.gnss import CONSTELLATIONS, OMEGA_EARTH, GnssReceiver, dop, gmst_rad

B = NumpyBackend()
J2000 = 946_728_000.0


class Sim:
    def __init__(self, world, n):
        self.B, self.n, self.world = B, n, world


def state(pos, vel=None, t=0.0):
    pos = np.atleast_2d(np.asarray(pos, float))
    s = type("S", (), {})()
    s.pos, s.vel, s.t = pos, np.zeros_like(pos) if vel is None else np.atleast_2d(np.asarray(vel, float)), t
    return s


def make(world=None, n=1, **kw):
    w = world or World()
    if w.origin is None:
        w.origin = LocalFrame(47.37, 8.54, 400.0)
    if w.utc is None:
        w.utc = "2026-06-21T10:00:00Z"
    g = GnssReceiver(**kw)
    g.bind(Sim(w, n), 3)
    return g


QUIET = dict(sisre_m=0.0, multipath_sigma_m=0.0, iono_vertical_m=0.0, tropo_residual=0.0, clock_bias_std_m=0.0, clock_drift_walk=0.0)


# ------------------------------------------------------------------------------------------ constellations
def test_orbit_geometry_matches_the_published_system_parameters():
    gps = CONSTELLATIONS["gps"]
    assert gps.n_sats == 30 and gps.period_s == pytest.approx(43078.0, abs=40.0)  # 11 h 58 min
    pos, vel = gps.ecef(J2000 + 1e7)
    inertial_r = np.linalg.norm(pos, axis=1)
    np.testing.assert_allclose(inertial_r, gps.semi_major_m, rtol=1e-12)  # circular orbit
    speed_inertial = np.linalg.norm(vel + np.cross([0, 0, OMEGA_EARTH], pos), axis=1)
    np.testing.assert_allclose(speed_inertial, gps.semi_major_m * gps.mean_motion, rtol=1e-9)
    assert speed_inertial[0] == pytest.approx(3874.0, abs=15.0)  # GPS orbital speed ~3.87 km/s
    lat_max = max(np.degrees(np.arcsin(np.abs(gps.ecef(J2000 + t)[0][:, 2]) / gps.semi_major_m)).max() for t in np.linspace(0, gps.period_s, 60))
    assert lat_max == pytest.approx(55.0, abs=0.7)  # the ground track reaches the orbital inclination
    for name, (n, inc) in {"galileo": (24, 56.0), "glonass": (24, 64.8), "beidou": (24, 55.0)}.items():
        assert CONSTELLATIONS[name].n_sats == n and CONSTELLATIONS[name].inclination_deg == inc
    assert CONSTELLATIONS["glonass"].period_s == pytest.approx(40544.0, abs=60.0)  # 11 h 15 min
    assert CONSTELLATIONS["galileo"].period_s == pytest.approx(50680.0, abs=80.0)  # 14 h 05 min


def test_velocity_is_the_derivative_of_position_in_the_earth_fixed_frame():
    for c in CONSTELLATIONS.values():
        t = J2000 + 5e6
        p0, v = c.ecef(t)
        p1, _ = c.ecef(t + 1.0)
        pm, _ = c.ecef(t - 1.0)
        np.testing.assert_allclose((p1 - pm) / 2.0, v, atol=0.01)


def test_gps_ground_track_repeats_every_sidereal_day():
    gps = CONSTELLATIONS["gps"]
    t = J2000 + 3e6
    p0, _ = gps.ecef(t)
    p1, _ = gps.ecef(t + 86164.0905)  # one sidereal day = two GPS orbits (to within the modelled period)
    assert np.max(np.linalg.norm(p1 - p0, axis=1)) < 80e3
    assert gmst_rad(J2000) == pytest.approx(math.radians(280.46061837), abs=1e-9)
    d = (gmst_rad(J2000 + 86164.0905) - gmst_rad(J2000)) % (2 * math.pi)
    assert min(d, 2 * math.pi - d) < 1e-5  # the sidereal angle advances by exactly one turn


def test_visibility_statistics_over_a_day():
    counts = {"gps": [], "all": []}
    g_gps, g_all = make(constellations=("gps",), urban=False), make(constellations=("gps", "galileo", "glonass", "beidou"), urban=False)
    for t in np.arange(0, 86400, 600.0):
        counts["gps"].append(int(g_gps.measure(state([0, 0, 10], t=t))["n_sat"][0]))
        counts["all"].append(int(g_all.measure(state([0, 0, 10], t=t))["n_sat"][0]))
    assert min(counts["gps"]) >= 6 and np.mean(counts["gps"]) == pytest.approx(9.5, abs=2.5)  # GPS alone: 8-12 above 10 degrees
    assert min(counts["all"]) >= 16 and np.mean(counts["all"]) > 25


# --------------------------------------------------------------------------------------------- the solution
def test_noise_free_receiver_recovers_position_velocity_and_clock_exactly():
    g = make(n=3, ideal=True, constellations=("gps", "galileo"))
    pos = [[0, 0, 50.0], [120.0, -40.0, 20.0], [-80.0, 300.0, 100.0]]
    vel = [[5.0, 0, 0], [0, -3.0, 1.0], [1.0, 2.0, -0.5]]
    r = g.measure(state(pos, vel))
    np.testing.assert_allclose(r["pos"], pos, atol=1e-6)
    np.testing.assert_allclose(r["vel"], vel, atol=1e-6)
    assert np.all(r["fix"] == 1.0) and np.all(r["n_sat"] >= 8) and np.all(r["clock_bias_m"] == 0.0)


def test_clock_bias_is_estimated():
    g = make(constellations=("gps", "galileo"), **{**QUIET, "clock_bias_std_m": 5000.0}, noise_sigma_m=1e-6, nlos_probability=0.0)
    g.sis[:] = 0.0
    g.mp_sigma = 0.0
    r = g.measure(state([0, 0, 30.0]))
    assert r["clock_bias_m"][0] == pytest.approx(g.clock_bias[0], abs=1e-3)
    assert np.linalg.norm(r["pos"][0] - [0, 0, 30.0]) < 1e-3  # a 5 km clock bias does not leak into the position


def test_dop_matches_the_closed_form_for_a_symmetric_constellation():
    # one satellite at zenith and three at elevation e, 120 deg apart: H rows [-e, 1]
    el = math.radians(20.0)
    dirs = [[0, 0, 1.0]] + [[math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)] for a in np.radians([0, 120, 240])]
    e = np.array(dirs)[None]
    d = dop(e, np.ones((1, 4), bool))
    H = np.concatenate([-e[0], np.ones((4, 1))], axis=1)
    Q = np.linalg.inv(H.T @ H)
    assert d["gdop"][0] == pytest.approx(math.sqrt(np.trace(Q))) and d["hdop"][0] == pytest.approx(math.sqrt(Q[0, 0] + Q[1, 1]))
    # for this symmetric geometry the horizontal DOP has the closed form 1 / sqrt(1.5 cos^2 e) per axis pair
    assert d["hdop"][0] == pytest.approx(math.sqrt(2.0 / (1.5 * math.cos(el) ** 2)), rel=1e-9)
    assert d["pdop"][0] ** 2 == pytest.approx(d["hdop"][0] ** 2 + d["vdop"][0] ** 2)
    assert d["gdop"][0] ** 2 == pytest.approx(d["pdop"][0] ** 2 + d["tdop"][0] ** 2)
    few = dop(e, np.array([[True, True, True, False]]))
    assert np.isinf(few["gdop"][0])
    flat = dop(np.array([[[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]], float), np.ones((1, 4), bool))  # all on the horizon plane: singular
    assert flat["vdop"][0] > 1e3 or np.isinf(flat["vdop"][0])


def test_position_error_statistics_follow_the_dilution_of_precision():
    """With only thermal noise, the error covariance of the weighted fix is (H^T W H)^-1, which we rebuild independently."""
    g = make(ideal=False, constellations=("gps", "galileo"), noise_sigma_m=1.0, nlos_probability=0.0, doppler_sigma_mps=0.001, **QUIET)
    g.sis[:] = 0.0
    errs, preds = [], []
    truth = np.array([0.0, 0.0, 20.0])
    for k in range(2500):
        r = g.measure(state(truth, t=0.0))
        errs.append(r["pos"][0] - truth)
        if k == 0:
            used = r["sat_used"][0]
            el = np.radians(r["sat_el_deg"][0])
            sat, _ = g._satellites(0.0)
            usr = g.frame.enu_to_ecef(truth[None])[0]
            e_enu = ((sat - usr) / np.linalg.norm(sat - usr, axis=1)[:, None]) @ g.frame.rotation.T
            cn0 = g.cn0_zenith - 9.0 * (1.0 - np.sin(np.maximum(el, 0)))
            w = np.where(used, 1.0 / (1.0 * 10.0 ** (-(cn0 - 45.0) / 20.0)) ** 2, 0.0)
            H = np.concatenate([-e_enu, np.ones((len(e_enu), 1))], axis=1)
            preds = np.sqrt(np.diag(np.linalg.inv(H.T @ (w[:, None] * H)))[:3])
    emp = np.std(np.array(errs), axis=0)
    np.testing.assert_allclose(emp, preds, rtol=0.07)
    assert abs(np.mean(np.array(errs))) < 0.1


def test_realistic_error_budget_and_determinism():
    kw = dict(constellations=("gps", "galileo"), rate=1.0)  # one epoch per second, as the test samples
    a, b = make(**kw), make(**kw)
    err_a = []
    for t in np.arange(0, 600, 1.0):
        ra, rb = a.measure(state([0, 0, 10.0], t=t)), b.measure(state([0, 0, 10.0], t=t))
        np.testing.assert_array_equal(ra["pos"], rb["pos"])  # same seed, same epochs, same numbers
        err_a.append(ra["pos"][0] - [0, 0, 10.0])
    vec = np.array(err_a)
    err = np.linalg.norm(vec, axis=1)
    assert 1.0 < np.percentile(err, 50) < 6.0 and np.percentile(err, 95) < 12.0  # single-frequency, no augmentation: metres
    assert np.linalg.norm(vec.mean(0)) > 2.0 * vec.std(0).max()  # dominated by a systematic part (atmosphere, orbit), not by white noise
    assert abs(vec.mean(0)[2]) > abs(vec.mean(0)[:2]).max()  # and, as in real receivers, the vertical error is the largest
    assert make(constellations=("gps",)).to_dict()["constellations"] == ["gps"]


def test_geometry_depends_on_utc_and_location():
    base = make(urban=False, constellations=("gps",))
    r0 = base.measure(state([0, 0, 0]))
    w2 = World()
    w2.utc, w2.origin = "2026-06-21T15:20:00Z", LocalFrame(47.37, 8.54, 400.0)
    other = make(w2, urban=False, constellations=("gps",))
    assert not np.array_equal(r0["sat_el_deg"], other.measure(state([0, 0, 0]))["sat_el_deg"])
    w3 = World()
    w3.utc, w3.origin = "2026-06-21T10:00:00Z", LocalFrame(-33.9, 151.2, 20.0)
    assert not np.array_equal(r0["sat_el_deg"], make(w3, urban=False, constellations=("gps",)).measure(state([0, 0, 0]))["sat_el_deg"])


# ---------------------------------------------------------------------------------------------- obstacles
def canyon_world():
    w = World()
    w.add_mesh(TriangleMesh.box((-35, 0, 12.5), (30, 300, 25)))  # a 40 m wide street with 25 m walls east and west of the vehicle
    w.add_mesh(TriangleMesh.box((35, 0, 12.5), (30, 300, 25)))
    return w


def test_urban_canyon_blocks_satellites_and_degrades_geometry():
    pos = [0.0, 0.0, 2.0]
    open_sky = make(urban=False, constellations=("gps", "galileo"))
    r_open = open_sky.measure(state(pos))
    g = make(canyon_world(), urban=True, constellations=("gps", "galileo"), nlos_probability=0.0)
    r = g.measure(state(pos))
    assert r["n_sat"][0] < r_open["n_sat"][0] - 3
    lost = r_open["sat_used"][0] & ~r["sat_used"][0]
    el_lost = r_open["sat_el_deg"][0][lost]
    el_kept = r["sat_el_deg"][0][r["sat_used"][0]]
    assert el_lost.mean() < el_kept.mean()  # the walls take the low-elevation satellites to the east and west


def test_reflected_signals_add_positive_range_bias_and_errors():
    many = []
    for p_nlos in (0.0, 1.0):
        g = make(canyon_world(), n=1, urban=True, constellations=("gps", "galileo", "glonass"), nlos_probability=p_nlos, **QUIET)
        g.sis[:] = 0.0
        errs = [
            np.linalg.norm(g.measure(state([0, 0, 2.0], t=t))["pos"][0] - [0, 0, 2.0])
            for t in np.arange(0, 600, 30.0)
            if g.measure(state([0, 0, 2.0], t=t)) is not None
        ]
        many.append(np.mean(errs))
    assert many[1] > 2.0 * many[0]  # accepting NLoS signals is worse than rejecting them


def test_terrain_and_walls_can_remove_the_fix_entirely():
    tunnel = World()
    tunnel.add_mesh(TriangleMesh.box((0, 0, 0), (100, 100, 60)).with_color((0.5, 0.5, 0.5)))
    g = make(tunnel, urban=True, nlos_probability=0.0, constellations=("gps", "galileo"))
    assert g.measure(state([0.0, 0.0, 0.0])) is None  # inside a solid block: nothing to receive
    from aevion.sensors.base import Sensor

    s = Sensor.update
    g.rate = 5.0
    assert s(g, state([0.0, 0.0, 0.0], t=0.0)) is False and g.latest is None  # an empty epoch is a gap, not a sample


def test_jamming_hook_lowers_cn0_and_loses_the_fix():
    pos = [0.0, 0.0, 20.0]
    g = make(constellations=("gps", "galileo"), urban=False)
    clean = g.measure(state(pos))
    g.interference_fn = lambda p: np.full(len(p), 16.0)
    weak = g.measure(state(pos))
    assert weak["cn0"][0] < clean["cn0"][0] - 8.0 and weak["n_sat"][0] < clean["n_sat"][0]  # low-elevation satellites drop below lock
    g.interference_fn = lambda p: np.full(len(p), 45.0)
    assert g.measure(state(pos)) is None


# ------------------------------------------------------------------------------------------------ engine
def test_gnss_in_the_engine_with_estimator_and_jamming_fault():
    from aevion.engine import Simulation

    cfg = {
        "duration": 8.0,
        "world": {"type": "scene", "params": {"origin": {"lat": 47.37, "lon": 8.54, "alt": 400.0}, "utc": "2026-06-21T10:00:00Z"}},
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 20.0], "radius": 10.0, "omega": 0.3}},
        "sensors": [
            "imu",
            "mag",
            "baro",
            {"type": "gnss", "name": "gps", "params": {"rate": 5.0, "constellations": ["gps", "galileo"]}, "record": True},
        ],
        "estimator": {"type": "mahony_kf", "handover_s": 2.0},
        "faults": [{"type": "gnss_jamming", "t_start": 4.0, "params": {"pos": [0, 0, 20.0], "power_dbm": 45.0, "sensor": "gps"}}],
    }
    arch = Simulation(cfg).run()
    rec = arch["sensors/gps"]  # (samples, N, 11): pos(3), vel(3), fix, n_sat, hdop, pdop, cn0
    t = arch["sensors/gps/t"] - arch.t[0]
    assert rec.shape[2] == 11 and rec.shape[0] >= 10
    before, after = rec[t < 3.5, 0], rec[t > 5.0, 0]
    assert before[:, 7].min() > 5 and np.all(before[:, 6] == 1.0)  # satellites in view, fix held before the jammer switches on
    assert len(after) < len(before) or np.mean(after[:, 10]) < np.mean(before[:, 10]) - 5.0  # jamming: fewer epochs or visibly lower C/N0
    ev = [e["type"] for e in arch.meta["events"]]
    assert "fault.start" in ev or "gnss_jamming" in ev


def test_validation_and_registry():
    for bad in (dict(constellations=()), dict(constellations=("gps", "nope")), dict(elevation_mask_deg=95.0), dict(nlos_probability=2.0)):
        with pytest.raises(ValueError):
            GnssReceiver(**bad)
    from aevion.registry import registries

    assert "gnss" in registries.sensors.names()
    assert isinstance(registries.sensors.create("gnss", rate=2.0), GnssReceiver)
