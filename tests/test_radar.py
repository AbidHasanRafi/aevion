"""FMCW radar."""

import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments.actors import LinearActor
from aevion.environments.world import World
from aevion.sensors.radar import C_LIGHT, K_BOLTZMANN, Radar, ca_cfar

B = NumpyBackend()


class Sim:
    def __init__(self, world, n):
        self.B, self.n, self.world = B, n, world


def st(pos=(0, 0, 5.0), vel=(0, 0, 0), t=0.0, n=1):
    s = type("S", (), {})()
    s.pos, s.vel = np.tile(np.asarray(pos, float), (n, 1)), np.tile(np.asarray(vel, float), (n, 1))
    s.quat, s.t = np.tile([1.0, 0, 0, 0], (n, 1)), t
    return s


def wall(x=30.0):
    w = World()
    w.add_box((x, 0, 5.0), (0.5, 60.0, 30.0))
    return w


def radar(world=None, n=1, **kw):
    kw.setdefault("ground_clutter", False)
    r = Radar(**kw)
    r.bind(Sim(world or wall(), n), 1)
    return r


# ----------------------------------------------------------------------------------------------- formulas
def test_waveform_figures_follow_the_fmcw_relations():
    r = Radar()
    lam = C_LIGHT / 77e9
    assert r.wavelength == pytest.approx(lam) and lam == pytest.approx(3.893e-3, rel=1e-3)
    assert r.range_resolution == pytest.approx(C_LIGHT / 2e9) and r.range_resolution == pytest.approx(0.15, rel=1e-3)
    assert r.velocity_resolution == pytest.approx(lam / (2 * 128 * 40e-6))
    assert r.max_unambiguous_velocity == pytest.approx(lam / (4 * 40e-6)) and r.max_unambiguous_velocity == pytest.approx(24.3, abs=0.1)
    wide = Radar(bandwidth_hz=4e9, n_chirps=512, chirp_s=20e-6)
    assert (
        wide.range_resolution < r.range_resolution
        and wide.velocity_resolution < r.velocity_resolution
        and wide.max_unambiguous_velocity > r.max_unambiguous_velocity
    )
    # velocity resolution and unambiguous velocity trade off through the number of chirps: v_max / dv = N / 2
    assert r.max_unambiguous_velocity / r.velocity_resolution == pytest.approx(r.n_chirps / 2)


def test_radar_equation_snr_and_detection_range():
    r = Radar()
    pt, g = 1e-3 * 10 ** (r.pt_dbm / 10), 10 ** (r.gain_dbi / 10)
    n0 = K_BOLTZMANN * 290.0 * 10 ** ((r.nf_db + r.losses_db) / 10)
    by_hand = 10 * math.log10(pt * g * g * r.wavelength**2 * 10.0 / ((4 * math.pi) ** 3 * 50.0**4) * (128 * 40e-6) / n0)
    assert float(r.snr_db(10.0, 50.0)) == pytest.approx(by_hand, abs=1e-9)
    assert float(r.snr_db(10.0, 100.0)) == pytest.approx(float(r.snr_db(10.0, 50.0)) - 40 * math.log10(2), abs=1e-9)  # R^-4: -12.04 dB per doubling
    assert float(r.snr_db(40.0, 50.0)) == pytest.approx(float(r.snr_db(10.0, 50.0)) + 6.0206, abs=1e-3)
    rmax = r.detection_range(10.0)
    assert float(r.snr_db(10.0, rmax)) == pytest.approx(r.thr_db, abs=1e-9)
    assert r.detection_range(0.01) < r.detection_range(10.0) and r.detection_range(10.0) / r.detection_range(0.01) == pytest.approx(1000.0**0.25)
    assert r.noise_psd == pytest.approx(n0) and r.integration_s == pytest.approx(128 * 40e-6)
    assert (
        150.0 < r.detection_range(10.0) < 500.0 and 50.0 < r.detection_range(1.0) < 300.0
    )  # plausible 77 GHz figures for a car and a pedestrian-sized target


def test_velocity_wraps_into_the_unambiguous_interval():
    r = Radar()
    vm = r.max_unambiguous_velocity
    np.testing.assert_allclose(r.wrap_velocity([0.0, 5.0, -5.0, vm + 3.0, -vm - 3.0, 30.0]), [0.0, 5.0, -5.0, -vm + 3.0, vm - 3.0, 30.0 - 2 * vm])
    assert float(r.wrap_velocity(vm)) == pytest.approx(-vm)  # the interval is [-v_max, v_max)


# ------------------------------------------------------------------------------------------ detections
def test_a_wall_is_detected_at_the_right_range_with_the_right_closing_speed():
    r = radar(wall(30.0))
    out = r.measure(st(vel=(5.0, 0, 0)))
    v = np.asarray(out["valid"][0])
    d = np.asarray(out["detections"][0])
    assert v.sum() >= 3
    centre = np.argmin(np.abs(r.beam_az) + np.abs(r.beam_el))
    assert v[centre] and d[centre, 0] == pytest.approx(29.75, abs=0.1)  # wall face at x = 29.75
    assert d[centre, 3] == pytest.approx(5.0, abs=0.1)  # closing at 5 m/s: positive
    away = np.asarray(radar(wall(30.0)).measure(st(vel=(-5.0, 0, 0)))["detections"][0])[centre, 3]
    assert away == pytest.approx(-5.0, abs=0.1)
    assert np.asarray(radar(wall(30.0)).measure(st(vel=(0, 4.0, 0)))["detections"][0])[centre, 3] == pytest.approx(
        0.0, abs=0.1
    )  # sideways: no Doppler on boresight
    assert np.all(np.asarray(out["detections"][0])[~v] == 0.0)  # undetected beams are zeroed


def test_off_boresight_beams_see_a_reduced_doppler():
    r = radar(wall(30.0), beams_h=15, beams_v=1, fov_h_deg=90.0, fov_v_deg=0.0)
    out = r.measure(st(vel=(8.0, 0, 0)))
    d, v = np.asarray(out["detections"][0]), np.asarray(out["valid"][0])
    for k in np.nonzero(v)[0]:
        assert d[k, 3] == pytest.approx(8.0 * math.cos(r.beam_az[k]), abs=0.15)  # radial speed = v cos(azimuth)


def test_small_targets_disappear_beyond_their_detection_range():
    r = radar(wall(30.0), rcs_m2=10.0)
    assert r.measure(st())["valid"].any()
    dim = radar(wall(30.0), rcs_m2=1e-4)  # a tiny reflector: detection range far below 30 m
    assert dim.detection_range(1e-4) < 29.0 and not np.asarray(dim.measure(st())["valid"]).any()
    close = radar(wall(30.0), rcs_m2=1e-4)
    assert np.asarray(close.measure(st(pos=(29.0 - close.detection_range(1e-4) * 0.5, 0, 5.0)))["valid"]).any()  # ... but visible when close enough


def test_incidence_angle_reduces_the_cross_section():
    w = World()
    w.add_box((30, 0, 5), (0.5, 60, 30))
    r = radar(w, beams_h=15, beams_v=1, fov_h_deg=60.0, fov_v_deg=0.0)
    d = np.asarray(r.measure(st())["detections"][0])
    inc = np.abs(r.beam_az)
    rcs = d[:, 5]
    mask = rcs > 0
    np.testing.assert_allclose(rcs[mask], 10.0 * np.cos(inc[mask]) ** 2, rtol=1e-6)  # specular cos^2 lobe


def test_moving_actor_doppler_adds_to_the_vehicle_motion():
    w = World()
    w.add_actor(LinearActor(start=(40.0, 0.0, 5.0), velocity=(-8.0, 0.0, 0.0), radius=1.5))
    r = radar(w, rcs_m2=100.0, beams_h=5, beams_v=1, fov_v_deg=0.0, fov_h_deg=10.0)
    w.advance(0.0)
    out = r.measure(st(vel=(0, 0, 0), t=0.0))
    d, v = np.asarray(out["detections"][0]), np.asarray(out["valid"][0])
    k = np.nonzero(v)[0][np.argmin(np.abs(r.beam_az[v]))]
    assert d[k, 3] == pytest.approx(8.0, abs=0.2)  # an oncoming target at 8 m/s approaches at +8
    both = r.measure(st(vel=(3.0, 0, 0), t=0.0))
    assert np.asarray(both["detections"][0])[k, 3] == pytest.approx(11.0, abs=0.2)


def test_velocity_ambiguity_aliases_like_the_hardware():
    r = radar(wall(30.0), beams_h=1, beams_v=1)
    vm = r.max_unambiguous_velocity
    out = r.measure(st(vel=(30.0, 0, 0)))
    assert np.asarray(out["detections"][0])[0, 3] == pytest.approx(30.0 - 2 * vm, abs=0.2)  # 30 m/s reads as -18.7 m/s
    unwrapped = Radar(chirp_s=10e-6)  # shorter chirps: larger unambiguous velocity
    assert unwrapped.max_unambiguous_velocity > 30.0


def test_measurement_noise_follows_the_cramer_rao_scaling():
    n = 3000
    r = radar(wall(20.0), n=n, beams_h=1, beams_v=1)
    out = r.measure(st(n=n))
    det = np.asarray(out["detections"])[:, 0]
    snr = 10 ** (det[0, 4] / 10)
    np.testing.assert_allclose(np.std(det[:, 0]), r.range_resolution / math.sqrt(2 * snr), rtol=0.08)
    np.testing.assert_allclose(np.std(det[:, 3]), r.velocity_resolution / math.sqrt(2 * snr), rtol=0.08)
    assert abs(np.mean(det[:, 0]) - 19.75) < 0.01 and snr > 1000.0
    far = radar(wall(80.0), n=n, beams_h=1, beams_v=1, rcs_m2=400.0)
    detf = np.asarray(far.measure(st(n=n))["detections"])[:, 0]
    assert np.std(detf[:, 0]) > np.std(det[:, 0])  # weaker echo, larger error


def test_ground_clutter_can_be_switched_off():
    w = World()
    down = dict(beams_h=3, beams_v=1, fov_h_deg=20.0, fov_v_deg=0.0)
    r = radar(w, ground_clutter=True, ground_rcs_m2=5.0, **down)
    r.dirs_body = np.tile([0.5, 0.0, -0.8660254], (3, 1))  # look 60 degrees down
    out = r.measure(st(pos=(0, 0, 8.0)))
    assert np.asarray(out["valid"]).any()
    quiet = radar(w, ground_clutter=False, **down)
    quiet.dirs_body = r.dirs_body
    assert not np.asarray(quiet.measure(st(pos=(0, 0, 8.0)))["valid"]).any()


# ---------------------------------------------------------------------------------------- range-Doppler
def test_range_doppler_map_peaks_at_the_target_cell_with_the_snr_power():
    w = wall(30.0)
    r = radar(w, beams_h=1, beams_v=1, map=True, n_range_bins=100)
    centre_range = (np.arange(100) + 0.5) * 1.0  # 1 m bins up to max_range = 100 m
    # choose a vehicle position so the wall face is exactly at a bin centre, and a speed on a Doppler bin centre
    pos_x = 29.75 - centre_range[24]
    v_bin = (80 - 64) * (2 * r.max_unambiguous_velocity / 128)  # Doppler bin 80 (bin 64 is zero velocity)
    out = r.measure(st(pos=(pos_x, 0, 5.0), vel=(v_bin, 0, 0)))
    m = np.asarray(out["rd_map"][0])
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    assert (ix, iy) == (24, 80)  # the wall's range cell and its Doppler cell
    snr_db = float(np.asarray(out["detections"][0])[0, 4])
    assert m.max() == pytest.approx(snr_db, abs=0.5)  # peak power = SNR
    floor = np.percentile(m, 50)
    assert floor == pytest.approx(0.0, abs=0.5)  # thermal floor sits at 0 dB
    assert m.shape == (128, 100)


def test_range_doppler_map_wraps_aliased_velocities():
    r = radar(wall(30.0), beams_h=1, beams_v=1, map=True, n_range_bins=100)
    vm = r.max_unambiguous_velocity
    m_alias = np.asarray(r.measure(st(vel=(vm + 4.0, 0, 0)))["rd_map"][0])
    iy, _ = np.unravel_index(np.argmax(m_alias), m_alias.shape)
    v_axis = (np.arange(128) - 64) * (2 * vm / 128)
    assert v_axis[iy] == pytest.approx(-vm + 4.0, abs=2 * vm / 128)  # the peak appears at the wrapped velocity


def test_ca_cfar_has_the_requested_false_alarm_rate_and_finds_targets():
    rng = np.random.default_rng(0)
    pfa = 1e-3
    noise_db = 10 * np.log10(rng.exponential(1.0, (250, 250)))  # exponential noise power = chi-square with 2 dof after the FFTs
    mask = ca_cfar(noise_db, pfa=pfa)
    assert mask.mean() == pytest.approx(pfa, rel=0.5)
    scaled = ca_cfar(noise_db + 20.0, pfa=pfa)  # constant false alarm: invariant to the noise level
    np.testing.assert_array_equal(mask, scaled)
    withtarget = noise_db.copy()
    withtarget[100, 120] = 10 * np.log10(10 ** (25 / 10) + 1.0)
    hit = ca_cfar(withtarget, pfa=pfa)
    assert hit[100, 120] and hit.sum() <= mask.sum() + 3
    cleaner = ca_cfar(noise_db, pfa=1e-5)
    assert cleaner.sum() < mask.sum()


def test_validation_registry_and_engine_integration():
    for bad in (dict(bandwidth_hz=0), dict(chirp_s=-1), dict(n_chirps=1), dict(max_range=0.1, min_range=0.5), dict(beams_h=0)):
        with pytest.raises(ValueError):
            Radar(**bad)
    from aevion.registry import registries

    assert "radar" in registries.sensors.names()
    from aevion.engine import Simulation

    cfg = {
        "duration": 1.0,
        "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [25, 0, 3], "size": [1, 30, 12]}]}},
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 3.0]}},
        "sensors": [{"type": "radar", "params": {"map": False, "ground_clutter": False}}],
    }
    sim = Simulation(cfg)
    sim.run()
    out = sim.suite["radar"].latest
    assert np.asarray(out["valid"]).any() and np.nanmin(
        np.where(np.asarray(out["valid"][0]), np.asarray(out["detections"][0])[:, 0], np.nan)
    ) == pytest.approx(24.5, abs=0.2)
