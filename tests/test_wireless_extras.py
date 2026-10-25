"""RIS, NTN, radio maps, RF threat models, wireless metrics and interoperability exporters."""

import csv
import math

import numpy as np
import pytest

from aevion.engine import Simulation
from aevion.environments.world import World
from aevion.wireless.interop import export_link_trace_csv, export_ns2_mobility, export_waypoints_csv, world_to_obj
from aevion.wireless.maps import (
    coverage_radius,
    empirical_coverage,
    empirical_variogram,
    fit_exponential_variogram,
    idw,
    kriging,
    mean_path_loss_db,
    optimal_altitude,
    outage_probability_from_pathloss,
    rbf,
    reconstruct,
    rmse,
)
from aevion.wireless.metrics import (
    LatencyBudget,
    aoi_from_archive,
    aoi_mm1,
    aoi_periodic_lossy,
    availability,
    average_aoi,
    ergodic_capacity_rayleigh,
    finite_blocklength_rate,
    jain_fairness,
    link_metrics,
    md1_wait_s,
    mm1_sojourn_s,
    outage_events,
    outage_probability,
    peak_aoi,
    rayleigh_outage_probability_db,
    reliability_after_harq,
    shannon_capacity_bps,
    simulate_mm1_aoi,
)
from aevion.wireless.metrics.latency import air_interface_latency_s
from aevion.wireless.ntn import (
    EARTH_RADIUS_M,
    SatelliteLink,
    doppler_shift_hz,
    gaseous_attenuation_db,
    orbital_period_s,
    orbital_speed_mps,
    pass_geometry,
    rain_attenuation_db,
    slant_range_m,
)
from aevion.wireless.phy import Numerology
from aevion.wireless.ris import RisPanel, optimal_phases, quantization_loss_db, quantize_phases, ris_channel, ris_received_power_dbm, scaling_summary
from aevion.wireless.security import (
    GNSS_NOMINAL_POWER_DBM,
    Jammer,
    anti_jam_margin_db,
    capture_probability,
    gnss_cn0_effective_dbhz,
    gnss_jam_to_signal_db,
    jammer_range_for_denial_m,
    pull_offset,
    sinr_with_jammer_db,
    spoofer_power_advantage_db,
)


def rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------------------------------------ RIS
TX, RX = [100.0, 20.0, 25.0], [60.0, -30.0, 10.0]


def panel(n=16, **kw):
    return RisPanel([0, 0, 30], normal=[1, 0, 0], nx=n, ny=n, dx=0.0375, dy=0.0375, **kw)  # lambda/2 at 4 GHz


def test_ris_gain_scales_with_the_square_of_the_element_count():
    """Optimal phases: received power ~ N^2 (coherent), so 4x the elements gives +12.04 dB (Tang et al. 2021)."""
    p8, p16, p32 = (float(ris_received_power_dbm(TX, RX, panel(n), 4e9)) for n in (8, 16, 32))
    assert p16 - p8 == pytest.approx(12.04, abs=0.1) and p32 - p16 == pytest.approx(12.04, abs=0.1)
    s = scaling_summary(panel(16), TX, RX, 4e9)
    assert s["gain_db"] == pytest.approx(10 * math.log10(256), abs=0.7)  # coherent N^2 vs incoherent N
    assert s["n_elements"] == 256


def test_ris_phase_quantisation_loss_matches_the_sinc_formula():
    assert [round(quantization_loss_db(b), 2) for b in (1, 2, 3)] == [3.92, 0.91, 0.22]
    r = panel(24, phase_bits=2)
    opt = float(ris_received_power_dbm(TX, RX, r, 4e9, mode="optimal"))
    q2 = float(ris_received_power_dbm(TX, RX, r, 4e9, mode="quantized"))
    assert opt - q2 == pytest.approx(quantization_loss_db(2), abs=0.25)
    ph = optimal_phases(TX, RX, r, 4e9)
    levels = np.unique(np.round(quantize_phases(ph, 2) / (math.pi / 2)).astype(int) % 4)
    assert set(levels) <= {0, 1, 2, 3}
    with pytest.raises(ValueError):
        quantize_phases(ph, 0)


def test_unconfigured_ris_is_far_worse_than_a_configured_one():
    r = panel(16)
    opt = float(ris_received_power_dbm(TX, RX, r, 4e9, mode="optimal"))
    rnd = float(ris_received_power_dbm(TX, RX, r, 4e9, mode="random", rng=rng(0)))
    off = float(ris_received_power_dbm(TX, RX, r, 4e9, mode="off"))
    assert opt > rnd + 10 and opt > off + 10
    with pytest.raises(ValueError):
        ris_received_power_dbm(TX, RX, r, 4e9, mode="magic")


def test_ris_power_falls_with_the_product_distance_and_with_back_side_illumination():
    base = float(ris_received_power_dbm([100, 0, 30], [100, 100, 30], panel(16), 4e9))
    far = float(ris_received_power_dbm([200, 0, 30], [200, 200, 30], panel(16), 4e9))
    assert far < base - 10  # doubling both distances: 1/(d1 d2)^2 -> -12 dB
    behind = float(np.abs(ris_channel([-100, 0, 30], [-100, 100, 30], panel(16), 4e9)))
    assert behind == 0.0  # transmitter behind the panel
    r = panel(4, reflection_amplitude=0.5)
    full = panel(4, reflection_amplitude=1.0)
    assert float(ris_received_power_dbm(TX, RX, r, 4e9)) == pytest.approx(float(ris_received_power_dbm(TX, RX, full, 4e9)) - 6.02, abs=0.01)


def test_ris_geometry_and_validation():
    r = panel(4)
    el = r.element_positions()
    assert el.shape == (16, 3) and np.allclose(el.mean(axis=0), [0, 0, 30]) and np.allclose(el[:, 0], 0.0)  # all on the x = 0 plane
    assert r.area_m2 == pytest.approx(16 * 0.0375**2) and r.far_field_distance_m(4e9) > 0
    assert r.element_gain_linear(4e9) == pytest.approx(4 * math.pi * 0.0375**2 / (299_792_458 / 4e9) ** 2)
    up = RisPanel([0, 0, 0], normal=[0, 0, 1], nx=2, ny=2)  # normal parallel to 'up' must still produce a valid grid
    assert np.linalg.matrix_rank(up.element_positions() - up.center) == 2
    with pytest.raises(ValueError):
        RisPanel([0, 0, 0], nx=0)
    with pytest.raises(ValueError):
        RisPanel([0, 0, 0], reflection_amplitude=1.5)


# ------------------------------------------------------------------------------------------------ NTN
def test_orbit_and_slant_range_reference_values():
    assert float(slant_range_m(90.0, 550e3)) == pytest.approx(550e3)
    assert float(slant_range_m(10.0, 550e3)) == pytest.approx(1815e3, rel=2e-3)
    assert float(slant_range_m(0.0, 35_786e3)) == pytest.approx(35_786e3 * 0 + 41_679e3, rel=2e-3)  # GEO at the horizon
    assert orbital_period_s(550e3) / 60 == pytest.approx(95.5, abs=0.1)
    assert orbital_speed_mps(550e3) == pytest.approx(7589.0, abs=2.0)
    assert orbital_period_s(35_786e3) / 3600 == pytest.approx(23.93, abs=0.05)  # sidereal day
    assert EARTH_RADIUS_M == 6_371_000.0


def test_leo_pass_geometry_duration_and_doppler():
    g = pass_geometry(550e3, 90.0, 1.0, 10.0)
    assert (
        g["elevation_deg"].max() == pytest.approx(90.0, abs=1.0)
        and g["elevation_deg"][0] == pytest.approx(10.0, abs=0.2)
        and g["elevation_deg"][-1] == pytest.approx(10.0, abs=0.2)
    )
    assert g["t"][-1] == pytest.approx(476.0, abs=8.0)  # an overhead Starlink-altitude pass above 10 degrees lasts about 8 minutes
    assert g["range_m"].min() == pytest.approx(550e3, rel=1e-3) and abs(g["range_rate_mps"][len(g["t"]) // 2]) < 100
    dop = doppler_shift_hz(g["range_rate_mps"], 2e9)
    assert dop[0] > 40e3 and dop[-1] < -40e3 and abs(dop.max() - SatelliteLink().max_doppler_hz()) < 4e3  # approaching: positive, receding: negative
    low = pass_geometry(550e3, 30.0, 1.0, 10.0)
    assert low["t"][-1] < g["t"][-1] and low["elevation_deg"].max() == pytest.approx(30.0, abs=1.5)


def test_satellite_link_budget_improves_with_elevation_and_degrades_in_rain():
    sl = SatelliteLink(freq_hz=2e9, eirp_dbw=34, g_over_t_dbk=-12, bandwidth_hz=5e6)
    b = sl.budget(np.array([10.0, 45.0, 90.0]))
    assert np.all(np.diff(b["snr_db"]) > 0) and np.all(np.diff(b["range_m"]) < 0)
    c = sl.budget(90.0)
    expected_cn0 = 34 - (float(c["fspl_db"]) + float(c["gas_db"]) + float(c["rain_db"]) + 0.5 + 2.0) - 12 + 228.6
    assert float(c["cn0_dbhz"]) == pytest.approx(expected_cn0, abs=0.01)  # -10 log10(k) = 228.6 dB
    ku = SatelliteLink(freq_hz=12e9, bandwidth_hz=36e6, rain_rate_mm_h=25.0)
    clear = SatelliteLink(freq_hz=12e9, bandwidth_hz=36e6)
    assert float(ku.budget(30.0)["snr_db"]) < float(clear.budget(30.0)["snr_db"]) - 3
    assert float(SatelliteLink().one_way_delay_s(90.0)) == pytest.approx(550e3 / 299_792_458)
    prof = SatelliteLink().pass_profile(90.0, 5.0)
    assert prof["snr_db"].max() > prof["snr_db"][0] and prof["rate_bps"].max() > 0 and len(prof["doppler_rate_hz_s"]) == len(prof["t"])


def test_atmospheric_terms():
    assert float(rain_attenuation_db(2.0, 50.0, 30.0)) == 0.0  # negligible below 3 GHz
    assert float(rain_attenuation_db(20.0, 0.0, 30.0)) == 0.0
    a20, a30 = float(rain_attenuation_db(20.0, 25.0, 30.0)), float(rain_attenuation_db(30.0, 25.0, 30.0))
    assert 2.0 < a20 < a30  # more rain loss at higher frequency
    assert float(rain_attenuation_db(20.0, 50.0, 30.0)) > a20 > float(rain_attenuation_db(20.0, 25.0, 60.0))
    with pytest.raises(ValueError):
        rain_attenuation_db(60.0, 10.0, 30.0)
    assert float(gaseous_attenuation_db(90.0, 12.0)) == pytest.approx(0.1) and float(gaseous_attenuation_db(30.0, 12.0)) == pytest.approx(0.2)


# ----------------------------------------------------------------------------------------------- maps
def test_uav_base_station_has_an_interior_optimal_altitude():
    """Al-Hourani 2014: for a fixed path-loss budget the coverage radius peaks at a finite altitude."""
    h_opt, r_opt = optimal_altitude(max_path_loss_db=100.0, freq_hz=2e9, env="urban")
    assert 150.0 < h_opt < 700.0 and r_opt > 0  # a few hundred metres, as in the paper
    for h in (h_opt * 0.25, h_opt * 4.0):
        assert coverage_radius(h, 100.0, 2e9, "urban") < r_opt
    # denser environments need a higher optimum and cover less
    h_urban, r_urban = optimal_altitude(100.0, 2e9, "urban")
    h_sub, r_sub = optimal_altitude(100.0, 2e9, "suburban")
    assert h_urban > h_sub and r_sub > r_urban
    assert coverage_radius(100.0, 60.0) == 0.0  # impossible budget
    assert float(mean_path_loss_db(500.0, 200.0)) > float(mean_path_loss_db(100.0, 200.0))
    assert 0.0 <= outage_probability_from_pathloss(115.0, 2e9, 500.0, 200.0) <= 1.0
    assert empirical_coverage([1, 2, 3, 4], 3) == 0.5 and empirical_coverage([1, 2, 3, 4], 3, weights=[1, 1, 1, 5]) == pytest.approx(6 / 8)


def test_map_reconstruction_recovers_a_smooth_field_and_kriging_reports_uncertainty():
    r = rng(0)
    f = lambda xy: -60 - 20 * np.log10(np.hypot(xy[:, 0] - 40, xy[:, 1] - 40) + 5) + 4 * np.sin(xy[:, 0] / 25.0)
    pts = r.uniform(0, 200, size=(150, 2))
    q = r.uniform(10, 190, size=(300, 2))
    truth = f(q)
    errs = {m: rmse(reconstruct(pts, f(pts), q, m).prediction, truth) for m in ("idw", "rbf", "kriging")}
    assert errs["rbf"] < 1.5 and errs["kriging"] < 2.0 and errs["idw"] < 5.0
    pred, std = reconstruct(pts, f(pts), q, "kriging").prediction, reconstruct(pts, f(pts), q, "kriging").std
    assert std is not None and std.shape == pred.shape and np.all(std >= 0)
    # kriging is exact at the sample points and uncertain far from them
    p_at, s_at = reconstruct(pts, f(pts), pts[:5], "kriging").prediction, reconstruct(pts, f(pts), pts[:5], "kriging").std
    assert np.allclose(p_at, f(pts[:5]), atol=0.05) and np.all(s_at < 0.5)
    far = reconstruct(pts, f(pts), np.array([[1000.0, 1000.0]]), "kriging").std
    assert far[0] > np.median(std)
    assert idw(pts, f(pts), pts[:3], k=None).shape == (3,) and rbf(pts, f(pts), q[:3]).shape == (3,)
    lags, gam = empirical_variogram(pts, f(pts))
    nug, sill, rg = fit_exponential_variogram(lags, gam)
    assert nug >= 0 and sill > 0 and rg > 0
    with pytest.raises(ValueError):
        reconstruct(pts, f(pts), q, "nearest")
    assert kriging(pts, f(pts), q[:2], variogram=(0.1, 10.0, 80.0))[0].shape == (2,)


# ----------------------------------------------------------------------------------------- security
def test_gnss_effective_cn0_follows_the_betz_formula():
    assert float(gnss_cn0_effective_dbhz(43.5, -100.0)) == pytest.approx(43.5, abs=1e-3)  # negligible jammer
    js = np.array([0.0, 20.0, 30.0, 40.0, 50.0])
    eff = gnss_cn0_effective_dbhz(43.5, js)
    assert np.all(np.diff(eff) < 0) and eff[0] > 43.0
    # J/S = 30 dB with Rc = 1.023 MHz: (C/N0)^-1 + (J/S)/Rc
    inv = 1 / 10**4.35 + 1000.0 / 1.023e6
    assert float(gnss_cn0_effective_dbhz(43.5, 30.0)) == pytest.approx(-10 * math.log10(inv), abs=1e-6)
    assert float(gnss_jam_to_signal_db(-98.5)) == pytest.approx(30.0)
    assert anti_jam_margin_db(43.0, 10.0, 3.0) == 30.0


def test_jammer_power_range_and_denial_distance():
    j = Jammer(pos=[0, 0, 0], power_dbm=30.0, freq_hz=1.57542e9, bandwidth_hz=20e6, duty=0.5)
    near, far = float(j.received_dbm([100, 0, 0])), float(j.received_dbm([1000, 0, 0]))
    assert near - far == pytest.approx(20.0, abs=1e-6)  # free space: 20 dB per decade
    assert float(j.received_dbm([100, 0, 0], rx_bandwidth_hz=2e6)) == pytest.approx(near - 10.0, abs=1e-6)  # only 10 % spectral overlap
    assert j.active(1.0) and not Jammer(pos=[0, 0, 0], t_start=2.0).active(1.0)
    r = jammer_range_for_denial_m(30.0)
    at_r = Jammer(pos=[0, 0, 0], power_dbm=30.0).received_dbm([r, 0, 0])
    assert float(gnss_cn0_effective_dbhz(43.5, gnss_jam_to_signal_db(at_r))) == pytest.approx(28.0, abs=0.05)
    assert 5e3 < r < 1e5 and jammer_range_for_denial_m(40.0) == pytest.approx(r * 10**0.5, rel=1e-6)
    assert float(sinr_with_jammer_db(-70.0, -100.0, -90.0)) < float(sinr_with_jammer_db(-70.0, -100.0, -110.0))
    assert float(sinr_with_jammer_db(-70.0, -100.0, -110.0, interference_dbm=-80.0)) < float(sinr_with_jammer_db(-70.0, -100.0, -110.0))


def test_spoofer_capture_logic_and_pull():
    adv = spoofer_power_advantage_db([0, 0, 0], [100, 0, 0], spoofer_power_dbm=10.0)
    assert float(adv) == pytest.approx(10.0 - 20 * math.log10(4 * math.pi * 100 * 1.57542e9 / 299_792_458) - GNSS_NOMINAL_POWER_DBM)
    p = capture_probability(np.array([-10.0, 2.0, 15.0]))
    assert p[0] < 0.01 and p[1] == pytest.approx(0.5) and p[2] > 0.99
    off = pull_offset(np.array([0.0, 5.0, 10.0, 100.0]), 5.0, 2.0, max_offset_m=30.0)
    assert list(off) == [0.0, 0.0, 10.0, 30.0]


NO_FAULT = None


def gps_cfg(fault, duration=8.0):
    return {
        "duration": duration,
        "seed": 4,
        "sensors": [{"type": "gps", "record": True, "params": {"rate": 10, "latency": 0.0}}],
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 10.0]}},
        "faults": [] if fault is None else [fault],
    }


def test_gnss_spoofing_fault_drags_the_reported_position():
    arch = Simulation(
        gps_cfg({"type": "gnss_spoofing", "t_start": 2.0, "params": {"rate_mps": 2.0, "direction": [1, 0, 0], "max_offset_m": 6.0}})
    ).run()
    base = Simulation(gps_cfg(NO_FAULT)).run()
    t = np.asarray(arch["sensors/gps/t"])
    delta = (np.asarray(arch["sensors/gps"]) - np.asarray(base["sensors/gps"]))[:, 0, :]
    assert np.abs(delta[t < 1.9]).max() < 1e-9
    assert delta[t > 7.5, 0].mean() == pytest.approx(6.0, abs=1e-6) and np.allclose(delta[t > 7.5, 1:], 0.0)  # saturated at max_offset
    mid = delta[(t > 3.95) & (t < 4.05), 0].mean()
    assert mid == pytest.approx(4.0, abs=0.25)  # 2 m/s for 2 s


def test_gnss_spoofing_needs_enough_power_to_capture():
    weak = gps_cfg(
        {
            "type": "gnss_spoofing",
            "t_start": 1.0,
            "params": {"spoofer_pos": [5000.0, 0.0, 0.0], "spoofer_power_dbm": -30.0, "rate_mps": 5.0, "max_offset_m": 50.0},
        },
        4.0,
    )
    strong = gps_cfg(
        {
            "type": "gnss_spoofing",
            "t_start": 1.0,
            "params": {"spoofer_pos": [50.0, 0.0, 0.0], "spoofer_power_dbm": 10.0, "rate_mps": 5.0, "max_offset_m": 50.0},
        },
        4.0,
    )
    base = np.asarray(Simulation(gps_cfg(NO_FAULT, 4.0)).run()["sensors/gps"])
    w = np.asarray(Simulation(weak).run()["sensors/gps"]) - base
    s = np.asarray(Simulation(strong).run()["sensors/gps"]) - base
    assert np.abs(w).max() == 0.0 and np.abs(s).max() > 5.0


def test_gnss_jamming_fault_freezes_the_output_then_recovers():
    arch = Simulation(gps_cfg({"type": "gnss_jamming", "t_start": 2.0, "t_end": 6.0, "params": {"pos": [0, 0, 0], "power_dbm": 20.0}})).run()
    t, g = np.asarray(arch["sensors/gps/t"]), np.asarray(arch["sensors/gps"])[:, 0, :]
    frozen = g[(t > 2.3) & (t < 5.9)]
    assert np.allclose(frozen, frozen[0]) and len(frozen) > 20  # lock lost: stale fix
    free = g[t > 6.5]
    assert free.std(axis=0).max() > 0.01 and np.abs(free - [0, 0, 10]).max() < 5.0  # normal noise again afterwards


def test_distant_weak_gnss_jammer_only_adds_noise():
    far = gps_cfg({"type": "gnss_jamming", "t_start": 1.0, "t_end": 7.0, "params": {"pos": [0, 0, 0], "power_dbm": -20.0}})
    base = np.asarray(Simulation(gps_cfg(NO_FAULT)).run()["sensors/gps"])
    arch = np.asarray(Simulation(far).run()["sensors/gps"])
    assert not np.allclose(arch[10:70], base[10:70])  # extra noise while the weak jammer is on
    assert np.abs(arch - base).max() < 20.0  # no lock loss: readings keep changing, just noisier
    assert arch[20:60, 0, :].std(axis=0).max() > 0.0


# ----------------------------------------------------------------------------------------------- metrics
def test_outage_statistics():
    o = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0], bool)
    st = outage_events(o, 0.1)
    assert st.n_events == 2 and st.probability == pytest.approx(0.4) and st.durations_s == pytest.approx([0.3, 0.1])
    assert st.longest_s == pytest.approx(0.3) and st.mean_duration_s == pytest.approx(0.2) and st.time_to_first_s == pytest.approx(0.2)
    assert outage_events(np.zeros(5, bool), 0.1).time_to_first_s == math.inf and outage_events([], 0.1).n_events == 0
    assert float(availability(o)) == pytest.approx(0.6)
    assert float(outage_probability([-10, 0, 10, 20], 5.0)) == 0.5
    # Rayleigh analytic outage against a Monte-Carlo draw
    snr_db = 10 * np.log10(rng(1).exponential(10.0, 200_000))
    assert float(outage_probability(snr_db, 3.0)) == pytest.approx(float(rayleigh_outage_probability_db(10.0, 3.0)), abs=0.005)


def test_capacity_measures():
    assert float(shannon_capacity_bps(20e6, 10.0)) == pytest.approx(20e6 * math.log2(11))
    g = 10.0
    ec = float(ergodic_capacity_rayleigh(g))
    mc = float(np.mean(np.log2(1 + g * rng(0).exponential(1.0, 400_000))))
    assert ec == pytest.approx(mc, abs=0.01) and ec < math.log2(1 + g)  # Jensen
    r = finite_blocklength_rate(g, 200, 1e-5)
    assert 0 < float(r) < math.log2(1 + g)
    assert float(finite_blocklength_rate(g, 20000, 1e-5)) > float(r) and float(finite_blocklength_rate(g, 200, 1e-1)) > float(
        r
    )  # longer blocks / looser targets
    assert float(finite_blocklength_rate(g, 10**9, 1e-5)) == pytest.approx(math.log2(1 + g), abs=1e-3)  # n -> inf: capacity
    assert jain_fairness([1, 1, 1, 1]) == 1.0 and jain_fairness([1, 0, 0, 0]) == 0.25 and jain_fairness([]) == 1.0


def test_age_of_information_closed_forms_match_simulation():
    assert aoi_mm1(0.5, 1.0) == pytest.approx(3.5)
    assert simulate_mm1_aoi(0.5, 1.0, 120_000, seed=0) == pytest.approx(aoi_mm1(0.5, 1.0), rel=0.03)
    assert simulate_mm1_aoi(0.8, 1.0, 120_000, seed=1) == pytest.approx(aoi_mm1(0.8, 1.0), rel=0.05)
    assert aoi_periodic_lossy(0.1, 1.0, 0.01) == pytest.approx(0.06) and aoi_periodic_lossy(0.1, 0.5, 0.0) == pytest.approx(0.15)
    with pytest.raises(ValueError):
        aoi_mm1(2.0, 1.0)
    with pytest.raises(ValueError):
        aoi_periodic_lossy(0.1, 0.0)
    # empirical: periodic updates every 0.1 s, each delivered after 0.02 s, every other one lost
    gen = np.arange(0.0, 200.0, 0.1)
    deliv = np.where(np.arange(len(gen)) % 2 == 0, gen + 0.02, np.inf)
    res = average_aoi(gen, deliv, 200.0, resolution=400_000)
    assert res["mean"] == pytest.approx(0.02 + 0.1 * 2 * 1.5 / 2 * 1.0 * 0 + (0.2 / 2), abs=0.01)  # D + T_eff/2 with T_eff = 0.2 s
    assert peak_aoi(gen, deliv) == pytest.approx(0.2 + 0.02, abs=1e-6)
    assert peak_aoi([0.0], [0.1]) == math.inf


def test_latency_helpers():
    n = Numerology(1)
    one = float(air_interface_latency_s(n, 1))
    assert one == pytest.approx((1.5 + 1 + 2.0) * 0.5e-3)
    assert float(air_interface_latency_s(n, 3)) == pytest.approx(one + 2 * 8 * 0.5e-3)
    assert mm1_sojourn_s(5.0, 10.0) == pytest.approx(0.2) and mm1_sojourn_s(10.0, 10.0) == math.inf
    assert md1_wait_s(5.0, 0.1) == pytest.approx(0.5 * 0.1 / (2 * 0.5)) and md1_wait_s(10.0, 0.1) == math.inf
    assert float(reliability_after_harq(0.1, 3)) == pytest.approx(0.999)
    b = LatencyBudget(0.005, 0.01, 0.005, 0.002)
    assert b.total_s == pytest.approx(0.022) and b.stable_for(0.05) and not b.stable_for(0.02)


def test_link_metrics_and_aoi_from_an_archive():
    cfg = {
        "duration": 8.0,
        "seed": 2,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 60.0], "radius": 40.0, "omega": 0.3}},
        "network": {"type": "nr_cellular", "rate_hz": 10, "params": {"sites": [{"pos": [0, 0, 25], "sectors": 1}], "los_mode": "mean"}},
    }
    arch = Simulation(cfg).run()
    m = link_metrics(arch)
    assert m["link_outage_frac"][0] == pytest.approx(float(arch["link_outage"].mean()))
    assert m["link_mean_rate_mbps"][0] == pytest.approx(float(arch["link_rate_mbps"].mean()))
    aoi = aoi_from_archive(arch, update_hz=10.0, packet_bits=8000.0)
    assert 0.04 < aoi["aoi_mean_s"][0] < 1.0 and aoi["aoi_peak_s"][0] >= aoi["aoi_mean_s"][0]
    bare = Simulation({"duration": 1.0}).run()
    assert link_metrics(bare) == {} and aoi_from_archive(bare) == {}


# ----------------------------------------------------------------------------------------------- interop
def test_ns3_exports(tmp_path):
    cfg = {
        "duration": 2.0,
        "vehicle": {"count": 2, "formation": {"type": "line", "params": {"spacing": 3.0}}},
        "network": {"type": "mesh", "rate_hz": 10, "params": {"gateway": 0}},
    }
    arch = Simulation(cfg).run()
    mob = export_ns2_mobility(arch, tmp_path / "m.tcl", every_s=0.5).read_text().splitlines()
    assert mob[0].startswith("$node_(0) set X_") and any(line.startswith("$ns_ at 0.5000 ") for line in mob)
    n_init = sum(1 for line in mob if not line.startswith("$ns_"))
    assert n_init == 2 * 3  # X, Y, Z for two nodes
    wp = list(csv.reader((export_waypoints_csv(arch, tmp_path / "w.csv", 0.5)).open()))
    assert wp[0] == ["t", "node", "x", "y", "z", "vx", "vy", "vz"] and len(wp) == 1 + 2 * 5 and float(wp[1][0]) == 0.0
    lt = list(csv.reader(export_link_trace_csv(arch, tmp_path / "l.csv").open()))
    assert lt[0][:2] == ["t", "node"] and "link_rate_mbps" in lt[0] and len(lt) == 1 + arch.n_steps * 2
    empty = list(csv.reader(export_link_trace_csv(Simulation({"duration": 0.5}).run(), tmp_path / "e.csv").open()))
    assert empty == [["t", "node"]]


def test_world_obj_export(tmp_path):
    w = World()
    w.add_box_corners([0, 0, 0], [10, 10, 20])
    w.add_cylinder((30.0, 0.0), 1.0, 0.0, 8.0)
    w.add_sphere((0, 30, 5), 2.0)
    w.set_heightmap(np.zeros((3, 3)), (-5.0, -5.0), 5.0)
    text = world_to_obj(w, tmp_path / "w.obj").read_text().splitlines()
    verts = [line for line in text if line.startswith("v ")]
    faces = [line for line in text if line.startswith("f ")]
    assert {"g box", "g cylinder", "g sphere", "g terrain"} <= set(text)
    assert len(verts) == 8 + 32 + 9 * 12 + 9
    n = len(verts)
    assert all(1 <= int(i) <= n for f in faces for i in f.split()[1:])  # every face index refers to an existing vertex
    assert len(faces) == 6 + (16 + 2) + 12 * 8 + 4
    assert world_to_obj(World(), tmp_path / "e.obj").read_text().startswith("# Aevion world")
