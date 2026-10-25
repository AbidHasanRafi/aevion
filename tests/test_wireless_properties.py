"""Property-based tests (hypothesis) for the wireless stack: invariants that must hold for any input."""

import math

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from aevion.wireless.antennas import HalfWaveDipole, Patch3GPP, SectorTR36814, UavAntenna, UniformLinearArray  # noqa: E402
from aevion.wireless.fading import rician_gain  # noqa: E402
from aevion.wireless.mac_network import (  # noqa: E402
    HandoverManager,
    NrCellularNetwork,
    RadioLinkMonitor,
    bianchi_fixed_point,
    bianchi_throughput_bps,
    compute_sinr,
)
from aevion.wireless.phy import BlerModel, get_table, harq_statistics  # noqa: E402
from aevion.wireless.propagation import TR38901, LinkGeometry, knife_edge_loss_db, make_model  # noqa: E402

heights = st.floats(1.5, 300.0)
dists = st.floats(10.0, 4000.0)
quat = st.tuples(*[st.floats(-1, 1)] * 4).filter(lambda q: sum(x * x for x in q) > 1e-3)


@given(st.sampled_from(["free_space", "two_ray", "log_distance", "al_hourani", "tr38901", "tr36777"]), dists, dists, heights)
def test_path_loss_is_finite_and_ordered_for_any_geometry(name, d1, d2, h_ut):
    kw = {"scenario": "uma"} if name == "tr38901" else {"scenario": "uma_av"} if name == "tr36777" else {}
    m = make_model(name, freq_hz=3.5e9, **kw)
    lo, hi = sorted((d1, d2))
    g = LinkGeometry.from_distances(np.array([lo, hi]), 25.0, h_ut)
    for los in (True, False):
        pl = m.path_loss_db(g, los)
        assert np.all(np.isfinite(pl)) and pl[1] >= pl[0] - 1e-6
    assert np.all(m.path_loss_db(g, False) >= m.path_loss_db(g, True) - 1e-9)
    assert np.all((m.los_probability(g) >= 0) & (m.los_probability(g) <= 1))


@given(st.floats(0.5, 100.0), st.floats(10.0, 5000.0), st.floats(1.5, 22.5))
def test_tr38901_los_probability_never_increases_with_distance(fc, d, h_ut):
    m = TR38901(fc * 1e9, "uma")
    p = m.los_probability(LinkGeometry.from_distances(np.array([d, d * 1.5]), 25.0, h_ut))
    assert p[1] <= p[0] + 1e-12


@given(st.floats(-5.0, 20.0), st.floats(0.0, 10.0))
def test_knife_edge_loss_is_monotone_in_the_fresnel_parameter(nu, delta):
    assert float(knife_edge_loss_db(nu + delta)) >= float(knife_edge_loss_db(nu)) - 1e-12
    assert float(knife_edge_loss_db(nu)) >= 0.0


@given(st.floats(0.0, 60.0), st.integers(1, 200))
@settings(max_examples=25)
def test_rician_gain_is_positive_with_unit_mean(k, n):
    g = rician_gain(k, n * 200, np.random.default_rng(n))
    assert np.all(g >= 0) and abs(g.mean() - 1.0) < 0.35


@given(quat, st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)).filter(lambda v: sum(x * x for x in v) > 1e-3))
def test_uav_antenna_gain_is_bounded_by_the_pattern_peak(q, d):
    for ant, peak in ((UavAntenna.preset("patch_down"), 8.0), (UavAntenna.preset("dipole"), HalfWaveDipole().peak_dbi)):
        g = float(ant.gain_dbi(np.array(q), np.array(d)))
        assert -40.0 - 1e-9 <= g <= peak + 1e-9


@given(st.floats(-math.pi, math.pi), st.floats(-math.pi / 2, math.pi / 2))
def test_patterns_are_bounded_and_finite(az, el):
    for pat in (Patch3GPP(), SectorTR36814(), HalfWaveDipole()):
        g = float(pat.gain_dbi(az, el))
        assert math.isfinite(g) and -40.0 <= g <= pat.peak_dbi + 1e-9


@given(st.floats(-math.pi / 3, math.pi / 3), st.integers(2, 32))
@settings(max_examples=40)
def test_ula_gain_never_exceeds_the_array_gain(az, n):
    ula = UniformLinearArray(n)
    w = ula.beam_weights(az, 0.0)
    probe = np.linspace(-math.pi / 2, math.pi / 2, 91)
    assert np.all(ula.array_factor_db(w, probe, 0.0) <= 10 * math.log10(n) + 1e-9)
    assert float(ula.array_factor_db(w, az, 0.0)) == pytest.approx(10 * math.log10(n), abs=1e-6)


@given(st.floats(-20.0, 40.0), st.integers(0, 28), st.integers(1, 6))
def test_harq_never_hurts_reliability(sinr, mcs, max_tx):
    m = BlerModel(get_table("mcs1"))
    h = harq_statistics(m, sinr, mcs, max_tx)
    assert float(h.residual_bler) <= float(h.first_tx_bler) + 1e-12
    assert 1.0 <= float(h.expected_transmissions) <= max_tx + 1e-9
    assert 0.0 <= float(h.first_tx_bler) <= 1.0


@given(st.integers(1, 200), st.sampled_from([8, 16, 32, 64, 128]), st.integers(1, 7))
def test_bianchi_always_has_a_valid_fixed_point(n, w, m):
    tau, p = bianchi_fixed_point(n, w, m)
    assert 0.0 < tau < 1.0 and 0.0 <= p < 1.0
    assert p == pytest.approx(1 - (1 - tau) ** (n - 1), abs=1e-7)
    assert 0.0 < bianchi_throughput_bps(n) < 1e6


@given(st.lists(st.floats(-110.0, -40.0), min_size=2, max_size=6), st.floats(-120.0, -90.0))
def test_sinr_is_below_snr_of_the_serving_cell_and_finite(powers, noise):
    rx = np.array([powers])
    serving = int(np.argmax(powers))
    res = compute_sinr(rx, [serving], noise)
    assert math.isfinite(float(res.sinr_db[0]))
    assert float(res.sinr_db[0]) <= powers[serving] - noise + 1e-9


@given(st.lists(st.tuples(st.floats(-120.0, -40.0), st.floats(-120.0, -40.0), st.floats(-30.0, 30.0)), min_size=5, max_size=60))
def test_handover_manager_invariants(samples):
    m = HandoverManager(1, 2, time_to_trigger_s=0.04, l3_filter_k=2)
    last = 0
    for k, (a, b, sinr) in enumerate(samples):
        serving, interrupted = m.step(k * 0.02, np.array([[a, b]]), 0.02, serving_sinr_db=sinr)
        assert serving[0] in (0, 1) and m.n_ho[0] >= last
        last = m.n_ho[0]
    assert m.n_pp[0] <= m.n_ho[0] and m.n_fail[0] <= m.n_ho[0] and len(m.events) == m.n_ho[0]


@given(st.lists(st.floats(-30.0, 30.0), min_size=5, max_size=80))
def test_rlf_monitor_invariants(sinrs):
    m = RadioLinkMonitor(1, t310_s=0.2, recovery_s=0.1, filter_s=0.01)
    for k, s in enumerate(sinrs):
        down = m.step(k * 0.02, s, 0.02)
        assert down.shape == (1,)
    n_events = sum(1 for e in m.events if e.kind == "rlf")
    assert m.n_rlf[0] == n_events
    starts = sum(1 for e in m.events if e.kind == "t310_start")
    ends = sum(1 for e in m.events if e.kind in ("t310_stop", "rlf"))
    assert 0 <= starts - ends <= 1  # a timer is either running or has ended


@given(
    st.lists(st.tuples(st.floats(-300.0, 900.0), st.floats(-300.0, 300.0), st.floats(2.0, 250.0)), min_size=1, max_size=4),
    st.integers(0, 2**16),
)
@settings(max_examples=25, deadline=None)
def test_network_report_is_always_well_formed(points, seed):
    n = NrCellularNetwork(sites=[{"pos": [0, 0, 25]}, {"pos": [600, 0, 25]}], seed=seed, background_users=1)
    pos = np.array(points, float)
    vel = np.zeros_like(pos)
    vel[:, 0] = 15.0
    for k in range(3):
        rep = n.evaluate(pos, t=0.1 * (k + 1), vel=vel, dt=0.1)
        assert np.all(np.isfinite(rep.sinr_db)) and np.all(rep.rate_bps >= 0) and np.all(np.isfinite(rep.rate_bps))
        assert np.all((rep.serving >= 0) & (rep.serving < n.n_cells))
        assert np.all(rep.rate_bps[rep.outage] == 0)
        for key, v in rep.extras.items():
            assert v.shape == (len(pos),) and not np.any(np.isnan(v)), key
        assert np.all((rep.extras["share"] > 0) & (rep.extras["share"] <= 1.0 + 1e-12))
    assume(True)
