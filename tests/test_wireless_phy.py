"""PHY abstraction: numerology, MCS tables, BLER model, HARQ, link adaptation and OFDM effective SINR."""

import math

import numpy as np
import pytest

from aevion.wireless.phy import (
    CQI_TABLE_1,
    CQI_TABLE_2,
    EESM_BETA,
    MCS_TABLE_1,
    MCS_TABLE_2,
    BlerModel,
    LinkAdaptation,
    Numerology,
    attenuated_shannon_se,
    eesm_effective_sinr,
    eesm_effective_sinr_db,
    get_table,
    harq_statistics,
    max_prb,
    peak_data_rate_bps,
    per_rb_sinr_db,
    re_per_prb,
    required_sinr_db,
    transport_block_bits,
)
from aevion.wireless.phy.ofdm_abstraction import mutual_information_gap_db


def rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------------------------- numerology
def test_numerology_basics():
    for mu, scs, slot_us in [(0, 15, 1000), (1, 30, 500), (2, 60, 250), (3, 120, 125)]:
        n = Numerology(mu)
        assert n.scs_khz == scs and n.slot_duration_s == pytest.approx(slot_us * 1e-6) and n.slots_per_subframe == 2**mu
        assert n.prb_bandwidth_hz == pytest.approx(12 * scs * 1e3)
        assert n.symbol_duration_s == pytest.approx(n.slot_duration_s / 14)
    with pytest.raises(ValueError):
        Numerology(7)


@pytest.mark.parametrize(
    "bw,scs,fr,expected",
    [
        (20, 15, "FR1", 106),
        (100, 30, "FR1", 273),
        (50, 15, "FR1", 270),
        (10, 15, "FR1", 52),
        (20, 30, "FR1", 51),
        (100, 60, "FR1", 135),
        (100, 120, "FR2", 66),
        (400, 120, "FR2", 264),
        (100, 60, "FR2", 132),
    ],
)
def test_prb_tables_match_ts_38_101(bw, scs, fr, expected):
    assert max_prb(bw, scs, fr) == expected


def test_nonexistent_configurations_raise_instead_of_interpolating():
    with pytest.raises(KeyError, match="no NR configuration"):
        max_prb(100, 15, "FR1")  # 100 MHz does not exist at 15 kHz SCS
    with pytest.raises(KeyError):
        max_prb(100, 120, "FR1")


def test_peak_rate_reproduces_the_well_known_figure():
    """100 MHz, 30 kHz SCS, 256-QAM, 4 layers: about 2.3 Gbit/s (3GPP TS 38.306 4.1.2 example)."""
    rate = peak_data_rate_bps(Numerology(1), max_prb(100, 30), qm=8, code_rate=948 / 1024, layers=4, overhead=0.14)
    assert rate == pytest.approx(2.337e9, rel=2e-3)
    assert Numerology(1).n_prb(100) == 273
    assert Numerology(1).occupied_bandwidth_hz(100) == pytest.approx(273 * 360e3)
    assert re_per_prb() == 156 and re_per_prb(symbols=7, dmrs_re=12) == 72


# --------------------------------------------------------------------------------------------- tables
def test_mcs_tables_have_the_standard_size_and_endpoints():
    assert len(MCS_TABLE_1) == 29 and len(MCS_TABLE_2) == 28 and len(CQI_TABLE_1) == 15 and len(CQI_TABLE_2) == 15
    assert MCS_TABLE_1[0] == (2, 120) and MCS_TABLE_1[28] == (6, 948)
    assert MCS_TABLE_2[27] == (8, 948) and MCS_TABLE_2[20] == (8, 682.5)
    assert CQI_TABLE_1[0] == (2, 78) and CQI_TABLE_1[14] == (6, 948) and CQI_TABLE_2[14] == (8, 948)
    for rows in (MCS_TABLE_1, MCS_TABLE_2, CQI_TABLE_1, CQI_TABLE_2):
        assert {q for q, _ in rows} <= {2, 4, 6, 8}
        eff = np.array([q * r / 1024 for q, r in rows])
        assert np.all(np.diff(eff) > -0.005)  # monotone except the tiny dips the standard has at modulation changes


def test_table_lookup_and_spectral_efficiency():
    t = get_table("mcs1")
    assert t[28].spectral_efficiency == pytest.approx(6 * 948 / 1024)
    assert t.highest_below(0.1) == -1
    assert t.highest_below(1.0) == 6 and t.highest_below(1.2) == 8  # QPSK R=449/1024 (0.88) and R=602/1024 (1.18)
    assert t.highest_below(99.0) == 28
    with pytest.raises(KeyError):
        get_table("mcs9")


def test_transport_block_size_is_proportional_to_resources():
    e = get_table("mcs1")[20]
    b1 = transport_block_bits(e, 10)
    assert transport_block_bits(e, 20) == pytest.approx(2 * b1)
    assert transport_block_bits(e, 10, layers=2) == pytest.approx(2 * b1)
    assert b1 == pytest.approx(156 * 10 * e.spectral_efficiency)


# ------------------------------------------------------------------------------------ BLER model
def test_attenuated_shannon_matches_tr_36_942_limits():
    assert float(attenuated_shannon_se(-20.0)) == 0.0  # below the minimum SINR nothing is decoded
    assert float(attenuated_shannon_se(60.0)) == pytest.approx(4.4)  # capped
    assert float(attenuated_shannon_se(10.0)) == pytest.approx(0.6 * math.log2(1 + 10.0))
    se = 1.7
    assert float(attenuated_shannon_se(required_sinr_db(se))) == pytest.approx(se, abs=1e-9)


def test_bler_curves_cross_the_reference_at_the_required_sinr_and_are_monotone():
    m = BlerModel(get_table("mcs1"), sigma_db=0.7, bler_ref=0.1)
    for mcs in (0, 5, 12, 20, 28):
        req = float(m.required_sinr_db(mcs))
        assert float(m.bler(req, mcs)) == pytest.approx(0.1, abs=1e-9)
        s = np.linspace(req - 5, req + 5, 50)
        b = m.bler(s, mcs)
        assert np.all(np.diff(b) < 0) and b[0] > 0.9 and b[-1] < 0.01
    # higher MCS needs more SINR
    assert np.all(np.diff(m.required_sinr_db(np.arange(29))) >= 0)


def test_best_mcs_respects_the_bler_target():
    m = BlerModel(get_table("mcs1"))
    sinr = np.linspace(-8, 30, 80)
    mcs = m.best_mcs(sinr, 0.1)
    assert (np.all(np.diff(mcs) >= 0) and mcs[0] == -1) or mcs[0] == 0
    ok = mcs >= 0
    assert np.all(m.bler(sinr[ok], mcs[ok]) <= 0.1 + 1e-9)
    # one MCS higher would violate the target (where one exists)
    higher = np.minimum(mcs[ok] + 1, 28)
    assert np.all((m.bler(sinr[ok], higher) > 0.1) | (higher == 28))
    # a stricter target never selects a higher MCS
    assert np.all(m.best_mcs(sinr, 0.01) <= mcs)
    gp = m.goodput_se(25.0, 20)
    assert 0 < float(gp) <= get_table("mcs1")[20].spectral_efficiency


def test_harq_improves_reliability_and_costs_transmissions():
    m = BlerModel(get_table("mcs1"))
    s = float(m.required_sinr_db(15))  # BLER 10 % on the first transmission
    one = harq_statistics(m, s, 15, max_tx=1)
    four = harq_statistics(m, s, 15, max_tx=4)
    assert float(one.residual_bler) == pytest.approx(0.1) and float(one.expected_transmissions) == 1.0
    assert float(four.residual_bler) < 1e-3 and 1.0 < float(four.expected_transmissions) < 1.2
    nocomb = harq_statistics(m, s, 15, max_tx=4, combining=False)
    assert float(nocomb.residual_bler) == pytest.approx(0.1**4, rel=1e-6) and float(four.residual_bler) < float(nocomb.residual_bler)


# ---------------------------------------------------------------------------------- link adaptation
def run_la(la, snr, n=600, dt=0.02, n_prb=106):
    out = []
    for k in range(n):
        out.append(la.step(k * dt, snr, dt, n_prb))
    return out


def test_link_adaptation_tracks_sinr_and_olla_holds_the_average_bler_at_the_target():
    n = Numerology(1)
    la = LinkAdaptation(3, n, rng=rng(0), target_bler=0.1, olla=True)
    res = run_la(la, np.array([2.0, 12.0, 24.0]))
    last = res[-1]
    assert list(last.mcs) == sorted(last.mcs) and last.mcs[0] < last.mcs[1] < last.mcs[2]
    assert np.all(last.rate_bps[1:] > last.rate_bps[:-1]) and np.all(last.rate_bps > 0)
    # MCS is discrete, so OLLA dithers between two neighbours; the *time average* BLER sits at the target
    mean_bler = np.mean([r.bler for r in res[200:]], axis=0)
    assert np.all(np.abs(mean_bler - 0.1) < 0.05), mean_bler
    assert np.all(np.abs(last.olla_offset_db) <= la.limit)
    assert np.all(last.residual_bler <= last.bler + 1e-12)


def test_olla_corrects_an_optimistic_cqi_report():
    """The reported SINR is 3 dB too high; without OLLA the BLER is far above the target, with OLLA it returns to it."""
    n = Numerology(1)
    plain = LinkAdaptation(1, n, rng=rng(1), olla=False, cqi_bias_db=3.0, cqi_delay_s=0.0)
    olla = LinkAdaptation(1, n, rng=rng(1), olla=True, cqi_bias_db=3.0, cqi_delay_s=0.0)
    rp = run_la(plain, [12.0], 800)
    ro = run_la(olla, [12.0], 800)
    bler_plain = float(np.mean([r.bler[0] for r in rp[300:]]))
    bler_olla = float(np.mean([r.bler[0] for r in ro[300:]]))
    assert bler_plain > 0.3 and abs(bler_olla - 0.1) < 0.06
    assert float(np.mean([r.olla_offset_db[0] for r in ro[300:]])) == pytest.approx(-3.0, abs=1.2)  # it learned the bias


def test_link_adaptation_scheduler_share_scales_goodput_and_shannon_phy_is_selectable():
    n = Numerology(1)
    a = LinkAdaptation(1, n, rng=rng(0), olla=False, cqi_delay_s=0.0)
    full = a.step(0.0, [15.0], 0.02, 106, slots_active=1.0).rate_bps[0]
    a2 = LinkAdaptation(1, n, rng=rng(0), olla=False, cqi_delay_s=0.0)
    half = a2.step(0.0, [15.0], 0.02, 106, slots_active=0.5).rate_bps[0]
    assert half == pytest.approx(0.5 * full, rel=1e-6)
    dead = LinkAdaptation(1, n, rng=rng(0)).step(0.0, [-30.0], 0.02, 106)
    assert dead.mcs[0] == -1 and dead.rate_bps[0] == 0.0


def test_la_is_reproducible():
    n = Numerology(1)
    out = []
    for _ in range(2):
        la = LinkAdaptation(2, n, rng=rng(5))
        out.append([la.step(k * 0.02, [8.0, 14.0], 0.02, 52).rate_bps.copy() for k in range(50)])
    assert np.array_equal(np.array(out[0]), np.array(out[1]))


# -------------------------------------------------------------------------------------------- OFDM
def test_eesm_limits_and_jensen_gap():
    flat = np.full(16, 10.0)
    assert float(eesm_effective_sinr(flat, 4.0)) == pytest.approx(10.0)  # flat channel: unchanged
    sel = np.array([1.0, 30.0] * 8)
    assert float(eesm_effective_sinr(sel, 1e6)) == pytest.approx(sel.mean(), rel=1e-3)  # beta -> inf: arithmetic mean
    assert float(eesm_effective_sinr(sel, 1e-3)) == pytest.approx(sel.min(), rel=1e-2)  # beta -> 0: the worst RB
    assert float(eesm_effective_sinr(sel, 4.0)) < sel.mean()
    sinr_db = np.array([20.0, -5.0, 20.0, -5.0])
    eff = float(eesm_effective_sinr_db(sinr_db, 4))
    mean_db = float(10 * np.log10(np.mean(10 ** (sinr_db / 10))))
    assert eff < mean_db
    assert mutual_information_gap_db(mean_db, sinr_db, 4) > 0
    assert set(EESM_BETA) == {2, 4, 6, 8}


def test_per_rb_sinr_from_a_frequency_response():
    h = np.ones(48, complex)
    assert np.allclose(per_rb_sinr_db(10.0, h, 4), 10.0)
    h2 = np.concatenate([np.ones(24), 0.1 * np.ones(24)]).astype(complex)
    r = per_rb_sinr_db(10.0, h2, 2)
    assert r[0] == pytest.approx(10.0) and r[1] == pytest.approx(-10.0)


def test_link_adaptation_averages_the_error_statistics_over_fading_samples():
    n = Numerology(1)
    model = BlerModel(get_table("mcs1"))
    la = LinkAdaptation(1, n, rng=rng(0), olla=False, cqi_delay_s=0.0)
    samples = np.array([[4.0, 10.0, 16.0, 22.0]])
    w = np.array([[0.1, 0.2, 0.3, 0.4]])
    r = la.step(0.0, [12.0], 0.02, 106, sinr_samples_db=samples, sample_weights=w)
    mcs = int(r.mcs[0])
    assert r.bler[0] == pytest.approx(float((w * model.bler(samples, mcs)).sum()))
    plain = LinkAdaptation(1, n, rng=rng(0), olla=False, cqi_delay_s=0.0).step(0.0, [12.0], 0.02, 106)
    assert plain.mcs[0] == mcs and plain.bler[0] != pytest.approx(r.bler[0])
    uniform = LinkAdaptation(1, n, rng=rng(0), olla=False, cqi_delay_s=0.0).step(0.0, [12.0], 0.02, 106, sinr_samples_db=samples)
    assert uniform.bler[0] == pytest.approx(float(model.bler(samples, mcs).mean()))
