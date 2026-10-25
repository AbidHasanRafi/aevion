"""Small-scale fading, shadowing and Doppler: distributions, correlation and analytic statistics."""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import j0

from aevion.wireless.fading import (
    CorrelatedShadowing,
    FadingProcess,
    JakesProcess,
    average_fade_duration,
    coherence_time,
    decorrelation_distance,
    jakes_autocorrelation,
    jakes_spectrum,
    level_crossing_rate,
    max_doppler_hz,
    nakagami_gain,
    nakagami_m_from_rician_k,
    rayleigh_gain,
    rayleigh_outage_probability,
    rician_gain,
    rician_k_from_nakagami_m,
    rician_outage_probability,
)


def rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------------------------------- distributions
def test_rayleigh_power_is_unit_mean_exponential():
    g = rayleigh_gain(200_000, rng())
    assert g.mean() == pytest.approx(1.0, abs=0.01)
    assert stats.kstest(g, "expon").pvalue > 1e-3
    assert rayleigh_outage_probability(0.1) == pytest.approx(1 - math.exp(-0.1))


@pytest.mark.parametrize("k", [0.0, 1.0, 5.0, 20.0])
def test_rician_power_has_unit_mean_and_the_right_cdf(k):
    g = rician_gain(k, 200_000, rng(1))
    assert g.mean() == pytest.approx(1.0, abs=0.01)
    # theoretical variance of the Rician power with unit mean: (1 + 2K) / (K + 1)^2
    assert g.var() == pytest.approx((1 + 2 * k) / (k + 1) ** 2, rel=0.05)
    for thr in (0.1, 0.5, 1.0):
        assert float((g < thr).mean()) == pytest.approx(float(rician_outage_probability(thr, k)), abs=0.006)


def test_rician_k_zero_is_rayleigh_and_large_k_approaches_a_constant():
    assert float(rician_outage_probability(0.3, 0.0)) == pytest.approx(float(rayleigh_outage_probability(0.3)), abs=1e-9)
    assert rician_gain(1000.0, 5000, rng()).std() < 0.07


@pytest.mark.parametrize("m", [0.5, 1.0, 3.0])
def test_nakagami_gain_is_gamma_with_unit_mean(m):
    g = nakagami_gain(m, 100_000, rng(2))
    assert g.mean() == pytest.approx(1.0, abs=0.02)
    assert g.var() == pytest.approx(1.0 / m, rel=0.05)
    assert stats.kstest(g, stats.gamma(m, scale=1.0 / m).cdf).pvalue > 1e-3
    with pytest.raises(ValueError):
        nakagami_gain(0.3)


def test_nakagami_rician_moment_matching_roundtrip():
    k = np.array([0.5, 2.0, 10.0])
    m = nakagami_m_from_rician_k(k)
    assert np.all(m >= 1.0)
    assert np.allclose(rician_k_from_nakagami_m(m), k, rtol=1e-9)
    assert float(nakagami_m_from_rician_k(0.0)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------- Doppler
def test_doppler_helpers():
    assert float(max_doppler_hz(30.0, 3.5e9)) == pytest.approx(30.0 * 3.5e9 / 299_792_458.0)
    assert float(jakes_autocorrelation(0.0, 100.0)) == 1.0
    tau = np.linspace(0, 0.05, 30)
    assert np.allclose(jakes_autocorrelation(tau, 100.0), j0(2 * math.pi * 100.0 * tau))
    # exact coherence time: autocorrelation equals the requested level
    for c in (0.5, 0.9):
        t = float(coherence_time(100.0, c))
        assert float(jakes_autocorrelation(t, 100.0)) == pytest.approx(c, abs=1e-9)
    assert float(coherence_time(100.0, 0.5)) == pytest.approx(0.2420e-2 * 1.0, rel=2e-3)
    with pytest.raises(ValueError):
        coherence_time(100.0, 1.5)


def test_jakes_spectrum_has_the_arcsine_mass_and_is_u_shaped():
    fd = 80.0
    f = np.linspace(-0.9 * fd, 0.9 * fd, 200_001)
    s = jakes_spectrum(f, fd)
    mass = float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(f)))
    assert mass == pytest.approx(2.0 / math.pi * math.asin(0.9), abs=1e-4)  # arcsine law: P(|f| < 0.9 fd)
    assert s[len(s) // 2] < s[10] and s[len(s) // 2] < s[-10]
    assert jakes_spectrum(np.array([0.0, 100.0]), fd)[1] == 0.0


def test_level_crossing_and_fade_duration_identities():
    fd = 50.0
    rho = np.array([0.1, 0.5, 1.0, 2.0])
    lcr, afd = level_crossing_rate(rho, fd), average_fade_duration(rho, fd)
    # outage probability = LCR * AFD  ->  1 - exp(-rho^2)
    assert np.allclose(lcr * afd, 1 - np.exp(-(rho**2)), rtol=1e-9)
    assert level_crossing_rate(1.0, fd) == pytest.approx(math.sqrt(2 * math.pi) * fd * math.exp(-1))


def test_jakes_process_has_unit_power_circular_symmetry_and_bessel_autocorrelation():
    n_links = 4000
    p = JakesProcess(n_links, rng=rng(3))
    fd, dt = 40.0, 1e-3
    h = [p.current()] + [p.step(fd, dt) for _ in range(40)]
    h = np.array(h)
    assert float(np.mean(np.abs(h) ** 2)) == pytest.approx(1.0, abs=0.04)
    assert abs(float(np.mean(h))) < 0.05
    assert abs(float(np.mean(h.real**2)) - float(np.mean(h.imag**2))) < 0.04  # equal I/Q power
    for lag in (0, 4, 8, 12, 20, 40):
        emp = float(np.mean(np.real(h[0] * np.conj(h[lag]))))
        assert emp == pytest.approx(float(jakes_autocorrelation(lag * dt, fd)), abs=0.06), lag
    # power gain is exponentially distributed (Rayleigh)
    assert stats.kstest(np.abs(h[0]) ** 2, "expon").pvalue > 1e-3


def test_jakes_process_follows_a_changing_doppler():
    p = JakesProcess(2000, rng=rng(4))
    still = [p.step(0.0, 1e-3) for _ in range(20)]
    assert np.allclose(still[0], still[-1])  # zero Doppler: frozen channel
    fast = [p.step(300.0, 1e-3) for _ in range(20)]
    assert float(np.mean(np.abs(fast[0] - fast[-1]))) > 0.5


def test_jakes_process_is_reproducible_and_validates():
    a, b = JakesProcess(5, rng=rng(7)), JakesProcess(5, rng=rng(7))
    assert np.array_equal(a.step(30.0, 0.01), b.step(30.0, 0.01))
    a.reset(rng(8))
    assert not np.array_equal(a.current(), b.current())
    with pytest.raises(ValueError):
        JakesProcess(0)
    with pytest.raises(ValueError):
        JakesProcess(3, n_osc=7)


def test_fading_process_los_links_fade_less_than_nlos():
    n = 3000
    fp = FadingProcess(n, rng(5))
    k_db = np.full(n, 10.0)
    los = np.zeros(n, bool)
    los[: n // 2] = True
    g = np.array([fp.step(los, k_db, 20.0, 5e-3) for _ in range(30)])
    assert g.mean() == pytest.approx(1.0, abs=0.05)
    assert g[:, : n // 2].var() < 0.35 * g[:, n // 2 :].var()  # K = 10 dB: far smaller spread than Rayleigh
    samples, t = FadingProcess(2, rng(6)).sample(5.0, 30.0, 0.01, 50)
    assert samples.shape == (50, 2) and t.shape == (50,)
    fp.reset(rng(1))


# ----------------------------------------------------------------------------------------------- shadowing
def test_shadowing_is_ar1_with_the_requested_correlation_and_variance():
    sh = CorrelatedShadowing((3000,), rng(9))
    sigma, step, d_corr = 6.0, 10.0, 50.0
    series = np.array([sh.step(sigma, step, d_corr) for _ in range(60)])
    assert float(series.std()) == pytest.approx(sigma, rel=0.05)
    lag1 = float(np.mean(series[:-1] * series[1:]) / np.mean(series**2))
    assert lag1 == pytest.approx(math.exp(-step / d_corr), abs=0.03)
    lag3 = float(np.mean(series[:-3] * series[3:]) / np.mean(series**2))
    assert lag3 == pytest.approx(math.exp(-3 * step / d_corr), abs=0.04)
    assert float(CorrelatedShadowing.autocorrelation(50.0, 50.0)) == pytest.approx(math.exp(-1))


def test_shadowing_does_not_change_when_the_terminal_does_not_move():
    sh = CorrelatedShadowing((10,), rng(0))
    a = sh.step(4.0, 0.0, 30.0)
    b = sh.step(4.0, 0.0, 30.0)
    assert np.allclose(a, b)
    far = sh.step(4.0, 1e6, 30.0)
    assert not np.allclose(a, far)
    sh.reset(rng(1))
    assert sh.step(4.0, 0.0, 30.0).shape == (10,)


def test_decorrelation_distances_follow_tr38901():
    assert float(decorrelation_distance("uma", True)) == 37.0 and float(decorrelation_distance("uma", False)) == 50.0
    assert float(decorrelation_distance("umi", True)) == 10.0
    assert float(decorrelation_distance("rma", False)) == 120.0
    with pytest.raises(KeyError):
        decorrelation_distance("space", True)
