"""Allan-variance analysis and the parameterised IMU (noise terms, 1/f bias instability, vibration, thermal drift)."""

import itertools
import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.sensors.allan import FLICKER_FLOOR, allan_deviation, allan_variance, fit_noise, gauss_markov_avar, log_taus, model_curve
from aevion.sensors.inertial import IMU, IMU_GRADES, _Flicker

B = NumpyBackend()


class Stub:
    """Minimal simulator/state pair: a stationary sensor platform."""

    def __init__(self, n=8, rotor=400.0, m=4):
        self.B, self.n = B, n
        self.state = type("S", (), {})()
        self.state.f_body = np.tile([0.0, 0.0, 9.80665], (n, 1))
        self.state.omega = np.zeros((n, 3))
        self.state.rotor = np.full((n, m), float(rotor))
        self.state.t = 0.0

    def hover_speed(self):
        return 400.0


def run(imu, n=8, steps=1000, rotor=400.0, m=4, f_body=None):
    sim = Stub(n, rotor, m)
    if f_body is not None:
        sim.state.f_body = np.tile(f_body, (n, 1))
    imu.bind(sim, 5)
    out = []
    for k in range(steps):
        sim.state.t = k * imu.dt
        r = imu.measure(sim.state)
        out.append((B.to_numpy(r["accel"]).copy(), B.to_numpy(r["gyro"]).copy()))
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])  # (T, n, 3) each


# ---------------------------------------------------------------------------------------------- Allan tool
def test_white_noise_has_slope_minus_half_and_recovers_n():
    rng = np.random.default_rng(0)
    dt, n_dens = 0.01, 3e-4
    x = rng.normal(0.0, n_dens / math.sqrt(dt), (100_000, 4))
    taus, ad = allan_deviation(x, dt)
    np.testing.assert_allclose(ad, n_dens / np.sqrt(taus), rtol=0.1)
    assert fit_noise(taus, ad, ("N",))["N"] == pytest.approx(n_dens, rel=0.03)


def test_random_walk_slope_plus_half_and_recovers_k():
    rng = np.random.default_rng(1)
    dt, k = 0.02, 5e-5
    walk = np.cumsum(rng.normal(0.0, k * math.sqrt(dt), (60_000, 16)), axis=0)  # a bias random walk
    taus, ad = allan_deviation(walk, dt)
    sel = (taus >= 1.0) & (taus <= 100.0)
    np.testing.assert_allclose(ad[sel], k * np.sqrt(taus[sel] / 3.0), rtol=0.15)
    assert fit_noise(taus, ad, ("K",))["K"] == pytest.approx(k, rel=0.08)


def test_gauss_markov_formula_matches_a_simulated_process():
    rng = np.random.default_rng(2)
    dt, tc = 0.05, 8.0
    phi = math.exp(-dt / tc)
    x = np.empty((40_000, 32))
    x[0] = rng.normal(size=32)
    for k in range(1, len(x)):
        x[k] = phi * x[k - 1] + math.sqrt(1 - phi * phi) * rng.normal(size=32)
    taus, av = allan_variance(x, dt)
    sel = (taus > 0.4) & (taus < 150)
    np.testing.assert_allclose(av[sel], gauss_markov_avar(taus[sel], 1.0, tc), rtol=0.12)
    assert gauss_markov_avar(np.array([8.0]), 2.0, tc)[0] == pytest.approx(2.0 * gauss_markov_avar(np.array([8.0]), 1.0, tc)[0])


def test_fit_decomposes_a_noise_free_mixture_exactly():
    taus = np.logspace(-2, 3, 40)
    truth = dict(N=2e-4, B=1.5e-4, K=1e-5, Q=0.0, R=0.0)
    got = fit_noise(taus, model_curve(taus, **truth), ("N", "B", "K"))
    for k in ("N", "B", "K"):
        assert got[k] == pytest.approx(truth[k], rel=1e-6)
    got_all = fit_noise(taus, model_curve(taus, N=1e-3, Q=2e-3, R=1e-6), ("Q", "N", "B", "K", "R"))
    assert got_all["N"] == pytest.approx(1e-3, rel=1e-4) and got_all["Q"] == pytest.approx(2e-3, rel=1e-4) and got_all["B"] < 1e-9
    with pytest.raises(ValueError):
        fit_noise(taus, taus, ("N", "X"))
    assert pytest.approx(0.6643, abs=1e-4) == FLICKER_FLOOR


def test_tau_grid_and_short_records():
    t = log_taus(10_000, 0.01)
    assert t[0] == pytest.approx(0.01) and np.all(np.diff(t) > 0) and t[-1] <= 10_000 * 0.01 / 4
    taus, av = allan_variance(np.arange(10.0), 1.0, taus=[1, 2, 3, 50])
    assert len(taus) == 3  # clusters longer than the record are dropped


# ---------------------------------------------------------------------------------------------------- IMU
def test_flicker_plateau_equals_the_requested_allan_floor():
    dt, floor = 0.05, 2e-4
    f = _Flicker(B, 64, dt, floor, (1.0, 100.0), gen=B.make_rng(3))
    x = np.empty((72_000, 64))
    for k in range(len(x)):
        x[k] = f.step()[:, 0]
    taus, ad = allan_deviation(x, dt)
    plateau = ad[(taus >= 2.0) & (taus <= 50.0)]
    assert plateau.min() > 0.85 * floor and plateau.max() < 1.2 * floor  # flat to within the ripple of a 4-process 1/f approximation
    with pytest.raises(ValueError):
        _Flicker(B, 2, dt, floor, (1.0, 5.0))


def test_imu_from_allan_converts_datasheet_units_and_reproduces_the_curve():
    imu = IMU.from_allan(
        gyro_arw=0.3,
        gyro_bias_instability=0.0,
        gyro_rrw=0.0,
        accel_vrw=0.05,
        accel_bias_instability=0.0,
        rate=20.0,
        gyro_bias_init=0.0,
        accel_bias_init=0.0,
        scale_factor_std=0.0,
    )
    assert imu.cfg["gn"] == pytest.approx(0.3 * math.pi / 180 / 60) and imu.cfg["an"] == pytest.approx(0.05 / 60)
    accel, gyro = run(imu, n=32, steps=20_000)
    taus, ad = allan_deviation(gyro[:, :, 0], imu.dt)
    np.testing.assert_allclose(ad[taus <= 20], imu.cfg["gn"] / np.sqrt(taus[taus <= 20]), rtol=0.08)  # ARW line, 0.3 deg/sqrt(h)
    taus, ad = allan_deviation(accel[:, :, 0], imu.dt)
    np.testing.assert_allclose(ad[taus <= 20], imu.cfg["an"] / np.sqrt(taus[taus <= 20]), rtol=0.08)


def test_imu_bias_instability_and_random_walk_appear_on_the_allan_curve():
    imu = IMU(
        rate=20.0,
        gyro_noise_density=1e-4,
        gyro_bias_walk=0.0,
        gyro_bias_init=0.0,
        scale_factor_std=0.0,
        gyro_bias_instability=3e-4,
        bias_tau_range=(1.0, 100.0),
    )
    _, gyro = run(imu, n=48, steps=60_000)
    taus, ad = allan_deviation(gyro[:, :, 1], imu.dt)
    plateau = ad[(taus >= 3.0) & (taus <= 40.0)]
    assert plateau.min() > 0.8 * 3e-4 and plateau.max() < 1.3 * 3e-4
    assert ad[0] == pytest.approx(1e-4 / math.sqrt(imu.dt), rel=0.1)  # white noise dominates at the shortest tau
    rw = IMU(rate=20.0, gyro_noise_density=1e-5, gyro_bias_walk=2e-5, gyro_bias_init=0.0, scale_factor_std=0.0)
    _, g2 = run(rw, n=48, steps=30_000)
    t2, a2 = allan_deviation(g2[:, :, 0], rw.dt)
    sel = (t2 > 20) & (t2 < 300)
    np.testing.assert_allclose(a2[sel], 2e-5 * np.sqrt(t2[sel] / 3.0), rtol=0.2)


def test_grades_are_ordered_from_noisy_to_precise():
    cfgs = [IMU.from_grade(g) for g in ("consumer", "industrial", "tactical", "navigation")]
    for a, b in itertools.pairwise(cfgs):
        assert a.cfg["gn"] > b.cfg["gn"] and a.bi[1] > b.bi[1] and a.cfg["an"] > b.cfg["an"]
    assert set(IMU_GRADES) == {"consumer", "industrial", "tactical", "navigation"}
    with pytest.raises(KeyError):
        IMU.from_grade("magic")


def test_defaults_are_unchanged_and_ideal_is_exact():
    imu = IMU(ideal=True)
    a, g = run(imu, n=2, steps=3)
    np.testing.assert_array_equal(a[0], np.tile([0, 0, 9.80665], (2, 1)))
    assert imu._flk is None and not imu._vib_on and imu._misalign is None
    plain = IMU(rate=100.0)
    a1, _ = run(plain, n=2, steps=5)
    again = IMU(rate=100.0)
    a2, _ = run(again, n=2, steps=5)
    np.testing.assert_array_equal(a1, a2)  # deterministic
    assert "accel_vibration" not in plain.to_dict() and "accel_bias_instability" not in plain.to_dict()


def test_vibration_amplitude_frequency_and_rotor_speed_scaling():
    fs, f_rot = 1000.0, 100.0
    omega = 2 * math.pi * f_rot
    imu = IMU(
        rate=fs,
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        accel_bias_walk=0.0,
        gyro_bias_walk=0.0,
        accel_bias_init=0.0,
        gyro_bias_init=0.0,
        scale_factor_std=0.0,
        accel_vibration=(0.0, 0.0, 0.6),
        gyro_vibration=(0.05, 0.0, 0.0),
    )
    imu.cfg.update(an=0.0, gn=0.0)
    sim = Stub(1, rotor=omega, m=1)
    sim.hover_speed = lambda: omega  # hover rotor speed == the speed under test: rel = 1
    imu.bind(sim, 1)
    acc = np.empty(4000)
    gyr = np.empty(4000)
    for k in range(4000):
        sim.state.t = k / fs
        r = imu.measure(sim.state)
        acc[k], gyr[k] = r["accel"][0, 2] - 9.80665, r["gyro"][0, 0]
    spec = np.abs(np.fft.rfft(acc * np.hanning(len(acc)))) * 2 / np.hanning(len(acc)).sum()
    f = np.fft.rfftfreq(len(acc), 1 / fs)
    assert f[np.argmax(spec)] == pytest.approx(f_rot, abs=0.5) and spec.max() == pytest.approx(0.6, rel=0.05)  # amplitude at hover
    assert np.max(np.abs(gyr)) == pytest.approx(0.05, rel=0.05)
    # 20 % above hover speed: amplitude x 1.44 (thrust ~ w^2); and the tone aliases like a real sampled sensor (700 Hz at 1 kHz -> 300 Hz)
    hot = IMU(
        rate=fs,
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        accel_bias_walk=0.0,
        gyro_bias_walk=0.0,
        accel_bias_init=0.0,
        gyro_bias_init=0.0,
        scale_factor_std=0.0,
        accel_vibration=(0.0, 0.0, 0.6),
    )
    hot.cfg.update(an=0.0, gn=0.0)
    sim2 = Stub(1, rotor=2 * math.pi * 700.0, m=1)
    sim2.hover_speed = lambda: 2 * math.pi * 700.0 / 1.2
    hot.bind(sim2, 1)
    tone = np.empty(4000)
    for k in range(4000):
        sim2.state.t = k / fs
        tone[k] = hot.measure(sim2.state)["accel"][0, 2] - 9.80665
    sp2 = np.abs(np.fft.rfft(tone * np.hanning(4000))) * 2 / np.hanning(4000).sum()
    assert np.fft.rfftfreq(4000, 1 / fs)[np.argmax(sp2)] == pytest.approx(300.0, abs=0.5) and sp2.max() == pytest.approx(0.6 * 1.44, rel=0.05)


def test_vibration_harmonics_and_multi_rotor_sums():
    kw = dict(
        rate=1000.0,
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        accel_bias_walk=0.0,
        gyro_bias_walk=0.0,
        accel_bias_init=0.0,
        gyro_bias_init=0.0,
        scale_factor_std=0.0,
        accel_vibration=(0.0, 0.0, 1.0),
    )
    one, _ = run(IMU(**kw), n=1, steps=2000, rotor=400.0, m=1)
    three, _ = run(IMU(vibration_harmonics=3, **kw), n=1, steps=2000, rotor=400.0, m=1)
    assert np.std(three - one) > 0.2  # harmonics add content
    many, _ = run(IMU(**kw), n=64, steps=800, rotor=400.0, m=4)
    rms = np.sqrt(np.mean((many[:, :, 2] - 9.80665) ** 2))
    assert 0.55 < rms < 0.9  # four incoherent rotors: total rms stays near the single-rotor value (1/sqrt(2))


def test_thermal_bias_drift_follows_the_warmup_curve():
    imu = IMU(
        rate=10.0,
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        accel_bias_walk=0.0,
        gyro_bias_walk=0.0,
        accel_bias_init=0.0,
        gyro_bias_init=0.0,
        scale_factor_std=0.0,
        temperature_c=35.0,
        warmup_c=20.0,
        warmup_tau=60.0,
        gyro_bias_tempco=2e-4,
    )
    imu.cfg.update(an=0.0, gn=0.0)
    _, gyro = run(imu, n=1, steps=3000)
    expected = 2e-4 * (10.0 + 20.0 * (1.0 - np.exp(-np.arange(3000) * 0.1 / 60.0)))
    np.testing.assert_allclose(gyro[:, 0, 0], expected, atol=1e-9)


def test_g_sensitivity_misalignment_and_quantisation():
    quiet = dict(
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        accel_bias_walk=0.0,
        gyro_bias_walk=0.0,
        accel_bias_init=0.0,
        gyro_bias_init=0.0,
        scale_factor_std=0.0,
    )
    gs = IMU(gyro_g_sensitivity=1e-4, **quiet)
    _, g = run(gs, n=1, steps=2, f_body=[0.0, 3.0, 9.0])
    np.testing.assert_allclose(g[0, 0], 1e-4 * np.array([0.0, 3.0, 9.0]), atol=1e-15)
    mis = IMU(misalignment_std_deg=0.5, **quiet)
    a, _ = run(mis, n=400, steps=1, f_body=[0.0, 0.0, 9.80665])
    leak = np.hypot(a[0, :, 0], a[0, :, 1]) / 9.80665
    assert np.mean(leak) == pytest.approx(0.5 * math.pi / 180 * math.sqrt(math.pi / 2) * 1.0, rel=0.25)  # Rayleigh mean of two 0.5 deg axes
    assert np.allclose(np.linalg.norm(a[0], axis=1), 9.80665, rtol=1e-3)  # small-angle skew matrix preserves the magnitude
    q = IMU(accel_resolution=0.05, gyro_resolution=0.001, **{**quiet, "accel_noise_density": 1e-2})
    qa, qg = run(q, n=4, steps=50)
    assert np.allclose(qa / 0.05, np.round(qa / 0.05)) and np.allclose(qg / 0.001, np.round(qg / 0.001))


def test_validation():
    for bad in (dict(gyro_bias_instability=-1.0), dict(warmup_tau=0.0), dict(misalignment_std_deg=-1.0)):
        with pytest.raises(ValueError):
            IMU(**bad)


def test_imu_runs_in_the_engine_and_vibrates_with_the_rotors():
    from aevion.engine import Simulation

    cfg = {
        "duration": 1.5,
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2]}},
        "sensors": [
            {
                "type": "imu",
                "params": {"rate": 400.0, "accel_vibration": [0.2, 0.2, 1.0], "gyro_bias_instability": 1e-4, "gyro_vibration": [0.02, 0.02, 0.0]},
                "record": True,
            }
        ],
    }
    arch = Simulation(cfg).run()
    data = arch["sensors/imu"]  # (samples, vehicles, 6): accel xyz then gyro xyz
    assert data.shape[0] > 100 and data.shape[2] == 6
    z = data[:, 0, 2]
    assert np.std(z[len(z) // 2 :]) > 0.2  # the vibration is visible on the vertical accelerometer
