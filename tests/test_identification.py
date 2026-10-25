"""System identification - recovering known parameters from (noisy) simulated flight logs."""

import numpy as np
import pytest

from aevion.core import rotation as rot
from aevion.core.backend import NumpyBackend
from aevion.engine import Simulation
from aevion.identification import PARAMETERS, apply_factors, identify, load_csv_log, simulate_windows
from aevion.vehicles.spec import VehicleSpec

B = NumpyBackend()
NOMINAL = VehicleSpec.generic_1kg()
SIGMA = dict(pos=0.005, vel=0.01, att=0.002, omega=0.005)


def flight(overrides, duration=30.0, seed=1):
    rng = np.random.default_rng(seed)
    pts = [[0, 0, 5]] + [
        [rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(3, 9)] for _ in range(14)
    ]  # a zig-zag that excites every axis, yaw included
    cfg = {
        "duration": duration,
        "seed": 3,
        "trajectory": {"type": "waypoints", "params": {"points": pts, "speed": 3.5, "yaw_follow": True}},
        "logging": {"record_every": 1},  # commands at the physics rate: needed to see the motor lag
        "vehicle": {"model": "generic_1kg", "overrides": overrides},
    }
    a = Simulation(cfg).run()
    log = {k: np.asarray(a[k]) for k in ("t", "pos", "vel", "quat", "omega", "rotor", "cmd")}
    noisy = dict(log)
    noisy["pos"] = log["pos"] + rng.normal(0, SIGMA["pos"], log["pos"].shape)
    noisy["vel"] = log["vel"] + rng.normal(0, SIGMA["vel"], log["vel"].shape)
    noisy["omega"] = log["omega"] + rng.normal(0, SIGMA["omega"], log["omega"].shape)
    noisy["quat"] = rot.qmul(B, log["quat"][:, 0], rot.from_rotvec(B, rng.normal(0, SIGMA["att"], log["quat"][:, 0].shape[:-1] + (3,))))[:, None, :]
    return noisy


TRUE = dict(kT=0.88, kQ=1.10, Jxy=1.25, drag_lin=1.6, tau=1.3)
OVERRIDES = dict(
    kT=NOMINAL.kT * TRUE["kT"],
    kQ=NOMINAL.kQ * TRUE["kQ"],
    inertia=[0.0105 * TRUE["Jxy"], 0.0105 * TRUE["Jxy"], 0.0198],
    drag_lin=list(np.asarray(NOMINAL.drag_lin) * TRUE["drag_lin"]),
    motor_tau=NOMINAL.motor_tau * TRUE["tau"],
)


@pytest.fixture(scope="module")
def log():
    return flight(OVERRIDES)


@pytest.fixture(scope="module")
def fit(log):
    return identify(NOMINAL, log, free=("kT", "kQ", "Jxy", "drag_lin", "tau"), noise=SIGMA)


def test_five_parameters_are_recovered_from_a_noisy_flight(fit):
    got = dict(zip(fit.names, fit.factors))
    tol = dict(kT=0.01, kQ=0.02, Jxy=0.02, drag_lin=0.04, tau=0.03)
    for k, t in TRUE.items():
        assert got[k] == pytest.approx(t, rel=tol[k]), (k, got[k])
    assert fit.success and fit.rms < 0.1 * fit.initial_rms  # the fitted model explains the log ~15x better than the nominal one
    assert fit.values["kT"] == pytest.approx(NOMINAL.kT * TRUE["kT"], rel=0.01) and fit.values["tau"] == pytest.approx(
        NOMINAL.motor_tau * TRUE["tau"], rel=0.03
    )
    assert fit.n_windows > 40 and fit.n_residuals > 50_000


def test_uncertainties_and_the_report_are_available(fit):
    assert np.all(np.isfinite(fit.stderr)) and np.all(fit.stderr > 0) and np.all(fit.stderr < 0.05)
    assert fit.correlation.shape == (5, 5) and np.allclose(np.diag(fit.correlation), 1.0, atol=1e-6)
    rep = fit.report()
    assert "kT" in rep and "windows" in rep and "warning" not in rep  # these five are separable from this flight


def test_a_degenerate_pair_is_flagged_instead_of_silently_misfitted(log):
    res = identify(NOMINAL, log, free=("kT", "mass"), noise=SIGMA, bounds=(0.3, 3.0))
    assert abs(res.correlation[0, 1]) > 0.95 and "not separable" in res.report()  # only kT / m is observable
    assert res.factors[0] / res.factors[1] == pytest.approx(TRUE["kT"], rel=0.05)  # what the data does fix: the ratio kT / m


def test_mass_is_identified_when_the_thrust_coefficient_is_known():
    heavy = flight(dict(mass=NOMINAL.mass * 1.15), duration=24.0, seed=5)
    res = identify(NOMINAL, heavy, free=("mass",), noise=SIGMA)
    assert res.factors[0] == pytest.approx(1.15, rel=0.02)


def test_measured_rotor_speeds_can_replace_the_commands_but_not_the_lag(log):
    no_cmd = {k: v for k, v in log.items() if k != "cmd"}
    res = identify(NOMINAL, no_cmd, free=("kT", "Jxy", "drag_lin"), noise=SIGMA)
    got = dict(zip(res.names, res.factors))
    assert (
        got["kT"] == pytest.approx(0.88, rel=0.02) and got["Jxy"] == pytest.approx(1.25, rel=0.04) and got["drag_lin"] == pytest.approx(1.6, rel=0.08)
    )


def test_the_windows_predict_the_log_with_the_true_parameters_and_not_with_the_nominal_ones(log):
    true_f = dict(TRUE)
    win = [(i, i + 100) for i in range(0, 5000, 500)]
    sim_true = simulate_windows(NOMINAL, log, true_f, win)
    sim_nom = simulate_windows(NOMINAL, log, {}, win)
    ref = np.asarray(log["pos"])[:, 0]
    idx = np.array([[a + j for j in range(101)] for a, _ in win])
    e_true = np.sqrt(np.mean((sim_true["pos"] - ref[idx]) ** 2))
    e_nom = np.sqrt(np.mean((sim_nom["pos"] - ref[idx]) ** 2))
    assert e_true < 0.02 and e_nom > 3.0 * e_true  # a 0.5 s window with the right parameters stays within the sensor noise


def test_apply_factors_changes_exactly_what_it_says():
    s = apply_factors(NOMINAL, dict(kT=0.9, Jxy=2.0, tau=1.5, drag_lin=3.0))
    assert s.kT == pytest.approx(NOMINAL.kT * 0.9) and s.motor_tau == pytest.approx(NOMINAL.motor_tau * 1.5)
    assert np.allclose(np.diag(s.inertia_matrix), [0.021, 0.021, 0.0198]) and np.allclose(s.drag_lin, np.asarray(NOMINAL.drag_lin) * 3.0)
    assert s.mass == NOMINAL.mass and NOMINAL.kT != s.kT  # a copy: the nominal spec is untouched
    assert set(PARAMETERS) >= {"mass", "kT", "kQ", "Jxy", "Jz", "tau", "drag_lin", "drag_quad", "ang_damp"}


def test_bad_requests_are_explained(log):
    with pytest.raises(ValueError, match="cannot identify"):
        identify(NOMINAL, log, free=("wingspan",))
    short = {k: v[:150] for k, v in log.items()}
    with pytest.raises(ValueError, match="too short"):
        identify(NOMINAL, short, free=("kT",))
    with pytest.raises(ValueError, match="rotor commands"):
        identify(NOMINAL, {k: v for k, v in log.items() if k not in ("cmd", "rotor")}, free=("kT",))


def test_csv_logs_load_with_default_or_custom_column_names(tmp_path, log):
    import csv

    t = log["t"][:60]
    p = tmp_path / "log.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "p", "q", "r", "m1", "m2", "m3", "m4"])
        for i in range(60):
            w.writerow([t[i], *log["pos"][i, 0], *log["vel"][i, 0], *log["quat"][i, 0], *log["omega"][i, 0], *log["cmd"][i, 0]])
    d = load_csv_log(p)
    assert d["pos"].shape == (60, 1, 3) and d["quat"].shape == (60, 1, 4) and d["cmd"].shape == (60, 1, 4) and np.allclose(d["t"], t)
    assert np.allclose(d["vel"][:, 0], log["vel"][:60, 0], atol=1e-12)
    no_cmd = load_csv_log(
        p,
        {"t": "t", "pos": ["x", "y", "z"], "vel": ["vx", "vy", "vz"], "quat": ["qw", "qx", "qy", "qz"], "omega": ["p", "q", "r"], "cmd": ["motor_a"]},
    )
    assert "cmd" not in no_cmd  # an optional channel that is absent is skipped
    with pytest.raises(ValueError, match="not found"):
        load_csv_log(p, {"t": "time"})
    empty = tmp_path / "e.csv"
    empty.write_text("t,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_csv_log(empty)
