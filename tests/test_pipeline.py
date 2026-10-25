import json
import math

import numpy as np
import pytest

from aevion import World
from aevion.datasets import RunArchive
from aevion.scenario import Scenario, make_formation, verify
from aevion.wireless import (
    AerialBaseStation,
    AirToGroundChannel,
    BaseStation,
    CellularNetwork,
    doppler_hz,
    fspl_db,
    los_probability,
    thermal_noise_dbm,
)


def short(**kw):
    base = dict(trajectory=dict(kind="circle", center=[0, 0, 2.5], radius=1.2, omega=1.0), duration=4.0, record_every=4)
    base.update(kw)
    return Scenario(**base)


# ------------------------------------------------------------------- scenario / archive
def test_scenario_json_roundtrip():
    sc = short(world={"kind": "forest", "n_trees": 5}, sensors=["imu"], wind={"type": "ConstantWind", "velocity": [1, 0, 0]})
    sc2 = Scenario.from_dict(json.loads(sc.to_json()))
    assert sc2.to_dict() == sc.to_dict()


def test_archive_roundtrip_and_manifest(tmp_path):
    a = short(sensors=["imu", "lidar"], record_sensors=["imu", "lidar"], world={"kind": "forest", "n_trees": 8, "half_extent": 10}).run()
    p = a.save(tmp_path / "r")
    assert p.suffix == ".aev"
    b = RunArchive.load(p)
    assert set(a.channels) == set(b.channels)
    for k in a.channels:
        np.testing.assert_array_equal(a[k], b[k])
    assert b.meta["schema"].startswith("aevion.run")
    assert b.meta["environment"]["numpy"] == np.__version__
    assert b.spec.name == "generic_1kg" and b.world.n_obstacles == 8
    b.to_csv(tmp_path / "r.csv")
    head = (tmp_path / "r.csv").read_text().splitlines()[0].split(",")
    assert head[0] == "t" and "pos[0]" in head
    b.to_hdf5(tmp_path / "r.h5")
    c = RunArchive.from_hdf5(tmp_path / "r.h5")
    np.testing.assert_array_equal(c["pos"], a["pos"])


def test_reproducibility_is_bit_exact_and_seed_sensitive():
    kw = dict(wind={"type": "DrydenTurbulence", "w20": 5.0, "severity": 1.0}, sensors=["imu", "gps"], seed=11)
    a, b = short(**kw).run(), short(**kw).run()
    np.testing.assert_array_equal(a["pos"], b["pos"])
    assert verify(a)["ok"]
    c = short(**{**kw, "seed": 12}).run()
    assert np.abs(a["pos"] - c["pos"]).max() > 1e-6


def test_fast_and_reference_kernels_agree_in_scenarios():
    a = short().run()
    b = short().run(kernel="reference")
    np.testing.assert_allclose(a["pos"], b["pos"], atol=1e-7)


def test_estimator_in_the_loop_flies_and_records():
    a = short(
        sensors=["imu", "gps", "baro", "mag"],
        use_estimator=True,
        estimator_handover_s=1.5,
        duration=6.0,
        trajectory=dict(kind="hover", pos=[0, 0, 2.0]),
    ).run()
    assert "est_pos" in a
    err = np.linalg.norm(a["pos"][-1, 0] - np.array([0, 0, 2.0]))
    assert err < 3.0 and np.isfinite(a["pos"]).all()


def test_rotor_failure_event_and_metrics():
    a = short(
        vehicle="hex_3kg",
        duration=3.0,
        perturbation=dict(kind="rotor_failure", t=1.0, vehicle=0, rotor=1, severity=1.0),
        trajectory=dict(kind="hover", pos=[0, 0, 5.0]),
    ).run()
    assert any(e["type"] == "rotor_failure" for e in a.meta["events"])


def test_formation_and_swarm_avoidance():
    f = make_formation("ring", 6, 1.5)
    d = np.linalg.norm(f[:, None] - f[None], axis=-1)
    d[np.arange(6), np.arange(6)] = np.inf
    assert d.min() == pytest.approx(1.5, rel=1e-9)
    from aevion import metrics

    a = short(
        n=4,
        formation=dict(kind="line", spacing=1.0),
        swarm=dict(avoid_radius=1.2, comm=dict(loss=0.2, delay_s=0.05)),
        trajectory=dict(kind="hover", pos=[0, 0, 2.0]),
    ).run()
    assert a.n_vehicles == 4
    assert 0.5 < a.meta["swarm_delivery_ratio"] < 1.0
    assert metrics.flight_metrics(a)["min_separation"][0] > 0.5


def test_metrics_values_are_consistent():
    from aevion import metrics

    a = short(duration=5.0).run()
    m = metrics.flight_metrics(a)
    pos = a["pos"][:, 0]
    assert m["path_length"][0] == pytest.approx(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    assert m["rmse_pos"][0] == pytest.approx(np.sqrt((np.linalg.norm(pos - a["ref_pos"][:, 0], axis=1) ** 2).mean()))
    assert m["energy_wh"][0] > 0 and 0 < m["soc_end"][0] < 1


# ---------------------------------------------------------------------------- experiments
def test_experiment_grid_paired_design_and_stats(tmp_path):
    from aevion.experiments import Experiment, ExperimentResult

    exp = Experiment(
        "t",
        short(duration=3.0),
        methods={"geo": {"controller": "geometric"}, "pid": {"controller": "cascaded"}},
        conditions={"calm": {}, "windy": {"wind": {"type": "ConstantWind", "velocity": [3, 0, 0]}}},
        seeds=[0, 1, 2],
        randomization={"mass": (0.9, 1.1), "wind_speed": (0, 1)},
        skip=1.0,
    )
    assert exp.n_trials() == 12
    r = exp.run(progress=False)
    assert len(r.rows) == 12 and r.methods == ["geo", "pid"] and r.conditions == ["calm", "windy"]
    # same random draw per seed across methods (paired)
    g = {(x["method"], x["condition"], x["seed"]): x["rnd_mass"] for x in r.rows}
    assert g[("geo", "calm", 1)] == g[("pid", "calm", 1)] != g[("geo", "calm", 2)]
    s = r.summary("rmse_pos")
    assert len(s) == 4 and all(x["ci_lo"] <= x["mean"] <= x["ci_hi"] for x in s)
    assert len(r.compare("rmse_pos")) == 2
    r.save(tmp_path / "e.json")
    r2 = ExperimentResult.load(tmp_path / "e.json")
    assert r2.rows == json.loads(json.dumps(r.rows))
    r.to_csv(tmp_path / "e.csv")
    assert "\\toprule" in r.to_latex("rmse_pos")


def test_experiment_parallel_matches_serial():
    from aevion.experiments import Experiment

    mk = lambda: Experiment("p", short(duration=2.0), methods={"geo": {"controller": "geometric"}}, seeds=[0, 1, 2, 3])
    a = mk().run(workers=1, progress=False)
    b = mk().run(workers=2, progress=False)
    np.testing.assert_allclose(a.values("rmse_pos", "geo"), b.values("rmse_pos", "geo"), rtol=1e-9)


def test_statistics_helpers():
    from aevion.analysis import stats as ST

    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 60), rng.normal(1.0, 1, 60)
    r = ST.compare_two(a, b)
    assert r["t_p"] < 1e-3 and r["cohens_d"] == pytest.approx(-1.0, abs=0.35) and r["cliffs_delta"] < -0.4
    m, lo, hi = ST.mean_ci(a)
    assert lo < m < hi
    est, lo, hi = ST.bootstrap_ci(a, seed=1)
    assert lo < est < hi
    p, lo, hi = ST.wilson_interval(8, 10)
    assert p == 0.8 and 0.49 < lo < 0.6 and 0.9 < hi < 0.97
    np.testing.assert_allclose(ST.holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert ST.sample_size_for_diff(1.0, 0.5) in range(60, 70)


# ------------------------------------------------------------------------------- wireless
def test_fspl_and_los_probability_reference_values():
    assert float(fspl_db(1000.0, 2.4e9)) == pytest.approx(100.05, abs=0.01)
    assert float(fspl_db(2000.0, 2.4e9) - fspl_db(1000.0, 2.4e9)) == pytest.approx(6.02, abs=0.01)
    th = np.array([5.0, 20.0, 45.0, 80.0])
    for env in ("suburban", "urban", "dense_urban", "highrise"):
        p = los_probability(th, env)
        assert (np.diff(p) > 0).all() and p[-1] > 0.6
    assert los_probability(80.0, "urban") > 0.99 and los_probability(5.0, "urban") < 0.1
    assert los_probability(30.0, "highrise") < los_probability(30.0, "urban") < los_probability(30.0, "suburban")
    assert thermal_noise_dbm(1.0, 0.0) == pytest.approx(-173.98, abs=0.05)
    assert float(doppler_hz([10.0, 0, 0], [100.0, 0, 0], [0.0, 0, 0], 2e9)) == pytest.approx(10 * 2e9 / 299792458.0, rel=1e-9)


def test_geometric_los_is_blocked_by_buildings():
    w = World()
    w.add_box((0, 0, 20), (10, 10, 40))
    ch = AirToGroundChannel("urban", los_mode="geometric", shadowing=False)
    pl_free, los_free, _ = ch.path_loss_db(np.array([[40.0, 0, 10]]), np.array([[40.0, 30, 10]]), world=w)
    pl_blk, los_blk, _ = ch.path_loss_db(np.array([[-40.0, 0, 10]]), np.array([[40.0, 0, 10]]), world=w)
    assert los_free[0] and not los_blk[0]
    assert pl_blk[0] - fspl_db(80.0, 2e9) == pytest.approx(20.0, abs=1e-9)  # urban NLoS excess loss


def test_network_sinr_math_and_serving_cell():
    bs = [BaseStation((0, 0, 25), antenna="isotropic", gain_dbi=0.0), BaseStation((500, 0, 25), antenna="isotropic", gain_dbi=0.0)]
    net = CellularNetwork(bs, AirToGroundChannel("urban", los_mode="mean", shadowing=False), bandwidth_hz=10e6)
    r = net.evaluate(np.array([[50.0, 0, 60.0], [450.0, 0, 60.0]]))
    assert list(r.serving) == [0, 1]
    assert r.sinr_db[0] > 10 and r.rate_bps[0] > 20e6
    # manual SINR for the first UAV
    rx = r.all_rx_dbm[0]
    s, i = 10 ** (rx[0] / 10), 10 ** (rx[1] / 10)
    n = 10 ** (net.noise_dbm / 10)
    assert r.sinr_db[0] == pytest.approx(10 * math.log10(s / (i + n)), abs=1e-9)
    xs, ys, m = net.coverage_map(60.0, (-100, 600, -100, 100), 50.0, samples=1)
    assert m.shape == (len(ys), len(xs)) and np.isfinite(m).all()


def test_sector_antenna_penalises_uavs_above_horizon():
    from aevion.wireless import sector_gain_dbi

    assert sector_gain_dbi(10.0, tilt_deg=10.0) == pytest.approx(8.0)
    assert sector_gain_dbi(-30.0, tilt_deg=10.0) < sector_gain_dbi(0.0, tilt_deg=10.0)


def test_aerial_base_station_coverage_decreases_with_range():
    users = np.stack([np.linspace(0, 400, 20), np.zeros(20), np.zeros(20)], 1)
    ab = AerialBaseStation(users, tx_power_dbm=10.0, snr_threshold_db=10.0)
    near = ab.evaluate([[200.0, 0, 50.0]])["coverage"][0]
    far = ab.evaluate([[5000.0, 0, 50.0]])["coverage"][0]
    assert near > far and far == 0.0


def test_wireless_scenario_records_link_channels():
    net = CellularNetwork([BaseStation((-30, -30, 20))], AirToGroundChannel("suburban", los_mode="mean"))
    a = short(wireless=net.to_dict(), duration=3.0).run()
    for k in ("link_sinr_db", "link_rate_mbps", "link_los", "link_serving"):
        assert k in a and a[k].shape == (a.n_steps, 1)
    assert a.meta["wireless"]["bs"][0]["pos"] == [-30.0, -30.0, 20.0]


# ------------------------------------------------------------------------------ estimation
def test_attitude_filter_converges_and_estimates_bias():
    from aevion import Multirotor
    from aevion.core import rotation as rot
    from aevion.estimation import Estimator
    from aevion.sensors import GPS, IMU, Barometer, Magnetometer, SensorSuite

    sim = Multirotor("generic_1kg", n=4, dt=0.005)
    suite = SensorSuite(IMU(), GPS(), Barometer(), Magnetometer()).bind(sim, 2)
    est = Estimator(sim, gps_aided=False)
    s = sim.reset(pos=[0, 0, 3.0], yaw=0.7)
    suite.reset(s)
    est.reset(None)
    for _ in range(int(15 / 0.005)):
        e = est.update(suite.update(s), suite.fresh())
        s = sim.step(s, sim.hover_command())
    B = sim.B
    d = np.degrees(np.abs(rot.to_euler(B, e.quat) - rot.to_euler(B, s.quat)))
    assert d[:, :2].max() < 1.0  # roll/pitch from gravity: tight
    assert d[:, 2].max() < 7.0  # yaw limited by the 2 uT magnetometer hard-iron bias (~4.6 deg on a 25 uT horizontal field)
    assert np.linalg.norm(e.pos - s.pos, axis=1).max() < 3.0


# --------------------------------------------------------------------------------- viewer
def test_viewer_payload_and_html(tmp_path):
    from aevion.viewer import build_payload, export_html

    a = short(sensors=["imu", "lidar"], record_sensors=["lidar"], world={"kind": "forest", "n_trees": 6, "half_extent": 8}).run()
    p = build_payload(a, max_frames=100)
    assert p["format"] == "aevion.view/2"
    assert p["meta"]["T"] <= 100 and p["meta"]["n"] == 1 and p["meta"]["M"] == 4
    assert p["world"]["cylinders"] and "lidar" in p
    out = export_html(a, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "importmap" in html and "aevion-payload" in html and "/*AEVION_BOOT*/" not in html and "<!--aevion:dev-->" not in html
    payload = json.loads(html.split('id="aevion-payload">')[1].split("</script>")[0])
    assert payload["meta"]["title"] == a.scenario["name"]


def test_viewer_javascript_parses_with_node(tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    from aevion.viewer.export import bundle_modules

    for i, (spec, src) in enumerate(bundle_modules().items()):
        if spec.startswith("@aevion/"):  # vendored three.js is checked by its sha256 manifest instead
            f = tmp_path / f"m{i}.mjs"
            f.write_text(src, encoding="utf-8")
            r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, f"{spec}: {r.stderr}"


# --------------------------------------------------------------------------- figures/report
def test_figure_style_settings():
    import matplotlib as mpl

    from aevion.analysis import style

    with style.style():
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["font.serif"][0] in (
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Nimbus Roman No9 L",
            "TeX Gyre Termes",
            "Liberation Serif",
            "FreeSerif",
            "STIXGeneral",
            "DejaVu Serif",
        )
        assert mpl.rcParams["xtick.direction"] == "in" and mpl.rcParams["xtick.top"] and mpl.rcParams["ytick.right"]
        assert mpl.rcParams["pdf.fonttype"] == 42 and mpl.rcParams["mathtext.fontset"] == "stix"
    assert style.figsize("single")[0] == 3.5 and style.figsize("double")[0] == 7.16
    assert style.figsize("single", venue="acm")[0] == 3.33


def test_report_generation(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from aevion.analysis.report import build_report, insights

    a = short(sensors=["imu"], record_sensors=["imu"], duration=5.0, wind={"type": "ConstantWind", "velocity": [2, 1, 0]}).run()
    ins = insights(a)
    assert {"Tracking accuracy", "Energy", "Wind"} <= {i["title"] for i in ins}
    p = build_report(a, tmp_path / "rep", formats=("png",), with_replay=True)
    assert p.exists() and (tmp_path / "rep" / "replay.html").exists() and (tmp_path / "rep" / "figures" / "overview.png").exists()
    assert "\\begin{tabular}" in (tmp_path / "rep" / "metrics.tex").read_text()


def test_stat_figures_render():
    import matplotlib

    matplotlib.use("Agg")
    from aevion.analysis import figures_stats as FS

    rng = np.random.default_rng(0)
    s = {"A": rng.normal(1, 0.1, 10), "B": rng.normal(1.5, 0.1, 10), "C": rng.normal(1.1, 0.1, 10)}
    for fig in (
        FS.fig_box(s, "x"),
        FS.fig_cdf(s, "x"),
        FS.fig_heatmap(rng.random((4, 5)), range(5), range(4)),
        FS.fig_pareto({"A": (1, 2), "B": (2, 1), "C": (3, 3)}, "x", "y"),
        FS.fig_line_ci(np.arange(4), {"A": rng.random((4, 5))}, "x", "y"),
    ):
        assert fig.get_axes()


# ------------------------------------------------------------------------------------- cli
def test_cli_smoke(tmp_path, capsys):
    from aevion.cli import main

    main(["info"])
    main(["vehicles"])
    out = capsys.readouterr().out
    assert "Aevion" in out and "crazyflie" in out
    main(["demo", "--world", "empty", "--wind", "0", "--duration", "3", "--trajectory", "hover", "--out", str(tmp_path / "d"), "--no-report"])
    assert (tmp_path / "d" / "run.aevd" / "manifest.json").exists()
    main(["inspect", str(tmp_path / "d" / "run.aevd")])
    with pytest.raises(SystemExit) as e:
        main(["verify", str(tmp_path / "d" / "run.aevd")])
    assert e.value.code == 0
    main(["replay", str(tmp_path / "d" / "run.aevd"), "--out", str(tmp_path / "x.html")])
    assert (tmp_path / "x.html").exists()
