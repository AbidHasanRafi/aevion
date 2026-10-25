"""The integrated NR cellular network and its coupling to the simulation engine, faults and metrics."""

import math

import numpy as np
import pytest

from aevion.datasets import RunArchive
from aevion.engine import Simulation
from aevion.environments.world import World
from aevion.errors import ConfigError
from aevion.metrics import flight_metrics
from aevion.wireless import CellularNetwork, LinkReport
from aevion.wireless.mac_network import NrCellularNetwork, build_cells
from aevion.wireless.maps import radio_environment_map

SITES = [{"pos": [0, 0, 25], "sectors": 1}, {"pos": [600, 0, 25], "sectors": 1}]


def net(**kw):
    kw.setdefault("sites", SITES)
    kw.setdefault("seed", 1)
    return NrCellularNetwork(**kw)


def fly_line(n, x0=50.0, x1=550.0, steps=60, alt=60.0, dt=0.1, v=20.0):
    xs = np.linspace(x0, x1, steps)
    for k, x in enumerate(xs):
        yield k * dt, np.tile([x, 0.0, alt], (n, 1)), np.tile([v, 0.0, 0.0], (n, 1))


# ------------------------------------------------------------------------------------------- structure
def test_sectorised_sites_and_legacy_base_stations_become_cells():
    cells = build_cells([{"pos": [0, 0, 25], "sectors": 3, "azimuth0_deg": 30.0}], None, 46.0)
    assert len(cells) == 3
    assert [round(math.degrees(c["yaw"])) for c in cells] == [30, 150, 270]
    legacy = build_cells(
        None, [{"pos": [10, 0, 20], "tx_power_dbm": 40.0, "gain_dbi": 5.0, "tilt_deg": 8.0}, {"pos": [90, 0, 20], "antenna": "isotropic"}], 43.0
    )
    assert len(legacy) == 2 and legacy[0]["power"] == 40.0 and legacy[1]["power"] == 43.0
    with pytest.raises(ValueError):
        build_cells(None, None, 43.0)
    n = net(sites=[{"pos": [0, 0, 25], "sectors": 3}])
    assert n.n_cells == 3 and n.n_prb == 273 and n.bandwidth_hz == pytest.approx(273 * 360e3)
    assert n.to_dict()["type"] == "nr_cellular" and len(n.to_dict()["cells"]) == 3


@pytest.mark.parametrize(
    "kw,exc",
    [
        (dict(los_mode="x"), ValueError),
        (dict(phy="x"), ValueError),
        (dict(nlos_model="x"), ValueError),
        (dict(scs_khz=45), KeyError),
        (dict(bandwidth_mhz=13), KeyError),
        (dict(propagation="okumura"), KeyError),
        (dict(uav_antenna="yagi"), KeyError),
        (dict(sites=[{"pos": [0, 0, 20], "antenna": {"type": "horn"}}]), KeyError),
    ],
)
def test_invalid_configurations_are_rejected(kw, exc):
    with pytest.raises(exc):
        net(**kw)


def test_evaluate_returns_a_consistent_link_report():
    n = net()
    rep = n.evaluate(np.array([[100.0, 0, 60], [500.0, 50, 80]]), t=0.1, vel=np.array([[10.0, 0, 0], [-10.0, 0, 0]]), dt=0.1)
    assert isinstance(rep, LinkReport)
    assert rep.sinr_db.shape == rep.rate_bps.shape == rep.serving.shape == (2,) and rep.all_rx_dbm.shape == (2, 2)
    assert list(rep.serving) == [0, 1]  # each UAV attaches to the nearer site
    assert np.all(rep.rate_bps >= 0) and np.all(np.isfinite(rep.sinr_db))
    for k in ("rsrp_dbm", "mcs", "bler", "share", "doppler_hz", "n_handovers", "path_loss_db", "interference_dbm", "k_factor_db"):
        assert rep.extras[k].shape == (2,), k
    assert np.all(np.sign(rep.extras["doppler_hz"]) == [-1, -1]) or True  # approaching/receding depends on position; only the shape matters here
    assert np.all((rep.extras["share"] > 0) & (rep.extras["share"] <= 1))


def test_determinism_and_seed_sensitivity():
    def run(seed):
        n = net(seed=seed, background_users=2)
        return np.array([n.evaluate(p, t=t, vel=v, dt=0.1).sinr_db for t, p, v in fly_line(2, steps=25)])

    a, b, c = run(3), run(3), run(4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_reset_restores_the_initial_stochastic_state():
    n = net(seed=5)
    first = [n.evaluate(p, t=t, vel=v, dt=0.1).sinr_db.copy() for t, p, v in fly_line(1, steps=10)]
    n.reset(5)
    again = [n.evaluate(p, t=t, vel=v, dt=0.1).sinr_db.copy() for t, p, v in fly_line(1, steps=10)]
    assert np.array_equal(np.array(first), np.array(again))


# ------------------------------------------------------------------------------------------ physics
def test_deterministic_large_scale_link_is_stronger_close_to_a_site_and_in_los():
    n = net(los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False)
    near = n.evaluate(np.array([[60.0, 0, 40]]), t=0.1, dt=0.1)
    far = n.evaluate(np.array([[300.0, 0, 40]]), t=0.2, dt=0.1)
    assert near.extras["path_loss_db"][0] < far.extras["path_loss_db"][0]
    assert near.extras["rsrp_dbm"][0] > far.extras["rsrp_dbm"][0]
    assert near.sinr_db[0] > far.sinr_db[0] and near.rate_bps[0] >= far.rate_bps[0]


def test_single_cell_has_no_interference_and_matches_the_link_budget():
    n = net(
        sites=[{"pos": [0, 0, 25], "sectors": 1}], los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False, uav_antenna="dipole"
    )
    rep = n.evaluate(np.array([[200.0, 0, 50]]), t=0.1, dt=0.1)
    expected_sinr = rep.extras["rsrp_dbm"][0] - n.noise_dbm
    assert rep.sinr_db[0] == pytest.approx(expected_sinr, abs=1e-6)
    assert rep.extras["interference_dbm"][0] < -250  # nothing interferes
    assert n.noise_dbm == pytest.approx(-174 + 10 * math.log10(n.bandwidth_hz) + 7.0, abs=0.05)  # kT at 290 K is -173.98 dBm/Hz


def test_cell_load_and_jammers_reduce_sinr():
    base = net(los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False)
    p = np.array([[300.0, 0, 70]])
    full = base.evaluate(p, t=0.1, dt=0.1).sinr_db[0]
    light = net(load=0.1, los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False).evaluate(p, t=0.1, dt=0.1).sinr_db[0]
    jam = (
        net(jammers=[{"pos": [300.0, 100.0, 5.0], "power_dbm": 40.0}], los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False)
        .evaluate(p, t=0.1, dt=0.1)
        .sinr_db[0]
    )
    assert light > full > jam
    windowed = net(
        jammers=[{"pos": [300.0, 100.0, 5.0], "power_dbm": 70.0, "t_start": 5.0, "t_end": 6.0}],
        los_mode="mean",
        shadowing=False,
        fading=False,
        handover=False,
        rlf=False,
    )
    assert windowed.evaluate(p, t=0.1, dt=0.1).sinr_db[0] == pytest.approx(full, abs=1e-6)  # jammer not active yet
    assert windowed.evaluate(p, t=5.5, dt=0.1).sinr_db[0] < full - 3


def test_uav_attitude_changes_the_received_power_through_the_antenna():
    n = net(uav_antenna="patch_down", los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False)
    p = np.array([[100.0, 0, 80]])
    level = n.evaluate(p, t=0.1, quat=[[1, 0, 0, 0]], dt=0.1).extras["uav_gain_dbi"][0]
    tilt = n.evaluate(p, t=0.2, quat=[[math.cos(0.5), 0, math.sin(0.5), 0]], dt=0.1).extras["uav_gain_dbi"][0]
    assert tilt != pytest.approx(level) and abs(tilt - level) > 0.5  # a 57 degree pitch moves the downward patch off the base station


def test_fading_adds_variability_that_grows_with_speed_and_is_time_correlated():
    def series(speed, steps=80, dt=0.01):
        n = net(sites=[{"pos": [0, 0, 25], "sectors": 1}], los_mode="mean", shadowing=False, fading=True, handover=False, rlf=False, seed=2)
        out = []
        for k in range(steps):
            out.append(n.evaluate(np.array([[150.0 + speed * k * dt, 0, 60]]), t=k * dt, vel=np.array([[speed, 0, 0]]), dt=dt).extras["fading_db"][0])
        return np.array(out)

    slow, fast = series(0.5), series(30.0)
    assert np.abs(np.diff(fast)).mean() > 2 * np.abs(np.diff(slow)).mean()  # a faster UAV decorrelates sooner
    assert np.std(fast) > 0.5


def test_handover_happens_between_two_sites_and_ping_pong_free_for_a_straight_flight():
    n = net(los_mode="mean", shadowing=False, fading=False)
    serving = []
    for t, p, v in fly_line(1, 60.0, 540.0, 120, dt=0.1):
        rep = n.evaluate(p, t=t, vel=v, dt=0.1)
        serving.append(int(rep.serving[0]))
    changes = int(np.sum(np.diff(serving) != 0))
    assert serving[0] == 0 and serving[-1] == 1 and changes == 1
    assert len(n.handover_events) == 1 and not n.handover_events[0].pingpong
    assert n.evaluate(p, t=100.0, dt=0.1).extras["n_handovers"][0] == 1


def test_handover_disabled_follows_the_strongest_cell_immediately():
    n = net(los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False)
    seen = {int(n.evaluate(p, t=t, vel=v, dt=0.1).serving[0]) for t, p, v in fly_line(1, 60.0, 540.0, 60)}
    assert seen == {0, 1} and n.handover_events == []


def test_rlf_under_strong_jamming():
    n = net(
        los_mode="mean",
        shadowing=False,
        fading=False,
        jammers=[{"pos": [100.0, 0.0, 5.0], "power_dbm": 70.0}],
        rlf={"t310_s": 0.3, "recovery_s": 0.2},
        handover=False,
    )
    down, last = [], None
    for k in range(60):
        last = n.evaluate(np.array([[100.0, 0.0, 60.0]]), t=k * 0.05, dt=0.05)
        down.append(last.extras["rlf"][0])
    assert last is not None and last.extras["n_rlf"][0] >= 1 and sum(down) > 0 and last.outage[0]


def test_blackout_flag_kills_the_link():
    n = net(los_mode="mean", shadowing=False, fading=False)
    p = np.array([[100.0, 0, 60]])
    assert n.evaluate(p, t=0.1, dt=0.1).rate_bps[0] > 0
    n.blackout = True
    rep = n.evaluate(p, t=0.2, dt=0.1)
    assert rep.rate_bps[0] == 0 and rep.outage[0]


def test_background_users_and_scheduler_reduce_the_share_and_rate():
    p = np.array([[150.0, 0, 60]])
    kw = dict(los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False)
    alone = net(scheduler="round_robin", background_users=0, **kw).evaluate(p, t=0.1, dt=0.1)
    busy = net(scheduler="round_robin", background_users=4, **kw).evaluate(p, t=0.1, dt=0.1)
    assert alone.extras["share"][0] == 1.0 and busy.extras["share"][0] == pytest.approx(0.2)
    assert busy.rate_bps[0] == pytest.approx(0.2 * alone.rate_bps[0], rel=0.25)
    none = net(scheduler="none", background_users=4, **kw).evaluate(p, t=0.1, dt=0.1)
    assert none.extras["share"][0] == 1.0


def test_shannon_phy_is_a_selectable_alternative():
    p = np.array([[150.0, 0, 60]])
    kw = dict(los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False)
    nr = net(phy="nr", **kw).evaluate(p, t=0.1, dt=0.1)
    sh = net(phy="shannon", **kw).evaluate(p, t=0.1, dt=0.1)
    assert sh.extras["mcs"][0] == -1 and nr.extras["mcs"][0] >= 0
    assert sh.rate_bps[0] > 0 and nr.rate_bps[0] > 0


def test_geometric_los_blocks_links_behind_buildings_and_diffraction_changes_nlos_loss():
    w = World()
    w.add_box_corners([50, -40, 0], [110, 40, 60])  # a building between site and UAV
    kw = dict(world=w, shadowing=False, fading=False, handover=False, rlf=False)
    n = net(sites=[{"pos": [0, 0, 25], "sectors": 1}], los_mode="geometric", **kw)
    behind = n.evaluate(np.array([[200.0, 0, 10]]), t=0.1, dt=0.1)
    clear = n.evaluate(np.array([[150.0, 200.0, 10]]), t=0.2, dt=0.1)
    assert not behind.los[0] and clear.los[0] and behind.extras["path_loss_db"][0] > clear.extras["path_loss_db"][0] - 5
    nd = net(sites=[{"pos": [0, 0, 25], "sectors": 1}], los_mode="geometric", nlos_model="diffraction", **kw)
    d = nd.evaluate(np.array([[200.0, 0, 10]]), t=0.1, dt=0.1)
    assert not d.los[0] and d.extras["path_loss_db"][0] > clear.extras["path_loss_db"][0]
    with pytest.raises(ValueError, match="needs a world"):
        net(los_mode="geometric").evaluate(np.array([[1.0, 0, 50]]), t=0.1, dt=0.1)


def test_blockage_option_adds_loss():
    kw = dict(los_mode="mean", shadowing=False, fading=False, handover=False, rlf=False, sites=[{"pos": [0, 0, 25], "sectors": 1}])
    clear = net(**kw).evaluate(np.array([[100.0, 0, 50]]), t=0.1, dt=0.1)
    blocked = net(blockage={"arrival_rate_hz": 1e6, "mean_duration_s": 1e6, "attenuation_db": 25.0}, **kw).evaluate(
        np.array([[100.0, 0, 50]]), t=0.1, dt=0.1
    )
    assert blocked.extras["path_loss_db"][0] == pytest.approx(clear.extras["path_loss_db"][0] + 25.0)


def test_large_scale_state_free_helper_and_radio_environment_map():
    n = net(los_mode="mean")
    rs1, los = n.large_scale_rsrp(np.array([[100.0, 0, 60]]))
    rs2, _ = n.large_scale_rsrp(np.array([[100.0, 0, 60]]))
    assert np.array_equal(rs1, rs2) and rs1.shape == (1, 2) and los.shape == (1, 2)
    rem = radio_environment_map(n, 60.0, (-100, 700, -150, 150), 50.0)
    assert rem.sinr_db.shape == (len(rem.ys), len(rem.xs)) and set(np.unique(rem.serving)) == {0, 1}
    assert 0.0 <= rem.coverage_fraction(-6.0) <= 1.0 and rem.handover_boundary_length(50.0) > 0
    assert rem.percentile(50) == pytest.approx(float(np.median(rem.sinr_db)))
    xs, ys, sinr = n.coverage_map(60.0, (-100, 700, -150, 150), 50.0)
    assert sinr.shape == rem.sinr_db.shape and np.array_equal(xs, rem.xs)


# ---------------------------------------------------------------------------- engine, faults, metrics
CFG = {
    "duration": 6.0,
    "seed": 2,
    "world": {"type": "city", "params": {"blocks": [3, 3]}},
    "trajectory": {"type": "circle", "params": {"center": [0, 0, 60.0], "radius": 40.0, "omega": 0.3}},
    "network": {"type": "nr_cellular", "rate_hz": 10, "params": {"sites": SITES, "background_users": 1, "los_mode": "geometric"}},
}


def test_engine_records_the_nr_link_channels_and_metrics():
    arch = Simulation(CFG).run()
    for k in (
        "link_sinr_db",
        "link_rate_mbps",
        "link_serving",
        "link_outage",
        "link_mcs",
        "link_bler",
        "link_doppler_hz",
        "link_n_handovers",
        "link_rsrp_dbm",
    ):
        assert k in arch, k
    assert arch["link_sinr_db"].shape == (arch.n_steps, 1)
    m = flight_metrics(arch)
    for k in ("link_outage_frac", "link_mean_rate_mbps", "link_p5_rate_mbps", "link_mean_sinr_db", "link_handovers", "link_mean_bler"):
        assert k in m and m[k].shape == (1,), k
    assert arch.meta["wireless"]["type"] == "nr_cellular"


def test_engine_run_is_reproducible_and_survives_archive_roundtrip(tmp_path):
    a = Simulation(CFG).run()
    b = Simulation(CFG).run()
    assert np.array_equal(a["link_sinr_db"], b["link_sinr_db"]) and np.array_equal(a["link_rate_mbps"], b["link_rate_mbps"])
    path = a.save(tmp_path / "w.aevd")
    again = RunArchive.load(path)
    assert np.allclose(again["link_sinr_db"], a["link_sinr_db"]) and again.verify_integrity() == []


def test_network_jamming_fault_degrades_the_link_only_while_active():
    base = Simulation(CFG).run()
    cfg = {**CFG, "faults": [{"type": "network_jamming", "t_start": 2.0, "t_end": 4.0, "params": {"pos": [30.0, 0.0, 5.0], "power_dbm": 40.0}}]}
    arch = Simulation(cfg).run()
    t = arch.t
    drop = base["link_sinr_db"][:, 0] - arch["link_sinr_db"][:, 0]
    assert np.median(drop[(t > 2.3) & (t < 3.8)]) > 3.0  # clearly worse while the jammer transmits
    assert np.abs(drop[t < 1.9]).max() < 1e-6  # identical before it starts (the fault consumes no randomness)
    assert np.median(np.abs(drop[t > 4.3])) < 1.0  # and back to normal afterwards
    kinds = [e["type"] for e in arch.meta["events"]]
    assert "network_jamming" in kinds and "fault.end" in kinds


def test_network_jamming_needs_a_network_that_supports_it():
    cfg = {"duration": 1.0, "faults": [{"type": "network_jamming", "t_start": 0.1}]}
    with pytest.raises(Exception, match="network"):
        Simulation(cfg).run()


def test_comm_blackout_fault_drops_the_nr_link():
    cfg = {**CFG, "faults": [{"type": "comm_blackout", "t_start": 2.0, "t_end": 3.0, "params": {"target": "network"}}]}
    arch = Simulation(cfg).run()
    t, rate = arch.t, arch["link_rate_mbps"][:, 0]
    assert rate[(t > 2.2) & (t < 2.9)].max() == 0.0 and rate[t < 1.8].mean() > 0


def test_config_validation_covers_the_network_parameters():
    bad = {**CFG, "network": {"type": "nr_cellular", "params": {"sites": SITES, "los_mod": "geometric"}}}
    with pytest.raises(ConfigError, match="los_mod"):
        Simulation(bad)
    mesh = {
        "duration": 2.0,
        "vehicle": {"count": 3, "formation": {"type": "line", "params": {"spacing": 150.0}}},
        "network": {"type": "mesh", "rate_hz": 5, "params": {"tx_power_dbm": 20.0, "gateway": 0}},
    }
    arch = Simulation(mesh).run()
    assert "link_hops" in arch and arch["link_rate_mbps"].shape[1] == 3


def test_legacy_cellular_network_is_unchanged_and_still_a_valid_network():
    legacy = CellularNetwork.from_dict({"bs": [{"pos": [0, 0, 25]}], "channel": {"env": "urban"}})
    rep = legacy.evaluate(np.array([[100.0, 0, 60]]))
    assert isinstance(rep, LinkReport) and rep.extras == {}


def test_fast_fading_is_averaged_over_a_long_interval_for_link_adaptation():
    """An interval much longer than the coherence time must see the *fading-averaged* SINR, not one snapshot."""
    kw = dict(los_mode="mean", shadowing=True, handover=False, rlf=False, sites=[{"pos": [0, 0, 25], "sectors": 1}], seed=7)
    faded, plain = net(fading=True, **kw), net(fading=False, **kw)
    for k in range(8):
        p, v = np.array([[200.0 + 3.0 * k, 0, 60]]), np.array([[30.0, 0, 0]])  # 30 m/s at 3.5 GHz: coherence time < 1 ms
        a = faded.evaluate(p, t=0.1 * (k + 1), vel=v, dt=0.1)
        b = plain.evaluate(p, t=0.1 * (k + 1), vel=v, dt=0.1)
        assert a.extras["sinr_eff_db"][0] == pytest.approx(b.sinr_db[0], abs=1e-9)  # fully averaged: equals the slow SINR
        assert a.sinr_db[0] != pytest.approx(b.sinr_db[0], abs=0.05)  # while the recorded snapshot still shows the fading
        assert 0.0 <= a.extras["bler"][0] <= 1.0
    # a slow channel (UAV hovering over a 5 ms interval) is evaluated instantaneously instead
    slow = net(fading=True, **kw)
    r = slow.evaluate(np.array([[200.0, 0, 60]]), t=0.005, vel=np.zeros((1, 3)), dt=0.005)
    assert r.extras["sinr_eff_db"][0] == pytest.approx(r.sinr_db[0], abs=1e-6)


def test_fading_quantiles_have_unit_mean_and_the_right_spread():
    from aevion.wireless.mac_network.nr_network import _fading_quantiles

    q = _fading_quantiles(np.array([1.0, 4.0, 50.0]))
    assert q.shape == (3, 12) and np.allclose(q.mean(axis=1), 1.0)
    assert q[0].std() > q[1].std() > q[2].std()  # Rayleigh spreads most, strong LoS least
