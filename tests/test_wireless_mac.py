"""MAC/network layer: schedulers, handover, radio-link failure, relay, mesh routing and CSMA."""

import math
from itertools import pairwise

import numpy as np
import pytest

from aevion.wireless.mac_network import (
    BianchiParams,
    CellScheduler,
    HandoverManager,
    MaxCI,
    MeshNetwork,
    ProportionalFair,
    RadioLinkMonitor,
    RelayLink,
    RoundRobin,
    best_rate,
    best_relay_position,
    bianchi_fixed_point,
    bianchi_throughput_bps,
    collision_probability,
    compute_sinr,
    connected_components,
    dijkstra,
    link_rate_matrix,
    make_scheduler,
    relay_rate,
    select_serving,
    simulate_cell,
)
from aevion.wireless.metrics import jain_fairness


def rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------------------- interference
def test_sinr_from_received_powers():
    rx = np.array([[-70.0, -80.0, -90.0]])  # dBm from three cells
    res = compute_sinr(rx, [0], noise_dbm=-100.0)
    sig = 10 ** (-7.0) * 1e-3
    interf = (10 ** (-8.0) + 10 ** (-9.0)) * 1e-3
    noise = 10 ** (-10.0) * 1e-3
    assert float(res.sinr_db[0]) == pytest.approx(10 * math.log10(sig / (interf + noise)), abs=1e-9)
    # lower neighbour load lowers interference
    assert compute_sinr(rx, [0], -100.0, load=[1.0, 0.1, 0.1]).sinr_db[0] > res.sinr_db[0]
    # a jammer adds interference
    assert compute_sinr(rx, [0], -100.0, jammer_dbm=[-75.0]).sinr_db[0] < res.sinr_db[0]
    assert compute_sinr(rx, [0], -100.0, jammer_dbm=np.array([[-75.0, -80.0]])).sinr_db[0] < res.sinr_db[0]
    assert list(select_serving(rx)) == [0] and list(select_serving(rx, bias_db=[0, 15, 0])) == [1]


# ------------------------------------------------------------------------------------------ schedulers
def test_scheduler_names_and_factory():
    assert (
        isinstance(make_scheduler("pf"), ProportionalFair)
        and isinstance(make_scheduler("rr"), RoundRobin)
        and isinstance(make_scheduler("max-ci"), MaxCI)
    )
    with pytest.raises(KeyError):
        make_scheduler("lottery")


def test_round_robin_gives_equal_time_proportional_fair_nearly_so_and_max_ci_starves_the_weak():
    sinr = 10 ** (np.array([20.0, 8.0, 0.0]) / 10)
    bw = 20e6
    rr = simulate_cell(sinr, bw, RoundRobin(), 4000, rng(0))
    pf = simulate_cell(sinr, bw, ProportionalFair(), 4000, rng(0))
    mx = simulate_cell(sinr, bw, MaxCI(), 4000, rng(0))
    assert np.allclose(rr[0], 1 / 3, atol=1e-3)
    assert np.all(pf[0] > 0.2) and pf[0].max() < 0.5  # PF keeps everybody in the game
    assert mx[0][0] > 0.9 and mx[0][2] < 0.02  # max C/I serves the strong user almost exclusively
    # multi-user diversity: max C/I > PF > RR in sum throughput
    assert mx[1].sum() > pf[1].sum() > rr[1].sum()
    assert jain_fairness(rr[0]) > jain_fairness(pf[0]) > jain_fairness(mx[0])


def test_pf_exploits_fading_peaks_of_equal_users():
    sinr = np.full(4, 10.0)
    _, thr_pf = simulate_cell(sinr, 20e6, ProportionalFair(), 3000, rng(1))
    _, thr_rr = simulate_cell(sinr, 20e6, RoundRobin(), 3000, rng(1))
    assert thr_pf.sum() > 1.08 * thr_rr.sum()  # scheduling on fading peaks gains several percent


def test_single_user_gets_the_whole_cell():
    share, thr = simulate_cell([10.0], 20e6, ProportionalFair())
    assert share[0] == 1.0 and thr[0] == pytest.approx(20e6 * math.log2(1 + 10 / 10**0.3))


def test_cell_scheduler_shares():
    s = CellScheduler(2, "round_robin", background_users=3, rng=rng(0))
    sh = s.shares([0, 0, 1], [10.0, 10.0, 10.0])
    assert np.allclose(sh, [1 / 5, 1 / 5, 1 / 4])  # cell 0: 2 UAVs + 3 background, cell 1: 1 UAV + 3 background
    none = CellScheduler(2, "none", background_users=3, rng=rng(0))
    assert np.all(none.shares([0, 1], [10.0, 10.0]) == 1.0)
    alone = CellScheduler(1, "proportional_fair", background_users=0, rng=rng(0))
    assert alone.shares([0], [10.0])[0] == 1.0
    for policy in ("proportional_fair", "max_ci"):
        sc = CellScheduler(1, policy, background_users=2, rng=rng(0))
        for _ in range(5):
            sh = sc.shares([0, 0], [20.0, 1.0])
        assert 0.0 <= sh.sum() <= 1.0 + 1e-12
    mx = CellScheduler(1, "max_ci", background_users=2, bg_sinr_db=(0.0, 0.1), rng=rng(0))
    assert mx.shares([0], [1000.0])[0] > 0.9  # a very strong UAV dominates max C/I
    with pytest.raises(KeyError):
        CellScheduler(1, "chaos")


# ------------------------------------------------------------------------------------------- handover
def synth_rsrp(t, speed_db_per_s=20.0):
    """Two cells: cell 0 fades away, cell 1 grows; crossover at t = 1 s."""
    return np.array([[-70.0 - speed_db_per_s * (t - 1.0) / 2, -70.0 + speed_db_per_s * (t - 1.0) / 2]])


def run_handover(manager, t_end=3.0, dt=0.02, sinr=10.0):
    served, intr = [], []
    for k in range(int(t_end / dt)):
        t = k * dt
        s, i = manager.step(t, synth_rsrp(t), dt, serving_sinr_db=sinr)
        served.append(int(s[0]))
        intr.append(bool(i[0]))
    return np.array(served), np.array(intr)


def test_a3_handover_triggers_after_hysteresis_and_time_to_trigger():
    m = HandoverManager(1, 2, hysteresis_db=3.0, time_to_trigger_s=0.16, l3_filter_k=0, interruption_s=0.05)
    served, intr = run_handover(m)
    assert served[0] == 0 and served[-1] == 1 and m.stats.n_handovers[0] == 1
    ev = m.events[0]
    # neighbour exceeds serving by 3 dB at t = 1 + 3/20 = 1.15 s; the A3 event fires after the 0.16 s time-to-trigger
    assert ev.t == pytest.approx(1.15 + 0.16, abs=0.05) and (ev.source, ev.target) == (0, 1)
    assert not ev.failed and not ev.pingpong
    # interruption lasts interruption_s
    assert intr.sum() * 0.02 == pytest.approx(0.05, abs=0.03)


def test_larger_hysteresis_or_ttt_delay_the_handover():
    times = {}
    for name, kw in {"base": dict(), "hys": dict(hysteresis_db=8.0), "ttt": dict(time_to_trigger_s=0.64)}.items():
        m = HandoverManager(1, 2, l3_filter_k=0, **{"hysteresis_db": 3.0, "time_to_trigger_s": 0.16, **kw})
        run_handover(m)
        times[name] = m.events[0].t
    assert times["hys"] > times["base"] + 0.2 and times["ttt"] > times["base"] + 0.4  # +5 dB of hysteresis = +0.25 s at 20 dB/s


def test_l3_filtering_delays_the_decision():
    fast = HandoverManager(1, 2, l3_filter_k=0, time_to_trigger_s=0.0)
    slow = HandoverManager(1, 2, l3_filter_k=19, time_to_trigger_s=0.0)  # alpha = 0.037: about 0.5 s of smoothing at 50 Hz
    run_handover(fast)
    run_handover(slow, t_end=6.0)
    assert slow.events[0].t > fast.events[0].t + 0.3


def test_no_handover_without_a_better_neighbour():
    m = HandoverManager(1, 2)
    for k in range(200):
        m.step(k * 0.02, np.array([[-70.0, -80.0 + np.sin(k)]]), 0.02)
    assert m.stats.n_handovers[0] == 0 and m.serving[0] == 0


def test_handover_failure_and_pingpong_are_classified():
    m = HandoverManager(
        1,
        2,
        l3_filter_k=0,
        time_to_trigger_s=0.04,
        hysteresis_db=1.0,
        failure_sinr_db=-8.0,
        failure_recovery_s=0.5,
        pingpong_window_s=1.0,
        interruption_s=0.02,
    )
    # attach to cell 0, hand over to 1, and straight back within a second
    seq = [(-60.0, -70.0)] * 5 + [(-70.0, -60.0)] * 15 + [(-60.0, -70.0)] * 30
    for k, (a, b) in enumerate(seq):
        m.step(k * 0.02, np.array([[a, b]]), 0.02, serving_sinr_db=10.0)
    assert m.stats.n_handovers[0] == 2 and m.stats.n_pingpong[0] == 1 and m.stats.n_failures[0] == 0
    # a command issued while the serving link is already below the failure threshold is a handover failure
    f = HandoverManager(1, 2, l3_filter_k=0, time_to_trigger_s=0.04, hysteresis_db=1.0, failure_sinr_db=-8.0, failure_recovery_s=0.5)
    interrupted = []
    for k in range(40):
        _, i = f.step(k * 0.02, np.array([[-70.0, -60.0]]) if k else np.array([[-60.0, -70.0]]), 0.02, serving_sinr_db=-12.0)
        interrupted.append(bool(i[0]))
    assert f.stats.n_failures[0] == 1 and sum(interrupted) * 0.02 == pytest.approx(0.5, abs=0.06)
    assert f.stats.rate_per_min(10.0)[0] == pytest.approx(6.0)


def test_handover_manager_validation_and_reselection():
    with pytest.raises(ValueError):
        HandoverManager(1, 0)
    m = HandoverManager(2, 3)
    m.step(0.0, np.array([[-60.0, -80, -90], [-90, -60, -80]]), 0.02)
    assert list(m.serving) == [0, 1]
    m.force_reselect([0], np.array([[-90.0, -90, -50], [-90, -60, -80]]))
    assert list(m.serving) == [2, 1]
    m.force_reselect([], np.zeros((2, 3)))
    m.reset()
    assert m.serving is None


# ----------------------------------------------------------------------------------------------- RLF
def test_rlf_declared_after_t310_and_recovers_after_the_recovery_time():
    mon = RadioLinkMonitor(1, t310_s=0.5, recovery_s=0.3, filter_s=0.001)
    t, down, kinds = 0.0, [], []
    for k in range(120):
        sinr = -15.0 if 10 <= k < 80 else 5.0
        down.append(bool(mon.step(t, sinr, 0.02)[0]))
        t += 0.02
    kinds = [e.kind for e in mon.events]
    assert kinds[:2] == ["t310_start", "rlf"] and "recovered" in kinds
    rlf = next(e for e in mon.events if e.kind == "rlf")
    start = next(e for e in mon.events if e.kind == "t310_start")
    assert rlf.t - start.t == pytest.approx(0.5, abs=0.04) and mon.n_rlf[0] >= 1
    assert sum(down) * 0.02 >= 0.3 - 0.04


def test_rlf_timer_is_stopped_by_recovery_in_time():
    mon = RadioLinkMonitor(1, t310_s=1.0, filter_s=0.001, n311=2)
    t = 0.0
    for k in range(60):
        sinr = -15.0 if 5 <= k < 25 else 5.0  # 0.4 s outage < T310
        mon.step(t, sinr, 0.02)
        t += 0.02
    kinds = [e.kind for e in mon.events]
    assert kinds == ["t310_start", "t310_stop"] and mon.n_rlf[0] == 0


def test_rlf_n310_requires_consecutive_out_of_sync_indications():
    mon = RadioLinkMonitor(1, n310=5, t310_s=0.2, filter_s=0.001)
    t = 0.0
    for k in range(30):
        mon.step(t, -15.0 if k % 3 == 0 else 5.0, 0.02)  # never five bad indications in a row
        t += 0.02
    assert mon.events == []
    with pytest.raises(ValueError):
        RadioLinkMonitor(1, q_out_db=-6.0, q_in_db=-8.0)


# --------------------------------------------------------------------------------------------- relay
def test_relay_rate_formulas():
    g1, g2, bw = 20.0, 5.0, 10e6
    assert float(relay_rate(g1, g2, bw, "df")) == pytest.approx(0.5 * bw * min(math.log2(1 + g1), math.log2(1 + g2)))
    assert float(relay_rate(g1, g2, bw, "df", half_duplex=False)) == pytest.approx(2 * float(relay_rate(g1, g2, bw, "df")))
    g_af = g1 * g2 / (g1 + g2 + 1)
    assert float(relay_rate(g1, g2, bw, "af")) == pytest.approx(0.5 * bw * math.log2(1 + g_af))
    assert float(relay_rate(g1, g2, bw, "af")) <= float(relay_rate(g1, g2, bw, "df")) + 1e-9
    with pytest.raises(ValueError):
        relay_rate(1, 1, bw, "magic")
    rate, used = best_rate(0.5, 100.0, 100.0, bw)
    assert used and float(rate) > bw * math.log2(1.5)
    rate, used = best_rate(1000.0, 5.0, 5.0, bw)
    assert not used


def test_best_relay_position_is_the_midpoint_when_the_direct_link_is_blocked():
    link = RelayLink(
        source=[0.0, 0.0, 30.0], destination=[2000.0, 0.0, 30.0], freq_hz=2.4e9, bandwidth_hz=20e6, tx_power_dbm=20.0, extra_loss_direct_db=60.0
    )
    pos, rate = best_relay_position(link, 100.0, (-500, 2500, -500, 500), 50.0)
    assert pos[0] == pytest.approx(1000.0, abs=60.0) and abs(pos[1]) <= 50.0 and rate > 0
    out = link.rates(pos)
    assert out["relay_bps"] > out["direct_bps"]
    assert abs(float(out["snr_sr_db"] - out["snr_rd_db"])) < 3.0


# ---------------------------------------------------------------------------------------------- mesh
def test_dijkstra_and_components():
    inf = np.inf
    w = np.array([[0, 1, 4, inf, inf], [1, 0, 2, 7, inf], [4, 2, 0, 3, inf], [inf, 7, 3, 0, inf], [inf, inf, inf, inf, 0]], float)
    w[w == 0] = inf
    d, pred = dijkstra(w, 0)
    assert list(d[:4]) == [0, 1, 3, 6] and d[4] == inf
    assert list(pred[:4]) == [-1, 0, 1, 2]
    adj = np.isfinite(w)
    assert list(connected_components(adj)) == [0, 0, 0, 0, 1]


def test_link_rate_matrix_symmetry_and_distance_ordering():
    pos = np.array([[0, 0, 50.0], [100, 0, 50], [1000, 0, 50], [10_000, 0, 50]])
    rate, sinr = link_rate_matrix(pos, 2.4e9, 20e6, 20.0)
    assert np.allclose(rate, rate.T) and np.all(np.diag(rate) == 0)
    assert rate[0, 1] > rate[0, 2] > rate[0, 3] and sinr[0, 1] > sinr[0, 2]


def chain(spacing, n=4):
    return np.array([[i * spacing, 0.0, 60.0] for i in range(n)])


def test_mesh_multi_hop_rate_and_hops():
    net = MeshNetwork(freq_hz=2.4e9, bandwidth_hz=20e6, tx_power_dbm=10.0, min_link_rate_bps=1e6, metric="airtime", gateway=0)
    rep = net.analyse(chain(400.0))
    assert (rep.connected.all() and list(rep.hops) == [0, 1, 2, 3]) or rep.hops[3] <= 3
    assert rep.hops[1] == 1 and rep.hops[3] >= 1
    rate, _ = link_rate_matrix(chain(400.0), 2.4e9, 20e6, 10.0)
    if rep.hops[2] == 2:
        assert rep.e2e_rate_bps[2] == pytest.approx(1.0 / (1.0 / rate[2, 1] + 1.0 / rate[1, 0]), rel=1e-6)
    assert rep.e2e_rate_bps[1] >= rep.e2e_rate_bps[2] >= 0
    assert rep.n_components == 1


def test_mesh_partition_and_external_gateway():
    net = MeshNetwork(freq_hz=2.4e9, tx_power_dbm=0.0, min_link_rate_bps=5e6, gateway=0)
    pos = np.array([[0, 0, 50.0], [200, 0, 50], [50_000, 0, 50]])
    rep = net.analyse(pos)
    assert list(rep.connected) == [True, True, False] and rep.hops[2] == -1 and rep.n_components == 2
    ext = MeshNetwork(freq_hz=2.4e9, tx_power_dbm=20.0, min_link_rate_bps=1e6, gateway=[0.0, 0.0, 10.0])
    r2 = ext.analyse(np.array([[300.0, 0, 50], [600.0, 0, 50]]))
    assert r2.connected.all() and r2.hops[0] == 1 and r2.hops[1] >= 1


def test_mesh_network_engine_interface():
    net = MeshNetwork(tx_power_dbm=20.0, gateway=0)
    rep = net.evaluate(chain(300.0))
    assert rep.rate_bps.shape == (4,) and rep.sinr_db.shape == (4,) and rep.all_rx_dbm.shape == (4, 4)
    assert set(rep.extras) == {"hops", "connected", "n_components"} and rep.extras["connected"].all()
    assert net.to_dict()["metric"] == "airtime"
    for metric in ("hops", "etx"):
        assert MeshNetwork(metric=metric).analyse(chain(300.0)).connected.all()
    with pytest.raises(ValueError):
        MeshNetwork(metric="random")


# ---------------------------------------------------------------------------------------------- CSMA
def test_bianchi_single_station_closed_form():
    tau, p = bianchi_fixed_point(1, 32, 3)
    assert tau == pytest.approx(2 / 33) and p == 0.0
    P = BianchiParams()
    s = bianchi_throughput_bps(1, P)
    mean_slot = (1 - tau) * P.slot_us + tau * P.t_success_basic
    assert s == pytest.approx(tau * P.payload_bits / (mean_slot * 1e-6), rel=1e-9)
    assert s < P.rate_bps


@pytest.mark.parametrize("n,w,m", [(2, 32, 3), (5, 16, 5), (10, 32, 3), (30, 64, 6)])
def test_bianchi_fixed_point_satisfies_both_equations(n, w, m):
    tau, p = bianchi_fixed_point(n, w, m)
    assert p == pytest.approx(1 - (1 - tau) ** (n - 1), abs=1e-9)
    assert tau == pytest.approx(2 * (1 - 2 * p) / ((1 - 2 * p) * (w + 1) + p * w * (1 - (2 * p) ** m)), rel=1e-6)
    assert 0 < tau < 1 and 0 <= p < 1


def test_bianchi_trends():
    p = [collision_probability(n) for n in (2, 5, 10, 20, 50)]
    assert all(a < b for a, b in pairwise(p))
    assert collision_probability(10, w=128) < collision_probability(10, w=16)  # larger window -> fewer collisions
    s_basic = [bianchi_throughput_bps(n) for n in (1, 5, 20, 50)]
    assert s_basic[1] > s_basic[2] > s_basic[3]  # collisions erode throughput as stations are added
    # RTS/CTS protects long packets when many stations contend
    assert bianchi_throughput_bps(50, rts_cts=True) > bianchi_throughput_bps(50)
    assert all(bianchi_throughput_bps(n) < BianchiParams().rate_bps for n in (1, 5, 50))
    with pytest.raises(ValueError):
        bianchi_fixed_point(0)
