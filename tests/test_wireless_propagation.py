"""Propagation models: reference values, structural properties, diffraction and image-method ray tracing."""

import math

import numpy as np
import pytest

from aevion.environments.world import World
from aevion.wireless.propagation import (
    MODELS,
    TR36777,
    TR38901,
    AlHourani,
    FreeSpace,
    LinkGeometry,
    LogDistance,
    TwoRayGround,
    breakpoint_distance,
    bullington_loss_db,
    fresnel_parameter,
    fspl_db,
    knife_edge_loss_db,
    make_model,
    realise,
    trace_paths,
    world_diffraction_loss_db,
)
from aevion.wireless.propagation.ray_tracing import complex_permittivity, fresnel_reflection

C = 299_792_458.0


def geo(d2d, h_bs=25.0, h_ut=1.5):
    return LinkGeometry.from_distances(np.atleast_1d(np.asarray(d2d, float)), h_bs, h_ut)


# ------------------------------------------------------------------------------------------- reference values
def test_free_space_reference_values():
    assert float(fspl_db(1000.0, 2.4e9)) == pytest.approx(32.44 + 20 * math.log10(1.0) + 20 * math.log10(2400.0), abs=0.02)
    assert float(fspl_db(1.0, 1e9)) == pytest.approx(20 * math.log10(4 * math.pi * 1e9 / C), abs=1e-9)
    # doubling the distance adds 6.02 dB
    assert float(fspl_db(200.0, 3e9) - fspl_db(100.0, 3e9)) == pytest.approx(6.0206, abs=1e-3)


def test_uma_los_matches_the_hand_calculation():
    g = geo(100.0, 25.0, 1.5)
    d3 = math.hypot(100.0, 23.5)
    expected = 28.0 + 22.0 * math.log10(d3) + 20.0 * math.log10(3.5)  # below the breakpoint (560 m)
    assert float(breakpoint_distance(25.0, 1.5, 3.5e9)) == pytest.approx(4 * 24 * 0.5 * 3.5e9 / C)
    assert float(TR38901(3.5e9, "uma").path_loss_db(g, True)[0]) == pytest.approx(expected, abs=1e-9)


def test_uma_los_beyond_the_breakpoint_uses_the_second_slope():
    m = TR38901(3.5e9, "uma")
    g = geo(1500.0, 25.0, 1.5)
    d3 = math.hypot(1500.0, 23.5)
    dbp = 4 * 24 * 0.5 * 3.5e9 / C
    expected = 28.0 + 40.0 * math.log10(d3) + 20.0 * math.log10(3.5) - 9.0 * math.log10(dbp**2 + 23.5**2)
    assert float(m.path_loss_db(g, True)[0]) == pytest.approx(expected, abs=1e-9)


def test_umi_nlos_is_the_larger_of_los_and_the_nlos_formula():
    m = TR38901(3.5e9, "umi")
    g = geo(100.0, 10.0, 1.5)
    d3 = math.hypot(100.0, 8.5)
    nlos_prime = 35.3 * math.log10(d3) + 22.4 + 21.3 * math.log10(3.5) - 0.3 * 0.0
    assert float(m.path_loss_db(g, False)[0]) == pytest.approx(max(nlos_prime, float(m.path_loss_db(g, True)[0])), abs=1e-9)


def test_inh_los_reference():
    m = TR38901(28e9, "inh")
    g = LinkGeometry.from_distances(np.array([20.0]), 3.0, 1.5)
    d3 = math.hypot(20.0, 1.5)
    assert float(m.path_loss_db(g, True)[0]) == pytest.approx(32.4 + 17.3 * math.log10(d3) + 20 * math.log10(28.0), abs=1e-9)


def test_los_probability_reference_points():
    uma = TR38901(3.5e9, "uma")
    assert float(uma.los_probability(geo(10.0))[0]) == 1.0  # <= 18 m
    p = float(uma.los_probability(geo(100.0))[0])
    assert p == pytest.approx(18 / 100 + math.exp(-100 / 63) * (1 - 18 / 100), abs=1e-9)
    umi = TR38901(3.5e9, "umi")
    assert float(umi.los_probability(geo(100.0, 10.0))[0]) == pytest.approx(18 / 100 + math.exp(-100 / 36) * (1 - 18 / 100), abs=1e-9)
    rma = TR38901(3.5e9, "rma")
    assert float(rma.los_probability(geo(510.0, 35.0))[0]) == pytest.approx(math.exp(-0.5), abs=1e-9)
    inh = TR38901(3.5e9, "inh")
    assert float(inh.los_probability(LinkGeometry.from_distances(np.array([1.0, 10.0]), 3.0, 1.5))[0]) == 1.0


def test_uav_height_changes_los_probability_in_tr36777():
    m = TR36777(2.0e9, "uma_av")
    low = float(m.los_probability(geo(500.0, 25.0, 10.0))[0])
    mid = float(m.los_probability(geo(500.0, 25.0, 60.0))[0])
    high = float(m.los_probability(geo(500.0, 25.0, 150.0))[0])
    assert low < mid < high == 1.0  # the higher the UAV, the more likely LoS


def test_tr36777_below_22m_equals_the_terrestrial_model():
    m, ground = TR36777(2.0e9, "umi_av"), TR38901(2.0e9, "umi")
    g = geo(300.0, 10.0, 15.0)
    for los in (True, False):
        assert np.allclose(m.path_loss_db(g, los), ground.path_loss_db(g, los))
    assert np.allclose(m.los_probability(g), ground.los_probability(g))


def test_tr36777_uma_av_formulas_for_a_drone():
    m = TR36777(2.0e9, "uma_av")
    h = 100.0
    g = geo(1000.0, 25.0, h)
    d3 = math.hypot(1000.0, 75.0)
    assert float(m.path_loss_db(g, True)[0]) == pytest.approx(28.0 + 22 * math.log10(d3) + 20 * math.log10(2.0), abs=1e-9)
    nlos = -17.5 + (46 - 7 * math.log10(h)) * math.log10(d3) + 20 * math.log10(40 * math.pi * 2.0 / 3)
    assert float(m.path_loss_db(g, False)[0]) == pytest.approx(max(nlos, float(m.path_loss_db(g, True)[0])), abs=1e-9)
    assert float(m.shadowing_std_db(g, True)[0]) == pytest.approx(4.64 * math.exp(-0.0066 * h))


def test_al_hourani_model_is_fspl_plus_the_environment_excess():
    from aevion.wireless.channel import ENVIRONMENTS

    m = AlHourani(2.0e9, "urban")
    g = geo(300.0, 0.0, 100.0)
    eta_l, eta_n = ENVIRONMENTS["urban"][2:]
    base = float(fspl_db(g.d3d, 2.0e9)[0])
    assert float(m.path_loss_db(g, True)[0]) == pytest.approx(base + eta_l)
    assert float(m.path_loss_db(g, False)[0]) == pytest.approx(base + eta_n)


def test_two_ray_follows_fourth_power_beyond_the_crossover():
    m = TwoRayGround(2.4e9)
    g_far = LinkGeometry.from_distances(np.array([5000.0, 10000.0]), 10.0, 2.0)  # beyond the 2 km crossover
    pl = m.path_loss_db(g_far, True)
    assert pl[1] - pl[0] == pytest.approx(12.04, abs=0.05)  # 40 dB/decade -> 12.04 dB per doubling
    near = LinkGeometry.from_distances(np.array([10.0, 20.0]), 10.0, 2.0)
    assert np.allclose(m.path_loss_db(near, True), fspl_db(near.d3d, 2.4e9))


def test_log_distance_exponent():
    m = LogDistance(2.4e9, n_los=2.0, n_nlos=4.0)
    g = LinkGeometry.from_distances(np.array([10.0, 100.0]), 10.0, 1.5)
    assert float(np.diff(m.path_loss_db(g, True))[0]) == pytest.approx(10 * 2.0 * math.log10(g.d3d[1] / g.d3d[0]), abs=1e-9)
    assert float(np.diff(m.path_loss_db(g, False))[0]) == pytest.approx(10 * 4.0 * math.log10(g.d3d[1] / g.d3d[0]), abs=1e-9)


# --------------------------------------------------------------------------------------- structural properties
MODEL_CASES = [
    ("free_space", {}),
    ("two_ray", {}),
    ("log_distance", {}),
    ("al_hourani", {}),
    ("tr38901", {"scenario": "uma"}),
    ("tr38901", {"scenario": "umi"}),
    ("tr38901", {"scenario": "rma"}),
    ("tr38901", {"scenario": "inh"}),
    ("tr36777", {"scenario": "uma_av"}),
    ("tr36777", {"scenario": "umi_av"}),
    ("tr36777", {"scenario": "rma_av"}),
]


@pytest.mark.parametrize("name,kw", MODEL_CASES, ids=lambda x: str(x))
def test_every_model_is_finite_monotone_and_consistent(name, kw):
    m = make_model(name, freq_hz=3.5e9 if "tr" in name else 2.0e9, **kw)
    d = np.geomspace(20.0, 3000.0, 40)
    g = LinkGeometry.from_distances(d, 25.0, 10.0)
    for los in (True, False):
        pl = m.path_loss_db(g, los)
        assert np.all(np.isfinite(pl)) and pl.shape == d.shape
        assert np.all(np.diff(pl) > -1e-6), f"{name} {kw} path loss decreases with distance (los={los})"
        assert np.all(m.shadowing_std_db(g, los) >= 0)
    assert np.all(m.path_loss_db(g, False) >= m.path_loss_db(g, True) - 1e-9)  # NLoS never better than LoS
    p = m.los_probability(g)
    assert np.all((p >= 0) & (p <= 1))
    mean_pl = m.mean_path_loss_db(g)
    assert np.all(mean_pl >= m.path_loss_db(g, True) - 1e-9) and np.all(mean_pl <= m.path_loss_db(g, False) + 1e-9)


@pytest.mark.parametrize("scenario", ["uma", "umi", "rma", "inh"])
def test_ground_los_probability_does_not_increase_with_distance(scenario):
    m = TR38901(3.5e9, scenario)
    p = m.los_probability(LinkGeometry.from_distances(np.geomspace(2.0, 3000.0, 50), 25.0, 1.5))
    assert np.all(np.diff(p) <= 1e-12)


def test_validity_flags_extrapolation():
    m = TR38901(3.5e9, "uma")
    ok = m.validity(geo(np.array([5.0, 100.0, 9000.0])))
    assert ok.tolist() == [False, True, False]
    assert not TR38901(150e9, "uma").validity(geo(100.0))[0]
    assert not TR36777(2e9, "uma_av").validity(geo(500.0, 25.0, 400.0))[0]


def test_unknown_names_and_parameters_are_rejected():
    with pytest.raises(KeyError, match="unknown propagation model"):
        make_model("okumura")
    with pytest.raises(ValueError):
        TR38901(3.5e9, "suburban")
    with pytest.raises(ValueError):
        TR36777(2e9, "nope")
    with pytest.raises(ValueError):
        FreeSpace(-1.0)
    assert set(MODELS) == {"free_space", "two_ray", "log_distance", "al_hourani", "tr38901", "tr36777"}


def test_realise_draws_los_and_shadowing_reproducibly():
    m = TR38901(3.5e9, "uma")
    g = geo(np.linspace(50, 400, 200))
    a = realise(m, g, np.random.default_rng(4))
    b = realise(m, g, np.random.default_rng(4))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    pl, los = realise(m, g, np.random.default_rng(1), shadowing=False)
    assert np.allclose(pl, m.path_loss_db(g, los))
    forced = realise(m, g, np.random.default_rng(1), los=True, shadowing=False)[0]
    assert np.allclose(forced, m.path_loss_db(g, True))
    # shadowing has the model's standard deviation
    many = LinkGeometry.from_distances(np.full(20000, 200.0), 25.0, 1.5)
    resid = realise(m, many, np.random.default_rng(2), los=True)[0] - m.path_loss_db(many, True)
    assert float(resid.std()) == pytest.approx(4.0, abs=0.15)


# -------------------------------------------------------------------------------------------- diffraction
def test_knife_edge_reference_values():
    assert float(knife_edge_loss_db(0.0)) == pytest.approx(6.03, abs=0.02)  # grazing incidence: ~6 dB
    assert float(knife_edge_loss_db(1.0)) == pytest.approx(13.9, abs=0.1)
    assert float(knife_edge_loss_db(2.4)) == pytest.approx(20.5, abs=0.1)
    assert float(knife_edge_loss_db(-1.0)) == 0.0  # well clear of the edge
    nu = np.linspace(-0.7, 5, 50)
    assert np.all(np.diff(knife_edge_loss_db(nu)) > 0)


def test_fresnel_parameter_scaling():
    lam = C / 2.4e9
    nu = float(fresnel_parameter(10.0, 50.0, 50.0, 2.4e9))
    assert nu == pytest.approx(10.0 * math.sqrt(2 * 100 / (lam * 2500)))
    assert float(fresnel_parameter(-5.0, 50.0, 50.0, 2.4e9)) < 0


def test_bullington_with_one_edge_equals_a_single_knife_edge():
    f = 2.4e9
    loss, nu = bullington_loss_db([10.0], [10.0], [100.0], [[40.0]], [[25.0]], [[True]], f)
    expected_nu = float(fresnel_parameter(15.0, 40.0, 60.0, f))
    assert float(nu[0]) == pytest.approx(expected_nu, rel=1e-9)
    assert float(loss[0]) == pytest.approx(float(knife_edge_loss_db(expected_nu)), abs=1e-9)


def test_bullington_without_edges_has_no_loss():
    loss, _ = bullington_loss_db([10.0], [10.0], [100.0], np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((1, 0), bool), 2.4e9)
    assert loss[0] == 0.0


def test_world_diffraction_over_a_building_roof():
    w = World()
    w.add_box_corners([45, -5, 0], [55, 5, 20])
    loss = float(world_diffraction_loss_db(w, [[0, 0, 10.0]], [[100, 0, 10.0]], 2.4e9)[0])
    expected = float(knife_edge_loss_db(float(fresnel_parameter(10.0, 45.0, 55.0, 2.4e9))))
    assert loss == pytest.approx(expected, abs=0.1)
    # a path that passes well above the roof is not attenuated
    assert float(world_diffraction_loss_db(w, [[0, 0, 60.0]], [[100, 0, 60.0]], 2.4e9)[0]) < 1.0
    # a path that misses the building sideways is not attenuated
    assert float(world_diffraction_loss_db(w, [[0, 30, 10.0]], [[100, 30, 10.0]], 2.4e9)[0]) == 0.0
    # taller building -> more loss
    w2 = World()
    w2.add_box_corners([45, -5, 0], [55, 5, 40])
    assert float(world_diffraction_loss_db(w2, [[0, 0, 10.0]], [[100, 0, 10.0]], 2.4e9)[0]) > loss


def test_diffraction_uses_terrain_profile():
    w = World()
    h = np.zeros((41, 41))
    h[:, 18:23] = 30.0  # a ridge across the path
    w.set_heightmap(h, origin=(0.0, -20.0), resolution=2.5)
    ridge = float(world_diffraction_loss_db(w, [[0, 0, 10.0]], [[100, 0, 10.0]], 2.4e9)[0])
    flat = float(world_diffraction_loss_db(World(), [[0, 0, 10.0]], [[100, 0, 10.0]], 2.4e9)[0])
    assert ridge > 10.0 and flat == 0.0


# ----------------------------------------------------------------------------------------------- ray tracing
def test_fresnel_reflection_limits():
    assert abs(fresnel_reflection(0.0, complex_permittivity("metal", 2.4e9))) == pytest.approx(1.0, abs=1e-3)
    # normal incidence on a dielectric: |(1 - sqrt(eps)) / (1 + sqrt(eps))|
    eps = 5.31 - 0.0j
    assert abs(fresnel_reflection(0.0, eps, "te")) == pytest.approx((math.sqrt(5.31) - 1) / (math.sqrt(5.31) + 1), abs=1e-9)
    # Brewster angle: the parallel component vanishes
    brewster = math.atan(math.sqrt(5.31))
    assert abs(fresnel_reflection(brewster, eps, "tm")) < 1e-9
    # grazing incidence reflects almost everything
    assert abs(fresnel_reflection(math.radians(89.5), eps, "te")) > 0.95
    with pytest.raises(KeyError):
        complex_permittivity("unobtainium", 1e9)
    with pytest.raises(ValueError):
        fresnel_reflection(0.3, eps, "circular")


def test_two_ray_ground_reflection_matches_the_analytic_interference_pattern():
    """Perfect-conductor ground: the coherent sum of direct and image paths is the textbook two-ray pattern."""
    f = 2.4e9
    lam, k = C / f, 2 * math.pi * f / C
    ht, hr = 20.0, 2.0
    d = np.array([50.0, 80.0, 120.0, 300.0, 700.0])
    tx = np.tile([0.0, 0.0, ht], (len(d), 1))
    rx = np.stack([d, np.zeros_like(d), np.full_like(d, hr)], axis=1)
    ps = trace_paths(World(), tx, rx, f, ground=True, ground_material="metal")
    d1, d2 = np.hypot(d, ht - hr), np.hypot(d, ht + hr)
    analytic = (lam / (4 * math.pi)) ** 2 * np.abs(np.exp(-1j * k * d1) / d1 - np.exp(-1j * k * d2) / d2) ** 2
    assert np.allclose(10 ** (ps.power_gain_db() / 10), analytic, rtol=1e-6)
    assert ps.valid.all() and ps.labels == ["direct", "ground"]
    # beyond the crossover the envelope falls with the fourth power of distance
    far = trace_paths(World(), [[0, 0, ht]] * 2, [[3000, 0, hr], [6000, 0, hr]], f, ground_material="metal")
    assert float(far.power_gain_db()[0] - far.power_gain_db()[1]) == pytest.approx(12.0, abs=1.0)


def test_building_blocks_the_direct_path_but_not_a_wall_reflection():
    w = World()
    w.add_box_corners([25, -10, 0], [35, 10, 20])  # between tx and rx
    tx = [[0, 0, 25.0]]
    blocked = trace_paths(w, tx, [[60, 0, 1.5]], 2.4e9)
    assert not blocked.valid[0, 0]
    assert blocked.k_factor_db()[0] == -np.inf
    clear = trace_paths(w, tx, [[0, 30, 1.5]], 2.4e9)
    assert clear.valid[0, 0] and clear.valid[0, 1]  # direct and ground
    assert clear.n_paths()[0] >= 2


def test_wall_reflection_has_the_image_path_length():
    w = World()
    w.add_box_corners([40, -50, 0], [50, 50, 30])  # a long wall facing -x
    tx = np.array([[0.0, -10.0, 20.0]])
    rx = np.array([[0.0, 10.0, 20.0]])
    ps = trace_paths(w, tx, rx, 5e9, ground=False)
    idx = ps.labels.index("box0:-x")
    assert ps.valid[0, idx]
    assert ps.length_m[0, idx] == pytest.approx(math.hypot(2 * 40.0, 20.0), rel=1e-9)  # mirror image of tx behind the wall plane
    assert ps.rms_delay_spread_s()[0] > 0
    assert math.isfinite(float(ps.k_factor_db()[0]))


def test_trace_paths_handles_an_empty_world_and_batches():
    ps = trace_paths(World(), np.zeros((4, 3)) + [0, 0, 10], np.random.default_rng(0).uniform(10, 100, (4, 3)) * [1, 1, 0.1], 3.5e9)
    assert ps.length_m.shape == (4, 2) and ps.valid[:, 0].all()
    assert np.all(np.isfinite(ps.power_gain_db()))


def test_validity_covers_the_terminal_height():
    uma = TR38901(3.5e9, "uma")
    assert uma.validity(geo(np.array([200.0, 200.0]), 25.0, 1.5))[0] and not uma.validity(geo(np.array([200.0]), 25.0, 60.0))[0]
    assert uma.h_ut_range_m == (1.5, 22.5) and TR38901(3.5e9, "rma").h_ut_range_m == (1.0, 10.0)
    assert TR36777(2e9, "uma_av").validity(geo(np.array([200.0]), 25.0, 60.0))[0]  # the aerial model covers UAV heights
    assert "h_ut_range_m" in uma.describe()


def test_terrestrial_model_for_a_high_uav_is_flagged_and_the_default_is_aerial():
    """Regression: TR 38.901 UMa NLoS contains -0.6 (h_UT - 1.5), which silently removes ~29 dB of loss at 50 m."""
    import warnings

    from aevion.wireless.mac_network import NrCellularNetwork

    site = [{"pos": [0, 0, 25], "sectors": 1}]
    p = np.array([[150.0, 0, 60.0]])
    with pytest.warns(UserWarning, match="validity range.*tr36777"):
        NrCellularNetwork(sites=site, propagation="tr38901", los_mode="mean", seed=0).evaluate(p, t=0.1, dt=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NrCellularNetwork(sites=site, los_mode="mean", seed=0).evaluate(p, t=0.1, dt=0.1)  # the default model is the aerial one
    g = LinkGeometry.from_distances(np.array([160.0]), 25.0, 50.0)
    assert float(TR36777(3.5e9, "uma_av").path_loss_db(g, False)[0]) > float(TR38901(3.5e9, "uma").path_loss_db(g, False)[0]) + 10
