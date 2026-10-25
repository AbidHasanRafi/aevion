"""Designs of experiments and global sensitivity analysis - against analytic benchmark functions."""

import math

import numpy as np
import pytest

from aevion.experiments.design import ParameterSpace, discrepancy, full_factorial, halton, latin_hypercube, min_distance, sobol
from aevion.experiments.sensitivity import morris, morris_analyze, morris_design, saltelli_design, sobol_analyze, sobol_indices


# ------------------------------------------------------------------------------------------------ designs
@pytest.mark.parametrize("n,d", [(5, 2), (16, 3), (50, 6)])
def test_latin_hypercube_has_exactly_one_point_in_every_stratum_of_every_coordinate(n, d):
    for optimize in (False, True):
        x = latin_hypercube(n, d, seed=3, optimize=optimize)
        assert x.shape == (n, d) and x.min() >= 0 and x.max() < 1
        for k in range(d):
            assert sorted(np.floor(x[:, k] * n).astype(int)) == list(range(n))


def test_the_maximin_optimisation_spreads_the_points_and_the_design_is_seeded():
    plain = latin_hypercube(30, 4, seed=1, optimize=False)
    opt = latin_hypercube(30, 4, seed=1, optimize=True, iterations=500)
    assert min_distance(opt) > min_distance(plain)
    assert np.array_equal(latin_hypercube(30, 4, seed=1), latin_hypercube(30, 4, seed=1)) and not np.array_equal(
        latin_hypercube(30, 4, seed=1), latin_hypercube(30, 4, seed=2)
    )
    assert min_distance(np.zeros((1, 2))) == math.inf
    with pytest.raises(ValueError):
        latin_hypercube(0, 2)


def test_low_discrepancy_sequences_cover_the_cube_more_evenly_than_random_points():
    n, d = 64, 4
    rnd = np.mean([discrepancy(np.random.default_rng(s).random((n, d))) for s in range(10)])
    assert discrepancy(sobol(n, d, seed=0)) < 0.5 * rnd
    assert discrepancy(halton(n, d)) < rnd
    assert discrepancy(latin_hypercube(n, d, seed=0)) < rnd
    s = sobol(64, 3, seed=5)  # a (0, m, 3)-net in base 2: one point in each of the 64 strata of every coordinate
    for k in range(3):
        assert sorted(np.floor(s[:, k] * 64).astype(int)) == list(range(64))


def test_halton_matches_the_radical_inverse_values():
    h = halton(6, 2)
    assert np.allclose(h[:3, 0], [0.5, 0.25, 0.75]) and np.allclose(h[:3, 1], [1 / 3, 2 / 3, 1 / 9])
    with pytest.raises(ValueError):
        halton(3, 50)
    assert full_factorial([[1, 2], ["a", "b", "c"]])[:3] == [(1, "a"), (1, "b"), (1, "c")] and len(full_factorial([[1, 2], [3, 4], [5, 6]])) == 8


def test_parameter_spaces_map_the_unit_cube_to_named_values_and_back():
    sp = ParameterSpace({"mass": (0.8, 1.2), "kp": ("log", 1.0, 100.0), "n": ("int", 1, 4), "mode": ["a", "b", "c"]})
    assert sp.dim == 4 and sp.names == ["mass", "kp", "n", "mode"]
    pts = sp.points(np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.999, 0.999, 0.999, 0.999]]))
    assert pts[0] == {"mass": 0.8, "kp": 1.0, "n": 1, "mode": "a"}
    assert pts[1]["kp"] == pytest.approx(10.0) and pts[1]["mass"] == pytest.approx(1.0) and pts[1]["n"] == 3 and pts[1]["mode"] == "b"
    assert pts[2]["n"] == 4 and pts[2]["mode"] == "c" and 99 < pts[2]["kp"] < 100
    big = sp.sample(256, "sobol", seed=0)
    assert {p["n"] for p in big} == {1, 2, 3, 4} and {p["mode"] for p in big} == {"a", "b", "c"}
    assert np.allclose(sp.to_unit(pts[1]), [0.5, 0.5, 0.5, 1 / 3], atol=1e-9)  # the lower edge of the cell the value came from
    for method in ("lhs", "halton", "random"):
        assert len(sp.sample(8, method, seed=1)) == 8
    for bad in ({}, {"a": (1, 1)}, {"a": ("log", 0, 1)}, {"a": []}, {"a": "x"}):
        with pytest.raises(ValueError):
            ParameterSpace(bad)
    with pytest.raises(ValueError):
        sp.sample(4, "grid")
    with pytest.raises(ValueError):
        sp.points(np.zeros((2, 3)))


# ------------------------------------------------------------------------------------------------ Morris
def test_morris_trajectories_change_one_factor_per_step_by_the_grid_step():
    X = morris_design(5, r=12, levels=4, seed=1)
    assert X.shape == (12 * 6, 5) and X.min() >= -1e-12 and X.max() <= 1 + 1e-12
    delta = 4 / (2 * 3)
    for t in range(12):
        P = X[t * 6 : (t + 1) * 6]
        steps = np.abs(np.diff(P, axis=0))
        assert np.allclose(steps.max(axis=1), delta) and np.all((steps > 1e-12).sum(axis=1) == 1)
        assert np.all((np.abs(np.diff(P, axis=0)) > 1e-12).sum(axis=0) == 1)  # each factor moves exactly once per trajectory
    with pytest.raises(ValueError):
        morris_design(3, levels=5)


def test_morris_recovers_a_linear_model_exactly_and_flags_interactions():
    f = lambda x: 3.0 * x[0] + 1.0 * x[1] + 0.0 * x[2]
    res = morris(f, [(0, 1)] * 3, r=15, seed=0)
    assert np.allclose(res.mu_star, [3, 1, 0], atol=1e-12) and np.allclose(res.sigma, 0, atol=1e-12) and list(res.ranking()[:2]) == [0, 1]
    g = lambda x: x[0] * x[1] + 0.01 * x[2]
    r2 = morris(g, [(0, 1)] * 3, r=40, seed=1)
    assert r2.sigma[0] > 0.1 and r2.sigma[1] > 0.1 and r2.sigma[2] < 1e-9  # interaction: the effect of x0 depends on x1
    scaled = morris(f, [(0, 2), (0, 1), (0, 1)], r=10, seed=0)
    assert scaled.mu_star[0] == pytest.approx(6.0)  # per unit of the unit cube: a factor with twice the range has twice the effect
    vec = morris(lambda X: X @ np.array([3.0, 1.0, 0.0]), [(0, 1)] * 3, r=15, seed=0, vectorized=True)
    assert np.allclose(vec.mu_star, res.mu_star)
    with pytest.raises(ValueError, match="match"):
        morris_analyze(morris_design(3, 5), np.zeros(7), 5)


# ------------------------------------------------------------------------------------------------ Sobol
def ishigami(X, a=7.0, b=0.1):
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def test_sobol_indices_of_the_ishigami_function_match_the_analytic_values():
    res = sobol_indices(ishigami, [(-math.pi, math.pi)] * 3, n=8192, seed=0, vectorized=True)
    s1 = [0.3139, 0.4424, 0.0]
    st = [0.5576, 0.4424, 0.2437]
    assert np.allclose(res.s1, s1, atol=0.03) and np.allclose(res.st, st, atol=0.03)
    assert res.variance == pytest.approx(13.8446, rel=0.03)
    assert list(res.ranking()) == [0, 1, 2] and res.interactions()[0] > 0.2 and abs(res.interactions()[1]) < 0.03
    assert (
        np.all(res.s1_ci[:, 0] <= res.s1)
        and np.all(res.s1 <= res.s1_ci[:, 1])
        and np.all(res.st_ci[:, 0] <= res.st)
        and np.all(res.st <= res.st_ci[:, 1])
    )


def g_function(X, a):
    return np.prod((np.abs(4 * X - 2) + a) / (1 + a), axis=1)


def test_sobol_indices_of_the_g_function_match_the_analytic_values():
    a = np.array([0.0, 1.0, 4.5, 9.0, 99.0])
    Vi = 1.0 / (3 * (1 + a) ** 2)
    V = np.prod(1 + Vi) - 1
    s1 = Vi / V
    st = Vi * np.prod(1 + Vi) / (1 + Vi) / V
    res = sobol_indices(lambda X: g_function(X, a), [(0, 1)] * 5, n=8192, seed=1, vectorized=True)
    assert np.allclose(res.s1, s1, atol=0.03) and np.allclose(res.st, st, atol=0.03)
    assert list(res.ranking()) == [0, 1, 2, 3, 4]  # the smaller the coefficient a, the more important the factor
    assert res.s1.sum() < 1.0 and res.st.sum() > res.s1.sum()  # interactions make the total effects exceed the first-order ones


def test_an_additive_model_has_no_interactions_and_the_design_has_the_documented_size():
    f = lambda x: 2 * x[0] + x[1]
    res = sobol_indices(f, [(0, 1)] * 3, n=1024, seed=2)
    assert np.allclose(res.s1, [0.8, 0.2, 0.0], atol=0.03) and np.allclose(res.st, res.s1, atol=0.03) and abs(res.st[2]) < 0.02
    X = saltelli_design(3, 64, seed=0)
    assert X.shape == ((3 + 2) * 64, 3) and X.min() >= 0 and X.max() <= 1
    assert np.allclose(X[2 * 64 : 3 * 64][:, 1:], X[:64][:, 1:]) and np.allclose(
        X[2 * 64 : 3 * 64][:, 0], X[64:128][:, 0]
    )  # AB_0 = A with column 0 from B
    with pytest.raises(ValueError, match="zero variance"):
        sobol_indices(lambda X: np.ones(len(X)), [(0, 1)] * 2, n=64, vectorized=True)
    with pytest.raises(ValueError, match="match"):
        sobol_analyze(np.zeros(10), 3, 64)
    with pytest.raises(ValueError, match="non-finite"):
        sobol_indices(lambda X: np.full(len(X), np.nan), [(0, 1)] * 2, n=16, vectorized=True)
