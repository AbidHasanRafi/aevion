"""Gaussian-process regression against the textbook formulas, and Bayesian optimisation on known test functions."""

import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from aevion.experiments.bayesopt import GaussianProcess, bayes_opt, expected_improvement, kernel, lower_confidence_bound, probability_of_improvement


# ------------------------------------------------------------------------------------------------ kernels
def test_kernels_have_the_closed_form_values_and_are_positive_semidefinite():
    r = 0.7
    x = np.array([[0.0], [r]])
    for kind, f in (
        ("rbf", lambda r: math.exp(-0.5 * r * r)),
        ("matern52", lambda r: (1 + math.sqrt(5) * r + 5 * r * r / 3) * math.exp(-math.sqrt(5) * r)),
        ("matern32", lambda r: (1 + math.sqrt(3) * r) * math.exp(-math.sqrt(3) * r)),
    ):
        K = kernel(x, x, 1.0, 2.0, kind)
        assert K[0, 0] == pytest.approx(2.0) and K[0, 1] == pytest.approx(2.0 * f(r)) and np.allclose(K, K.T)
        pts = np.random.default_rng(0).normal(size=(30, 3))
        assert np.linalg.eigvalsh(kernel(pts, pts, [0.5, 1.0, 2.0], 1.0, kind)).min() > -1e-9
    assert kernel(x, x, 2.0, 1.0, "rbf")[0, 1] == pytest.approx(math.exp(-0.5 * (r / 2.0) ** 2))  # a longer length scale correlates more
    with pytest.raises(ValueError):
        kernel(x, x, 1.0, 1.0, "periodic")
    with pytest.raises(ValueError):
        GaussianProcess("periodic")


# ------------------------------------------------------------------------------------------------ the GP
def test_log_marginal_likelihood_and_posterior_equal_the_textbook_formulas():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (12, 2))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1]
    ls, var, noise = np.array([0.4, 0.8]), 1.3, 1e-3
    gp = GaussianProcess("rbf", noise=noise, jitter=0.0)
    K = kernel(X, X, ls, var, "rbf") + noise * np.eye(12)
    assert gp.log_marginal_likelihood(X, y, ls, var, noise) == pytest.approx(multivariate_normal(mean=np.zeros(12), cov=K).logpdf(y), rel=1e-10)
    # posterior with fixed hyper-parameters, on standardised data
    gp.set_hyperparameters(ls, var, noise).fit(X, y, optimize_hypers=False)
    Xs = rng.uniform(0, 1, (5, 2))
    mean, std = gp.predict(Xs)
    z = (y - y.mean()) / y.std()
    Ks, Kss = kernel(Xs, X, ls, var, "rbf"), kernel(Xs, Xs, ls, var, "rbf")
    want_mean = Ks @ np.linalg.solve(K, z) * y.std() + y.mean()
    want_var = np.diag(Kss - Ks @ np.linalg.solve(K, Ks.T))
    assert np.allclose(mean, want_mean, atol=1e-8) and np.allclose(std, np.sqrt(want_var) * y.std(), atol=1e-6)


def test_noise_free_data_is_interpolated_and_far_away_the_prior_returns():
    X = np.linspace(0, 1, 8)[:, None]
    y = np.sin(6 * X[:, 0])
    gp = GaussianProcess("matern52", noise=1e-8).fit(X, y)
    m, s = gp.predict(X)
    assert np.allclose(m, y, atol=1e-3) and s.max() < 5e-2
    far_m, far_s = gp.predict(np.array([[40.0]]))
    assert far_m[0] == pytest.approx(y.mean(), abs=1e-3) and far_s[0] == pytest.approx(gp.variance**0.5 * y.std(), rel=1e-3)  # reverts to the prior
    mid_m, mid_s = gp.predict(np.array([[0.5 / 7 + 3 / 7]]))
    assert mid_s[0] > s.max()  # between observations the model is less certain than at them


def test_hyperparameter_fitting_finds_the_scale_of_the_data_and_the_noise_level():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 10, (40, 1))
    y_smooth = np.sin(X[:, 0] / 2.0)
    short = GaussianProcess("rbf").fit(X, np.sin(2.0 * X[:, 0]))
    long_ = GaussianProcess("rbf").fit(X, y_smooth)
    assert long_.lengthscale[0] > 2.5 * short.lengthscale[0]  # slowly varying data gets a longer length scale
    noisy = y_smooth + rng.normal(0, 0.3, 40)
    gp = GaussianProcess("rbf", noise=1e-2, fit_noise=True).fit(X, noisy)
    assert 0.04 < gp.noise < 0.4  # the noise variance on standardised outputs: about 0.09 / var(y) = 0.15
    X2 = rng.uniform(0, 1, (80, 2))
    y2 = np.sin(8 * X2[:, 0])  # depends on the first input only
    g2 = GaussianProcess("matern52").fit(X2, y2)
    assert g2.lengthscale[1] > 3 * g2.lengthscale[0]  # ARD: the irrelevant input gets a long length scale


def test_gp_input_validation():
    gp = GaussianProcess()
    with pytest.raises(RuntimeError, match="fit"):
        gp.predict([[0.0]])
    with pytest.raises(ValueError):
        gp.fit([[0.0]], [1.0])
    with pytest.raises(ValueError, match="non-finite"):
        gp.fit([[0.0], [1.0]], [1.0, np.nan])
    with pytest.raises(ValueError):
        gp.fit([[0.0], [1.0]], [1.0])


# ------------------------------------------------------------------------------------------------ acquisition functions
def test_acquisition_functions_have_their_defining_properties():
    mean, std = np.array([0.0, 1.0, 2.0, 0.0]), np.array([1.0, 1.0, 1.0, 0.0])
    ei = expected_improvement(mean, std, best=0.5)
    assert np.all(ei >= 0) and ei[0] > ei[1] > ei[2]  # lower predicted mean = more improvement
    z = (0.5 - 0.0) / 1.0
    assert ei[0] == pytest.approx(0.5 * norm.cdf(z) + norm.pdf(z))  # the closed form
    assert ei[3] == pytest.approx(0.5)  # no uncertainty, mean below the incumbent: the improvement is deterministic
    assert expected_improvement(np.array([2.0]), np.array([0.0]), 0.5)[0] == 0.0  # no uncertainty, mean above: nothing to gain
    wide = expected_improvement(np.array([1.0, 1.0]), np.array([0.1, 2.0]), best=0.5)
    assert wide[1] > wide[0]  # at the same mean, more uncertainty = more expected improvement
    assert np.allclose(lower_confidence_bound(mean, std, 2.0), -(mean - 2 * std))
    pi = probability_of_improvement(mean, std, 0.5)
    assert pi[0] == pytest.approx(norm.cdf(0.5)) and pi[3] == 1.0 and 0 <= pi[2] <= pi[1] <= pi[0] <= 1


# ------------------------------------------------------------------------------------------------ the optimiser
def branin(x):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi**2), 5 / math.pi, 6.0, 10.0, 1 / (8 * math.pi)
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * math.cos(x[0]) + s


BRANIN = [(-5.0, 10.0), (0.0, 15.0)]
BRANIN_MIN = 0.397887


def test_bayesian_optimisation_finds_the_branin_minimum_with_a_small_budget_and_beats_random_search():
    bo = [bayes_opt(branin, BRANIN, n_init=8, n_iter=22, seed=s).y_best for s in range(3)]
    rng = np.random.default_rng(0)
    rand = [min(branin(rng.uniform([-5, 0], [10, 15])) for _ in range(30)) for _ in range(3)]
    assert np.median(bo) < BRANIN_MIN + 0.3 and np.median(bo) < np.median(rand)
    res = bayes_opt(branin, BRANIN, n_init=8, n_iter=22, seed=0)
    assert len(res.y) == 30 and res.best_so_far[-1] == res.y_best and np.all(np.diff(res.best_so_far) <= 1e-12)  # the running best never gets worse
    assert res.y_best == pytest.approx(branin(res.x_best)) and np.all(res.X >= [-5, 0]) and np.all(res.X <= [10, 15])
    again = bayes_opt(branin, BRANIN, n_init=8, n_iter=22, seed=0)
    assert np.array_equal(res.y, again.y)  # seeded


def test_maximisation_the_other_acquisitions_batches_and_a_start_point():
    f = lambda x: -((x[0] - 0.3) ** 2) - 2 * (x[1] + 0.2) ** 2
    for acq in ("ei", "lcb", "pi"):
        r = bayes_opt(f, [(-1, 1), (-1, 1)], n_init=6, n_iter=14, seed=1, acquisition=acq, maximize=True)
        assert r.y_best > -0.02 and np.allclose(r.x_best, [0.3, -0.2], atol=0.15), acq
    b = bayes_opt(f, [(-1, 1), (-1, 1)], n_init=6, n_iter=5, seed=1, maximize=True, batch=3)
    assert len(b.y) == 6 + 5 * 3 and len({tuple(x) for x in np.round(b.X, 9)}) == len(b.X)  # distinct points
    calls = []
    s = bayes_opt(
        f,
        [(-1, 1), (-1, 1)],
        n_init=3,
        n_iter=2,
        seed=0,
        maximize=True,
        x0=np.array([0.3, -0.2]),
        evaluate_batch=lambda X: (calls.append(len(X)), [f(x) for x in X])[1],
    )
    assert s.y_best == pytest.approx(0.0, abs=1e-12) and calls[0] == 3 and calls[1:] == [1, 1]
    seen = []
    bayes_opt(f, [(-1, 1)] * 2, n_init=3, n_iter=2, seed=0, maximize=True, callback=lambda n, x, y: seen.append(n))
    assert seen == [4, 5]


def test_failed_evaluations_do_not_stop_the_search():
    def flaky(x):
        return float("nan") if x[0] > 6 else branin(x)

    res = bayes_opt(flaky, BRANIN, n_init=10, n_iter=15, seed=2)
    assert np.isfinite(res.y).all() and res.failures >= 1 and res.y_best < 5.0
    with pytest.raises(RuntimeError, match="every initial"):
        bayes_opt(lambda x: float("inf"), BRANIN, n_init=4, n_iter=1)


def test_optimiser_input_validation():
    for bad in ([(1, 0)], [(0, 1, 2)], []):
        with pytest.raises(ValueError):
            bayes_opt(lambda x: 0.0, bad)
    with pytest.raises(ValueError, match="acquisition"):
        bayes_opt(lambda x: 0.0, [(0, 1)], acquisition="thompson")
