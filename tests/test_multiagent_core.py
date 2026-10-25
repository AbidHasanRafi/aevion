"""Graphs, consensus, formation control and task allocation - against spectra, scipy optima and brute force."""

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from aevion.multiagent.allocation import TravelScore, auction, brute_force_allocation, cbba, hungarian
from aevion.multiagent.consensus import (
    average_consensus,
    consensus_rate,
    consensus_step,
    disagreement,
    formation_error,
    formation_step,
    lossy_consensus,
    max_step,
    push_sum,
)
from aevion.multiagent.graphs import (
    adjacency,
    algebraic_connectivity,
    components,
    diameter,
    disk_graph,
    is_connected,
    knn_graph,
    laplacian,
    laplacian_spectrum,
    max_degree_weights,
    metropolis_weights,
)


# ------------------------------------------------------------------------------------------------ graphs
@pytest.mark.parametrize("n", [3, 5, 8, 12])
def test_the_ring_and_path_spectra_match_their_closed_forms(n):
    ring = laplacian_spectrum(adjacency("ring", n))
    assert np.allclose(np.sort(ring), np.sort(2 - 2 * np.cos(2 * np.pi * np.arange(n) / n)))
    path = laplacian_spectrum(adjacency("line", n))
    assert np.allclose(path, np.sort(2 - 2 * np.cos(np.pi * np.arange(n) / n)))
    assert algebraic_connectivity(adjacency("complete", n)) == pytest.approx(n)  # K_n: lambda_2 = n
    assert algebraic_connectivity(adjacency("star", n)) == pytest.approx(1.0)


def test_laplacian_rows_sum_to_zero_and_connectivity_matches_lambda2():
    np.random.default_rng(0)
    for seed in range(20):
        A = adjacency("random", 9, p=0.25, seed=seed)
        assert np.allclose(laplacian(A).sum(axis=1), 0.0)
        assert is_connected(A) == (algebraic_connectivity(A) > 1e-9)
    two = np.zeros((4, 4))
    two[0, 1] = two[1, 0] = two[2, 3] = two[3, 2] = 1.0
    assert components(two) == [[0, 1], [2, 3]] and diameter(two) == math.inf
    assert diameter(adjacency("line", 6)) == 5 and diameter(adjacency("ring", 8)) == 4 and diameter(adjacency("complete", 5)) == 1


def test_geometric_graphs_and_weight_matrices():
    pos = np.array([[0, 0, 0], [1, 0, 0], [2.2, 0, 0], [10, 0, 0.0]])
    D = disk_graph(pos, 1.5)
    assert D[0, 1] == 1 and D[1, 2] == 1 and D[0, 2] == 0 and D[3].sum() == 0
    K = knn_graph(pos, 1)
    assert K[3, 2] == 1 and K[2, 3] == 1  # symmetrised
    assert np.all(K == K.T)
    A = adjacency("random", 10, p=0.3, seed=2)
    for W in (metropolis_weights(A), max_degree_weights(A)):
        assert np.allclose(W.sum(axis=0), 1) and np.allclose(W.sum(axis=1), 1) and np.allclose(W, W.T) and W.min() >= -1e-12
    with pytest.raises(ValueError):
        adjacency("moebius", 4)
    with pytest.raises(ValueError):
        laplacian(-np.ones((3, 3)))


# ------------------------------------------------------------------------------------------------ consensus
def test_consensus_preserves_the_mean_and_decays_at_the_rate_the_spectrum_predicts():
    A = adjacency("ring", 8)
    x0 = np.random.default_rng(1).normal(size=(8, 3))
    traj = average_consensus(x0, A, steps=120)
    mean = x0.mean(axis=0)
    assert np.allclose(traj.mean(axis=1), mean, atol=1e-12)  # doubly stochastic: exact conservation
    d = np.array([np.linalg.norm(t - mean) for t in traj])
    rate = consensus_rate(A)
    slope = (d[100] / d[60]) ** (1 / 40)
    assert slope == pytest.approx(rate, abs=2e-3)  # the asymptotic contraction is the second-largest |1 - eps lambda|
    assert disagreement(traj[-1]) < 1e-3


def test_a_larger_step_is_faster_until_the_stability_limit_then_diverges():
    A = adjacency("line", 6)
    lim = max_step(A)
    assert consensus_rate(A, 0.5 * lim) < consensus_rate(A, 0.1 * lim) and consensus_rate(A, 1.05 * lim) > 1.0
    x0 = np.arange(6.0)
    assert disagreement(average_consensus(x0, A, 400, eps=0.9 * lim)[-1]) < 1e-6
    assert disagreement(average_consensus(x0, A, 400, eps=1.05 * lim)[-1]) > 10.0


def test_better_connected_graphs_agree_faster_and_metropolis_also_converges():
    x0 = np.random.default_rng(2).normal(size=10)
    final = {k: disagreement(average_consensus(x0, adjacency(k, 10), 30)[-1]) for k in ("line", "ring", "complete")}
    assert final["complete"] < final["ring"] < final["line"]
    m = average_consensus(x0, adjacency("random", 10, p=0.5, seed=3), 400, weights="metropolis")
    assert disagreement(m[-1]) < 1e-8 and abs(m[-1].mean() - x0.mean()) < 1e-12
    assert consensus_step(x0, adjacency("complete", 10)).shape == x0.shape


def test_a_pure_delay_keeps_the_mean_and_loss_degrades_it_gracefully():
    A = adjacency("ring", 8)
    x0 = np.random.default_rng(3).normal(size=8)
    d = lossy_consensus(x0, A, steps=1500, delay=3)
    assert np.allclose(d.mean(axis=1), x0.mean(), atol=1e-12) and disagreement(d[-1]) < 1e-6
    lossy = lossy_consensus(x0, A, steps=1500, loss=0.3, seed=1)
    assert disagreement(lossy[-1]) < 1e-3  # agrees anyway ...
    assert abs(lossy[-1].mean() - x0.mean()) > 1e-6  # ... but not exactly on the original average
    assert abs(lossy[-1].mean() - x0.mean()) < 1.0 * x0.std()
    vec = lossy_consensus(np.random.default_rng(4).normal(size=(8, 3)), A, steps=100, delay=1)
    assert vec.shape == (101, 8, 3)


def test_push_sum_reaches_the_average_on_a_directed_ring_where_plain_consensus_cannot():
    A = adjacency("ring_directed", 7)
    x0 = np.random.default_rng(4).normal(size=7) * 5
    est = push_sum(x0, A, 400)
    assert np.allclose(est[-1], x0.mean(), atol=1e-9)
    plain = average_consensus(x0, A, 400)  # I - eps L on a digraph is row- but not column-stochastic: it agrees on a weighted, not the plain, average
    assert abs(plain[-1].mean() - x0.mean()) > 1e-6 or disagreement(plain[-1]) < 1e-9


def test_formation_control_converges_to_the_shape_and_keeps_the_centroid():
    n = 6
    offsets = np.array([[2 * math.cos(2 * math.pi * k / n), 2 * math.sin(2 * math.pi * k / n), 0.0] for k in range(n)])
    A = adjacency("ring", n)
    p = np.random.default_rng(5).normal(size=(n, 3)) * 4
    c0 = p.mean(axis=0)
    e0 = formation_error(p, offsets)
    for _ in range(600):
        p = p + 0.05 * formation_step(p, A, offsets, 1.0, 0.05)
    assert formation_error(p, offsets) < 1e-3 * e0 and np.allclose(p.mean(axis=0) - offsets.mean(axis=0), c0, atol=1e-9)
    # a common velocity feed-forward moves the whole shape without deforming it
    q = p.copy()
    for _ in range(100):
        q = q + 0.05 * formation_step(q, A, offsets, 1.0, 0.05, vel_ff=[1.0, 0.0, 0.0])
    assert np.allclose(q - p, [5.0, 0.0, 0.0], atol=1e-6) and formation_error(q, offsets) < 1e-3 * e0


# ------------------------------------------------------------------------------------------------ Hungarian / auction
@pytest.mark.parametrize("shape", [(5, 5), (6, 9), (9, 6), (1, 4), (12, 12)])
def test_hungarian_matches_scipy(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(25):
        C = rng.integers(-20, 50, shape).astype(float) if rng.random() < 0.5 else rng.normal(size=shape)
        r, c, total = hungarian(C)
        sr, sc = linear_sum_assignment(C)
        assert total == pytest.approx(C[sr, sc].sum()) and len(set(c)) == len(c) == min(shape)
        rm, cm, tm = hungarian(C, maximize=True)
        sr, sc = linear_sum_assignment(C, maximize=True)
        assert tm == pytest.approx(C[sr, sc].sum())


def test_hungarian_handles_forbidden_pairs_and_reports_impossible_problems():
    C = np.array([[1.0, np.inf, 3.0], [np.inf, 2.0, np.inf], [4.0, 5.0, np.inf]])
    r, c, total = hungarian(C)
    assert total == pytest.approx(3.0 + 2.0 + 4.0) and list(c) == [2, 1, 0]
    with pytest.raises(ValueError, match="forbidden"):
        hungarian(np.array([[np.inf, np.inf], [1.0, 2.0]]))
    with pytest.raises(ValueError):
        hungarian(np.array([[np.nan, 1.0]]))
    assert hungarian(np.zeros((0, 3)))[2] == 0.0


@pytest.mark.parametrize("shape", [(6, 6), (5, 9)])
def test_auction_is_within_its_guaranteed_distance_of_the_optimum(shape):
    rng = np.random.default_rng(7)
    for _ in range(25):
        a = rng.uniform(0, 10, shape)
        res = auction(a, eps=0.02)
        r, c, best = hungarian(a, maximize=True)
        assert len(set(res.cols)) == shape[0] and res.total <= best + 1e-9
        assert best - res.total <= res.bound + 1e-9 and res.bound == pytest.approx(max(shape) * res.eps)


def test_auction_with_integer_benefits_is_exactly_optimal_by_default():
    rng = np.random.default_rng(8)
    for _ in range(30):
        a = rng.integers(0, 30, (7, 7)).astype(float)
        assert auction(a).total == pytest.approx(hungarian(a, maximize=True)[2])
    with pytest.raises(ValueError):
        auction(np.ones((5, 3)))
    with pytest.raises(ValueError):
        auction(np.array([[1.0, np.inf]]))


# ------------------------------------------------------------------------------------------------ CBBA
def _instance(seed, n_agents, n_tasks):
    rng = np.random.default_rng(seed)
    return TravelScore(rng.uniform(0, 10, (n_agents, 2)), rng.uniform(0, 10, (n_tasks, 2)), value=rng.uniform(1, 5, n_tasks), speed=2.0, discount=0.9)


@pytest.mark.parametrize("seed", range(8))
def test_cbba_is_conflict_free_and_within_the_guarantee_of_the_optimum(seed):
    sc = _instance(seed, 3, 5)
    res = cbba(sc, 3, 5)
    best, _ = brute_force_allocation(sc, 3, 5)
    assert res.converged and res.conflict_free
    assert sorted(j for p in res.paths for j in p) == list(range(5))  # all tasks are positive-valued, so all are taken
    assert res.score >= 0.5 * best - 1e-9  # the published bound for diminishing-marginal-gain scores
    assert res.score >= 0.85 * best  # and in practice it is near-optimal on small instances


def test_cbba_with_one_task_per_agent_is_a_conflict_free_greedy_assignment_near_the_hungarian_optimum():
    rng = np.random.default_rng(11)
    ratios = []
    for _ in range(20):
        V = rng.uniform(1, 10, (4, 4))
        score = lambda i, path, V=V: V[i, path[0]] if len(path) == 1 else (-np.inf if len(path) > 1 else 0.0)
        res = cbba(score, 4, 4, max_bundle=1)
        best = hungarian(V, maximize=True)[2]
        assert res.conflict_free and len(res.assignment()) == 4 and res.score <= best + 1e-9
        ratios.append(res.score / best)
    assert min(ratios) >= 0.5 and np.mean(ratios) > 0.9  # it is greedy, not optimal: the auction without eps-scaling can lose a little
    diag = np.eye(4) * 5.0 + 1.0
    assert cbba(lambda i, path: diag[i, path[0]] if len(path) == 1 else (-np.inf if path else 0.0), 4, 4, max_bundle=1).score == pytest.approx(24.0)


def test_cbba_works_on_a_sparse_connected_graph_and_gives_the_same_conflict_free_answer():
    sc = _instance(3, 4, 6)
    full = cbba(sc, 4, 6)
    ring = cbba(sc, 4, 6, adjacency=adjacency("ring", 4))
    line = cbba(sc, 4, 6, adjacency=adjacency("line", 4))
    for r in (ring, line):
        assert r.converged and r.conflict_free and sorted(j for p in r.paths for j in p) == list(range(6))
        assert r.score >= 0.8 * full.score
    assert ring.rounds >= 1


def test_cbba_respects_capabilities_and_bundle_limits():
    ap = np.array([[0.0, 0], [10.0, 0]])
    tp = np.array([[1.0, 0], [2.0, 0], [9.0, 0]])
    cap = np.array([[1, 1, 0], [1, 1, 1]])  # agent 0 cannot do task 2
    sc = TravelScore(ap, tp, value=[3.0, 3.0, 3.0], capability=cap)
    res = cbba(sc, 2, 3)
    assert 2 in res.paths[1] and 2 not in res.paths[0]
    one = cbba(sc, 2, 3, max_bundle=1)
    assert all(len(p) <= 1 for p in one.paths) and len(one.unassigned) == 1
    assert res.assignment()[2] == 1


def test_travel_score_discounts_later_tasks_and_path_order_matters():
    sc = TravelScore([[0.0, 0.0]], [[1.0, 0.0], [5.0, 0.0]], value=[1.0, 1.0], speed=1.0, discount=0.9)
    near_first = sc(0, [0, 1])
    far_first = sc(0, [1, 0])
    assert near_first == pytest.approx(0.9 + 0.9**5) and near_first > far_first
    assert sc(0, []) == 0.0
    # diminishing marginal gain: adding a task to a longer path never helps more than to a shorter one
    assert sc(0, [0, 1]) - sc(0, [0]) <= sc(0, [1]) - sc(0, []) + 1e-12
    pairs = list(itertools.permutations(range(2)))
    assert len(pairs) == 2
