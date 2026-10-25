"""Voronoi coverage and UTM-style deconfliction - against closed forms and exact geometry."""

import itertools
import math

import numpy as np
import pytest

from aevion.multiagent.coverage import CoverageDomain, cell_masses, centroids, coverage_cost, lloyd, lloyd_step, voronoi_assign
from aevion.multiagent.utm import SpaceTimePlanner, Trajectory4D, closest_approach, detect_conflicts, predict_conflicts, schedule_departures
from aevion.planning import PlanningSpace


# ------------------------------------------------------------------------------------------------ coverage
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_lloyd_on_a_uniform_segment_finds_the_analytic_optimum(n):
    dom = CoverageDomain([0.0, 1.0], 1.0 / 4000)
    rng = np.random.default_rng(n)
    res = lloyd(np.sort(rng.uniform(0, 1, (n, 1)), axis=0), dom, iterations=3000, tol=1e-10)
    opt = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    assert np.allclose(np.sort(res.sites[:, 0]), opt, atol=2e-3)
    assert coverage_cost(res.sites, dom) == pytest.approx(1.0 / (12 * n * n), rel=5e-3)


def test_the_locational_cost_never_increases_and_one_site_converges_to_the_centroid():
    dom = CoverageDomain([0, 0, 4, 2], 0.05)
    assert dom.mass == pytest.approx(8.0) and dom.dim == 2
    res = lloyd(np.random.default_rng(1).uniform(0, 2, (6, 2)), dom, iterations=60)
    assert np.all(np.diff(res.cost) <= 1e-12) and res.cost[-1] < 0.6 * res.cost[0]
    one = lloyd([[0.1, 0.1]], dom, iterations=5)
    assert np.allclose(one.sites[0], [2.0, 1.0], atol=1e-9)  # the centroid of the rectangle
    assert coverage_cost(one.sites, dom) == pytest.approx((4.0**2 + 2.0**2) / 12 * 8.0, rel=1e-3)  # second moment about the centre


def test_cells_partition_the_mass_and_equal_masses_emerge_for_uniform_density_on_a_symmetric_layout():
    dom = CoverageDomain([0, 0, 1, 1], 0.01)
    sites = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    m = cell_masses(sites, dom)
    assert m.sum() == pytest.approx(1.0) and np.allclose(m, 0.25, atol=1e-9)
    assert np.allclose(centroids(sites, dom), sites, atol=1e-9)  # already a centroidal Voronoi tessellation
    assert np.array_equal(voronoi_assign(sites, [[0.1, 0.1], [0.9, 0.9]]), [0, 3])


def test_density_pulls_agents_towards_where_the_mass_is_and_masks_exclude_regions():
    peak = lambda q: 0.05 + np.exp(-(((q[:, 0] - 0.8) ** 2 + (q[:, 1] - 0.5) ** 2) / (2 * 0.1**2)))
    dom = CoverageDomain([0, 0, 1, 1], 0.02, density=peak)
    res = lloyd(np.random.default_rng(2).uniform(0, 1, (5, 2)), dom, iterations=200)
    uniform = lloyd(res.history[0], CoverageDomain([0, 0, 1, 1], 0.02), iterations=200)
    near = lambda r: (np.linalg.norm(r.sites - [0.8, 0.5], axis=1) < 0.25).sum()
    assert near(res) > near(uniform)  # more agents serve the hot spot than under a uniform density
    assert np.linalg.norm(res.sites - [0.8, 0.5], axis=1).min() < np.linalg.norm(uniform.sites - [0.8, 0.5], axis=1).min()
    ring = CoverageDomain([0, 0, 1, 1], 0.02, mask=lambda q: np.linalg.norm(q - 0.5, axis=1) > 0.2)
    assert ring.mass < 1.0 - math.pi * 0.2**2 + 0.02 and np.all(np.linalg.norm(ring.points - 0.5, axis=1) > 0.2)


def test_partial_steps_and_move_limits_converge_the_same_way_with_slower_motion():
    dom = CoverageDomain([0, 0, 1, 1], 0.05)
    s0 = np.random.default_rng(3).uniform(0, 1, (4, 2))
    full = lloyd_step(s0, dom, 1.0)
    half = lloyd_step(s0, dom, 0.5)
    assert np.allclose(half - s0, 0.5 * (full - s0))
    capped = lloyd_step(s0, dom, 1.0, max_move=0.01)
    assert np.linalg.norm(capped - s0, axis=1).max() <= 0.01 + 1e-12
    assert lloyd(s0, dom, 400, gain=0.3, max_move=0.05).cost[-1] == pytest.approx(lloyd(s0, dom, 400).cost[-1], rel=0.15)


def test_coverage_input_validation():
    with pytest.raises(ValueError):
        CoverageDomain([0, 0, 0], 0.1)
    with pytest.raises(ValueError):
        CoverageDomain([0, 1], 0.0)
    with pytest.raises(ValueError):
        CoverageDomain([0, 1], 0.1, density=lambda q: -np.ones(len(q)))
    with pytest.raises(ValueError, match="no mass"):
        CoverageDomain([0, 1], 0.1, mask=lambda q: np.zeros(len(q), bool))


# ------------------------------------------------------------------------------------------------ conflicts
def test_the_closest_approach_of_two_crossing_straight_flights_is_exact():
    a = Trajectory4D([0.0, 10.0], [[0, 0, 0], [10, 0, 0]])
    b = Trajectory4D([0.0, 10.0], [[5, 5, 1], [5, -5, 1]])
    d, t = closest_approach(a, b)
    assert d == pytest.approx(1.0) and t == pytest.approx(5.0)  # both reach x = y = 0 height-separated by 1 m at t = 5
    offset = Trajectory4D([0.0, 10.0], [[5, 5, 1], [5, -5, 1]]).shifted(2.0)
    d2, t2 = closest_approach(a, offset)
    # offset arrives at the crossing at t = 7 while a is at x = 7; relative geometry is analytic
    rel = lambda t: np.linalg.norm(a.position(t) - offset.position(t))
    ts = np.linspace(2, 10, 200001)
    assert d2 == pytest.approx(min(rel(x) for x in ts[::50]), abs=1e-3)
    assert closest_approach(a, Trajectory4D([20.0, 30.0], [[0, 0, 0], [1, 0, 0]])) == (math.inf, pytest.approx(math.nan, nan_ok=True))


def test_conflict_intervals_are_the_roots_of_the_quadratic():
    # head-on at relative speed 2 m/s along x; separation 3 m; they meet at t = 5
    a = Trajectory4D([0.0, 10.0], [[0, 0, 0], [10, 0, 0]])
    b = Trajectory4D([0.0, 10.0], [[10, 0, 0], [0, 0, 0]])
    (c,) = detect_conflicts([a, b], 3.0)
    assert (c.i, c.j) == (0, 1)
    # the gap closes at 2 m/s: 10 m at t=0, 3 m at t=3.5, 0 at t=5, 3 m again at t=6.5
    assert (
        c.t_start == pytest.approx(3.5)
        and c.t_end == pytest.approx(6.5)
        and c.d_min == pytest.approx(0.0, abs=1e-9)
        and c.t_min == pytest.approx(5.0)
    )
    assert detect_conflicts([a, b], 0.0) == []
    far = Trajectory4D([0.0, 10.0], [[0, 20, 0], [10, 20, 0]])
    assert detect_conflicts([a, far], 5.0) == []


def test_brief_encounters_between_breakpoints_are_not_missed_and_presence_windows_are_respected():
    a = Trajectory4D([0.0, 100.0], [[0, 0, 0], [1000, 0, 0]])
    b = Trajectory4D([0.0, 100.0], [[500, 30, 0], [500, -30, 0]])  # crosses a's line at t = 50 when a is at x = 500
    (c,) = detect_conflicts([a, b], 5.0)
    assert c.t_min == pytest.approx(50.0) and c.d_min == pytest.approx(0.0, abs=1e-9)
    gone = Trajectory4D([60.0, 70.0], [[500, 0, 0], [510, 0, 0]])  # appears after a has passed
    assert detect_conflicts([a, gone], 5.0) == []
    parked = Trajectory4D([0.0, 10.0], [[500, 0, 0], [500, 0, 0.5]], hold_after=200.0)
    assert detect_conflicts([a, parked], 5.0)[0].t_min == pytest.approx(50.0, abs=0.1)  # the parked vehicle still occupies the point
    with pytest.raises(ValueError):
        Trajectory4D([1.0, 1.0], [[0, 0, 0], [1, 0, 0]])


def test_constant_velocity_prediction_orders_threats_by_time():
    pos = np.array([[0.0, 0, 0], [100.0, 0, 0], [0.0, 60.0, 0], [500.0, 500, 0]])
    vel = np.array([[10.0, 0, 0], [-10.0, 0, 0], [0.0, -6.0, 0], [0.0, 0, 0]])
    c = predict_conflicts(pos, vel, 20.0, horizon=30.0)
    assert len(c) >= 2
    first = c[0]
    assert (first["i"], first["j"]) == (0, 1) and first["t_cpa"] == pytest.approx(5.0) and first["d_cpa"] == pytest.approx(0.0, abs=1e-9)
    assert first["t_violation"] == pytest.approx(4.0)  # 100 m closing at 20 m/s reaches 20 m after 4 s
    assert all(x["t_violation"] <= y["t_violation"] for x, y in itertools.pairwise(c))
    assert predict_conflicts(pos[[3]], vel[[3]], 20.0) == []


def test_strategic_deconfliction_by_ground_delay():
    a = Trajectory4D([0.0, 10.0], [[0, 0, 0], [100, 0, 0]])
    b = Trajectory4D([0.0, 10.0], [[100, 0, 0], [0, 0, 0]])
    c = Trajectory4D([0.0, 10.0], [[0, 50, 0], [100, 50, 0]])
    delays, ok = schedule_departures([a, b, c], separation=10.0, max_delay=60.0, step=1.0)
    assert ok and delays[0] == 0.0 and delays[1] > 0.0  # the second head-on flight is held on the ground
    shifted = [r.shifted(d) for r, d in zip([a, b, c], delays)]
    assert detect_conflicts(shifted, 10.0) == []
    _, ok2 = schedule_departures([a, b], separation=10.0, max_delay=3.0, step=1.0)
    assert not ok2  # not enough time to resolve within the allowed delay
    d_rev, _ = schedule_departures([a, b], separation=10.0, order=[1, 0])
    assert d_rev[1] == 0.0 and d_rev[0] > 0.0  # priority decides who waits


# ------------------------------------------------------------------------------------------------ space-time planner
def space(shape=(14, 14, 3), res=1.0, blocks=()):
    occ = np.zeros(shape, bool)
    for b in blocks:
        occ[b] = True
    return PlanningSpace(occ, (0.0, 0.0, 0.0), res)


def test_two_vehicles_swap_places_in_a_corridor_wide_enough_to_pass():
    sp = space((12, 5, 3))
    plan = SpaceTimePlanner(sp, dt=1.0, separation=1.5, connectivity=26).plan(
        [[0.5, 2.5, 1.5], [10.5, 2.5, 1.5]], [[10.5, 2.5, 1.5], [0.5, 2.5, 1.5]]
    )
    assert plan.success and plan.conflicts(1.5) == []
    for t, goal in zip(plan.trajectories, ([10.5, 2.5, 1.5], [0.5, 2.5, 1.5])):
        assert np.allclose(t.p[-1], goal) and t.max_speed() <= math.sqrt(3) + 1e-9
    assert plan.makespan() >= 10  # nobody can beat the straight-line travel time


def test_a_vehicle_waits_for_the_other_in_a_single_lane_and_the_result_is_conflict_free():
    blocks = [(slice(None), slice(0, 2), slice(None)), (slice(None), slice(3, 5), slice(None))]  # one free row in y (z has two layers)
    sp = space((10, 5, 2), blocks=blocks)
    # two vehicles with crossing goals on a 1-cell-wide lane: the second must wait or fail; both orders are tried
    planner = SpaceTimePlanner(sp, dt=1.0, separation=1.0, connectivity=6)
    plan = planner.plan([[0.5, 2.5, 0.5], [3.5, 2.5, 0.5]], [[9.5, 2.5, 0.5], [8.5, 2.5, 0.5]])
    assert plan.success and plan.conflicts(1.0) == []
    swap = SpaceTimePlanner(sp, dt=1.0, separation=1.5, connectivity=6).plan(
        [[0.5, 2.5, 0.5], [9.5, 2.5, 0.5]], [[9.5, 2.5, 0.5], [0.5, 2.5, 0.5]], retries=1
    )
    assert not swap.success and "no conflict-free path" in " ".join(
        swap.reasons.values()
    )  # genuinely impossible: the two z layers are only 1 m apart, less than the 1.5 m separation
    assert swap.expansions > 0


def test_obstacles_are_avoided_and_arrived_vehicles_keep_blocking_their_goal():
    sp = space((14, 14, 3), blocks=[(slice(5, 9), slice(0, 10), slice(None))])  # a wall with a gap at y >= 10
    planner = SpaceTimePlanner(sp, dt=1.0, separation=1.2)
    starts = [[1.5, 3.5, 1.5], [12.5, 3.5, 1.5], [1.5, 7.5, 1.5]]
    goals = [[12.5, 3.5, 1.5], [1.5, 3.5, 1.5], [12.5, 8.5, 1.5]]
    plan = planner.plan(starts, goals)
    assert plan.success and plan.conflicts(1.2) == []
    for t in plan.trajectories:
        pts = np.vstack([t.position(x) for x in np.linspace(t.start, t.t[-1], 60)])
        assert sp.is_free(pts).all()
        assert all(sp.segment_free(a, b) for a, b in itertools.pairwise(pts))
    assert all(np.allclose(t.p[-1], g) for t, g in zip(plan.trajectories, goals))


def test_the_planner_reports_bad_inputs_instead_of_failing_silently():
    sp = space((8, 8, 2), blocks=[(3, 3, slice(None))])
    planner = SpaceTimePlanner(sp, dt=1.0, separation=1.0)
    bad_goal = planner.plan([[0.5, 0.5, 0.5]], [[3.5, 3.5, 0.5]])
    assert not bad_goal.success and "blocked" in bad_goal.reasons[0]
    close = planner.plan([[0.5, 0.5, 0.5], [0.9, 0.5, 0.5]], [[5.5, 5.5, 0.5], [6.5, 5.5, 0.5]], retries=0)
    assert not close.success and "separation" in " ".join(close.reasons.values())
    with pytest.raises(ValueError):
        planner.plan([[0, 0, 0]], [[1, 1, 1], [2, 2, 2]])
    with pytest.raises(ValueError):
        SpaceTimePlanner(sp, dt=0.0)
    single = planner.plan([[0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5]])
    assert single.success and single.makespan() == 0.0
