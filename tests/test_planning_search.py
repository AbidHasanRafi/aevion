"""Planning space, A*, weighted A*, Theta*, D* Lite, RRT*, PRM - against brute force and closed forms."""

import math

import numpy as np
import pytest

from aevion.environments.world import World
from aevion.planning import PRM, DStarLite, PlanningSpace, astar, path_length, rrt_star, segment_free_batch, theta_star
from aevion.planning.graph import neighbour_offsets


def random_space(shape=(12, 12, 6), density=0.18, seed=0, res=1.0):
    rng = np.random.default_rng(seed)
    occ = rng.random(shape) < density
    return PlanningSpace(occ, (0.0, 0.0, 0.0), res)


def free_cell(space, rng):
    while True:
        c = tuple(int(rng.integers(0, s)) for s in space.shape)
        if not space.occ[c]:
            return c


def centre(space, c):
    return space.center(np.array(c))


def dijkstra_cost(space, a, b, connectivity=26):
    """Reference: shortest path on the same graph (same no-corner-cutting rule) with scipy."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    nx, ny, nz = space.shape
    occ = space.occ
    rows, cols, w = [], [], []
    offs = neighbour_offsets(connectivity)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for dx, dy, dz in offs:
                    ok = True
                    for ax in (0, 1):
                        for ay in (0, 1):
                            for az in (0, 1):
                                cx, cy, cz = x + (dx if ax else 0), y + (dy if ay else 0), z + (dz if az else 0)
                                if not (0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz) or occ[cx, cy, cz]:
                                    ok = False
                    if ok:
                        rows.append((x * ny + y) * nz + z)
                        cols.append(((x + dx) * ny + y + dy) * nz + z + dz)
                        w.append(space.res * math.sqrt(dx * dx + dy * dy + dz * dz))
    G = coo_matrix((w, (rows, cols)), shape=(nx * ny * nz,) * 2).tocsr()
    i = (a[0] * ny + a[1]) * nz + a[2]
    j = (b[0] * ny + b[1]) * nz + b[2]
    return float(dijkstra(G, indices=i)[j])


# ------------------------------------------------------------------------------------------------ space
def test_free_space_closed_forms_for_every_connectivity():
    s = PlanningSpace.empty((0, 0, 0, 20, 20, 20), 1.0)
    a, b = (0, 0, 0), (10, 6, 3)
    p, q = centre(s, a), centre(s, b)
    octile = 3 * math.sqrt(3) + 3 * math.sqrt(2) + 4  # 3 body diagonals, 3 face diagonals, 4 straight
    assert astar(s, p, q, 26).cost == pytest.approx(octile)
    assert astar(s, p, q, 6).cost == pytest.approx(10 + 6 + 3)  # Manhattan
    assert astar(s, p, q, 18).cost == pytest.approx(9 * math.sqrt(2) + 1)  # nine face diagonals pair up 18 of the 19 units of displacement


def test_the_segment_test_agrees_with_dense_sampling_of_the_voxels():
    s = random_space((30, 30, 10), 0.12, seed=1)
    rng = np.random.default_rng(2)
    A = rng.uniform(0.5, [29.5, 29.5, 9.5], (300, 3))
    Bq = rng.uniform(0.5, [29.5, 29.5, 9.5], (300, 3))
    fast = segment_free_batch(s, A, Bq)
    dense = [not s.occ[tuple(np.floor(a + np.linspace(0, 1, 800)[:, None] * (b - a)).astype(int).T)].any() for a, b in zip(A, Bq)]
    agree = np.mean(np.array(dense) == fast)  # a dense sample can step over a voxel corner that the exact walk touches
    assert agree > 0.99
    assert not s.segment_free([-1, 1, 1], [5, 5, 5]) and not s.segment_free([1, 1, 1], [5, 5, 50])  # leaving the grid is blocked


def test_from_world_inflates_obstacles_by_the_requested_radius():
    w = World()
    w.add_box((10, 10, 5), (4, 4, 10))
    s = PlanningSpace.from_world(w, bounds=(0, 0, 0, 20, 20, 12), resolution=0.5, inflation=1.0, ground=False)
    assert not s.is_free([[10.0, 10.0, 5.0]])[0] and not s.is_free([[10.0, 12.9, 5.0]])[0]  # inside / within 1 m of the face
    assert s.is_free([[10.0, 13.6, 5.0]])[0] and s.is_free([[2.0, 2.0, 5.0]])[0]
    assert s.clearance is not None and s.clearance.shape == s.shape
    cell = s.index([10.0, 13.1, 5.0])
    assert s.clearance[tuple(cell)] == pytest.approx(1.1, abs=0.3)


def test_nearest_free_and_sampling():
    s = random_space(seed=3)
    s.occ[5:8, 5:8, 2:4] = True
    p = s.nearest_free([6.5, 6.5, 3.0])
    assert p is not None and s.is_free(p)[0] and np.linalg.norm(p - [6.5, 6.5, 3.0]) < 3.0
    pts = s.sample_free(200, np.random.default_rng(0))
    assert s.is_free(pts).all() and pts.shape == (200, 3)
    with pytest.raises(ValueError):
        PlanningSpace(np.zeros((1, 5, 5), bool), (0, 0, 0), 1.0)


# ------------------------------------------------------------------------------------------------ A*
@pytest.mark.parametrize("seed", range(6))
def test_astar_is_optimal_against_dijkstra_on_random_grids(seed):
    s = random_space((10, 10, 5), 0.25, seed=seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(3):
        a, b = free_cell(s, rng), free_cell(s, rng)
        ref = dijkstra_cost(s, a, b)
        r = astar(s, centre(s, a), centre(s, b))
        if math.isinf(ref):
            assert not r.success and r.reason == "no path"
        else:
            assert r.success and r.cost == pytest.approx(ref, rel=1e-9)
            assert path_length(r.path) == pytest.approx(ref, rel=1e-9) and s.path_free(r.path)  # the polyline is itself collision-free


def test_weighted_astar_trades_optimality_for_effort():
    s = random_space((40, 40, 8), 0.2, seed=7)
    rng = np.random.default_rng(5)
    a, b = free_cell(s, rng), free_cell(s, rng)
    opt = astar(s, centre(s, a), centre(s, b))
    if not opt.success:
        pytest.skip("random instance is blocked")
    fast = astar(s, centre(s, a), centre(s, b), weight=3.0)
    assert fast.success and opt.cost <= fast.cost <= 3.0 * opt.cost * (1 + 1e-9)
    assert fast.expansions <= opt.expansions
    with pytest.raises(ValueError):
        astar(s, centre(s, a), centre(s, b), weight=0.5)


def test_diagonal_moves_may_not_cut_corners():
    occ = np.zeros((4, 4, 3), bool)
    occ[1, 0, :] = occ[0, 1, :] = True  # a diagonal 'pinch' next to the corner cell
    s = PlanningSpace(occ, (0, 0, 0), 1.0)
    r = astar(s, centre(s, (0, 0, 1)), centre(s, (1, 1, 1)))
    assert not r.success  # (0,0)->(1,1) would squeeze between two blocked cells
    r2 = astar(s, centre(s, (0, 0, 1)), centre(s, (1, 1, 2)))
    assert not r2.success


def test_start_or_goal_inside_an_obstacle_is_reported_or_snapped():
    s = random_space(seed=4)
    s.occ[6, 6, 3] = True
    inside = centre(s, (6, 6, 3))
    r = astar(s, centre(s, (0, 0, 0)) if not s.occ[0, 0, 0] else centre(s, (11, 11, 5)), inside)
    assert not r.success and "obstacle" in r.reason
    s.occ[0, 0, 0] = False
    snapped = astar(s, centre(s, (0, 0, 0)), inside, snap_to_free=True)
    assert snapped.success and "nearest free" in snapped.reason and s.is_free(snapped.path[-1])[0]
    out = astar(s, [-5, 0, 0], centre(s, (1, 1, 1)))
    assert not out.success and "outside" in out.reason
    assert not astar(s, [np.nan, 0, 0], centre(s, (1, 1, 1))).success


def test_unreachable_goals_and_expansion_limits():
    s = random_space((15, 15, 5), 0.0, seed=0)
    s.occ[7, :, :] = True  # a wall splits the space
    r = astar(s, centre(s, (1, 1, 1)), centre(s, (13, 13, 1)))
    assert not r.success and r.reason == "no path" and r.expansions > 100
    big = random_space((40, 40, 10), 0.0)
    r2 = astar(big, centre(big, (0, 0, 0)), centre(big, (39, 39, 9)), max_expansions=50)
    assert not r2.success and "limit" in r2.reason


def test_clearance_weighting_trades_length_for_distance_from_obstacles():
    w = World()
    w.add_box((15.0, 10.0, 5.0), (2.0, 12.0, 10.0))  # a wall with free passage on both ends
    s = PlanningSpace.from_world(w, bounds=(0, 0, 0, 30, 20, 10), resolution=0.5, inflation=0.5, ground=False)
    a, b = [2.0, 10.0, 5.0], [28.0, 10.0, 5.0]
    plain = astar(s, a, b)
    s.clearance_cost(safe_distance=3.0, weight=4.0)
    safe = astar(s, a, b)
    assert plain.success and safe.success
    cl = lambda p: float(s.clearance[tuple(s.index(p).T)].min())
    assert cl(safe.path) >= cl(plain.path) - 1e-9
    assert path_length(safe.path) >= path_length(plain.path) - 1e-9  # the safer route is not shorter


# ------------------------------------------------------------------------------------------------ Theta*
def test_theta_star_is_any_angle_shorter_than_the_grid_path_and_collision_free():
    s = random_space((40, 40, 8), 0.12, seed=11)
    rng = np.random.default_rng(3)
    shorter = 0
    for _ in range(8):
        a, b = free_cell(s, rng), free_cell(s, rng)
        g = astar(s, centre(s, a), centre(s, b))
        t = theta_star(s, centre(s, a), centre(s, b))
        assert g.success == t.success
        if g.success:
            assert s.path_free(t.path)
            assert path_length(t.path) <= path_length(g.path) + 1e-9
            shorter += path_length(t.path) < path_length(g.path) - 1e-6
            assert len(t.path) <= len(g.path)
    assert shorter >= 3
    e = PlanningSpace.empty((0, 0, 0, 20, 20, 20), 1.0)
    straight = theta_star(e, [0.5, 0.5, 0.5], [18.5, 7.5, 3.5])
    assert len(straight.path) == 2 and straight.cost == pytest.approx(np.linalg.norm([18, 7, 3]))
    with pytest.raises(ValueError):
        e.clearance = np.zeros(e.shape)
        e.clearance_cost(1.0)
        theta_star(e, [0.5, 0.5, 0.5], [5.5, 5.5, 5.5])


# ------------------------------------------------------------------------------------------------ D* Lite
def test_dstar_lite_matches_a_fresh_search_after_every_map_change():
    s = random_space((14, 14, 5), 0.1, seed=21)
    rng = np.random.default_rng(8)
    a, b = free_cell(s, rng), free_cell(s, rng)
    d = DStarLite(s, centre(s, a), centre(s, b))
    first = d.plan()
    ref = astar(PlanningSpace(s.occ.copy(), s.lo, s.res), centre(s, a), centre(s, b))
    assert first.success == ref.success
    if not ref.success:
        pytest.skip("random instance is blocked")
    assert first.cost == pytest.approx(ref.cost, rel=1e-9)
    base_expansions = first.expansions
    for _step in range(5):
        cells = [free_cell(s, rng) for _ in range(6)]
        cells = [c for c in cells if c not in (a, b)]
        d.update_cells(cells, occupied=True)
        again = d.plan()
        fresh = astar(PlanningSpace(s.occ.copy(), s.lo, s.res), centre(s, a), centre(s, b))
        assert again.success == fresh.success
        if fresh.success:
            assert again.cost == pytest.approx(fresh.cost, rel=1e-9)
            assert s.path_free(again.path)
        else:
            break
        assert again.expansions < base_expansions  # repairing is cheaper than starting over
    # removing obstacles can only shorten the path
    blocked = np.argwhere(s.occ)
    d.update_cells(blocked[:20], occupied=False)
    freed = d.plan()
    fresh = astar(PlanningSpace(s.occ.copy(), s.lo, s.res), centre(s, a), centre(s, b))
    assert freed.cost == pytest.approx(fresh.cost, rel=1e-9)


def test_dstar_lite_replans_from_a_moving_start():
    s = random_space((16, 16, 4), 0.1, seed=31)
    rng = np.random.default_rng(2)
    a, b = free_cell(s, rng), free_cell(s, rng)
    d = DStarLite(s, centre(s, a), centre(s, b))
    r = d.plan()
    if not r.success or len(r.path) < 5:
        pytest.skip("trivial instance")
    mid = r.path[len(r.path) // 2]
    d.move_start(mid)
    r2 = d.plan()
    ref = astar(PlanningSpace(s.occ.copy(), s.lo, s.res), mid, centre(s, b))
    assert r2.success and r2.cost == pytest.approx(ref.cost, rel=1e-9)


# ------------------------------------------------------------------------------------------------ sampling planners
def cluttered():
    s = PlanningSpace.empty((0, 0, 0, 30, 30, 12), 0.5)
    rng = np.random.default_rng(4)
    for _ in range(14):
        c = rng.uniform([6, 6, 0], [24, 24, 8])
        r = rng.uniform(1.0, 2.5)
        X, Y, Z = np.meshgrid(*[(np.arange(n) + 0.5) * 0.5 for n in s.shape], indexing="ij")
        s.occ |= (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < r * r
    return s


def test_rrt_star_converges_towards_the_optimum_in_free_space():
    s = PlanningSpace.empty((0, 0, 0, 30, 30, 12), 0.5)
    a, b = [2.0, 2.0, 2.0], [26.0, 24.0, 9.0]
    straight = float(np.linalg.norm(np.subtract(b, a)))
    r = rrt_star(s, a, b, max_iter=1500, step=4.0, seed=1)
    assert r.success and s.path_free(r.path)
    assert r.cost >= straight - 1e-9 and r.cost < 1.1 * straight
    assert np.allclose(r.path[0], a) and np.allclose(r.path[-1], b)


def test_rrt_star_improves_with_more_samples_and_stays_collision_free():
    s = cluttered()
    a, b = [1.0, 1.0, 2.0], [28.0, 28.0, 6.0]
    assert s.is_free(np.array([a, b])).all()
    first = rrt_star(s, a, b, max_iter=4000, step=3.0, seed=2, stop_at_first=True)
    best = rrt_star(s, a, b, max_iter=2500, step=3.0, seed=2)
    assert first.success and best.success and s.path_free(first.path) and s.path_free(best.path)
    ref = theta_star(s, a, b)
    assert best.cost <= first.cost + 1e-9 and best.cost >= ref.cost * 0.97  # never shorter than the any-angle reference (to discretisation)
    assert best.cost < 1.25 * ref.cost


def test_rrt_star_reports_failure_cleanly():
    s = PlanningSpace.empty((0, 0, 0, 20, 20, 10), 0.5)
    s.occ[18:22, :, :] = True  # a wall across the whole space (x = 9 .. 11 m)
    r = rrt_star(s, [1, 1, 1], [19.5, 10, 5], max_iter=200, step=2.0)
    assert not r.success and "no path" in r.reason
    near = rrt_star(s, [1, 1, 1], [4.0, 2.0, 1.5], max_iter=0)
    assert near.success and len(near.path) == 2  # within one step and visible: the direct connection needs no samples
    s.occ[0:3, 0:3, 0:3] = True
    assert not rrt_star(s, [1, 1, 1], [10, 10, 5]).success


def test_prm_answers_many_queries_from_one_roadmap_and_repairs_after_changes():
    s = cluttered()
    prm = PRM(s, seed=0).build(n_samples=1200, k=12)
    rng = np.random.default_rng(6)
    ratios = []
    for _ in range(6):
        a, b = s.sample_free(2, rng)
        res = prm.query(a, b)
        ref = theta_star(s, a, b)
        if not ref.success:
            continue
        assert res.success and s.path_free(res.path)
        ratios.append(res.cost / ref.cost)
    assert len(ratios) >= 4 and min(ratios) >= 0.95 and np.median(ratios) < 1.5
    before = len(prm.edges)
    s.occ[10:14, :, :] = True  # a new wall
    removed = prm.revalidate()
    assert removed > 0 and len(prm.edges) == before - removed
    a, b = np.array([1.0, 15.0, 2.0]), np.array([28.0, 15.0, 2.0])
    r = prm.query(a, b)
    assert not r.success or s.path_free(r.path)
    prm.add_samples(400)
    assert len(prm.nodes) == 1600


def test_prm_refuses_queries_in_obstacles_or_before_building():
    s = cluttered()
    prm = PRM(s)
    assert not prm.query([1, 1, 1], [5, 5, 5]).success and "empty" in prm.query([1, 1, 1], [5, 5, 5]).reason
    prm.build(200, 8)
    blocked = np.argwhere(s.occ)[0]
    assert not prm.query(s.center(blocked), [5, 5, 5]).success
