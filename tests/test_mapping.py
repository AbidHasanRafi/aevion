"""Log-odds voxel mapping - exact ray traversal, Bayes updates, clipping, mapping a real obstacle, distance fields."""

import numpy as np
import pytest

from aevion.environments.world import World
from aevion.estimation.mapping import OccupancyGrid
from aevion.sensors.raycast import cast_rays


def grid(res=0.5, **kw):
    return OccupancyGrid((0, 0, 0, 20, 20, 10), res, **kw)


def test_an_axis_aligned_ray_marks_free_space_and_an_obstacle_at_its_end():
    g = grid()
    n = g.integrate([[0.25, 5.25, 5.25]], [[10.25, 5.25, 5.25]], [True])
    row = g.logodds[:, 10, 10]
    assert n == 21 and np.all(row[:20] == pytest.approx(-0.4)) and row[20] == pytest.approx(0.85)
    assert g.occupied().sum() == 1 and g.occupied([[10.3, 5.3, 5.3]])[0] and not g.occupied([[5.3, 5.3, 5.3]])[0]
    assert g.coverage() == pytest.approx(21 / g.logodds.size)


def test_the_traversal_visits_exactly_the_voxels_a_dense_sampling_of_the_ray_touches():
    rng = np.random.default_rng(0)
    g = grid(0.5)
    for _ in range(40):
        o = rng.uniform(1, 19, 3) * np.array([1, 1, 0.5])
        e = rng.uniform(1, 19, 3) * np.array([1, 1, 0.5])
        g.clear()
        g.integrate([o], [e], [False])
        visited = {tuple(i) for i in np.argwhere(g.logodds != 0)}
        s = np.linspace(0, 1, 4000)[:, None]
        pts = o + s * (e - o)
        sampled = {tuple(i) for i in np.floor(pts / 0.5).astype(int)}
        assert sampled <= visited  # nothing the ray crosses is missed
        assert len(visited - sampled) <= 2  # and only corner-grazing voxels are added


def test_every_voxel_receives_one_update_per_ray_and_exactly_one_is_a_hit():
    g = grid(0.25)
    o, e = np.array([1.0, 2.0, 1.0]), np.array([17.3, 14.1, 8.2])
    n = g.integrate([o], [e], [True])
    idx = np.argwhere(g.logodds != 0)
    assert n == len(idx)  # no voxel is visited twice by one ray
    assert np.isclose(g.logodds, 0.85).sum() == 1 and np.isclose(g.logodds, -0.4).sum() == n - 1
    assert g.occupied([e])[0]  # the end voxel is the one that contains the end point


def test_evidence_accumulates_saturates_and_can_be_reversed():
    g = grid(l_min=-2.0, l_max=3.5)
    for _ in range(30):
        g.integrate([[0.25, 0.25, 0.25]], [[2.25, 0.25, 0.25]], [True])
    assert g.logodds[4, 0, 0] == pytest.approx(3.5) and g.probability([[2.3, 0.3, 0.3]])[0] == pytest.approx(0.9707, abs=1e-3)
    n_back = 0
    while g.occupied([[2.3, 0.3, 0.3]])[0] and n_back < 100:  # the same voxel is later seen *through* (the obstacle moved away)
        g.integrate([[0.25, 0.25, 0.25]], [[5.25, 0.25, 0.25]], [True])
        n_back += 1
    assert n_back == int(np.ceil((3.5 - g.thr) / 0.4)) and g.logodds[4, 0, 0] < g.thr


def test_a_miss_marks_the_whole_ray_free_including_its_last_voxel():
    g = grid()
    g.integrate([[0.25, 0.25, 0.25]], [[4.25, 0.25, 0.25]], [False])
    assert g.logodds[:, 0, 0][:9].tolist() == pytest.approx([-0.4] * 9) and g.occupied().sum() == 0


def test_rays_are_clipped_to_the_grid_and_an_outside_end_is_never_an_obstacle():
    g = grid()
    g.integrate([[-5.0, 5.25, 5.25]], [[10.25, 5.25, 5.25]], [True])  # starts outside, ends inside: a hit
    assert g.logodds[20, 10, 10] == pytest.approx(0.85) and g.logodds[0, 10, 10] == pytest.approx(-0.4)
    g.clear()
    g.integrate([[10.25, 5.25, 5.25]], [[40.0, 5.25, 5.25]], [True])  # ends outside: the 'hit' is not in the map
    assert g.occupied().sum() == 0 and g.logodds[39, 10, 10] == pytest.approx(-0.4)
    g.clear()
    assert g.integrate([[-9.0, -9.0, 5.0]], [[-1.0, -1.0, 5.0]], [True]) == 0 and not g.known().any()  # entirely outside
    assert g.integrate([[1.0, 1.0, 1.0]], [[np.nan, 1.0, 1.0]], [True]) == 0  # non-finite input is skipped, not propagated


def test_a_ray_grazing_a_grid_face_and_degenerate_rays_do_not_crash_or_loop():
    g = grid()
    g.integrate([[0.0, 0.0, 0.0]], [[20.0, 0.0, 0.0]], [True])  # along an edge of the box
    g.integrate([[5.0, 5.0, 5.0]], [[5.0, 5.0, 5.0]], [True])  # zero length
    g.integrate([[5.0, 5.0, 5.0]], [[5.0, 5.0, 9.99]], [True])  # vertical
    assert np.all(np.isfinite(g.logodds))


def test_mapping_a_box_from_several_viewpoints_recovers_its_surface():
    w = World()
    w.add_box((10.1, 10.1, 2.6), (4.0, 6.0, 5.2))  # x 8.1..12.1, y 7.1..13.1, z 0..5.2 (off the voxel boundaries on purpose)
    g = grid(0.5)
    rng = np.random.default_rng(1)
    hit_points = []
    for origin in ([1, 10, 4], [10, 1, 4], [19, 10, 4], [10, 19, 4], [3, 3, 8], [17, 17, 8]):
        o = np.asarray(origin, float)
        d = rng.normal(size=(3000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        hit = cast_rays(w, o, d, 40.0)
        t, valid = np.asarray(hit.t, float), np.asarray(hit.prim != -1)
        g.insert_ranges(o, d, np.where(valid, t, 0.0), valid, max_range=40.0)
        on_box = valid & (np.asarray(hit.prim) == 0)  # primitive 0 is the box (the ground is a plane, not a primitive)
        hit_points.append((o + d * t[:, None])[on_box])
    pts = np.concatenate(hit_points)
    assert len(pts) > 400
    vox = np.unique(np.floor((pts - g.lo) / g.res).astype(int), axis=0)
    recall = g.occupied()[vox[:, 0], vox[:, 1], vox[:, 2]].mean()  # voxels that really contain a measured surface point
    assert recall > 0.85
    c = g.centers()
    above = c[c[:, 2] > 0.8]  # ignore the ground, which the rays also hit
    near_box = (np.abs(above[:, 0] - 10.1) <= 2.0 + 0.5) & (np.abs(above[:, 1] - 10.1) <= 3.0 + 0.5) & (above[:, 2] <= 5.2 + 0.5)
    assert near_box.mean() > 0.97  # nothing is hallucinated away from the box (one voxel of discretisation slack)
    assert 0.0 < g.coverage() < 1.0 and g.entropy() < g.logodds.size  # entropy falls below the all-unknown 1 bit per voxel


def test_the_distance_field_of_the_map_matches_the_analytic_distance_to_the_box():
    g = OccupancyGrid((0, 0, 0, 20, 20, 10), 0.5)
    occ = np.zeros(g.shape, bool)
    occ[16:24, 14:26, 0:10] = True  # the box x 8..12, y 7..13, z 0..5
    g.logodds[occ] = 3.0
    sdf = g.to_sdf()
    pts = np.array([[4.0, 10.0, 2.0], [10.0, 17.0, 2.0], [15.0, 15.0, 2.0], [10.0, 10.0, 2.5]])
    want = np.array([4.0, 4.0, np.hypot(3.0, 2.0), -2.0])  # the last is inside: negative, 2 m to the nearest face
    got = sdf.distance(pts)
    assert np.all(np.abs(got - want) <= 0.5 + 1e-9)  # within one voxel
    cons = g.to_sdf(treat_unknown_as_free=False)
    assert cons.distance(np.array([[4.0, 10.0, 2.0]]))[0] < 1.0  # unknown space counts as an obstacle when asked


def test_ranges_at_or_beyond_the_maximum_range_are_misses():
    g = grid()
    d = np.array([[1.0, 0.0, 0.0]])
    g.insert_ranges([0.25, 5.25, 5.25], d, [20.0], [True], max_range=5.0)
    assert g.occupied().sum() == 0 and g.logodds[5, 10, 10] == pytest.approx(-0.4)
    g.insert_ranges([0.25, 5.25, 5.25], d, [3.0], [True], max_range=5.0)
    assert g.occupied().sum() == 0 or g.occupied([[3.3, 5.3, 5.3]])[0]
    g.clear()
    g.insert_points([0.25, 5.25, 5.25], [[6.3, 5.3, 5.3]])
    assert g.occupied([[6.3, 5.3, 5.3]])[0]


def test_invalid_grids_are_refused():
    for bad in ((0, 0, 0, -1, 1, 1), (0, 0, 0, 1, 1), (0, 0, 0, np.inf, 1, 1)):
        with pytest.raises(ValueError):
            OccupancyGrid(bad, 0.1)
    with pytest.raises(ValueError, match="resolution"):
        OccupancyGrid((0, 0, 0, 1, 1, 1), 0.0)
    with pytest.raises(ValueError, match="voxels"):
        OccupancyGrid((0, 0, 0, 1000, 1000, 1000), 0.1)
    with pytest.raises(ValueError, match="l_hit"):
        OccupancyGrid((0, 0, 0, 1, 1, 1), 0.1, l_hit=-1.0)
