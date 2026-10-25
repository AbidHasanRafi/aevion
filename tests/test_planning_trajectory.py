"""Path clean-up, time-optimal speed profiles, minimum-snap trajectories, corridor repair."""

import math

import numpy as np
import pytest

from aevion.control.reference import Waypoints
from aevion.planning import (
    ArcCurve,
    MinSnap,
    PlanningSpace,
    plan_path,
    plan_trajectory,
    prune_path,
    resample_path,
    shortcut_path,
    theta_star,
    time_optimal_profile,
)
from aevion.planning.graph import path_length
from aevion.planning.trajectory import trajectory_limits


# ------------------------------------------------------------------------------------------------ minimum snap
def test_one_segment_rest_to_rest_is_the_classic_seventh_order_polynomial():
    D, T = 6.0, 2.5
    snap = MinSnap([[0, 0, 0], [D, 0, 0]], [T])
    tau = np.linspace(0, 1, 50)
    want = D * (35 * tau**4 - 84 * tau**5 + 70 * tau**6 - 20 * tau**7)  # the unique polynomial with zero v, a, j at both ends
    got = snap.sample_many(tau * T)[:, 0]
    assert np.allclose(got, want, atol=1e-9)
    for o in (1, 2, 3):
        assert np.allclose(snap.sample(0.0, o), 0.0, atol=1e-9) and np.allclose(snap.sample(T, o), 0.0, atol=1e-8)


def test_waypoints_are_hit_and_derivatives_are_continuous_through_jerk():
    P = np.array([[0, 0, 1.0], [3, 1, 2.0], [5, 4, 2.5], [9, 4, 1.0]])
    T = np.array([2.0, 1.5, 2.5])
    snap = MinSnap(P, T)
    for k, t in enumerate(snap.knots):
        assert np.allclose(snap.sample(t), P[k], atol=1e-8)
    for t in snap.knots[1:-1]:
        for o in (1, 2, 3):  # derivatives agree from the left and the right of every interior knot
            assert np.allclose(snap.sample(t - 1e-7, o), snap.sample(t + 1e-7, o), atol=1e-4)
    assert np.allclose(snap.sample(0.0, 1), 0.0, atol=1e-9) and np.allclose(snap.sample(snap.duration, 2), 0.0, atol=1e-8)


def test_free_boundary_derivatives_are_honoured():
    snap = MinSnap([[0, 0, 0], [4, 0, 0]], [3.0], start_derivs=[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]], end_derivs=[[0.5, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.allclose(snap.sample(0.0, 1), [1.0, 0, 0]) and np.allclose(snap.sample(3.0, 1), [0.5, 0, 0], atol=1e-8)


def _snap_integral(f, T, n=4000):
    ts = np.linspace(0.0, T, n)
    return float(np.trapezoid(np.sum(np.array([f(t) for t in ts]) ** 2, axis=1), ts))


def test_minimum_snap_has_lower_snap_than_the_other_smooth_interpolant():
    """The C^6 interpolant used by Waypoints is feasible for the same constraints; minimum snap must not exceed its snap energy."""
    P = np.array([[0, 0, 0], [2, 3, 1], [6, 3, 2], [8, 0, 0]], float)
    T = np.array([1.5, 2.0, 1.5])
    snap = MinSnap(P, T)
    other = Waypoints(P, times=T)

    def snap_of_waypoints(t):
        s = min(int(np.searchsorted(other.knots, t, side="right") - 1), len(T) - 1)
        return other._basis(t - other.knots[s], 4) @ other.coef[s]

    e_min = _snap_integral(lambda t: snap.sample(t, 4) if False else _fd4(snap, t), snap.duration)
    e_other = _snap_integral(snap_of_waypoints, other.knots[-1])
    assert e_min <= e_other * (1 + 1e-6)
    assert snap.cost == pytest.approx(e_min, rel=2e-3)  # the closed-form cost equals the numerical integral


def _fd4(snap, t, h=1e-3):
    """Fourth derivative by the exact polynomial basis of the segment containing t."""
    s = min(int(np.searchsorted(snap.knots, t, side="right") - 1), len(snap.T) - 1)
    tau = (t - snap.knots[s]) / snap.T[s]
    from aevion.planning.trajectory import _basis

    return (_basis(tau, 4) @ snap.c[s]) / snap.T[s] ** 4


def test_time_scaling_scales_speed_and_acceleration_exactly():
    P = np.array([[0, 0, 0], [3, 2, 1], [7, 2, 1]], float)
    T = np.array([1.2, 1.6])
    a, b = MinSnap(P, T), MinSnap(P, 2.0 * T)
    va, aa = trajectory_limits(a)
    vb, ab = trajectory_limits(b)
    assert vb == pytest.approx(va / 2.0, rel=2e-3) and ab == pytest.approx(aa / 4.0, rel=2e-3)  # same shape, stretched in time
    ts = np.linspace(0, a.duration, 20)
    assert np.allclose(a.sample_many(ts), b.sample_many(2 * ts), atol=1e-8)


def test_minsnap_input_validation():
    with pytest.raises(ValueError):
        MinSnap([[0, 0, 0]], [])
    with pytest.raises(ValueError):
        MinSnap([[0, 0, 0], [1, 0, 0]], [-1.0])
    with pytest.raises(ValueError):
        MinSnap([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [1.0])


# ------------------------------------------------------------------------------------------------ time-optimal speed profile
def test_straight_line_profile_is_the_trapezoid_of_bang_bang_control():
    curve = ArcCurve([[0, 0, 0], [20, 0, 0]])
    prof = time_optimal_profile(curve, v_max=4.0, a_max=2.0)
    assert prof.duration == pytest.approx(20 / 4.0 + 4.0 / 2.0, rel=5e-3)  # accelerate 2 s, cruise, decelerate 2 s
    assert prof.v.max() == pytest.approx(4.0, abs=1e-3) and prof.v[0] == 0.0 and prof.v[-1] < 0.05
    short = time_optimal_profile(ArcCurve([[0, 0, 0], [2, 0, 0]]), v_max=10.0, a_max=2.0)  # never reaches v_max: a triangle
    assert short.duration == pytest.approx(2 * math.sqrt(2.0 / 2.0), rel=5e-3) and short.v.max() == pytest.approx(2.0, abs=0.02)


def test_corner_speed_is_limited_by_the_curvature_and_total_acceleration_never_exceeds_the_bound():
    R, a_max, v_max = 5.0, 2.0, 6.0
    th = np.linspace(0, math.pi, 60)
    pts = np.stack([R * np.cos(th), R * np.sin(th), np.zeros_like(th)], axis=1)
    curve = ArcCurve(pts)
    prof = time_optimal_profile(curve, v_max, a_max)
    mid = (prof.s > 0.3 * curve.length) & (prof.s < 0.7 * curve.length)
    assert np.allclose(prof.v[mid], math.sqrt(a_max * R), rtol=0.03)  # κ v² = a_max on the arc: v = sqrt(a R) = 3.16 m/s
    _, _, pss = curve.sample(prof.s)
    kappa = np.linalg.norm(pss, axis=1)
    a_t = np.gradient(prof.v**2, prof.s) / 2.0
    a_n = kappa * prof.v**2
    assert np.all(np.hypot(a_t, a_n) <= a_max * 1.03)  # total acceleration respects the bound everywhere (numerical derivative slack)
    assert np.isclose(curve.length, math.pi * R, rtol=2e-3)


def test_profile_is_never_slower_than_a_scaled_alternative():
    """Optimality check: no smaller duration exists for a v_max / a_max pair that is only slightly relaxed in the wrong direction."""
    curve = ArcCurve([[0, 0, 0], [5, 0, 0], [5, 5, 0], [10, 5, 0]])
    a = time_optimal_profile(curve, 3.0, 2.0)
    b = time_optimal_profile(curve, 3.0, 2.2)
    c = time_optimal_profile(curve, 3.3, 2.0)
    assert b.duration < a.duration and c.duration <= a.duration + 1e-9  # more authority never costs time
    with pytest.raises(ValueError):
        time_optimal_profile(curve, 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ path clean-up
def _cluttered(seed=4):
    s = PlanningSpace.empty((0, 0, 0, 30, 30, 12), 0.5)
    rng = np.random.default_rng(seed)
    X, Y, Z = np.meshgrid(*[(np.arange(n) + 0.5) * 0.5 for n in s.shape], indexing="ij")
    for _ in range(16):
        c = rng.uniform([5, 5, 0], [25, 25, 9])
        r = rng.uniform(1.0, 2.5)
        s.occ |= (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < r * r
    return s


def test_pruning_and_shortcutting_keep_paths_free_and_never_lengthen_them():
    s = _cluttered()
    a, b = [1.0, 1.0, 2.0], [28.0, 28.0, 6.0]
    grid = plan_path(s, a, b, "astar")
    assert grid.success
    pr = prune_path(s, grid.path)
    sc = shortcut_path(s, grid.path, 300, seed=1)
    for p in (pr, sc):
        assert s.path_free(p) and np.allclose(p[0], a) and np.allclose(p[-1], b)
        assert path_length(p) <= path_length(grid.path) + 1e-9 and len(p) <= len(grid.path)
    assert len(pr) < len(grid.path) / 3
    assert path_length(sc) <= path_length(pr) + 1e-9


def test_resample_path_spacing_and_endpoints():
    p = resample_path([[0, 0, 0], [10, 0, 0], [10, 10, 0]], 1.5)
    assert np.allclose(p[0], [0, 0, 0]) and np.allclose(p[-1], [10, 10, 0])
    assert np.all(np.linalg.norm(np.diff(p, axis=0), axis=1) <= 1.5 + 1e-9)


# ------------------------------------------------------------------------------------------------ planned trajectories
def test_a_planned_trajectory_is_collision_free_and_meets_its_limits_tightly():
    s = _cluttered()
    a, b = [1.0, 1.0, 2.0], [28.0, 28.0, 6.0]
    path = theta_star(s, a, b).path
    tr = plan_trajectory(s, path, v_max=3.0, a_max=2.5)
    pts = tr.path(0.02)
    assert s.is_free(pts).all()
    vp, ap = tr.info["peak_speed"], tr.info["peak_acceleration"]
    assert max(vp / 3.0, math.sqrt(ap / 2.5)) == pytest.approx(1.0, abs=0.02)  # limits are met, and one of them is binding
    assert vp <= 3.0 * 1.02 and ap <= 2.5 * 1.02
    p0, v0 = tr.sample(0.0)[:2]
    p1, v1 = tr.sample(tr.duration)[:2]
    assert np.allclose(p0, a, atol=1e-6) and np.allclose(p1, b, atol=1e-6) and np.allclose(v0, 0.0) and np.allclose(v1, 0.0, atol=1e-6)
    assert np.allclose(tr.sample(tr.duration + 5.0)[0], b, atol=1e-6)  # holds the goal after the end
    loose = plan_trajectory(s, path, v_max=3.0, a_max=2.5, tighten=False)
    assert tr.duration <= loose.duration + 1e-9  # tightening never slows it


def test_corridor_repair_pulls_the_smooth_curve_back_into_a_narrow_passage():
    s = PlanningSpace.empty((0, 0, 0, 30, 20, 6), 0.25)
    s.occ[:, :, :] = True
    s.occ[:, 36:44, :] = False  # an E-W corridor, 2 m wide, y in 9..11
    s.occ[80:88, 36:76, :] = False  # then a N-S branch (x in 20..22) opening to the north
    s.occ[80:116, 68:76, :] = False  # and an E-W branch at the end
    a, b = [1.0, 10.0, 3.0], [27.0, 18.0, 3.0]
    path = theta_star(s, a, b).path
    assert len(path) >= 3
    tr = plan_trajectory(s, path, v_max=4.0, a_max=4.0, margin=0.0)
    assert s.is_free(tr.path(0.02)).all()
    assert tr.info["repairs"] >= 1 or len(tr.waypoints) > len(path)  # the smooth curve cut a corner at least once and was repaired
    with pytest.raises(RuntimeError, match="free space"):
        plan_trajectory(s, path, v_max=4.0, a_max=4.0, max_repairs=0)


def test_clearance_margin_is_respected_when_the_space_knows_its_clearances():
    from aevion.environments.world import World

    w = World()
    w.add_box((15.0, 10.0, 5.0), (2.0, 12.0, 10.0))
    s = PlanningSpace.from_world(w, bounds=(0, 0, 0, 30, 20, 10), resolution=0.5, inflation=0.5, ground=False)
    a, b = [2.0, 10.0, 5.0], [28.0, 10.0, 5.0]
    path = theta_star(s, a, b).path
    tr = plan_trajectory(s, path, v_max=3.0, a_max=3.0, margin=0.5)
    idx = s.index(tr.path(0.02))
    assert s.clearance[idx[:, 0], idx[:, 1], idx[:, 2]].min() >= 0.5 - 1e-9


def test_plan_path_dispatches_by_name_and_rejects_unknown_planners():
    s = PlanningSpace.empty((0, 0, 0, 20, 20, 10), 0.5)
    for m in ("astar", "theta_star", "Theta*", "rrt_star", "prm"):
        kw = dict(max_iter=300, step=4.0) if m == "rrt_star" else dict(n_samples=200, k=8) if m == "prm" else {}
        r = plan_path(s, [1, 1, 1], [18, 18, 8], m, seed=1, **kw)
        assert r.success and s.path_free(r.path), m
    with pytest.raises(ValueError, match="unknown planner"):
        plan_path(s, [1, 1, 1], [5, 5, 5], "dijkstra_of_doom")


def test_yaw_follow_points_along_the_motion():
    tr = plan_trajectory(None, [[0, 0, 2], [10, 10, 2]], v_max=3.0, a_max=2.0, yaw_follow=True)
    assert tr.sample(tr.duration / 2)[4] == pytest.approx(math.pi / 4, abs=1e-6)
