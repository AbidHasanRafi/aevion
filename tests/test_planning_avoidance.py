"""The QP tools, ORCA-3D and control-barrier-function filters - against solvers, geometry and simulation."""

import numpy as np
import pytest

from aevion.environments.sdf import SignedDistanceField, sphere_sdf
from aevion.planning.avoidance import ORCA3D, CBFSafetyFilter, orca_half_space
from aevion.planning.qp import project, solve_qp


# ------------------------------------------------------------------------------------------------ projection
def _scipy_projection(x0, A, b, radius=None):
    from scipy.optimize import minimize

    cons = [{"type": "ineq", "fun": lambda x, a=a, bb=bb: a @ x - bb} for a, bb in zip(A, b)]
    if radius is not None:
        cons.append({"type": "ineq", "fun": lambda x: radius**2 - x @ x})
    return minimize(
        lambda x: 0.5 * np.sum((x - x0) ** 2), x0, jac=lambda x: x - x0, constraints=cons, method="SLSQP", options=dict(ftol=1e-14, maxiter=500)
    ).x


def test_projection_onto_one_half_space_is_the_orthogonal_foot_point():
    r = project([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0]], [2.0])
    assert np.allclose(r.x, [2.0, 0.0, 0.0]) and r.feasible
    r2 = project([5.0, 1.0, 1.0], [[1.0, 0.0, 0.0]], [2.0])  # already satisfied: unchanged
    assert np.allclose(r2.x, [5.0, 1.0, 1.0])


@pytest.mark.parametrize("seed", range(5))
def test_projection_matches_a_generic_solver_on_random_polytopes_and_balls(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(6, 3))
    x_ok = rng.normal(size=3) * 0.3
    b = A @ x_ok - rng.uniform(0.05, 0.5, 6)  # x_ok is strictly feasible, so the set is non-empty
    x0 = rng.normal(size=3) * 3
    got = project(x0, A, b)
    assert got.feasible and np.all(A @ got.x >= b - 1e-7)
    assert np.allclose(got.x, _scipy_projection(x0, A, b), atol=1e-5)
    with_ball = project(x0, A, b, np.zeros(3), 2.5)
    if np.linalg.norm(x_ok) < 2.5:  # the ball and the polytope intersect: the projection is exact
        assert with_ball.feasible and np.linalg.norm(with_ball.x) <= 2.5 + 1e-6 and np.all(A @ with_ball.x >= b - 1e-5)
        assert np.allclose(with_ball.x, _scipy_projection(x0, A, b, 2.5), atol=1e-4)


def test_inconsistent_constraints_are_reported_as_infeasible():
    r = project([0, 0, 0], [[1.0, 0, 0], [-1.0, 0, 0]], [1.0, 1.0])  # x >= 1 and x <= -1
    assert not r.feasible and r.violation > 0.5
    r2 = project([0, 0, 0], [[1.0, 0, 0]], [5.0], np.zeros(3), 1.0)  # a ball that does not reach the half-space
    assert not r2.feasible
    assert np.allclose(project([3.0, 4.0, 0.0], None, None, np.zeros(3), 1.0).x, [0.6, 0.8, 0.0])  # a pure ball projection
    assert project([1, 2, 3], np.zeros((1, 3)), [1.0]).feasible  # a zero row is ignored


# ------------------------------------------------------------------------------------------------ ADMM QP
def test_qp_unconstrained_and_box_constrained_closed_forms():
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, -2.0])
    free = solve_qp(P, q)
    assert free.status == "solved" and np.allclose(free.x, -np.linalg.solve(P, q), atol=1e-4)
    box = solve_qp(P, q, np.eye(2), [-0.2, -0.2], [0.2, 0.2])
    assert box.status == "solved" and np.all(np.abs(box.x) <= 0.2 + 1e-5) and abs(box.x[1] - 0.2) < 1e-4  # the pull towards +y is active


@pytest.mark.parametrize("seed", range(4))
def test_qp_matches_slsqp_with_mixed_equality_and_inequality_rows(seed):
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    n = 8
    M = rng.normal(size=(n, n))
    P = M @ M.T + 0.1 * np.eye(n)
    q = rng.normal(size=n)
    Ain = rng.normal(size=(5, n))
    Aeq = rng.normal(size=(2, n))
    x_feas = rng.normal(size=n) * 0.3
    u_in = Ain @ x_feas + rng.uniform(0.1, 1.0, 5)
    beq = Aeq @ x_feas
    A = np.vstack([Ain, Aeq])
    l = np.concatenate([np.full(5, -np.inf), beq])
    u = np.concatenate([u_in, beq])
    res = solve_qp(P, q, A, l, u, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000)
    ref = minimize(
        lambda x: 0.5 * x @ P @ x + q @ x,
        x_feas,
        jac=lambda x: P @ x + q,
        constraints=[{"type": "ineq", "fun": lambda x: u_in - Ain @ x}, {"type": "eq", "fun": lambda x: Aeq @ x - beq}],
        method="SLSQP",
        options=dict(ftol=1e-14, maxiter=500),
    )
    assert res.status == "solved"
    assert np.allclose(Aeq @ res.x, beq, atol=1e-4) and np.all(Ain @ res.x <= u_in + 1e-4)
    assert res.objective == pytest.approx(ref.fun, rel=1e-4, abs=1e-4)


def test_qp_warm_start_converges_faster_and_bad_bounds_are_refused():
    rng = np.random.default_rng(1)
    n = 30
    M = rng.normal(size=(n, n))
    P = M @ M.T + np.eye(n)
    q = rng.normal(size=n)
    A = np.vstack([np.eye(n)])
    cold = solve_qp(P, q, A, np.full(n, -0.1), np.full(n, 0.1))
    warm = solve_qp(P, q * 1.01, A, np.full(n, -0.1), np.full(n, 0.1), x0=cold.x, y0=cold.y)
    assert cold.status == warm.status == "solved" and warm.iterations <= cold.iterations
    with pytest.raises(ValueError):
        solve_qp(P, q, A, np.full(n, 1.0), np.full(n, 0.0))


# ------------------------------------------------------------------------------------------------ ORCA geometry
def _collides(p, v, R, tau):
    """Brute force: does a relative motion p - v t come within R during [0, tau]?"""
    t = np.linspace(0.0, tau, 4001)
    return bool(np.min(np.linalg.norm(p[None, :] - v[None, :] * t[:, None], axis=1)) < R - 1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_orca_half_space_lands_on_the_boundary_of_the_velocity_obstacle(seed):
    rng = np.random.default_rng(seed)
    R, tau = 1.0, 3.0
    p = rng.normal(size=3)
    p *= rng.uniform(2.5, 8.0) / np.linalg.norm(p)
    toward = p / np.linalg.norm(p)
    v = toward * rng.uniform(0.1, 3.0) + rng.normal(size=3) * 0.3
    if not _collides(p, v, R, tau):
        v = toward * 2.5  # make sure the sample starts inside the VO
    assert _collides(p, v, R, tau)
    u, n = orca_half_space(p, v, R, tau, 0.05)
    assert np.linalg.norm(n) == pytest.approx(1.0) and u @ n > 0.0  # a colliding relative velocity must be pushed outwards
    edge = v + u
    eps = 1e-3
    assert not _collides(p, edge + eps * n, R, tau)  # just outside the boundary: safe
    assert _collides(p, edge - eps * n, R, tau)  # just inside: still colliding


def test_orca_half_space_leaves_a_safe_relative_velocity_alone_but_gives_it_room():
    p, v = np.array([6.0, 0.0, 0.0]), np.array([-0.5, 3.0, 0.0])  # sliding past: clearly outside the VO
    assert not _collides(p, v, 1.0, 3.0)
    u, n = orca_half_space(p, v, 1.0, 3.0, 0.05)
    assert u @ n < 0.0  # u points *towards* the boundary from outside; the allowed half-space still contains the current velocity
    assert (v - (v + 0.5 * u)) @ n > 0.0
    over = orca_half_space(np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0, 3.0, 0.05)  # already overlapping: separate
    assert over[1] @ np.array([1.0, 0, 0]) < 0 or np.linalg.norm(over[0]) > 0


# ------------------------------------------------------------------------------------------------ ORCA in simulation
def _simulate(orca, pos, goals, speed=1.0, steps=800, dt=0.05, radius=0.4, avoid=True):
    P = np.array(pos, float)
    V = np.zeros_like(P)
    min_d = np.inf
    for _ in range(steps):
        to_goal = np.asarray(goals) - P
        dist = np.linalg.norm(to_goal, axis=1, keepdims=True)
        pref = np.where(dist > 0.05, to_goal / np.maximum(dist, 1e-9) * np.minimum(speed, dist / dt), 0.0)
        V = orca.compute(P, V, pref) if avoid else pref
        P = P + V * dt
        D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P)) * 1e9
        min_d = min(min_d, float(D.min()))
        if np.all(np.linalg.norm(np.asarray(goals) - P, axis=1) < 0.1):
            break
    return P, min_d


def test_two_agents_head_on_with_a_small_offset_pass_each_other_without_touching():
    orca = ORCA3D(radius=0.4, time_horizon=3.0, max_speed=1.5)
    pos = [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.05]]
    goals = [[5.0, 0.0, 0.0], [-5.0, 0.0, 0.05]]
    P, dmin = _simulate(orca, pos, goals, speed=1.0)
    assert dmin >= 2 * 0.4 - 1e-3 and np.all(np.linalg.norm(P - np.array(goals), axis=1) < 0.2)
    _, d_straight = _simulate(orca, pos, goals, avoid=False)
    assert d_straight < 0.2  # without avoidance they fly through each other (sanity of the scenario)


def test_ten_agents_swap_places_across_a_sphere_without_a_collision():
    n = 10
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pos, goals = 6.0 * dirs, -6.0 * dirs  # every agent flies to the antipode: all paths cross near the centre
    orca = ORCA3D(radius=0.4, time_horizon=4.0, max_speed=1.5, margin=0.1)
    P, dmin = _simulate(orca, pos, goals, speed=1.0, steps=1500)
    assert dmin >= 2 * 0.4 - 0.02
    assert np.all(np.linalg.norm(P - goals, axis=1) < 0.3)
    _, d_straight = _simulate(orca, pos, goals, steps=1500, avoid=False)
    assert d_straight < 0.8  # the unavoided flow does collide at the centre
    assert orca.infeasible_count <= 5  # the shorter-horizon fallback is rarely needed


def test_a_static_obstacle_field_keeps_agents_out_and_lets_them_through_around_it():
    sdf = SignedDistanceField.from_function(lambda p: sphere_sdf(p, (0, 0, 0), 2.0), (-8, -8, -8, 8, 8, 8), 0.2)
    orca = ORCA3D(radius=0.4, time_horizon=3.0, max_speed=1.5, sdf=sdf, margin=0.0)
    P = np.array([[-6.0, 0.1, 0.1]])
    goal = np.array([[6.0, 0.0, 0.0]])
    V = np.zeros((1, 3))
    dmin = np.inf
    for _ in range(1500):
        to = goal - P
        pref = to / max(np.linalg.norm(to), 1e-9) * min(1.0, np.linalg.norm(to) / 0.05)
        V = orca.compute(P, V, pref)
        P = P + V * 0.05
        dmin = min(dmin, float(sdf.distance(P)[0]))
        if np.linalg.norm(goal - P) < 0.15:
            break
    assert dmin >= 0.4 - 0.05  # the agent's surface never penetrated the obstacle
    assert np.linalg.norm(goal - P) < 0.3


def test_an_unresponsive_moving_obstacle_is_avoided_with_full_responsibility():
    orca = ORCA3D(radius=0.4, time_horizon=4.0, max_speed=1.5)
    P, V = np.array([[-6.0, 0.0, 0.0]]), np.zeros((1, 3))
    other_p, other_v = np.array([[6.0, 0.05, 0.0]]), np.array([[-1.0, 0.0, 0.0]])  # flies straight at us and does not react
    dmin = np.inf
    for _ in range(400):
        pref = np.array([[1.0, 0.0, 0.0]])
        V = orca.compute(P, V, pref, other_pos=other_p, other_vel=other_v, other_radius=0.5)
        P = P + V * 0.05
        other_p = other_p + other_v * 0.05
        dmin = min(dmin, float(np.linalg.norm(P - other_p)))
    assert dmin >= 0.4 + 0.5 - 0.02


def test_per_agent_radii_and_validation():
    orca = ORCA3D(radius=[0.3, 0.9], time_horizon=3.0, max_speed=1.5)
    P, _ = _simulate(orca, [[-5.0, 0, 0], [5.0, 0, 0.1]], [[5.0, 0, 0], [-5.0, 0, 0.1]], speed=1.0)
    assert np.all(np.isfinite(P))
    with pytest.raises(ValueError):
        ORCA3D(time_horizon=0.0)


# ------------------------------------------------------------------------------------------------ CBF
def _sphere_field(radius=2.0):
    return SignedDistanceField.from_function(lambda p: sphere_sdf(p, (0, 0, 0), radius), (-9, -9, -9, 9, 9, 9), 0.15)


def test_the_velocity_filter_keeps_a_single_integrator_outside_the_unsafe_set():
    sdf = _sphere_field()
    flt = CBFSafetyFilter(sdf, radius=0.4, margin=0.1, alpha=2.0, v_max=3.0)
    p = np.array([[-7.0, 0.05, 0.05]])
    h_min = np.inf
    for _ in range(1200):
        v_des = np.array([[2.5, 0.0, 0.0]])  # straight through the obstacle
        v = flt.filter_velocity(p, v_des)
        p = p + v * 0.02
        h_min = min(h_min, float(sdf.distance(p)[0] - 0.5))
    assert h_min >= -0.02  # forward invariance of {d - r - margin >= 0}, up to the field's interpolation error
    naive = np.array([[-7.0, 0.05, 0.05]])
    reached = np.inf
    for _ in range(600):
        naive = naive + np.array([[2.5, 0.0, 0.0]]) * 0.02
        reached = min(reached, float(sdf.distance(naive)[0]))
    assert reached < 0.0  # without the filter the vehicle flies into the obstacle


def test_the_filter_does_not_touch_commands_that_are_already_safe():
    sdf = _sphere_field()
    flt = CBFSafetyFilter(sdf, radius=0.4, margin=0.1, alpha=2.0, v_max=3.0, activation_distance=4.0)
    far = np.array([[-8.0, 0.0, 0.0]])
    assert np.allclose(flt.filter_velocity(far, [[1.0, 0.0, 0.0]]), [[1.0, 0.0, 0.0]])  # beyond the activation distance
    near = np.array([[-2.8, 0.0, 0.0]])  # 0.8 m from the surface: h = 0.8 - 0.5 = 0.3
    away = flt.filter_velocity(near, [[-1.0, 0.0, 0.0]])  # moving away from the obstacle is always allowed
    assert np.allclose(away, [[-1.0, 0.0, 0.0]], atol=1e-9)
    toward = flt.filter_velocity(near, [[1.0, 0.0, 0.0]])
    assert toward[0, 0] == pytest.approx(2.0 * 0.3, abs=0.03) and abs(toward[0, 1]) < 0.05  # u_x <= alpha h: slowed, not deflected sideways
    mild = flt.filter_velocity(np.array([[-3.5, 0.0, 0.0]]), [[1.0, 0.0, 0.0]])  # h = 1.0 allows 2 m/s: 1 m/s passes untouched
    assert np.allclose(mild, [[1.0, 0.0, 0.0]], atol=1e-9)


def test_the_acceleration_filter_prevents_a_double_integrator_from_hitting_the_obstacle():
    sdf = _sphere_field()
    flt = CBFSafetyFilter(sdf, radius=0.4, margin=0.1, a_max=6.0)
    p, v = np.array([[-7.0, 0.05, 0.0]]), np.array([[2.5, 0.0, 0.0]])
    h_min = np.inf
    for _ in range(1500):
        a_nom = 3.0 * (np.array([[1.0, 0.0, 0.0]]) * 3.0 - v)  # a velocity controller that insists on flying straight on
        a = flt.filter_acceleration(p, v, a_nom, alpha1=1.5, alpha2=1.5)
        v = v + a * 0.01
        p = p + v * 0.01
        h_min = min(h_min, float(sdf.distance(p)[0] - 0.5))
    assert h_min >= -0.03


def test_moving_vehicles_are_treated_as_unresponsive_obstacles():
    flt = CBFSafetyFilter(None, radius=0.4, margin=0.1, alpha=2.0, v_max=2.0)
    p, other = np.array([[-6.0, 0.0, 0.0]]), np.array([[6.0, 0.02, 0.0]])
    ov = np.array([[-1.0, 0.0, 0.0]])
    d_min = np.inf
    for _ in range(600):
        v = flt.filter_velocity(p, [[1.5, 0.0, 0.0]], other_pos=other, other_vel=ov, other_radius=0.4)
        p, other = p + v * 0.02, other + ov * 0.02
        d_min = min(d_min, float(np.linalg.norm(p - other)))
    assert d_min >= 0.9 - 0.02
    with pytest.raises(ValueError):
        CBFSafetyFilter(alpha=0.0)
