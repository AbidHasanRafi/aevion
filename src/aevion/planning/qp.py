"""Small convex-optimisation tools used by the avoidance filters and the model-predictive controller.

* :func:`project` - Euclidean projection of a point onto ``{x : A x >= b} ∩ {|x - c| <= r}``: Hildreth's dual coordinate ascent for
  half-spaces alone, Dykstra's alternating projections when a ball is included. This is the small QP behind ORCA and control-barrier
  safety filters (``min |u - u_des|²`` subject to linear constraints).
* :func:`solve_qp` - a dense OSQP-style ADMM solver (Stellato et al. 2020) for ``min ½ xᵀPx + qᵀx  s.t.  l <= A x <= u``, with
  equality rows, one-sided bounds and warm starts. It is accurate to a few digits, which is what a receding-horizon controller needs.

Everything is plain NumPy and deterministic, so the solvers run on any device the package supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = ["project", "ProjectionResult", "solve_qp", "QPResult"]


@dataclass
class ProjectionResult:
    x: np.ndarray
    feasible: bool
    violation: float  # largest constraint violation of ``x`` (0 when feasible)
    iterations: int


def project(
    x0: Any,
    A: Any = None,
    b: Any = None,
    ball_center: Any = None,
    ball_radius: Optional[float] = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> ProjectionResult:
    """Closest point to ``x0`` satisfying ``A x >= b`` and ``|x - ball_center| <= ball_radius``.

    When the constraints are inconsistent the iterates settle on a compromise: ``feasible`` is ``False`` and ``violation`` says
    by how much. Rows of ``A`` that are (numerically) zero are ignored.
    """
    x0 = np.asarray(x0, float)
    A = np.zeros((0, x0.size)) if A is None else np.atleast_2d(np.asarray(A, float))
    b = np.zeros(0) if b is None else np.asarray(b, float).reshape(-1)
    norms2 = np.sum(A * A, axis=1)
    keep = norms2 > 1e-18
    A, b, norms2 = A[keep], b[keep], norms2[keep]
    c = np.zeros_like(x0) if ball_center is None else np.asarray(ball_center, float)

    def violation(x: np.ndarray) -> float:
        v = 0.0 if not len(A) else float(np.max(b - A @ x, initial=0.0)) / 1.0
        if ball_radius is not None:
            v = max(v, float(np.linalg.norm(x - c) - ball_radius))
        return max(v, 0.0)

    if not len(A):
        x = x0.copy()
        if ball_radius is not None and np.linalg.norm(x - c) > ball_radius:
            x = c + (x - c) * (ball_radius / np.linalg.norm(x - c))
        return ProjectionResult(x, True, 0.0, 0)
    if ball_radius is None:  # Hildreth: coordinate ascent on the dual
        lam = np.zeros(len(A))
        x = x0.copy()
        for it in range(1, max_iter + 1):  # noqa: B007
            moved = 0.0
            for i in range(len(A)):
                step = (b[i] - A[i] @ x) / norms2[i]
                new = max(0.0, lam[i] + step)
                d = new - lam[i]
                if d != 0.0:
                    x = x + d * A[i]
                    lam[i] = new
                    moved = max(moved, abs(d) * np.sqrt(norms2[i]))
            if moved < tol:
                break
        v = violation(x)
        return ProjectionResult(x, v <= 1e-6 * max(1.0, float(np.abs(b).max(initial=0.0))), v, it)
    # Dykstra: exact projection onto the intersection of the half-spaces and the ball
    x = x0.copy()
    inc = [np.zeros_like(x0) for _ in range(len(A) + 1)]
    for it in range(1, max_iter + 1):  # noqa: B007
        x_prev = x.copy()
        moved = 0.0
        for i in range(len(A) + 1):
            y = x + inc[i]
            if i < len(A):
                gap = b[i] - A[i] @ y
                p = y + (gap / norms2[i]) * A[i] if gap > 0 else y
            else:
                r = np.linalg.norm(y - c)
                p = y if r <= ball_radius else c + (y - c) * (ball_radius / r)
            new_inc = y - p
            moved = max(moved, float(np.linalg.norm(new_inc - inc[i])))
            inc[i] = new_inc
            x = p
        if np.linalg.norm(x - x_prev) < tol and moved < tol:  # neither the iterate nor any correction term still changes
            break
    v = violation(x)
    return ProjectionResult(x, v <= 1e-5 * max(1.0, ball_radius or 1.0), v, it)


# ======================================================================================================== ADMM QP
@dataclass
class QPResult:
    x: np.ndarray
    y: np.ndarray  # dual variables of the constraint rows
    z: np.ndarray
    status: str  # "solved" or "max_iter"
    iterations: int
    objective: float
    primal_residual: float
    dual_residual: float


def solve_qp(
    P: Any,
    q: Any,
    A: Any = None,
    l: Any = None,
    u: Any = None,
    x0: Any = None,
    y0: Any = None,
    rho: float = 0.1,
    sigma: float = 1e-6,
    alpha: float = 1.6,
    max_iter: int = 4000,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    check_every: int = 10,
    adaptive_rho: bool = True,
) -> QPResult:
    """Solve ``min ½ xᵀ P x + qᵀ x`` subject to ``l <= A x <= u`` (``±inf`` allowed; ``l == u`` makes an equality row).

    ``P`` must be positive semidefinite. ``x0`` / ``y0`` warm-start the iteration (a receding-horizon controller passes the
    previous solution). The returned ``status`` is ``"solved"`` when the primal and dual residuals meet the tolerances.
    """
    P = np.atleast_2d(np.asarray(P, float))
    q = np.asarray(q, float).reshape(-1)
    n = q.size
    P = 0.5 * (P + P.T)
    if A is None:
        A = np.zeros((0, n))
        l = np.zeros(0)
        u = np.zeros(0)
    A = np.atleast_2d(np.asarray(A, float))
    m = A.shape[0]
    l = np.full(m, -np.inf) if l is None else np.asarray(l, float).reshape(-1)
    u = np.full(m, np.inf) if u is None else np.asarray(u, float).reshape(-1)
    if np.any(l > u + 1e-12):
        raise ValueError("lower bound above upper bound")
    eq = np.isfinite(l) & np.isfinite(u) & (np.abs(u - l) < 1e-10)
    rho_vec = np.where(eq, 1e3 * rho, rho)

    def factor(rv: np.ndarray) -> Any:
        from scipy.linalg import cho_factor

        K = P + sigma * np.eye(n) + A.T @ (rv[:, None] * A)
        return cho_factor(K, lower=True)

    from scipy.linalg import cho_solve

    chol = factor(rho_vec)
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    y = np.zeros(m) if y0 is None else np.asarray(y0, float).copy()
    z = np.clip(A @ x, l, u) if m else np.zeros(0)
    pr = dr = np.inf
    status, it = "max_iter", 0
    for it in range(1, max_iter + 1):
        rhs = sigma * x - q + A.T @ (rho_vec * z - y)
        xt = cho_solve(chol, rhs)
        zt = A @ xt
        x_new = alpha * xt + (1.0 - alpha) * x
        z_relaxed = alpha * zt + (1.0 - alpha) * z
        z_new = np.clip(z_relaxed + y / rho_vec, l, u) if m else z
        y = y + rho_vec * (z_relaxed - z_new)
        x, z = x_new, z_new
        if it % check_every == 0 or it == max_iter:
            Ax = A @ x if m else np.zeros(0)
            Px = P @ x
            pr = float(np.max(np.abs(Ax - z), initial=0.0))
            dr = float(np.max(np.abs(Px + q + A.T @ y)))
            eps_p = eps_abs + eps_rel * max(float(np.max(np.abs(Ax), initial=0.0)), float(np.max(np.abs(z), initial=0.0)))
            eps_d = eps_abs + eps_rel * max(float(np.max(np.abs(Px))), float(np.max(np.abs(A.T @ y), initial=0.0)), float(np.max(np.abs(q))))
            if pr <= eps_p and dr <= eps_d:
                status = "solved"
                break
            if adaptive_rho and m and it % (check_every * 5) == 0 and pr > 0 and dr > 0:
                ratio = np.sqrt((pr / max(eps_p, 1e-12)) / max(dr / max(eps_d, 1e-12), 1e-12))
                if ratio > 5.0 or ratio < 0.2:
                    scale = float(np.clip(ratio, 1e-2, 1e2))
                    rho_vec = np.clip(rho_vec * scale, 1e-6, 1e6)
                    chol = factor(rho_vec)
    obj = float(0.5 * x @ P @ x + q @ x)
    return QPResult(x, y, z, status, it, obj, pr, dr)
