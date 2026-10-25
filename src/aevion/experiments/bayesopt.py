"""Gaussian-process regression and Bayesian optimisation.

:class:`GaussianProcess` is a standard zero-mean GP on standardised outputs with a Matérn-5/2 or squared-exponential kernel (one length scale per
input dimension), fitted by maximising the log marginal likelihood (Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006, ch. 2 and 5);
the tests compare its likelihood and posterior with the textbook formulas. :func:`bayes_opt` minimises (or maximises) an expensive function with
Expected Improvement (Jones, Schonlau and Welch 1998), lower confidence bound or probability of improvement, starting from a Sobol design, optionally
proposing a batch per iteration with the constant-liar heuristic (Ginsbourger et al. 2010).

Intended for the 2–15 dimensions of controller gains and scenario parameters; the cubic cost in the number of evaluations is negligible next to a simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, optimize
from scipy.stats import norm

from .design import sobol

__all__ = ["GaussianProcess", "kernel", "expected_improvement", "lower_confidence_bound", "probability_of_improvement", "bayes_opt", "BOResult"]


def kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: Any, variance: float, kind: str = "matern52") -> np.ndarray:
    """Covariance matrix between two point sets (inputs divided by the length scale(s) first)."""
    ls = np.asarray(lengthscale, float)
    A, B = np.asarray(X1, float) / ls, np.asarray(X2, float) / ls
    d2 = np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T, 0.0)
    if kind == "rbf":
        return variance * np.exp(-0.5 * d2)
    if kind == "matern52":
        r = np.sqrt(5.0 * d2)
        return variance * (1.0 + r + r * r / 3.0) * np.exp(-r)
    if kind == "matern32":
        r = np.sqrt(3.0 * d2)
        return variance * (1.0 + r) * np.exp(-r)
    raise ValueError("kernel must be 'matern52', 'matern32' or 'rbf'")


class GaussianProcess:
    """GP regression ``y = f(x) + ε``; ``fit`` standardises ``y`` and (optionally) optimises length scales, signal variance and noise variance."""

    def __init__(self, kind: str = "matern52", noise: float = 1e-6, fit_noise: bool = False, jitter: float = 1e-9) -> None:
        kernel(np.zeros((1, 1)), np.zeros((1, 1)), 1.0, 1.0, kind)  # validates the name
        self.kind, self.noise, self.fit_noise, self.jitter = kind, float(noise), bool(fit_noise), float(jitter)
        self.lengthscale: np.ndarray = np.ones(1)
        self.variance = 1.0
        self.X: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    # ------------------------------------------------------------------------------------------- likelihood
    def _unpack(self, theta: np.ndarray, d: int) -> Tuple[np.ndarray, float, float]:
        ls = np.exp(theta[:d])
        var = float(np.exp(theta[d]))
        noise = float(np.exp(theta[d + 1])) if self.fit_noise else self.noise
        return ls, var, noise

    def _nll(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        d = X.shape[1]
        ls, var, noise = self._unpack(theta, d)
        K = kernel(X, X, ls, var, self.kind) + (noise + self.jitter) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e25
        a = linalg.cho_solve((L, True), y)
        return float(0.5 * y @ a + np.log(np.diag(L)).sum() + 0.5 * len(y) * math.log(2 * math.pi))

    def log_marginal_likelihood(self, X: Any, y: Any, lengthscale: Any, variance: float, noise: Optional[float] = None) -> float:
        """``log p(y | X)`` for given hyper-parameters (``y`` used as it is, no standardisation)."""
        X = np.atleast_2d(np.asarray(X, float))
        ls = np.broadcast_to(np.asarray(lengthscale, float), (X.shape[1],))
        theta = np.concatenate([np.log(ls), [math.log(variance)], [math.log(noise if noise is not None else self.noise)]])
        saved = self.fit_noise
        self.fit_noise = True
        try:
            return -self._nll(theta, X, np.asarray(y, float))
        finally:
            self.fit_noise = saved

    # ------------------------------------------------------------------------------------------------ fitting
    def fit(self, X: Any, y: Any, optimize_hypers: bool = True, restarts: int = 4, seed: int = 0) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).reshape(-1)
        if len(X) != len(y) or len(X) < 2:
            raise ValueError("need at least two points and one output per point")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("the training data contain non-finite values")
        d = X.shape[1]
        self.y_mean, self.y_std = float(y.mean()), float(y.std()) or 1.0
        z = (y - self.y_mean) / self.y_std
        span = np.maximum(X.max(0) - X.min(0), 1e-9)
        if optimize_hypers:
            rng = np.random.default_rng(seed)
            n_par = d + 1 + (1 if self.fit_noise else 0)
            lo = np.concatenate([np.log(span * 0.02), [math.log(1e-2)], [math.log(1e-8)] if self.fit_noise else []])
            hi = np.concatenate([np.log(span * 20.0), [math.log(1e2)], [math.log(1.0)] if self.fit_noise else []])
            best_t, best_v = None, np.inf
            for k in range(max(restarts, 1)):
                t0 = np.concatenate([np.log(span * 0.5), [0.0], [math.log(1e-4)] if self.fit_noise else []]) if k == 0 else rng.uniform(lo, hi)
                res = optimize.minimize(self._nll, t0, args=(X, z), method="L-BFGS-B", bounds=list(zip(lo, hi)))
                if res.fun < best_v:
                    best_t, best_v = res.x, res.fun
            assert best_t is not None and len(best_t) == n_par
            self.lengthscale, self.variance, noise = self._unpack(best_t, d)
            if self.fit_noise:
                self.noise = noise
        else:
            self.lengthscale = np.broadcast_to(np.asarray(self.lengthscale, float), (d,)).copy()
        self.X = X
        K = kernel(X, X, self.lengthscale, self.variance, self.kind) + (self.noise + self.jitter) * np.eye(len(X))
        self._L = np.linalg.cholesky(K)
        self._alpha = linalg.cho_solve((self._L, True), z)
        return self

    def set_hyperparameters(self, lengthscale: Any, variance: float, noise: float) -> "GaussianProcess":
        self.lengthscale, self.variance, self.noise = np.asarray(lengthscale, float), float(variance), float(noise)
        return self

    # ------------------------------------------------------------------------------------------- prediction
    def predict(self, Xs: Any, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the *latent function* at ``Xs`` (add the noise variance for new observations)."""
        if self.X is None or self._L is None or self._alpha is None:
            raise RuntimeError("fit the model first")
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = kernel(Xs, self.X, self.lengthscale, self.variance, self.kind)
        mean = Ks @ self._alpha * self.y_std + self.y_mean
        if not return_std:
            return mean, np.zeros_like(mean)
        v = linalg.solve_triangular(self._L, Ks.T, lower=True)
        var = np.maximum(self.variance - (v**2).sum(0), 0.0)
        return mean, np.sqrt(var) * self.y_std


# ------------------------------------------------------------------------------------------------ acquisition
def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    """Expected improvement below ``best`` (minimisation): ``(b − μ − ξ) Φ(z) + σ φ(z)``, ``z = (b − μ − ξ) / σ``; zero where ``σ = 0`` and ``μ ≥ b − ξ``."""
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    imp = best - mean - xi
    z = np.divide(imp, std, out=np.zeros_like(imp), where=std > 1e-12)
    ei = imp * norm.cdf(z) + std * norm.pdf(z)
    return np.where(std > 1e-12, ei, np.maximum(imp, 0.0))


def probability_of_improvement(mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    imp = best - np.asarray(mean, float) - xi
    return np.where(std > 1e-12, norm.cdf(np.divide(imp, std, out=np.zeros_like(imp), where=std > 1e-12)), (imp > 0).astype(float))


def lower_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """``−(μ − κ σ)``: larger is more promising (so every acquisition is maximised)."""
    return -(np.asarray(mean, float) - kappa * np.asarray(std, float))


# ------------------------------------------------------------------------------------------------ optimisation loop
@dataclass
class BOResult:
    x_best: np.ndarray
    y_best: float
    X: np.ndarray
    y: np.ndarray
    best_so_far: np.ndarray
    gp: GaussianProcess
    failures: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


def bayes_opt(
    f: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    n_init: int = 8,
    n_iter: int = 25,
    seed: int = 0,
    acquisition: str = "ei",
    maximize: bool = False,
    xi: float = 0.01,
    kappa: float = 2.0,
    n_candidates: int = 2000,
    batch: int = 1,
    gp_kind: str = "matern52",
    noise: float = 1e-6,
    fit_noise: bool = False,
    evaluate_batch: Optional[Callable[[np.ndarray], Sequence[float]]] = None,
    x0: Optional[np.ndarray] = None,
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> BOResult:
    """Minimise ``f`` over a box (maximise with ``maximize=True``) with ``n_init`` Sobol points followed by ``n_iter`` model-based proposals.

    ``f`` takes a vector of the real parameters. ``batch > 1`` proposes that many points per iteration (constant liar); ``evaluate_batch(X)`` may evaluate
    them in parallel. A non-finite value (a diverged simulation) is replaced by the worst finite value seen so far and counted in ``failures``; the optimiser
    never proposes the same point twice.
    """
    b = np.asarray(bounds, float)
    if b.ndim != 2 or b.shape[1] != 2 or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("bounds must be (d, 2) with upper > lower")
    if acquisition not in ("ei", "lcb", "pi"):
        raise ValueError("acquisition must be 'ei', 'lcb' or 'pi'")
    d, sign = len(b), (-1.0 if maximize else 1.0)
    rng = np.random.default_rng(seed)
    to_real = lambda u: b[:, 0] + np.asarray(u) * (b[:, 1] - b[:, 0])  # noqa: E731

    def run(Ureal: np.ndarray) -> List[float]:
        vals = evaluate_batch(Ureal) if evaluate_batch is not None else [f(x) for x in Ureal]
        return [float(v) for v in vals]

    U = sobol(max(n_init, 2), d, seed)
    if x0 is not None:
        U = np.vstack([(np.atleast_2d(x0) - b[:, 0]) / (b[:, 1] - b[:, 0]), U])[: max(n_init, 2)]
    raw = run(to_real(U))
    failures = 0
    finite = [v for v in raw if np.isfinite(v)]
    if not finite:
        raise RuntimeError("every initial evaluation returned a non-finite value")
    worst = max(sign * v for v in finite)
    ys = []
    for v in raw:
        if np.isfinite(v):
            ys.append(sign * v)
        else:
            ys.append(worst)
            failures += 1
    Xu = U.copy()
    y = np.array(ys)
    best_trace = [float(y.min())]
    hist: List[Dict[str, Any]] = []
    gp = GaussianProcess(gp_kind, noise=noise, fit_noise=fit_noise)
    for it in range(int(n_iter)):
        gp.fit(Xu, y, seed=seed + it)
        Xaug, yaug = Xu.copy(), y.copy()
        proposals: List[np.ndarray] = []
        for q in range(max(batch, 1)):
            gpq = (
                gp
                if q == 0
                else GaussianProcess(gp_kind, noise=noise, fit_noise=False)
                .set_hyperparameters(gp.lengthscale, gp.variance, gp.noise)
                .fit(Xaug, yaug, optimize_hypers=False)
            )
            best_y = float(yaug.min())
            cand = np.vstack([rng.random((n_candidates, d)), np.clip(Xu[np.argmin(y)] + rng.normal(0, 0.05, (n_candidates // 4, d)), 0, 1)])
            m, s = gpq.predict(cand)
            if acquisition == "ei":
                a = expected_improvement(m, s, best_y, xi * gpq.y_std)
            elif acquisition == "pi":
                a = probability_of_improvement(m, s, best_y, xi * gpq.y_std)
            else:
                a = lower_confidence_bound(m, s, kappa)
            top = cand[np.argsort(-a)[:3]]
            best_u, best_a = top[0], float(np.max(a))
            for u0 in top:  # refine the best few candidates with a bounded local search

                def neg(u: np.ndarray, gpq: GaussianProcess = gpq, best_y: float = best_y) -> float:
                    mm, ss = gpq.predict(u[None])
                    if acquisition == "ei":
                        return -float(expected_improvement(mm, ss, best_y, xi * gpq.y_std)[0])
                    if acquisition == "pi":
                        return -float(probability_of_improvement(mm, ss, best_y, xi * gpq.y_std)[0])
                    return -float(lower_confidence_bound(mm, ss, kappa)[0])

                res = optimize.minimize(neg, u0, method="L-BFGS-B", bounds=[(0, 1)] * d, options=dict(maxiter=30))
                if -res.fun > best_a:
                    best_u, best_a = np.clip(res.x, 0, 1), -float(res.fun)
            dmin = np.min(np.linalg.norm(Xaug - best_u, axis=1))
            if dmin < 1e-6:  # never re-evaluate a point: take a random one instead
                best_u = rng.random(d)
            proposals.append(best_u)
            Xaug = np.vstack([Xaug, best_u])
            yaug = np.append(yaug, float(yaug.min()))  # constant liar: pretend the proposal achieved the current best
        P = np.array(proposals)
        vals = run(to_real(P))
        for u, v in zip(P, vals):
            if np.isfinite(v):
                yv = sign * v
            else:
                yv = float(y.max())
                failures += 1
            Xu = np.vstack([Xu, u])
            y = np.append(y, yv)
            hist.append(dict(iteration=it, y=sign * yv, x=to_real(u).tolist()))
            if callback is not None:
                callback(len(y), to_real(u), sign * yv)
        best_trace.append(float(y.min()))
    k = int(np.argmin(y))
    return BOResult(to_real(Xu[k]), float(sign * y[k]), to_real(Xu), sign * y, sign * np.array(best_trace), gp, failures, hist)
