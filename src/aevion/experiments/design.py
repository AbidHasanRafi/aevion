"""Designs of experiments: Latin hypercube, Sobol, Halton and full factorial samples, and named parameter spaces.

Every sampler returns points in the unit hypercube ``[0, 1]^d``; :class:`ParameterSpace` maps them to named parameters (uniform, log-uniform, integer,
categorical) so a design can drive any configuration. Space-filling designs cover a parameter box far more evenly than the same number of random draws
(their discrepancy is lower), which is why a 64-point Sobol design resolves a response surface that 64 random runs only roughly sample.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["latin_hypercube", "sobol", "halton", "full_factorial", "min_distance", "discrepancy", "ParameterSpace", "Parameter"]


def _check(n: int, d: int) -> None:
    if n < 1 or d < 1:
        raise ValueError("n and d must be at least 1")


def min_distance(x: np.ndarray) -> float:
    """Smallest pairwise Euclidean distance of the points (the quantity a maximin design maximises)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("inf")
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    d[np.diag_indices(len(x))] = np.inf
    return float(d.min())


def discrepancy(x: np.ndarray, method: str = "CD") -> float:
    """Centred L2 discrepancy of a point set in the unit cube (lower = more uniform)."""
    from scipy.stats import qmc

    return float(qmc.discrepancy(np.asarray(x, float), method=method))


def latin_hypercube(n: int, d: int, seed: int = 0, optimize: bool = True, iterations: int = 200) -> np.ndarray:
    """``n`` points with exactly one point in each of the ``n`` equal strata of every coordinate.

    With ``optimize`` the design is improved by swapping coordinates between pairs of points whenever that increases the minimum pairwise distance
    (a maximin Latin hypercube); the stratification is preserved by every swap.
    """
    _check(n, d)
    rng = np.random.default_rng(seed)
    x = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(d)], axis=1)
    if not optimize or n < 3 or d < 2:
        return x
    best = min_distance(x)
    for _ in range(int(iterations)):
        col = int(rng.integers(d))
        i, j = rng.choice(n, 2, replace=False)
        y = x.copy()
        y[i, col], y[j, col] = x[j, col], x[i, col]
        m = min_distance(y)
        if m > best:
            x, best = y, m
    return x


def sobol(n: int, d: int, seed: int = 0, scramble: bool = True, skip: int = 0) -> np.ndarray:
    """The first ``n`` points of a (scrambled) Sobol sequence (balance properties hold for ``n`` a power of two)."""
    _check(n, d)
    from scipy.stats import qmc

    s = qmc.Sobol(d, scramble=scramble, seed=seed)
    if skip:
        s.fast_forward(skip)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the power-of-two balance warning is the caller's business
        return np.asarray(s.random(n))


def _radical_inverse(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]


def halton(n: int, d: int, skip: int = 1) -> np.ndarray:
    """Halton points (radical inverses in the first ``d`` primes); ``skip`` drops the leading zero point."""
    _check(n, d)
    if d > len(_PRIMES):
        raise ValueError(f"Halton sequences are provided up to {len(_PRIMES)} dimensions")
    return np.array([[_radical_inverse(i + skip, _PRIMES[k]) for k in range(d)] for i in range(n)])


def full_factorial(levels: Sequence[Sequence[Any]]) -> List[Tuple[Any, ...]]:
    """Every combination of the given per-factor levels."""
    return list(itertools.product(*levels))


@dataclass
class Parameter:
    name: str
    kind: str  # "float" | "log" | "int" | "categorical"
    low: float = 0.0
    high: float = 1.0
    choices: Optional[Sequence[Any]] = None

    def map(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, float), 0.0, 1.0 - 1e-12)
        if self.kind == "float":
            return self.low + (self.high - self.low) * u
        if self.kind == "log":
            return np.exp(math.log(self.low) + (math.log(self.high) - math.log(self.low)) * u)
        if self.kind == "int":
            return np.floor(self.low + (self.high - self.low + 1) * u).astype(int)
        assert self.choices is not None
        idx = np.floor(u * len(self.choices)).astype(int)
        return np.array([self.choices[i] for i in idx], dtype=object)

    def unmap(self, value: Any) -> float:
        """Inverse of :meth:`map` (the lower edge of the value's cell for discrete parameters)."""
        if self.kind == "float":
            return float((value - self.low) / (self.high - self.low))
        if self.kind == "log":
            return float((math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low)))
        if self.kind == "int":
            return float((value - self.low) / (self.high - self.low + 1))
        assert self.choices is not None
        return float(list(self.choices).index(value) / len(self.choices))


class ParameterSpace:
    """Named parameters: ``ParameterSpace({"mass": (0.8, 1.2), "kp": ("log", 1, 100), "n": ("int", 1, 4), "mode": ["a", "b"]})``.

    A 2-tuple ``(lo, hi)`` is uniform, ``("log", lo, hi)`` log-uniform, ``("int", lo, hi)`` an integer in ``[lo, hi]``, and a list a categorical choice.
    """

    def __init__(self, spec: Dict[str, Any]) -> None:
        if not spec:
            raise ValueError("a parameter space needs at least one parameter")
        self.params: List[Parameter] = []
        for name, s in spec.items():
            if isinstance(s, list):
                if not s:
                    raise ValueError(f"'{name}' has no choices")
                self.params.append(Parameter(name, "categorical", choices=s))
            elif isinstance(s, tuple) and len(s) == 2:
                lo, hi = float(s[0]), float(s[1])
                if not hi > lo:
                    raise ValueError(f"'{name}': the upper bound must exceed the lower bound")
                self.params.append(Parameter(name, "float", lo, hi))
            elif isinstance(s, tuple) and len(s) == 3 and s[0] in ("log", "int"):
                lo, hi = float(s[1]), float(s[2])
                if not hi > lo or (s[0] == "log" and lo <= 0):
                    raise ValueError(f"'{name}': need hi > lo (and lo > 0 for a log-uniform parameter)")
                self.params.append(Parameter(name, s[0], lo, hi))
            else:
                raise ValueError(f"cannot interpret the specification of '{name}': {s!r}")
        self.names = [p.name for p in self.params]

    @property
    def dim(self) -> int:
        return len(self.params)

    def points(self, unit: np.ndarray) -> List[Dict[str, Any]]:
        """Map unit-cube points ``(n, d)`` to a list of ``{name: value}`` dictionaries."""
        u = np.atleast_2d(np.asarray(unit, float))
        if u.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} columns, got {u.shape[1]}")
        cols = [p.map(u[:, k]) for k, p in enumerate(self.params)]
        return [{p.name: (c[i].item() if hasattr(c[i], "item") else c[i]) for p, c in zip(self.params, cols)} for i in range(len(u))]

    def sample(self, n: int, method: str = "sobol", seed: int = 0) -> List[Dict[str, Any]]:
        """``n`` design points: ``method`` is ``sobol``, ``lhs``, ``halton`` or ``random``."""
        d = self.dim
        if method == "sobol":
            u = sobol(n, d, seed)
        elif method == "lhs":
            u = latin_hypercube(n, d, seed)
        elif method == "halton":
            u = halton(n, d)
        elif method == "random":
            u = np.random.default_rng(seed).random((n, d))
        else:
            raise ValueError("method must be sobol, lhs, halton or random")
        return self.points(u)

    def to_unit(self, point: Dict[str, Any]) -> np.ndarray:
        return np.array([p.unmap(point[p.name]) for p in self.params])
