"""Numerical building blocks of the diagnostic figures: frequency response and stability margins, ROC curves, kernel density estimates.

They are separate from the plotting code so the numbers can be tested against closed forms (a first-order lag is -3.01 dB and -45 degrees at its corner;
the Gaussian-score ROC has area ``Φ(d′/√2)``) and reused in tables.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, signal
from scipy.stats import norm

__all__ = ["tf_from", "freqresp", "margins", "roc_curve", "auc", "auc_ci", "kde_curve", "bandwidth", "poles_zeros"]

TF = Tuple[Sequence[float], Sequence[float]]


def tf_from(system: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``(num, den)`` polynomials (descending powers of ``s``) of a transfer function, ``(A, B, C, D)`` state space or a ``scipy.signal`` LTI object (single input/output)."""
    if hasattr(system, "num") and hasattr(system, "den"):
        return np.atleast_1d(system.num).astype(float), np.atleast_1d(system.den).astype(float)
    if isinstance(system, (tuple, list)) and len(system) == 2:
        return np.atleast_1d(np.asarray(system[0], float)), np.atleast_1d(np.asarray(system[1], float))
    if isinstance(system, (tuple, list)) and len(system) == 4:
        num, den = signal.ss2tf(*[np.atleast_2d(np.asarray(m, float)) for m in system])
        return np.trim_zeros(num[0], "f") if np.any(num[0]) else np.zeros(1), den
    raise ValueError("a system is (num, den), (A, B, C, D) or a scipy.signal transfer function")


def poles_zeros(system: Any) -> Tuple[np.ndarray, np.ndarray]:
    num, den = tf_from(system)
    return np.roots(den), np.roots(num) if len(num) > 1 else np.zeros(0)


def freqresp(system: Any, w: np.ndarray) -> np.ndarray:
    """``H(jω)`` for angular frequencies ``w`` [rad/s]."""
    num, den = tf_from(system)
    s = 1j * np.asarray(w, float)
    return np.polyval(num, s) / np.polyval(den, s)


def margins(system: Any, w_lo: float = 1e-3, w_hi: float = 1e3, n: int = 20000) -> Dict[str, float]:
    """Gain and phase margins of the open loop ``L(s)``: ``gain_margin`` (linear, ``1 / |L|`` at the phase crossover), ``phase_margin_deg``, and the two crossover frequencies.

    A missing crossover gives ``inf`` (gain margin) or ``nan`` (phase margin). The phase is unwrapped along the frequency grid.
    """
    w = np.logspace(np.log10(w_lo), np.log10(w_hi), n)
    H = freqresp(system, w)
    logmag = np.log(np.abs(H))
    phase = np.unwrap(np.angle(H))
    out = dict(gain_margin=float("inf"), phase_margin_deg=float("nan"), w_gc=float("nan"), w_pc=float("nan"))
    lw = np.log(w)
    # gain crossover: |L| passes through 1
    cross = np.nonzero(np.diff(np.sign(logmag)) != 0)[0]
    if len(cross):
        i = int(cross[0])
        wgc = optimize.brentq(lambda x: float(np.log(np.abs(freqresp(system, np.array([x]))[0]))), w[i], w[i + 1], xtol=1e-12)
        ph = float(np.degrees(np.interp(np.log(wgc), lw, phase)))
        pm = 180.0 + ph
        out["w_gc"] = float(wgc)
        out["phase_margin_deg"] = float(pm - 360.0 * np.round(pm / 360.0))
    # phase crossover: the unwrapped phase passes through -180 degrees (or -180 - 360 k)
    k = np.floor((phase + np.pi) / (2 * np.pi))
    jumps = np.nonzero(k[:-1] != k[1:])[0]
    if len(jumps):
        i = int(jumps[0])
        target = -np.pi + 2 * np.pi * max(k[i], k[i + 1])
        wpc = optimize.brentq(lambda x: float(np.interp(np.log(x), lw, phase) - target), w[i], w[i + 1], xtol=1e-12)
        out["w_pc"] = float(wpc)
        out["gain_margin"] = float(1.0 / abs(freqresp(system, np.array([wpc]))[0]))
    return out


# ------------------------------------------------------------------------------------------------ ROC
def roc_curve(y_true: Any, score: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False- and true-positive rates for every distinct threshold (higher score = more likely positive); the curve starts at (0, 0) and ends at (1, 1)."""
    y = np.asarray(y_true).astype(bool).reshape(-1)
    s = np.asarray(score, float).reshape(-1)
    if len(y) != len(s) or len(y) == 0:
        raise ValueError("y_true and score must have the same non-zero length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    P, N = int(y.sum()), int((~y).sum())
    if P == 0 or N == 0:
        raise ValueError("a ROC curve needs both positive and negative examples")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    distinct = np.nonzero(np.diff(s))[0]
    idx = np.append(distinct, len(s) - 1)
    tp = np.cumsum(y)[idx]
    fp = (idx + 1) - tp
    fpr = np.concatenate([[0.0], fp / N])
    tpr = np.concatenate([[0.0], tp / P])
    thr = np.concatenate([[np.inf], s[idx]])
    return fpr, tpr, thr


def auc(y_true: Any, score: Any) -> float:
    """Area under the ROC curve (trapezoid rule; equals the probability that a random positive outscores a random negative, ties counting half)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc_ci(y_true: Any, score: Any, level: float = 0.95, n_boot: int = 500, seed: int = 0) -> Tuple[float, float, float]:
    """AUC with a stratified-bootstrap percentile interval (positives and negatives are resampled separately)."""
    y = np.asarray(y_true).astype(bool).reshape(-1)
    s = np.asarray(score, float).reshape(-1)
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
    vals = []
    for _ in range(int(n_boot)):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(auc(y[idx], s[idx]))
    a = (1 - level) / 2
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return auc(y, s), float(lo), float(hi)


# ------------------------------------------------------------------------------------------------ KDE
def bandwidth(x: Any, rule: Union[str, float] = "scott") -> float:
    x = np.asarray(x, float)
    if isinstance(rule, (int, float)):
        return float(rule)
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    if rule == "scott":
        return 1.06 * sd * len(x) ** (-0.2) if sd > 0 else 1.0
    if rule == "silverman":
        s = min(sd, iqr / 1.34) if iqr > 0 else sd
        return 0.9 * s * len(x) ** (-0.2) if s > 0 else 1.0
    raise ValueError("bandwidth rule must be 'scott', 'silverman' or a number")


def kde_curve(
    x: Any, grid: Optional[np.ndarray] = None, bw: Union[str, float] = "scott", n: int = 256, pad: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian kernel density estimate on ``grid`` (default: the data range padded by ``pad`` bandwidths); returns ``(grid, density)``, which integrates to one."""
    x = np.asarray(x, float).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError("a density estimate needs at least two finite values")
    h = bandwidth(x, bw)
    if grid is None:
        grid = np.linspace(x.min() - pad * h, x.max() + pad * h, n)
    z = (grid[:, None] - x[None, :]) / h
    return grid, norm.pdf(z).sum(axis=1) / (len(x) * h)
