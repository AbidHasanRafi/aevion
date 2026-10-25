"""Statistics for comparing methods across seeds: intervals, tests, effect sizes."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from scipy import stats as sps


def mean_ci(x: Any, level: float = 0.95) -> tuple:
    """Mean with a Student-t confidence interval. Returns ``(mean, lo, hi)``."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (float("nan"),) * 3
    m = x.mean()
    if n < 2:
        return m, m, m
    h = sps.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def bootstrap_ci(x: Any, stat=np.mean, level: float = 0.95, n_boot: int = 4000, seed: int = 0) -> tuple:
    """Percentile bootstrap CI for any statistic. Returns ``(estimate, lo, hi)``."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(1)
    a = (1 - level) / 2
    return float(stat(x)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple:
    """Wilson score interval for a proportion (success rates). Returns ``(p, lo, hi)``."""
    if n == 0:
        return (float("nan"),) * 3
    z = sps.norm.ppf(0.5 + level / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), min(c + h, 1.0)


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]: P(a>b) - P(a<b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm_correction(p: Any) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def compare_two(a, b) -> dict:
    """Welch t-test, Mann-Whitney U, and effect sizes for two samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = sps.ttest_ind(a, b, equal_var=False)
    try:
        u = sps.mannwhitneyu(a, b, alternative="two-sided")
        pu = float(u.pvalue)
    except ValueError:
        pu = float("nan")
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()), t_p=float(t.pvalue), u_p=pu, cohens_d=cohens_d(a, b), cliffs_delta=cliffs_delta(a, b))


def pairwise_table(samples: Dict[str, Any], baseline: str | None = None) -> list:
    """Pairwise comparison rows with Holm-adjusted Welch p-values (against ``baseline`` or all pairs)."""
    names = list(samples)
    pairs = [(baseline, n) for n in names if n != baseline] if baseline else [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    rows = []
    for a, b in pairs:
        r = compare_two(samples[a], samples[b])
        rows.append(dict(a=a, b=b, **r))
    if rows:
        adj = holm_correction([r["t_p"] for r in rows])
        for r, p in zip(rows, adj):
            r["t_p_holm"] = float(p)
    return rows


def significance_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def sample_size_for_diff(sd: float, delta: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Per-group sample size to detect a mean difference ``delta`` with a two-sample t-test (normal approx.)."""
    za, zb = sps.norm.ppf(1 - alpha / 2), sps.norm.ppf(power)
    return math.ceil(2 * ((za + zb) * sd / delta) ** 2)
