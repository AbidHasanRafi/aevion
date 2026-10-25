"""Age of information (AoI): how stale is the freshest delivered update?

``Delta(t) = t - u(t)`` where ``u(t)`` is the generation time of the most recent update delivered by time ``t``. The time-average
AoI is the area under the sawtooth divided by the horizon. Closed forms are given for the textbook cases so simulations can be
checked against them.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

__all__ = ["aoi_mm1", "aoi_periodic_lossy", "average_aoi", "peak_aoi", "simulate_mm1_aoi"]


def _freshest(gen: np.ndarray, deliv: np.ndarray, t: np.ndarray) -> np.ndarray:
    order = np.argsort(deliv)
    d, g = deliv[order], gen[order]
    best = np.maximum.accumulate(g)  # freshest generation among everything delivered so far
    idx = np.searchsorted(d, t, side="right") - 1
    out = np.full(t.shape, -np.inf)
    ok = idx >= 0
    out[ok] = best[idx[ok]]
    return out


def average_aoi(gen_times: Any, delivery_times: Any, horizon: float, t0: float = 0.0, resolution: int = 20000) -> Dict[str, float]:
    """Time-average and peak AoI over ``[t0, horizon]`` from update generation and delivery times (lost updates: ``inf`` delivery).

    Before the first delivery the age is measured from ``t0`` (the system starts fresh at ``t0``).
    """
    gen = np.asarray(gen_times, float)
    deliv = np.asarray(delivery_times, float)
    keep = np.isfinite(deliv)
    t = np.linspace(t0, horizon, resolution)
    u = _freshest(gen[keep], deliv[keep], t)
    u = np.where(np.isfinite(u), u, t0)
    age = t - u
    area = float(np.sum(0.5 * (age[1:] + age[:-1]) * np.diff(t)))  # trapezoid rule (np.trapz was renamed in NumPy 2)
    return dict(mean=area / (horizon - t0), peak=float(age.max()))


def peak_aoi(gen_times: Any, delivery_times: Any) -> float:
    """Largest age reached just before a delivery: ``max_k (D_k - u_{k-1}^*)`` over consecutive deliveries."""
    gen = np.asarray(gen_times, float)
    deliv = np.asarray(delivery_times, float)
    keep = np.isfinite(deliv)
    order = np.argsort(deliv[keep])
    g, d = gen[keep][order], deliv[keep][order]
    if len(d) < 2:
        return float("inf")
    fresh = np.maximum.accumulate(g)
    return float(np.max(d[1:] - fresh[:-1]))


def aoi_periodic_lossy(period_s: float, success_prob: float, delay_s: float = 0.0) -> float:
    """Mean AoI of periodic updates (every ``T``) delivered independently with probability ``p`` after a fixed delay ``D``:
    ``D + T (2 - p) / (2 p)``  (``D + T/2`` when nothing is lost)."""
    if not 0.0 < success_prob <= 1.0:
        raise ValueError("success_prob must be in (0, 1]")
    return delay_s + period_s * (2.0 - success_prob) / (2.0 * success_prob)


def aoi_mm1(arrival_rate: float, service_rate: float) -> float:
    """Mean AoI of an M/M/1 FCFS queue (Kaul, Yates, Gruteser, INFOCOM 2012): ``(1/mu)(1 + 1/rho + rho^2/(1 - rho))``, ``rho = lambda / mu < 1``."""
    rho = arrival_rate / service_rate
    if not 0.0 < rho < 1.0:
        raise ValueError("need 0 < arrival_rate < service_rate")
    return (1.0 / service_rate) * (1.0 + 1.0 / rho + rho**2 / (1.0 - rho))


def simulate_mm1_aoi(arrival_rate: float, service_rate: float, n: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo mean AoI of an M/M/1 FCFS queue (for validating :func:`aoi_mm1`)."""
    rng = np.random.default_rng(seed)
    gen = np.cumsum(rng.exponential(1.0 / arrival_rate, n))
    serv = rng.exponential(1.0 / service_rate, n)
    deliv = np.empty(n)
    t_free = 0.0
    for k in range(n):
        start = max(gen[k], t_free)
        t_free = start + serv[k]
        deliv[k] = t_free
    # exact integral of the sawtooth between deliveries
    age_start = deliv[:-1] - gen[:-1]
    inter = np.diff(deliv)
    area = np.sum(age_start * inter + 0.5 * inter**2)
    return float(area / (deliv[-1] - deliv[0]))
