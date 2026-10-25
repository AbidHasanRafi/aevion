"""Parallel data collection across worker processes.

Each worker builds its own environment (``make_env(seed)``) and policy (``make_policy(env)``), rolls it out and returns a :class:`Dataset`; the parent
concatenates them with distinct environment ids. Both factories must be importable top-level callables (or ``functools.partial`` of one) so they can be
sent to worker processes, which is also what makes a run reproducible: worker ``i`` always uses seed ``seed + i``. With ``n_workers = 1`` the work runs in
the calling process.

This is the single-machine form of distributed rollouts. Ray, Dask and multi-node runners are not provided; the factories above are the same units of
work such a runner would distribute.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional, Tuple

from .dataset import Dataset, collect

__all__ = ["parallel_collect"]


def _work(args: Tuple[Callable[[int], Any], Callable[[Any], Any], Optional[Callable[[Any], Any]], int, int, int, float]) -> Dataset:
    make_env, make_policy, make_labeler, steps, seed, worker, noise = args
    env = make_env(seed)
    pol = make_policy(env)
    lab = make_labeler(env) if make_labeler is not None else None
    d = collect(env, pol, steps, labeler=lab, noise=noise, seed=seed, reset_seed=seed)
    d.env_ids = d.env_ids + worker * env.n
    d.meta["worker"] = worker
    return d


def parallel_collect(
    make_env: Callable[[int], Any],
    make_policy: Callable[[Any], Callable[[Any], Any]],
    steps: int,
    n_workers: int = 2,
    seed: int = 0,
    make_labeler: Optional[Callable[[Any], Callable[[Any], Any]]] = None,
    noise: float = 0.0,
) -> Dataset:
    """Collect ``steps`` control steps per sub-environment in each of ``n_workers`` processes and merge the results (worker order, so the output is deterministic)."""
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")
    jobs = [(make_env, make_policy, make_labeler, int(steps), seed + w, w, float(noise)) for w in range(n_workers)]
    if n_workers == 1:
        parts: List[Dataset] = [_work(jobs[0])]
    else:
        leaked = "NUMBA_NUM_THREADS" not in os.environ
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # one numba thread per worker process
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                parts = list(ex.map(_work, jobs))
        finally:
            if leaked:
                os.environ.pop("NUMBA_NUM_THREADS", None)
    out = parts[0]
    for p in parts[1:]:
        out = out.concat(p)
    out.meta["workers"] = n_workers
    return out
