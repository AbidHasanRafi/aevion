"""Runners: where the trials of an experiment execute.

A runner maps a top-level function over a list of picklable jobs and yields ``(index, result)`` as results become available, so the caller can store each
result the moment it exists (that is what makes an interrupted run resumable). :class:`SerialRunner` runs in the calling process (easy to debug, no
pickling), :class:`ProcessRunner` uses a pool of worker processes on this machine.

For a cluster, :func:`slurm_script` writes a Slurm *job array* whose tasks each run one shard of the experiment (``trial_index % n_shards == array_task_id``)
and write to a shared result cache; merging is then a cache union (:meth:`ResultCache.merge_from`). Ray and Dask runners are not provided; any object with
an ``imap(fn, jobs)`` method of the same shape can be passed instead.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Runner", "SerialRunner", "ProcessRunner", "slurm_script", "shard_indices"]


class Runner:
    name = "runner"

    def imap(self, fn: Callable[[Any], Any], jobs: Sequence[Any]) -> Iterator[Tuple[int, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def map(self, fn: Callable[[Any], Any], jobs: Sequence[Any]) -> List[Any]:
        out: List[Any] = [None] * len(jobs)
        for i, r in self.imap(fn, jobs):
            out[i] = r
        return out


class SerialRunner(Runner):
    name = "serial"

    def imap(self, fn: Callable[[Any], Any], jobs: Sequence[Any]) -> Iterator[Tuple[int, Any]]:
        for i, j in enumerate(jobs):
            yield i, fn(j)


class ProcessRunner(Runner):
    """A pool of ``workers`` processes (one numba thread each); results are yielded in job order."""

    name = "process"

    def __init__(self, workers: int = 2) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = int(workers)

    def imap(self, fn: Callable[[Any], Any], jobs: Sequence[Any]) -> Iterator[Tuple[int, Any]]:
        if not jobs:
            return
        leaked = "NUMBA_NUM_THREADS" not in os.environ
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        try:
            with ProcessPoolExecutor(max_workers=min(self.workers, len(jobs))) as ex:
                yield from enumerate(ex.map(fn, jobs, chunksize=1))
        finally:
            if leaked:
                os.environ.pop("NUMBA_NUM_THREADS", None)


def shard_indices(n: int, shard: int, n_shards: int) -> List[int]:
    """Indices ``i`` of ``range(n)`` with ``i % n_shards == shard`` (the assignment is independent of what is already cached)."""
    if n_shards < 1 or not 0 <= shard < n_shards:
        raise ValueError("need 0 <= shard < n_shards")
    return [i for i in range(n) if i % n_shards == shard]


def slurm_script(
    spec_path: str,
    cache_dir: str,
    n_shards: int,
    job_name: str = "aevion-exp",
    partition: Optional[str] = None,
    time: str = "01:00:00",
    mem: str = "4G",
    cpus: int = 1,
    python: str = "python",
    concurrent: Optional[int] = None,
    log_dir: str = "logs",
) -> str:
    """Text of an ``sbatch`` script that runs ``n_shards`` array tasks, task ``k`` executing shard ``k`` of the experiment in ``spec_path`` into ``cache_dir``."""
    if n_shards < 1 or cpus < 1:
        raise ValueError("n_shards and cpus must be at least 1")
    if any(c in spec_path + cache_dir + job_name + python for c in ";&|`$\n"):
        raise ValueError("paths and names must not contain shell metacharacters")
    arr = f"0-{n_shards - 1}" + (f"%{concurrent}" if concurrent else "")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={arr}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={log_dir}/%x-%A_%a.out",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines += [
        "",
        "set -euo pipefail",
        f"mkdir -p {log_dir}",
        "export NUMBA_NUM_THREADS=1",
        f'{python} -m aevion.experiments.worker "{spec_path}" --cache "{cache_dir}" --shard "$SLURM_ARRAY_TASK_ID" --of {n_shards}',
        "",
    ]
    return "\n".join(lines)
