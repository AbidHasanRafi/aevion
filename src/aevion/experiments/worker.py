"""Command-line worker: run one shard of a saved experiment into a result cache (what a Slurm array task executes).

    python -m aevion.experiments.worker experiment.json --cache results/cache --shard 3 --of 16

The specification is plain JSON describing configurations and a *registered* metric name; the file is read as data and nothing in it is executed.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .cache import ResultCache
from .config_experiment import ConfigExperiment
from .runners import ProcessRunner, SerialRunner


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="aevion.experiments.worker", description=(__doc__ or "").splitlines()[0])
    ap.add_argument("spec", help="experiment specification (JSON written by ConfigExperiment.save_spec)")
    ap.add_argument("--cache", required=True, help="result cache directory (shared between shards)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1, help="number of shards")
    ap.add_argument("--workers", type=int, default=1, help="processes on this machine")
    args = ap.parse_args(argv)
    exp = ConfigExperiment.load_spec(args.spec)
    cache = ResultCache(args.cache)
    res = exp.run(runner=ProcessRunner(args.workers) if args.workers > 1 else SerialRunner(), cache=cache, shard=(args.shard, args.of))
    m = res.meta
    print(  # noqa: T201
        f"{exp.name}: shard {args.shard}/{args.of}: {m['trials']} trials ({m['cached']} cached, {m['computed']} computed, {m['failed']} failed) in {m['wall_s']:.1f} s"
    )
    return 1 if m["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
