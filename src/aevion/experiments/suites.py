"""Benchmark suites: fixed sets of scenarios and a protocol, so methods from different studies are compared on the same footing.

A :class:`BenchmarkSuite` names a handful of scenarios (complete configurations), the metric that ranks methods and the seeds. :func:`run_suite` evaluates any
set of *methods* (configuration overrides, typically a controller or an estimator) on every scenario and returns a :class:`SuiteResult`: one leaderboard per
scenario (with significance, see :func:`~aevion.experiments.database.leaderboard`) and an overall ranking by mean rank across scenarios. Trials go through
the usual cache, so re-running the suite after adding one method only simulates the new method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .cache import ResultCache
from .config_experiment import ConfigExperiment
from .database import format_leaderboard, leaderboard
from .runners import Runner

__all__ = ["BenchmarkSuite", "SuiteResult", "run_suite", "SUITES", "get_suite"]


@dataclass
class BenchmarkSuite:
    name: str
    scenarios: Dict[str, Dict[str, Any]]
    primary: str = "rmse_pos"
    lower_is_better: bool = True
    seeds: Sequence[int] = (0, 1, 2)
    metric: str = "flight"
    metric_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def experiment(self, scenario: str, methods: Dict[str, Dict[str, Any]], seeds: Optional[Sequence[int]] = None) -> ConfigExperiment:
        if scenario not in self.scenarios:
            raise KeyError(f"suite {self.name!r} has no scenario {scenario!r}; it has {sorted(self.scenarios)}")
        return ConfigExperiment(
            f"{self.name}/{scenario}", self.scenarios[scenario], methods, None, seeds or self.seeds, self.metric, self.metric_params
        )


@dataclass
class SuiteResult:
    suite: str
    primary: str
    lower_is_better: bool
    boards: Dict[str, List[Dict[str, Any]]]
    overall: List[Dict[str, Any]]
    n_trials: int
    failed: int

    def table(self) -> str:
        direction = "lower" if self.lower_is_better else "higher"
        parts = [f"suite {self.suite!r}: ranked by {self.primary} ({direction} is better), {self.n_trials} trials, {self.failed} failed"]
        for sc, board in self.boards.items():
            parts.append(f"\n[{sc}]\n" + format_leaderboard(board, self.primary))
        w = max(len(o["name"]) for o in self.overall)
        parts.append("\n[overall: mean rank over scenarios]")
        parts += [f" {o['rank']:>2}  {o['name']:<{w}}  {o['mean_rank']:.2f}" for o in self.overall]
        return "\n".join(parts)

    def to_json(self) -> str:
        data = dict(
            suite=self.suite,
            primary=self.primary,
            lower_is_better=self.lower_is_better,
            boards=self.boards,
            overall=self.overall,
            trials=self.n_trials,
            failed=self.failed,
        )
        return json.dumps(data, indent=1, default=float)

    def save(self, path: Any) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p


def run_suite(
    suite: BenchmarkSuite,
    methods: Dict[str, Dict[str, Any]],
    runner: Optional[Runner] = None,
    cache: Optional[ResultCache] = None,
    db: Any = None,
    seeds: Optional[Sequence[int]] = None,
    scenarios: Optional[Sequence[str]] = None,
) -> SuiteResult:
    if not methods:
        raise ValueError("give at least one method")
    boards: Dict[str, List[Dict[str, Any]]] = {}
    ranks: Dict[str, List[float]] = {m: [] for m in methods}
    n_trials = failed = 0
    for sc in scenarios or list(suite.scenarios):
        res = suite.experiment(sc, methods, seeds).run(runner=runner, cache=cache, db=db)
        n_trials += res.meta["trials"]
        failed += res.meta["failed"]
        board = leaderboard(res.rows, suite.primary, suite.lower_is_better)
        boards[sc] = board
        scored = {b["name"]: b["rank"] for b in board}  # a method whose every trial failed has no score: rank it last
        for m in methods:
            ranks[m].append(float(scored.get(m, len(methods))))
    order = sorted(methods, key=lambda m: float(np.mean(ranks[m])))
    overall = [dict(rank=i + 1, name=m, mean_rank=float(np.mean(ranks[m])), ranks=ranks[m]) for i, m in enumerate(order)]
    return SuiteResult(suite.name, suite.primary, suite.lower_is_better, boards, overall, n_trials, failed)


def _tracking_scenarios() -> Dict[str, Dict[str, Any]]:
    base = {"duration": 6.0}
    return {
        "circle": {**base, "trajectory": {"type": "circle", "params": {"center": [0, 0, 4.0], "radius": 3.0, "omega": 0.8}}},
        "figure8": {**base, "trajectory": {"type": "figure8", "params": {"center": [0, 0, 4.0], "size": 3.0, "period": 7.0}}},
        "hover_step": {**base, "trajectory": {"type": "hover", "params": {"pos": [3, 2, 4.0]}}},
    }


def _disturbance_scenarios() -> Dict[str, Dict[str, Any]]:
    traj = {"type": "circle", "params": {"center": [0, 0, 4.0], "radius": 2.5, "omega": 0.7}}
    return {
        "steady_wind": {"duration": 8.0, "trajectory": traj, "wind": {"type": "constant", "velocity": [5.0, 2.0, 0.0]}},
        "payload_drop": {"duration": 8.0, "trajectory": traj, "faults": [{"type": "payload_change", "t_start": 3.0, "params": {"mass": 0.4}}]},
        "turbulence": {"duration": 8.0, "trajectory": traj, "wind": {"type": "dryden", "w20": 6.0}},
    }


SUITES: Dict[str, BenchmarkSuite] = {
    "tracking": BenchmarkSuite(
        "tracking",
        _tracking_scenarios(),
        metric_params={"skip": 1.0},
        description="nominal trajectory tracking: circle, figure-eight and a hover step",
    ),
    "disturbance": BenchmarkSuite(
        "disturbance", _disturbance_scenarios(), metric_params={"skip": 2.0}, description="rejection of steady wind, a payload change and turbulence"
    ),
}


def get_suite(name: str) -> BenchmarkSuite:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; available: {sorted(SUITES)}")
    return SUITES[name]
