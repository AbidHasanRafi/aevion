"""Experiments over full simulation configurations, with caching, sharding and a result database.

A :class:`ConfigExperiment` is a grid of trials ``method × condition × seed``. Each trial is a complete ``Simulation`` configuration:
the base configuration deep-merged with the condition's overrides, then the method's, plus the seed. Anything a configuration can
express (controller, estimator, swarm law, faults, wind, sensors, hooks) can therefore be a factor, unlike the original
:class:`~aevion.experiments.Experiment`, which is kept unchanged.

Trials run on a :class:`~aevion.experiments.runners.Runner`, are looked up in and stored to a
:class:`~aevion.experiments.cache.ResultCache` as they finish, and can be written to a :class:`~aevion.experiments.database.ResultDB`.
The metric is a registered name (``metrics`` registry), so the experiment description is plain data that can be saved to JSON,
shipped to a cluster node and loaded without executing anything from the file.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__
from .cache import ResultCache, trial_key
from .design import ParameterSpace
from .engine import ExperimentResult
from .runners import Runner, SerialRunner, shard_indices

__all__ = ["ConfigExperiment", "Trial", "run_config_trial", "deep_merge", "nest", "get_path"]


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """``base`` updated with ``over``: nested dictionaries merge, everything else (lists included) is replaced.

    A component block (one with a ``type``) whose type *differs* from the base's is replaced as a whole: the parameters of a circle mean nothing to a planned
    trajectory, so merging them would only produce an invalid configuration.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not ("type" in v and "type" in out[k] and v["type"] != out[k]["type"]):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def nest(assignments: Dict[str, Any]) -> Dict[str, Any]:
    """``{"controller.params.wn_pos": 5}`` -> ``{"controller": {"params": {"wn_pos": 5}}}`` (dictionaries only; lists cannot be addressed)."""
    out: Dict[str, Any] = {}
    for path, v in assignments.items():
        node = out
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"path {path!r} runs into a value that is not a dictionary")
        node[parts[-1]] = v
    return out


def get_path(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in path.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


@dataclass
class Trial:
    method: str
    condition: str
    seed: int
    config: Dict[str, Any]
    key: str

    def tags(self) -> Dict[str, Any]:
        return dict(method=self.method, condition=self.condition, seed=self.seed)


def run_config_trial(job: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level (picklable) worker: simulate one configuration and reduce it to a row of metrics; a failing trial becomes a ``failed`` row, not an exception."""
    from ..engine import Simulation
    from ..registry import registries

    row: Dict[str, Any] = dict(key=job["key"], **job["tags"])
    t0 = time.perf_counter()
    try:
        arch = Simulation(job["config"]).run()
        m = registries.metrics.create(job["metric"], arch, **job["metric_params"])
        row.update({k: float(v) for k, v in m.items()})
        row["status"] = "ok"
    except Exception as ex:  # a diverged or invalid trial must not abort the other thousand
        row.update(status="failed", error=f"{type(ex).__name__}: {ex}")
    row["wall_s"] = time.perf_counter() - t0
    return row


def _say(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


class ConfigExperiment:
    """Grid ``methods × conditions × seeds`` of simulation configurations.

    Parameters
    ----------
    base: a ``Simulation`` configuration dictionary (anything ``Simulation(...)`` accepts).
    methods: ``{name: overrides}`` - e.g. ``{"geometric": {"controller": "geometric"}, "mpc": {"controller": {"type": "mpc"}}}``.
    conditions: ``{name: overrides}`` - e.g. ``{"calm": {}, "gusty": {"wind": {"type": "dryden", "w20": 6}}}``.
    seeds: one trial per seed (the seed is written to the configuration's ``seed``).
    metric / metric_params: a name from the ``metrics`` registry (``"flight"``) and its keyword arguments (``{"skip": 1.0}``).
    """

    def __init__(
        self,
        name: str,
        base: Dict[str, Any],
        methods: Optional[Dict[str, Dict[str, Any]]] = None,
        conditions: Optional[Dict[str, Dict[str, Any]]] = None,
        seeds: Sequence[int] = (0, 1, 2),
        metric: str = "flight",
        metric_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(base, dict):
            raise TypeError("base must be a configuration dictionary")
        self.name, self.base = name, copy.deepcopy(base)
        self.methods = methods or {"default": {}}
        self.conditions = conditions or {"nominal": {}}
        self.seeds = [int(s) for s in seeds]
        self.metric, self.metric_params = metric, dict(metric_params or {})
        if not self.seeds:
            raise ValueError("an experiment needs at least one seed")

    # ---------------------------------------------------------------------------------------------- trials
    def trials(self) -> List[Trial]:
        out: List[Trial] = []
        for m, mo in self.methods.items():
            for c, co in self.conditions.items():
                for s in self.seeds:
                    cfg = deep_merge(deep_merge(self.base, co), mo)
                    cfg["seed"] = s
                    key = trial_key(dict(config=cfg, metric=self.metric, metric_params=self.metric_params), salt=__version__)
                    out.append(Trial(m, c, s, cfg, key))
        return out

    def n_trials(self) -> int:
        return len(self.methods) * len(self.conditions) * len(self.seeds)

    def validate(self) -> None:
        """Check one configuration per method × condition against the schema and the metric name, so mistakes surface before any trial runs."""
        from ..config import SimulationConfig
        from ..registry import registries

        if self.metric not in registries.metrics:
            raise ValueError(f"unknown metric {self.metric!r}; registered: {registries.metrics.names()}")
        seen = set()
        for t in self.trials():
            if (t.method, t.condition) in seen:
                continue
            seen.add((t.method, t.condition))
            try:
                SimulationConfig.from_dict(copy.deepcopy(t.config))
            except Exception as ex:
                raise ValueError(f"method {t.method!r} / condition {t.condition!r}: {ex}") from ex

    # ---------------------------------------------------------------------------------------------- running
    def run(
        self,
        runner: Optional[Runner] = None,
        cache: Optional[ResultCache] = None,
        db: Any = None,
        shard: Optional[Tuple[int, int]] = None,
        progress: bool = False,
        validate: bool = True,
        raise_on_error: bool = False,
    ) -> ExperimentResult:
        """Run (or fetch from the cache) every trial and return the rows in trial order.

        ``shard=(k, n)`` restricts the run to trials ``i`` with ``i % n == k`` (for cluster arrays). Failed trials are returned with ``status="failed"`` and an
        ``error`` message and are *not* cached, so a re-run retries them.
        """
        if validate:
            self.validate()
        runner = runner or SerialRunner()
        t0 = time.perf_counter()
        trials = self.trials()
        idx = list(range(len(trials))) if shard is None else shard_indices(len(trials), shard[0], shard[1])
        rows: Dict[int, Dict[str, Any]] = {}
        pending: List[int] = []
        for i in idx:
            hit = cache.get(trials[i].key) if cache is not None else None
            if hit is not None:
                rows[i] = hit
            else:
                pending.append(i)
        n_cached = len(rows)
        jobs = [
            dict(config=trials[i].config, tags=trials[i].tags(), key=trials[i].key, metric=self.metric, metric_params=self.metric_params)
            for i in pending
        ]
        for done, (j, row) in enumerate(runner.imap(run_config_trial, jobs), start=1):
            i = pending[j]
            rows[i] = row
            if row["status"] == "ok" and cache is not None:
                cache.put(trials[i].key, row)
            if progress:
                _say(f"\r  [{done + n_cached}/{len(idx)}] {row['method']} / {row['condition']} / seed {row['seed']}      ")
            if raise_on_error and row["status"] != "ok":
                raise RuntimeError(f"trial {row['method']}/{row['condition']}/seed {row['seed']} failed: {row.get('error')}")
        if progress:
            _say("\n")
        ordered = [rows[i] for i in idx]
        failed = sum(1 for r in ordered if r["status"] != "ok")
        meta = dict(
            name=self.name,
            base=self.base,
            methods=list(self.methods),
            conditions=list(self.conditions),
            seeds=self.seeds,
            metric=self.metric,
            metric_params=self.metric_params,
            version=__version__,
            runner=getattr(runner, "name", "custom"),
            trials=len(ordered),
            cached=n_cached,
            computed=len(pending),
            failed=failed,
            shard=list(shard) if shard else None,
            wall_s=time.perf_counter() - t0,
        )
        result = ExperimentResult(self.name, ordered, meta)
        if db is not None:
            db.add_result(result)
        return result

    # ---------------------------------------------------------------------------------------------- (de)serialisation
    def to_spec(self) -> Dict[str, Any]:
        return dict(
            aevion_experiment=1,
            name=self.name,
            base=self.base,
            methods=self.methods,
            conditions=self.conditions,
            seeds=self.seeds,
            metric=self.metric,
            metric_params=self.metric_params,
        )

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "ConfigExperiment":
        if spec.get("aevion_experiment") != 1:
            raise ValueError("not an Aevion experiment specification")
        return cls(
            spec["name"],
            spec["base"],
            spec.get("methods"),
            spec.get("conditions"),
            spec.get("seeds", (0,)),
            spec.get("metric", "flight"),
            spec.get("metric_params"),
        )

    def save_spec(self, path: Any) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_spec(), indent=1), encoding="utf-8")
        return p

    @classmethod
    def load_spec(cls, path: Any) -> "ConfigExperiment":
        return cls.from_spec(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------------------------------------------------------------------------------------------- designs
    @classmethod
    def from_design(
        cls,
        name: str,
        base: Dict[str, Any],
        space: ParameterSpace,
        paths: Dict[str, str],
        n: int,
        design: str = "sobol",
        design_seed: int = 0,
        methods: Optional[Dict[str, Dict[str, Any]]] = None,
        seeds: Sequence[int] = (0,),
        metric: str = "flight",
        metric_params: Optional[Dict[str, Any]] = None,
    ) -> "ConfigExperiment":
        """One condition per design point: ``paths`` maps each parameter name to the dotted configuration path it sets (``{"wn": "controller.params.wn_pos"}``)."""
        missing = [p for p in space.names if p not in paths]
        if missing:
            raise ValueError(f"no configuration path for parameter(s) {missing}")
        conditions = {}
        for i, pt in enumerate(space.sample(n, design, design_seed)):
            conditions[f"d{i:03d}"] = nest({paths[k]: v for k, v in pt.items()})
        exp = cls(name, base, methods, conditions, seeds, metric, metric_params)
        exp.design_points = dict(zip(conditions, space.sample(n, design, design_seed)))  # type: ignore[attr-defined]
        return exp

    def design_table(self, result: ExperimentResult, metric: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """``(X, y, names)`` of a design experiment: the parameter values of each design point and the mean of ``metric`` over its trials."""
        pts: Dict[str, Dict[str, Any]] = getattr(self, "design_points", {})
        if not pts:
            raise ValueError("this experiment was not built with from_design")
        names = list(next(iter(pts.values())))
        X, y = [], []
        for c, p in pts.items():
            v = result.values(metric, condition=c)
            if len(v):
                X.append([float(p[k]) for k in names])
                y.append(float(v.mean()))
        return np.array(X), np.array(y), names
