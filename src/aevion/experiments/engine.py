"""Experiment engine: methods × conditions × seeds with randomisation, statistics and export.

An :class:`Experiment` is a grid of *trials*. Each trial is a :class:`~aevion.scenario.Scenario`
built from a base scenario plus overrides from one *method* (e.g. a controller) and one
*condition* (e.g. a wind level), a per-trial random draw of plant parameters, and a seed. Trials
are independent, so they run in parallel across processes.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..analysis import stats as ST
from ..scenario import Scenario
from ..vehicles.spec import VehicleSpec, get_vehicle

# per-trial randomisation keys (relative scale factors unless noted)
VEHICLE_SCALES = {"mass": "mass", "kT": "kT", "kQ": "kQ", "motor_tau": "motor_tau", "w_max": "w_max", "rotor_radius": "rotor_radius"}


def _deep_update(d: dict, upd: dict) -> dict:
    out = copy.deepcopy(d)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_randomization(sc: dict, rnd: Dict[str, tuple], rng: np.random.Generator) -> dict:
    """Return a scenario dict with randomised vehicle parameters, wind and initial offset."""
    sc = copy.deepcopy(sc)
    veh = sc["vehicle"]
    spec = (VehicleSpec.from_dict(veh) if isinstance(veh, dict) else get_vehicle(veh)).to_dict()
    drawn = {}
    for key, (lo, hi) in rnd.items():
        v = float(rng.uniform(lo, hi))
        drawn[key] = v
        if key in VEHICLE_SCALES:
            spec[VEHICLE_SCALES[key]] = spec[VEHICLE_SCALES[key]] * v
        elif key == "inertia":
            spec["inertia"] = [x * v for x in np.asarray(spec["inertia"], float).ravel()]
        elif key == "drag":
            spec["drag_lin"] = [x * v for x in spec["drag_lin"]]
            spec["drag_quad"] = [x * v for x in spec["drag_quad"]]
        elif key == "wind_speed":
            th = rng.uniform(0, 2 * np.pi)
            models: List[Dict[str, Any]] = [dict(type="ConstantWind", velocity=[v * math.cos(th), v * math.sin(th), 0.0])]
            if sc.get("wind"):
                models.append(sc["wind"])
            sc["wind"] = dict(type="CompositeWind", models=models)
        elif key == "turbulence":
            models = [dict(type="DrydenTurbulence", w20=max(v, 0.1), severity=1.0)]
            if sc.get("wind"):
                models.append(sc["wind"])
            sc["wind"] = dict(type="CompositeWind", models=models)
        elif key == "start_offset":
            p0 = np.asarray(sc.get("initial", {}).get("pos") or [0, 0, 0], float)
            sc.setdefault("initial", {})
            sc["initial"]["pos"] = (p0 + rng.uniform(-v, v, 3) * np.array([1, 1, 0.5])).tolist()
        else:
            raise ValueError(f"unknown randomization key {key!r}")
    sc["vehicle"] = spec
    sc["_randomization"] = drawn
    return sc


def _say(text: str) -> None:
    """Progress output goes to stderr so stdout stays clean for results."""
    sys.stderr.write(text)
    sys.stderr.flush()


def _run_trial(args):
    """Top-level (picklable) worker: run one scenario and return its metrics."""

    sc_dict, tags, metric_fn_name, keep_archive_dir = args
    from ..metrics import aggregate, flight_metrics
    from ..scenario import Scenario as _S

    drawn = sc_dict.pop("_randomization", {})
    skip = sc_dict.pop("_skip", 0.0)
    t0 = time.perf_counter()
    arch = _S.from_dict(sc_dict).run()
    m = aggregate(flight_metrics(arch, skip=skip))
    row = dict(tags)
    row.update({f"rnd_{k}": v for k, v in drawn.items()})
    row.update(m)
    row["wall_s"] = time.perf_counter() - t0
    if keep_archive_dir:
        Path(keep_archive_dir).mkdir(parents=True, exist_ok=True)
        arch.save(Path(keep_archive_dir) / f"{tags['method']}__{tags['condition']}__s{tags['seed']}.aev")
    return row


@dataclass
class ExperimentResult:
    name: str
    rows: List[dict]
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- views
    @property
    def methods(self) -> list:
        return list(dict.fromkeys(r["method"] for r in self.rows))

    @property
    def conditions(self) -> list:
        return list(dict.fromkeys(r["condition"] for r in self.rows))

    def metric_names(self) -> list:
        skip = {"method", "condition", "seed"}
        return [k for k in self.rows[0] if k not in skip and not k.startswith("rnd_") and isinstance(self.rows[0][k], (int, float))]

    def values(self, metric: str, method: Optional[str] = None, condition: Optional[str] = None) -> np.ndarray:
        return np.array(
            [
                r[metric]
                for r in self.rows
                if (method is None or r["method"] == method) and (condition is None or r["condition"] == condition) and metric in r
            ],
            float,
        )

    def summary(self, metric: str, level: float = 0.95) -> list:
        out = []
        for m in self.methods:
            for c in self.conditions:
                v = self.values(metric, m, c)
                if len(v) == 0:
                    continue
                mean, lo, hi = ST.mean_ci(v, level)
                out.append(
                    dict(
                        method=m,
                        condition=c,
                        n=len(v),
                        mean=mean,
                        std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        ci_lo=lo,
                        ci_hi=hi,
                        median=float(np.median(v)),
                        min=float(v.min()),
                        max=float(v.max()),
                    )
                )
        return out

    def compare(self, metric: str, condition: Optional[str] = None, baseline: Optional[str] = None) -> list:
        conds = [condition] if condition else self.conditions
        rows = []
        for c in conds:
            samples = {m: self.values(metric, m, c) for m in self.methods}
            for r in ST.pairwise_table(samples, baseline):
                rows.append(dict(condition=c, **r))
        return rows

    def table_text(self, metric: str, fmt: str = "{:.4g}") -> str:
        s = self.summary(metric)
        w = max(len(r["method"]) for r in s)
        lines = [f"{metric}  (mean [95% CI], n)"]
        for r in s:
            lines.append(
                f"  {r['method']:<{w}} | {r['condition']:<14} {fmt.format(r['mean'])} [{fmt.format(r['ci_lo'])}, {fmt.format(r['ci_hi'])}]  n={r['n']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------- export
    def to_csv(self, path):
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(self.rows)

    def to_latex(self, metric: str, caption: str = "", label: str = "", fmt: str = "{:.3g}", scale: float = 1.0) -> str:
        """Booktabs table (LaTeX): rows = methods, columns = conditions, cells = mean ± std."""
        conds = self.conditions
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{caption or metric}}}",
            f"\\label{{{label or 'tab:' + metric}}}",
            r"\begin{tabular}{l" + "c" * len(conds) + "}",
            r"\toprule",
            "Method & " + " & ".join(c.replace("_", r"\_") for c in conds) + r" \\",
            r"\midrule",
        ]
        for m in self.methods:
            cells = []
            for c in conds:
                v = self.values(metric, m, c) * scale
                cells.append("--" if len(v) == 0 else f"{fmt.format(v.mean())} $\\pm$ {fmt.format(v.std(ddof=1) if len(v) > 1 else 0.0)}")
            lines.append(m.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def save(self, path):
        Path(path).write_text(json.dumps(dict(name=self.name, meta=self.meta, rows=self.rows), default=float), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["name"], d["rows"], d.get("meta", {}))


class Experiment:
    """Grid experiment.

    Parameters
    ----------
    base : Scenario used as the template.
    methods : ``{name: overrides}``; overrides are scenario-dict fragments, e.g. ``{"controller": "cascaded"}``.
    conditions : ``{name: overrides}``, e.g. ``{"calm": {}, "gusty": {"wind": {...}}}``.
    seeds : seeds (one trial per method × condition × seed).
    randomization : ``{"mass": (0.8, 1.2), "wind_speed": (0, 6), ...}``, drawn per trial from the seed.
    """

    def __init__(
        self,
        name: str,
        base: Scenario,
        methods: Dict[str, dict],
        conditions: Optional[Dict[str, dict]] = None,
        seeds: Sequence[int] = range(5),
        randomization: Optional[Dict[str, tuple]] = None,
        skip: float = 0.0,
    ):
        self.name, self.base = name, base
        self.methods = methods
        self.conditions = conditions or {"nominal": {}}
        self.seeds = list(seeds)
        self.randomization = randomization or {}
        self.skip = skip

    def trials(self):
        for m, mo in self.methods.items():
            for c, co in self.conditions.items():
                for s in self.seeds:
                    sc = _deep_update(self.base.to_dict(), _deep_update(co, {}))
                    sc = _deep_update(sc, mo)
                    sc["seed"] = int(s)
                    rng = np.random.default_rng(1_000_003 * int(s) + 17)  # same draw for every method: paired design
                    if self.randomization:
                        sc = apply_randomization(sc, self.randomization, rng)
                    sc["_skip"] = self.skip
                    yield sc, dict(method=m, condition=c, seed=int(s))

    def n_trials(self) -> int:
        return len(self.methods) * len(self.conditions) * len(self.seeds)

    def run(self, workers: int = 1, progress: bool = True, archive_dir: Optional[str] = None) -> ExperimentResult:
        jobs = [(sc, tags, "default", archive_dir) for sc, tags in self.trials()]
        t0 = time.perf_counter()
        rows = []
        if workers <= 1:
            for i, j in enumerate(jobs):
                rows.append(_run_trial(j))
                if progress:
                    _say(f"\r  [{i + 1}/{len(jobs)}] {rows[-1]['method']} / {rows[-1]['condition']} / seed {rows[-1]['seed']}      ")
        else:
            # one numba thread per worker process; restored afterwards, because numba re-reads its configuration when the
            # environment changes and refuses a different thread count once this process has launched its own threads
            leaked = "NUMBA_NUM_THREADS" not in os.environ
            os.environ.setdefault("NUMBA_NUM_THREADS", "1")
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for i, row in enumerate(ex.map(_run_trial, jobs, chunksize=1)):
                        rows.append(row)
                        if progress:
                            _say(f"\r  [{i + 1}/{len(jobs)}]      ")
            finally:
                if leaked:
                    os.environ.pop("NUMBA_NUM_THREADS", None)
        if progress:
            _say(f"\n  {len(jobs)} trials in {time.perf_counter() - t0:.1f}s\n")
        meta = dict(
            base=self.base.to_dict(),
            methods=list(self.methods),
            conditions=list(self.conditions),
            seeds=self.seeds,
            randomization={k: list(v) for k, v in self.randomization.items()},
            wall_s=time.perf_counter() - t0,
            workers=workers,
        )
        return ExperimentResult(self.name, rows, meta)
