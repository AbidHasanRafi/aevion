"""An auto-generated methods appendix: what was simulated, how the experiment was designed, which statistics were used, and how to reproduce it.

The text is assembled from the *actual* configuration and experiment record (not from a template with blanks), so the numbers in a paper's methods section
are the numbers that were run. Output is Markdown or LaTeX; the wording is plain and factual, meant as a starting point to edit rather than final prose.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__
from .templates import latex_escape

__all__ = ["methods_appendix", "MethodsAppendix", "METRIC_GLOSSARY"]

METRIC_GLOSSARY = {
    "rmse_pos": "root-mean-square 3-D position error against the reference",
    "rmse_xy": "root-mean-square horizontal position error",
    "rmse_z": "root-mean-square vertical position error",
    "rmse_steady": "position RMSE over the second half of the run",
    "mean_err": "mean 3-D position error",
    "max_err": "maximum 3-D position error",
    "final_err": "position error at the end of the run",
    "path_length": "length of the flown path",
    "max_tilt_deg": "largest roll or pitch angle",
    "max_rate_deg_s": "largest body angular rate",
    "max_acc": "largest acceleration magnitude",
    "rms_jerk": "root-mean-square jerk",
    "energy_wh": "electrical energy drawn from the battery",
    "mean_power_w": "mean electrical power",
    "soc_end": "battery state of charge at the end",
    "rotor_saturation_frac": "fraction of time any rotor was within 2 % of its speed limit",
    "thrust_margin": "unused fraction of the maximum thrust at the peak demand",
    "min_separation": "smallest distance between any two vehicles",
    "min_clearance": "smallest distance to an obstacle",
    "collisions": "number of collision events",
    "crashed": "1 if the vehicle was on the ground after the first second",
}


@dataclass
class MethodsAppendix:
    sections: List[Tuple[str, str]]

    def markdown(self) -> str:
        return "\n\n".join(f"## {t}\n\n{b}" for t, b in self.sections) + "\n"

    def latex(self) -> str:
        out = []
        for t, b in self.sections:
            paras = [latex_escape(p).replace("\n- ", "\n\\item ") for p in b.split("\n\n")]
            body = []
            for p in paras:
                if "\\item " in p:
                    head, *items = p.split("\n\\item ")
                    body.append(head + "\n\\begin{itemize}\n" + "\n".join("\\item " + i for i in items) + "\n\\end{itemize}")
                else:
                    body.append(p)
            out.append(f"\\subsection*{{{latex_escape(t)}}}\n" + "\n\n".join(body))
        return "\n\n".join(out) + "\n"

    def text(self, fmt: str = "markdown") -> str:
        if fmt not in ("markdown", "latex"):
            raise ValueError("fmt must be 'markdown' or 'latex'")
        return self.markdown() if fmt == "markdown" else self.latex()


def _comp(c: Optional[Dict[str, Any]]) -> str:
    if not c:
        return "none"
    params = ", ".join(f"{k} = {v}" for k, v in (c.get("params") or {}).items())
    extra = {k: v for k, v in c.items() if k not in ("type", "params") and v not in (None, False, "", [], {})}
    bits = [str(c.get("type"))] + ([f"({params})"] if params else []) + ([f"[{', '.join(f'{k}={v}' for k, v in extra.items())}]"] if extra else [])
    return " ".join(bits)


def _simulation_section(cfg: Dict[str, Any]) -> str:
    from ..vehicles.spec import VehicleSpec, get_vehicle

    eng = cfg["engine"]
    veh = cfg["vehicle"]
    try:
        spec = get_vehicle(veh["model"]) if isinstance(veh["model"], str) else VehicleSpec.from_dict(veh["model"])
        over = veh.get("overrides") or {}
        vtxt = f"vehicle `{veh['model']}` ({spec.n_rotors} rotors, mass {over.get('mass', spec.mass):g} kg, maximum thrust {spec.max_thrust():.1f} N)"
    except Exception:  # an unregistered model name: describe what the configuration says
        vtxt = f"vehicle `{veh['model']}`"
    lines = [
        f"Simulations used Aevion {__version__} (Python {platform.python_version()}, NumPy {np.__version__}). The rigid-body dynamics (Newton–Euler on unit quaternions with first-order rotor lag, "
        f"rotor gyroscopics, drag and ground effect) were integrated with `{eng['integrator']}` at a fixed step of {eng['dt'] * 1000:g} ms for {cfg['duration']:g} s per run.",
        f"The run used {vtxt}" + (f", {veh['count']} vehicles" if veh.get("count", 1) > 1 else "") + ".",
        "- trajectory: " + _comp(cfg.get("trajectory")),
        "- controller: " + _comp(cfg.get("controller")),
        "- estimator: " + (_comp(cfg.get("estimator")) if cfg.get("estimator") else "none (the controller used the true state)"),
        "- wind: " + _comp(cfg.get("wind")),
        "- world: " + _comp(cfg.get("world")),
    ]
    sens = cfg.get("sensors") or []
    lines.append("- sensors: " + (", ".join(_comp(s) for s in sens) if sens else "none"))
    faults = cfg.get("faults") or []
    if faults:
        lines.append("- faults: " + "; ".join(f"{_comp(f)} from t = {f.get('t_start', 0):g} s" for f in faults))
    if cfg.get("swarm"):
        sw = cfg["swarm"]
        lines.append(
            f"- inter-agent link: loss {sw['comm']['loss']:g}, delay {sw['comm']['delay_s']:g} s, laws: "
            + (", ".join(_comp(x) for x in sw.get("laws", [])) or "collision avoidance only")
        )
    return "\n".join(lines[:2]) + "\n\n" + "\n".join(lines[2:])


def methods_appendix(
    config: Optional[Dict[str, Any]] = None,
    result: Any = None,
    metrics: Optional[Sequence[str]] = None,
    experiment: Any = None,
) -> MethodsAppendix:
    """Describe a study.

    ``config`` is a simulation configuration dictionary (defaults are filled in from the schema); ``result`` an :class:`~aevion.experiments.ExperimentResult` (the trial counts,
    methods, conditions, seeds and any cache statistics come from it); ``experiment`` a :class:`~aevion.experiments.ConfigExperiment` (its base configuration is used when
    ``config`` is omitted). ``metrics`` limits the glossary to the metrics the paper reports.
    """
    from ..config import SimulationConfig
    from ..experiments.cache import trial_key

    sections: List[Tuple[str, str]] = []
    if config is None and experiment is not None:
        config = experiment.base
    cfg_full: Optional[Dict[str, Any]] = None
    if config is not None:
        cfg_full = SimulationConfig.from_dict(config).to_dict()
        sections.append(("Simulation", _simulation_section(cfg_full)))
    meta: Dict[str, Any] = dict(result.meta) if result is not None else {}
    if experiment is not None:
        meta.setdefault("methods", list(experiment.methods))
        meta.setdefault("conditions", list(experiment.conditions))
        meta.setdefault("seeds", experiment.seeds)
        meta.setdefault("metric", experiment.metric)
    if result is not None or experiment is not None:
        methods, conds, seeds = meta.get("methods", []), meta.get("conditions", []), meta.get("seeds", [])
        n = meta.get("trials") or (len(methods) * len(conds) * len(seeds))
        txt = (
            f"The experiment crossed {len(methods)} method(s) ({', '.join(f'`{m}`' for m in methods)}) with {len(conds)} condition(s) ({', '.join(f'`{c}`' for c in conds)}) "
            f"and {len(seeds)} random seed(s), giving {n} simulation runs. "
        )
        txt += "Every method saw the same seeds (a paired design), and a run is deterministic given its configuration and seed, so any run can be repeated bit for bit."
        if meta.get("failed"):
            txt += f" {meta['failed']} run(s) failed and are excluded from the statistics."
        sections.append(("Experimental design", txt))
    names = list(metrics) if metrics is not None else (result.metric_names() if result is not None else [])
    known = [m for m in names if m in METRIC_GLOSSARY]
    if known:
        sections.append(("Metrics", "\n".join(f"- `{m}`: {METRIC_GLOSSARY[m]}" for m in known)))
    sections.append(
        (
            "Statistical analysis",
            "Results are reported as the mean over seeds with a 95 % Student-t confidence interval. Methods are compared pairwise with Welch's t-test and the Mann–Whitney U test; "
            "p-values are adjusted for multiple comparisons with the Holm–Bonferroni procedure, and effect sizes are given as Cohen's d and Cliff's delta. "
            "A difference is called significant when the adjusted p-value is below 0.05; with few seeds the tests have little power, and absence of significance is not evidence of equivalence.",
        )
    )
    repro = [f"Aevion {__version__}, Python {platform.python_version()} on {platform.system()}, NumPy {np.__version__}."]
    if cfg_full is not None:
        repro.append("The full configuration is part of every run archive; its content hash is `" + trial_key(cfg_full)[:16] + "`.")
    repro.append(
        "Trial results are cached by a hash of the complete configuration, metric and Aevion version, so a changed setting is never served a stale result."
    )
    sections.append(("Reproducibility", " ".join(repro)))
    return MethodsAppendix(sections)
