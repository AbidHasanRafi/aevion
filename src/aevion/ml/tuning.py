"""Hyper-parameter search with Optuna.

:func:`optuna_search` takes a plain search space (``{"kp": ("float", 0.5, 5.0), "lr": ("float", 1e-4, 1e-1, "log"), ...}``) and an objective
``f(params) -> float``, so the same function can tune controller gains, reward weights or learning rates. The study is seeded, so a run repeats; a
:class:`~aevion.ml.tracking.Logger` receives every trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["optuna_search", "TuneResult"]


@dataclass
class TuneResult:
    best_params: Dict[str, Any]
    best_value: float
    trials: List[Dict[str, Any]] = field(default_factory=list)
    study: Any = None


def _suggest(trial: Any, name: str, spec: Sequence[Any]) -> Any:
    kind = spec[0]
    if kind == "float":
        return trial.suggest_float(name, float(spec[1]), float(spec[2]), log=len(spec) > 3 and spec[3] == "log")
    if kind == "int":
        return trial.suggest_int(name, int(spec[1]), int(spec[2]), log=len(spec) > 3 and spec[3] == "log")
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec[1]))
    raise ValueError(f"unknown parameter kind {kind!r} for '{name}' (use float, int or categorical)")


def optuna_search(
    objective: Callable[[Dict[str, Any]], float],
    space: Dict[str, Sequence[Any]],
    n_trials: int = 20,
    direction: str = "minimize",
    seed: int = 0,
    sampler: str = "tpe",
    logger: Optional[Any] = None,
    timeout: Optional[float] = None,
) -> TuneResult:
    """Search ``space`` for the parameters that optimise ``objective``; trials that raise are recorded as failed and skipped."""
    try:
        import optuna
    except ImportError as ex:
        raise ImportError("optuna_search needs Optuna: pip install optuna") from ex
    if direction not in ("minimize", "maximize"):
        raise ValueError("direction must be 'minimize' or 'maximize'")
    if not space:
        raise ValueError("the search space is empty")
    for name, spec in space.items():  # reject a malformed space before any trial runs
        if spec[0] not in ("float", "int", "categorical") or len(spec) < 2:
            raise ValueError(f"unknown parameter kind {spec[0]!r} for '{name}' (use float, int or categorical)")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    samplers = {"tpe": optuna.samplers.TPESampler(seed=seed), "random": optuna.samplers.RandomSampler(seed=seed)}
    if sampler not in samplers:
        raise ValueError(f"sampler must be one of {sorted(samplers)}")
    study = optuna.create_study(direction=direction, sampler=samplers[sampler])
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    def obj(trial: Any) -> float:
        params = {k: _suggest(trial, k, v) for k, v in space.items()}
        try:
            value = float(objective(params))
        except (ArithmeticError, ValueError, RuntimeError) as ex:
            errors.append(f"{type(ex).__name__}: {ex}")
            raise
        rows.append(dict(number=trial.number, value=value, **params))
        if logger is not None:
            logger.log(trial.number, objective=value, **{k: v for k, v in params.items() if isinstance(v, (int, float))})
        return value

    study.optimize(obj, n_trials=n_trials, timeout=timeout, catch=(ArithmeticError, ValueError, RuntimeError))
    if not rows:
        raise RuntimeError(f"every trial failed; the last error was {errors[-1] if errors else 'unknown'}")
    return TuneResult(dict(study.best_params), float(study.best_value), rows, study)
