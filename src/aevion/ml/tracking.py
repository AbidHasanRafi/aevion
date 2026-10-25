"""Experiment tracking for training runs: JSONL and CSV always, TensorBoard / MLflow / Weights & Biases when they are installed.

Every logger has the same two methods, ``log(step, **metrics)`` and ``close()``. :class:`JsonlLogger` and :class:`CsvLogger` need nothing; the others
import their library when constructed and raise an ``ImportError`` that says what to install. :class:`MultiLogger` fans one call out to several.
:class:`LoggingCallback` adapts a logger to Stable-Baselines3 training so rollout statistics appear without any extra code.

The TensorBoard, MLflow and W&B adapters are written to the libraries' documented APIs and are tested here against stand-ins for them, **not
against the real packages** (none is installed in the development environment).
"""

from __future__ import annotations

import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Logger", "JsonlLogger", "CsvLogger", "TensorBoardLogger", "MLflowLogger", "WandbLogger", "MultiLogger", "LoggingCallback", "read_jsonl"]


def _scalar(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return v


class Logger:
    def log(self, step: int, **metrics: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class JsonlLogger(Logger):
    """One JSON object per line: ``{"step": 10, "time": ..., "loss": 0.3}``; flushed on every call so a crash loses nothing."""

    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115  (kept open between log calls, closed in close())

    def log(self, step: int, **metrics: Any) -> None:
        row = {"step": int(step), "time": time.time(), **{k: _scalar(v) for k, v in metrics.items()}}
        self._f.write(json.dumps(row, default=float) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


def read_jsonl(path: Any) -> List[Dict[str, Any]]:
    """Read a :class:`JsonlLogger` file (a truncated last line, e.g. after a crash, is ignored)."""
    out: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


class CsvLogger(Logger):
    """CSV with a header; new metric names that appear later extend the header (the file is rewritten at close)."""

    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []

    def log(self, step: int, **metrics: Any) -> None:
        self.rows.append({"step": int(step), **{k: _scalar(v) for k, v in metrics.items()}})
        self._write()

    def _write(self) -> None:
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(self.rows)

    def close(self) -> None:
        self._write()


def _need(module: str, hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as ex:
        raise ImportError(f"{module} is not installed: {hint}") from ex


class TensorBoardLogger(Logger):
    def __init__(self, logdir: Any) -> None:
        mod = None
        for name in ("torch.utils.tensorboard", "tensorboardX"):
            try:
                mod = importlib.import_module(name)
                break
            except ImportError:
                continue
        if mod is None:
            raise ImportError("TensorBoard logging needs the 'tensorboard' package (pip install tensorboard) or 'tensorboardX'")
        self.writer = mod.SummaryWriter(str(logdir))

    def log(self, step: int, **metrics: Any) -> None:
        for k, v in metrics.items():
            v = _scalar(v)
            if isinstance(v, (int, float, bool)):
                self.writer.add_scalar(k, float(v), int(step))

    def close(self) -> None:
        self.writer.close()


class MLflowLogger(Logger):
    def __init__(self, run_name: Optional[str] = None, experiment: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        self.mlflow = _need("mlflow", "pip install mlflow")
        if experiment:
            self.mlflow.set_experiment(experiment)
        self.run = self.mlflow.start_run(run_name=run_name)
        if params:
            self.mlflow.log_params({k: _scalar(v) for k, v in params.items()})

    def log(self, step: int, **metrics: Any) -> None:
        for k, v in metrics.items():
            v = _scalar(v)
            if isinstance(v, (int, float, bool)):
                self.mlflow.log_metric(k, float(v), step=int(step))

    def close(self) -> None:
        self.mlflow.end_run()


class WandbLogger(Logger):
    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, mode: str = "online") -> None:
        self.wandb = _need("wandb", "pip install wandb")
        self.run = self.wandb.init(project=project, name=name, config=config or {}, mode=mode)

    def log(self, step: int, **metrics: Any) -> None:
        self.wandb.log({k: _scalar(v) for k, v in metrics.items()}, step=int(step))

    def close(self) -> None:
        self.wandb.finish()


class MultiLogger(Logger):
    def __init__(self, loggers: Sequence[Logger]) -> None:
        self.loggers = list(loggers)

    def log(self, step: int, **metrics: Any) -> None:
        for lg in self.loggers:
            lg.log(step, **metrics)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()


class LoggingCallback:
    """Stable-Baselines3 callback that logs episode return, length and success to a :class:`Logger` every ``every`` rollouts."""

    def __new__(cls, logger: Logger, every: int = 1) -> Any:  # build a real BaseCallback subclass lazily so SB3 stays optional
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.k = 0
                self._ret: List[float] = []
                self._len: List[float] = []
                self._succ: List[float] = []

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    ep = info.get("episode")
                    if ep:
                        self._ret.append(ep["r"])
                        self._len.append(ep["l"])
                        if "is_success" in info:
                            self._succ.append(float(info["is_success"]))
                return True

            def _on_rollout_end(self) -> None:
                self.k += 1
                if self.k % every == 0 and self._ret:
                    m: Dict[str, Any] = dict(ep_return=float(np.mean(self._ret)), ep_length=float(np.mean(self._len)))
                    if self._succ:
                        m["success_rate"] = float(np.mean(self._succ))
                    logger.log(self.num_timesteps, **m)
                    self._ret, self._len, self._succ = [], [], []

        return _CB()
