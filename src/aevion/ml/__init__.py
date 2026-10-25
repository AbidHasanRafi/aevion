"""Learning tools: datasets, imitation (behaviour cloning, DAgger), policies and their export, curricula, experiment tracking, tuning, parallel rollouts.

Everything that needs PyTorch (policies, imitation) is imported on first use, so ``import aevion.ml`` works without it.
"""

from __future__ import annotations

import importlib
from typing import Any

from .curriculum import Curriculum
from .dataset import Dataset, Normalizer, collect
from .tracking import CsvLogger, JsonlLogger, Logger, LoggingCallback, MLflowLogger, MultiLogger, TensorBoardLogger, WandbLogger, read_jsonl

_LAZY = {
    "MLPPolicy": "policies",
    "load_policy": "policies",
    "export_onnx": "policies",
    "export_torchscript": "policies",
    "verify_onnx": "policies",
    "behavior_cloning": "imitation",
    "dagger": "imitation",
    "DaggerResult": "imitation",
    "BCResult": "imitation",
    "evaluate_policy_rollouts": "imitation",
    "optuna_search": "tuning",
    "TuneResult": "tuning",
    "parallel_collect": "rollouts",
}

__all__ = [
    "Curriculum",
    "Dataset",
    "Normalizer",
    "collect",
    "Logger",
    "JsonlLogger",
    "CsvLogger",
    "TensorBoardLogger",
    "MLflowLogger",
    "WandbLogger",
    "MultiLogger",
    "LoggingCallback",
    "read_jsonl",
    *_LAZY,
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
