"""Clock model for sensors whose timestamps are not perfectly synchronised with the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ClockModel:
    """Reported time ``= (1 + drift_ppm·1e-6) · t_true + offset_s + jitter``.

    ``jitter_std_s`` draws independent Gaussian timestamp jitter per sample (reproducible through ``seed``).
    """

    offset_s: float = 0.0
    drift_ppm: float = 0.0
    jitter_std_s: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def stamp(self, t_true: float) -> float:
        t = (1.0 + self.drift_ppm * 1e-6) * t_true + self.offset_s
        if self.jitter_std_s > 0.0:
            t += float(self._rng.normal(0.0, self.jitter_std_s))
        return t

    def true_time(self, t_reported: float) -> float:
        """Invert the deterministic part of the model (offset and drift)."""
        return (t_reported - self.offset_s) / (1.0 + self.drift_ppm * 1e-6)

    def error_at(self, t_true: float) -> float:
        """Deterministic timestamp error (reported − true) at ``t_true``."""
        return self.drift_ppm * 1e-6 * t_true + self.offset_s

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_any(cls, spec: Optional[dict]) -> Optional["ClockModel"]:
        return None if not spec else cls(**spec)
