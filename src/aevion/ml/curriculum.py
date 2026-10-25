"""Curriculum learning: raise a task's difficulty when the agent masters the current level.

A :class:`Curriculum` is a list of stages (a level in ``[0, 1]`` passed to the task's ``set_difficulty``, or a dictionary of task attributes to set). It tracks
the recent success of finished episodes and promotes the agent to the next stage once the success rate over a window of ``window`` episodes reaches
``promote_at`` (and demotes it below ``demote_at``, if given). It never advances faster than one stage per window, so a lucky streak cannot skip levels.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["Curriculum"]

Stage = Union[float, Dict[str, Any]]


class Curriculum:
    def __init__(self, stages: Sequence[Stage], promote_at: float = 0.8, window: int = 50, demote_at: Optional[float] = None, start: int = 0) -> None:
        if len(stages) < 1:
            raise ValueError("a curriculum needs at least one stage")
        if not 0.0 < promote_at <= 1.0 or window < 1:
            raise ValueError("promote_at must be in (0, 1] and window at least 1")
        if demote_at is not None and not 0.0 <= demote_at < promote_at:
            raise ValueError("demote_at must be below promote_at")
        self.stages: List[Stage] = list(stages)
        self.promote_at, self.window, self.demote_at = promote_at, int(window), demote_at
        self.level = int(np.clip(start, 0, len(self.stages) - 1))
        self._recent: Deque[float] = deque(maxlen=self.window)
        self.history: List[Dict[str, float]] = []
        self.episodes = 0

    @property
    def stage(self) -> Stage:
        return self.stages[self.level]

    @property
    def finished(self) -> bool:
        return self.level == len(self.stages) - 1

    def success_rate(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0

    def apply(self, env_or_task: Any) -> None:
        """Set the current stage on a task (or on the ``task`` of an environment)."""
        task = getattr(env_or_task, "task", env_or_task)
        st = self.stage
        if isinstance(st, dict):
            for k, v in st.items():
                if not hasattr(task, k):
                    raise AttributeError(f"task {getattr(task, 'name', task)!r} has no attribute {k!r}")
                setattr(task, k, v)
        else:
            if not hasattr(task, "set_difficulty"):
                raise AttributeError(f"task {getattr(task, 'name', task)!r} does not support set_difficulty()")
            task.set_difficulty(float(st))

    def report(self, successes: Any) -> bool:
        """Record finished episodes (booleans); returns ``True`` if the stage changed (call :meth:`apply` again then)."""
        for s in np.atleast_1d(np.asarray(successes, float)).ravel():
            self._recent.append(float(s))
            self.episodes += 1
        if len(self._recent) < self.window:
            return False
        rate = self.success_rate()
        old = self.level
        if rate >= self.promote_at and not self.finished:
            self.level += 1
        elif self.demote_at is not None and rate < self.demote_at and self.level > 0:
            self.level -= 1
        if self.level != old:
            self.history.append(dict(episodes=self.episodes, from_level=old, to_level=self.level, success_rate=rate))
            self._recent.clear()  # judge the new stage on its own episodes
            return True
        return False
