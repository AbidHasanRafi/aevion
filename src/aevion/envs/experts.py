"""Privileged scripted experts for the extended tasks (velocity action mode).

An expert may read the simulator state directly (gate normals, the platform's velocity, which surface points are still unseen) while the learner
sees only its observation: the teacher-student set-up that behaviour cloning and DAgger rely on. They are also the performance references used in
the tests and the examples.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .tasks_ext import _np

__all__ = ["make_expert", "EXPERTS"]


def _limit(v: np.ndarray, vmax: float) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, vmax / np.maximum(n, 1e-9))


def _action(env: Any, v: np.ndarray) -> Any:
    a = np.zeros((env.n, 4))
    a[:, :3] = np.clip(v / env.cfg.max_speed, -1.0, 1.0)
    return env.B.asarray(a)


def gate_racing(env: Any, speed: float = 3.0) -> Callable[[Any], Any]:
    task = env.task

    def policy(obs: Any) -> Any:
        pos = _np(env, env.state.pos)
        c, nrm, _ = task._current(env)
        rel = c - pos
        along = (rel * nrm).sum(-1, keepdims=True)
        lateral = rel - along * nrm
        lat = np.linalg.norm(lateral, axis=-1, keepdims=True)
        forward = speed * np.clip(1.0 - lat / (1.5 * task.radius), 0.15, 1.0)  # slow down until lined up with the gate
        v = 3.0 * lateral + forward * nrm * (along > -1.0)
        far = along > 1.5
        v = np.where(far, 2.0 * rel + 0.0 * nrm, v)  # far from the plane: head for the gate; near it: line up and go through along the normal
        return _action(env, _limit(v, speed))

    return policy


def landing(env: Any) -> Callable[[Any], Any]:
    """Follow the platform in the plane and descend along a funnel: the allowed height shrinks with the lateral error, so the commands are continuous (a hard switch between
    'hold' and 'descend' is difficult to imitate)."""

    def policy(obs: Any) -> Any:
        s = env.state
        pos = _np(env, s.pos)
        tgt, tv = _np(env, env.target), _np(env, env.target_vel)
        rel_xy = tgt[:, :2] - pos[:, :2]
        err = np.linalg.norm(rel_xy, axis=1)
        v = np.zeros((env.n, 3))
        v[:, :2] = tv[:, :2] + 2.5 * rel_xy
        z_des = np.maximum(3.0 * err - 0.1, -0.3)
        v[:, 2] = np.clip(1.5 * (z_des - pos[:, 2]), -1.0, 0.8)
        return _action(env, v)

    return policy


def target_tracking(env: Any) -> Callable[[Any], Any]:
    def policy(obs: Any) -> Any:
        pos = _np(env, env.state.pos)
        return _action(env, _np(env, env.target_vel) + 4.0 * (_np(env, env.target) - pos))

    return policy


def inspection(env: Any, speed: float = 2.0) -> Callable[[Any], Any]:
    task = env.task

    def policy(obs: Any) -> Any:
        pos = _np(env, env.state.pos)
        rel = _np(env, env.target) - pos
        v = _limit(2.0 * rel, speed)
        radial = pos[:, :2]
        r = np.linalg.norm(radial, axis=1, keepdims=True)
        keep = task.structure_radius + 0.9
        push = np.where((r < keep) & (pos[:, 2:3] < task.structure_height + 0.5), (keep - r) / np.maximum(r, 1e-6), 0.0)
        v[:, :2] += 3.0 * push * radial
        return _action(env, v)

    return policy


def area_coverage(env: Any, speed: float = 3.0) -> Callable[[Any], Any]:
    def policy(obs: Any) -> Any:
        pos = _np(env, env.state.pos)
        return _action(env, _limit(1.5 * (_np(env, env.target) - pos), speed))

    return policy


EXPERTS = {
    "gate_racing": gate_racing,
    "landing": landing,
    "target_tracking": target_tracking,
    "inspection": inspection,
    "area_coverage": area_coverage,
}


def make_expert(env: Any, **kw: Any) -> Callable[[Any], Any]:
    """The scripted expert for ``env``'s task (the environment must use ``action_mode="velocity"``)."""
    name = env.task.name
    if name not in EXPERTS:
        raise ValueError(f"no scripted expert for task {name!r}; available: {sorted(EXPERTS)}")
    if env.cfg.action_mode != "velocity":
        raise ValueError("the experts command velocity: create the environment with action_mode='velocity'")
    return EXPERTS[name](env, **kw)
