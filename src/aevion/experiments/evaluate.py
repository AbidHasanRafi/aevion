"""Evaluate learned (or scripted) policies on :class:`VecQuadEnv` across seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ..analysis import stats as ST
from ..envs.vec_env import VecQuadEnv


@dataclass
class EvalResult:
    """Per-seed arrays; each entry has one value per seed (so statistics are over independent seeds)."""

    per_seed: Dict[str, np.ndarray]
    episodes: Dict[str, np.ndarray]
    seeds: list

    def summary(self, level: float = 0.95) -> dict:
        out = {}
        for k, v in self.per_seed.items():
            m, lo, hi = ST.mean_ci(v, level)
            out[k] = dict(mean=m, lo=lo, hi=hi, n=len(v))
        return out

    def text(self) -> str:
        return "\n".join(f"  {k:<14} {d['mean']:9.4f}  [{d['lo']:.4f}, {d['hi']:.4f}]  (n={d['n']} seeds)" for k, d in self.summary().items())


def evaluate_policy(
    policy: Callable,
    env_factory: Callable[[int], VecQuadEnv],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    episodes_per_seed: int = 32,
    max_steps: Optional[int] = None,
) -> EvalResult:
    """Roll out ``policy(obs) -> action`` until ``episodes_per_seed`` episodes finish for each seed.

    ``env_factory(seed)`` builds a fresh environment (so the seed controls both the environment
    randomisation and the initial states).
    """
    per_seed: Dict[str, list] = {k: [] for k in ("return", "success_rate", "crash_rate", "pos_error", "ep_length")}
    episodes: Dict[str, list] = {k: [] for k in ("return", "success", "crashed", "length")}
    for seed in seeds:
        env = env_factory(seed)
        B = env.B
        obs = env.reset(seed=seed)
        done_eps, rets, succ, crash, lens, errs = 0, [], [], [], [], []
        steps = 0
        limit = max_steps or env.max_steps * (episodes_per_seed // env.n + 3)
        while done_eps < episodes_per_seed and steps < limit:
            act = policy(obs)
            obs, r, term, trunc, info = env.step(act)
            d = B.to_numpy(info["done"]).astype(bool)
            if d.any():
                rets += B.to_numpy(info["final_return"])[d].tolist()
                succ += B.to_numpy(info["success"])[d].astype(float).tolist()
                crash += B.to_numpy(info["crashed"])[d].astype(float).tolist()
                lens += B.to_numpy(info["episode_length"])[d].tolist()
                errs += B.to_numpy(info["pos_error"])[d].tolist()
                done_eps += int(d.sum())
            steps += 1
        k = episodes_per_seed
        per_seed["return"].append(np.mean(rets[:k]))
        per_seed["success_rate"].append(np.mean(succ[:k]))
        per_seed["crash_rate"].append(np.mean(crash[:k]))
        per_seed["pos_error"].append(np.mean(errs[:k]))
        per_seed["ep_length"].append(np.mean(lens[:k]))
        for key, v in (("return", rets), ("success", succ), ("crashed", crash), ("length", lens)):
            episodes[key].append(np.array(v[:k]))
    return EvalResult({k: np.array(v) for k, v in per_seed.items()}, {k: np.concatenate(v) for k, v in episodes.items()}, list(seeds))


class ProportionalPolicy:
    """Simple scripted baseline for ``velocity`` / ``position`` action modes (go to target, damp velocity)."""

    def __init__(self, env: VecQuadEnv, gain: float = 5.0):
        if env.cfg.action_mode not in ("velocity", "position"):
            raise ValueError("ProportionalPolicy needs action_mode 'velocity' or 'position'")
        self.env, self.gain = env, gain
        self.scale = env.cfg.obs_scales[0] / (env.cfg.max_speed if env.cfg.action_mode == "velocity" else env.cfg.max_step)

    def __call__(self, obs):
        B = self.env.B
        a = B.clip(obs[:, 0:3] * self.scale * (self.gain / 5.0), -1.0, 1.0)
        return B.cat([a, a[:, :1] * 0.0], axis=-1)


class TrainingLog:
    """Collects ``(timesteps, metric)`` curves over several seeds for learning-curve figures."""

    def __init__(self):
        self.runs: Dict[str, list] = {}

    def add(self, method: str, timesteps, values):
        self.runs.setdefault(method, []).append((np.asarray(timesteps, float), np.asarray(values, float)))

    def aggregate(self, method: str, n_points: int = 100):
        """Interpolate all runs of a method onto a common grid → (x, mean, lo, hi)."""
        runs = self.runs[method]
        x = np.linspace(max(r[0].min() for r in runs), min(r[0].max() for r in runs), n_points)
        Y = np.stack([np.interp(x, t, v) for t, v in runs])
        stats = np.array([ST.mean_ci(Y[:, i]) for i in range(n_points)])
        return x, stats[:, 0], stats[:, 1], stats[:, 2]
