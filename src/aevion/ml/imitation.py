"""Behaviour cloning and DAgger.

**Behaviour cloning** fits a policy to ``(observation, expert action)`` pairs by regression. Trained on states the expert visits it makes errors that
carry it to states the expert never showed (covariate shift), and the errors compound (Ross and Bagnell 2010).

**DAgger** (Ross, Gordon and Bagnell, AISTATS 2011) removes the mismatch: iteration 0 collects expert demonstrations; each later iteration runs the
*learner* (mixed with the expert with probability ``β_i = β₀ᶦ``), has the expert **label every visited state**, aggregates the labelled states
into the dataset and retrains. :func:`dagger` returns the policy and a history of dataset sizes, losses and, if an evaluation environment is given,
the success rate after each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .dataset import Dataset, collect
from .policies import MLPPolicy

__all__ = ["behavior_cloning", "dagger", "DaggerResult", "evaluate_policy_rollouts", "BCResult"]


@dataclass
class BCResult:
    policy: MLPPolicy
    losses: List[float]
    val_loss: Optional[float]


def behavior_cloning(
    data: Dataset,
    hidden: Sequence[int] = (128, 128),
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_fraction: float = 0.0,
    seed: int = 0,
    policy: Optional[MLPPolicy] = None,
    loss: str = "mse",
) -> BCResult:
    """Fit ``tanh(MLP)`` to the dataset's actions. Pass ``policy`` to continue training an existing network (its normaliser is kept)."""
    if len(data) < 2:
        raise ValueError("the dataset is too small to train on")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    n_val = int(len(data) * val_fraction)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    if policy is None:
        nz = data.normalizer()
        policy = MLPPolicy(data.obs_dim, data.act_dim, hidden, nz.mean, nz.std)
    elif policy.obs_dim != data.obs_dim or policy.act_dim != data.act_dim:
        raise ValueError("the policy and the dataset disagree on observation or action size")
    X = torch.as_tensor(data.observations)
    Y = torch.as_tensor(data.actions)
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    crit = {"mse": nn.MSELoss(), "huber": nn.SmoothL1Loss()}[loss]
    policy.train()
    losses: List[float] = []
    for _ in range(epochs):
        order = rng.permutation(tr_idx)
        total, count = 0.0, 0
        for s in range(0, len(order), batch_size):
            b = torch.as_tensor(order[s : s + batch_size])
            opt.zero_grad()
            l = crit(policy.forward(X[b]), Y[b])
            l.backward()
            opt.step()
            total += float(l.detach()) * len(b)
            count += len(b)
        losses.append(total / max(count, 1))
    policy.eval()
    val = None
    if n_val:
        with torch.no_grad():
            val = float(crit(policy.forward(X[torch.as_tensor(val_idx)]), Y[torch.as_tensor(val_idx)]))
    return BCResult(policy, losses, val)


def evaluate_policy_rollouts(
    env: Any, policy: Callable[[Any], Any], episodes: int = 16, seed: int = 0, max_steps: Optional[int] = None
) -> Dict[str, float]:
    """Success rate, crash rate and mean return over (at least) ``episodes`` finished episodes of the vectorised environment."""
    B = env.B
    obs = env.reset(seed=seed)
    succ: List[float] = []
    crash: List[float] = []
    ret: List[float] = []
    limit = max_steps or env.max_steps * (episodes // env.n + 3)
    for _ in range(limit):
        obs, r, term, trunc, info = env.step(policy(obs))
        d = np.asarray(B.to_numpy(info["done"])).astype(bool)
        if d.any():
            succ += np.asarray(B.to_numpy(info["success"]))[d].astype(float).tolist()
            crash += np.asarray(B.to_numpy(info["crashed"]))[d].astype(float).tolist()
            ret += np.asarray(B.to_numpy(info["final_return"]))[d].astype(float).tolist()
        if len(succ) >= episodes:
            break
    k = max(episodes, 1)
    return dict(
        success_rate=float(np.mean(succ[:k])) if succ else 0.0,
        crash_rate=float(np.mean(crash[:k])) if crash else 0.0,
        mean_return=float(np.mean(ret[:k])) if ret else float("nan"),
        episodes=len(succ[:k]),
    )


@dataclass
class DaggerResult:
    policy: MLPPolicy
    dataset: Dataset
    history: List[Dict[str, float]] = field(default_factory=list)


def dagger(
    env: Any,
    expert: Callable[[Any], Any],
    iterations: int = 5,
    steps_per_iter: int = 300,
    beta0: float = 0.5,
    hidden: Sequence[int] = (128, 128),
    epochs: int = 40,
    lr: float = 1e-3,
    noise: float = 0.0,
    eval_env: Any = None,
    eval_episodes: int = 16,
    seed: int = 0,
    retrain_from_scratch: bool = False,
) -> DaggerResult:
    """DAgger with an exponentially decaying mixture ``β_i = β₀ᶦ`` (iteration 0 is pure expert demonstration, ``β = 1``)."""
    data: Optional[Dataset] = None
    policy: Optional[MLPPolicy] = None
    hist: List[Dict[str, float]] = []
    B = env.B
    for it in range(iterations + 1):
        beta = 1.0 if it == 0 else beta0**it
        runner = expert if policy is None else policy  # iteration 0 demonstrates; later iterations let the (mixed) learner drive
        part = collect(env, runner, steps_per_iter, labeler=expert, beta=beta, noise=noise, seed=seed + it, reset_seed=seed + 1000 * it)
        data = part if data is None else data.concat(part)
        res = behavior_cloning(
            data, hidden=hidden, epochs=epochs, lr=lr, seed=seed + it, policy=None if (policy is None or retrain_from_scratch) else policy
        )
        policy = res.policy
        row: Dict[str, float] = dict(iteration=it, beta=beta, dataset=len(data), loss=res.losses[-1])
        if eval_env is not None:

            def act(o: Any, p: MLPPolicy = policy) -> Any:
                return B.asarray(p(B.to_numpy(o)))

            row.update(evaluate_policy_rollouts(eval_env, act, eval_episodes, seed=seed + 7))
        hist.append(row)
    assert data is not None and policy is not None
    return DaggerResult(policy, data, hist)
