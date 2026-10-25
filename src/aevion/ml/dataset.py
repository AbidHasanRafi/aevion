"""Transition datasets for imitation and offline reinforcement learning, in the layout D4RL uses.

A :class:`Dataset` stores ``(observation, action, reward, terminal, timeout, next_observation)`` rows plus the index of the environment that produced each
row, so batched environments can be stored in step order and still be split back into episodes. ``terminals`` mark true terminations (a crash, a
success) and ``timeouts`` mark truncation by the time limit; offline RL needs the difference (bootstrap after a timeout, not after a crash).
:func:`collect` rolls a policy out in a :class:`~aevion.envs.vec_env.VecQuadEnv` and optionally labels the visited states with another policy's
actions, which is the data-aggregation step of DAgger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = ["Dataset", "collect", "Normalizer"]

_ARRAYS = ("observations", "actions", "rewards", "terminals", "timeouts", "next_observations", "env_ids")


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x: Any) -> np.ndarray:
        return (np.asarray(x, np.float32) - self.mean) / self.std


@dataclass
class Dataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    next_observations: np.ndarray
    env_ids: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.observations)
        for name in _ARRAYS:
            if len(getattr(self, name)) != n:
                raise ValueError(f"'{name}' has {len(getattr(self, name))} rows, expected {n}")
        self.observations = np.asarray(self.observations, np.float32)
        self.next_observations = np.asarray(self.next_observations, np.float32)
        self.actions = np.asarray(self.actions, np.float32)
        self.rewards = np.asarray(self.rewards, np.float32)
        self.terminals = np.asarray(self.terminals, bool)
        self.timeouts = np.asarray(self.timeouts, bool)
        self.env_ids = np.asarray(self.env_ids, np.int32)

    def __len__(self) -> int:
        return len(self.observations)

    # ----------------------------------------------------------------------------------------------- structure
    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[1])

    def episode_ids(self) -> np.ndarray:
        """Episode index of every row (rows of different environments are interleaved in step order)."""
        end = self.terminals | self.timeouts
        out = np.zeros(len(self), np.int64)
        counter: Dict[int, int] = {}
        base = 0
        ids: Dict[Tuple[int, int], int] = {}
        for i, (e, d) in enumerate(zip(self.env_ids, end)):
            k = counter.get(int(e), 0)
            key = (int(e), k)
            if key not in ids:
                ids[key] = base
                base += 1
            out[i] = ids[key]
            if d:
                counter[int(e)] = k + 1
        return out

    def episodes(self) -> Iterator["Dataset"]:
        eid = self.episode_ids()
        for k in np.unique(eid):
            yield self.take(np.nonzero(eid == k)[0])

    def returns(self) -> np.ndarray:
        """Undiscounted return of every episode that finished (a trailing unfinished episode is excluded)."""
        eid = self.episode_ids()
        out = []
        for k in np.unique(eid):
            rows = np.nonzero(eid == k)[0]
            if self.terminals[rows[-1]] or self.timeouts[rows[-1]]:
                out.append(float(self.rewards[rows].sum()))
        return np.asarray(out)

    def take(self, idx: Any) -> "Dataset":
        return Dataset(**{n: getattr(self, n)[idx] for n in _ARRAYS}, meta=dict(self.meta))

    def concat(self, other: "Dataset") -> "Dataset":
        if self.obs_dim != other.obs_dim or self.act_dim != other.act_dim:
            raise ValueError("datasets have different observation or action sizes")
        return Dataset(**{n: np.concatenate([getattr(self, n), getattr(other, n)]) for n in _ARRAYS}, meta=dict(self.meta))

    def normalizer(self, floor: float = 0.05) -> Normalizer:
        """Per-feature mean and standard deviation, with the deviation floored at ``floor``: a feature that never varied in the demonstrations (an unused
        action channel fed back as an input, say) would otherwise be amplified enormously the first time the learner moves it."""
        return Normalizer(self.observations.mean(0), np.maximum(self.observations.std(0), floor))

    def batches(self, size: int, rng: np.random.Generator, epochs: int = 1) -> Iterator[np.ndarray]:
        for _ in range(epochs):
            order = rng.permutation(len(self))
            for s in range(0, len(order), size):
                yield order[s : s + size]

    # ------------------------------------------------------------------------------------------------- I/O
    def to_d4rl(self) -> Dict[str, np.ndarray]:
        """The dictionary D4RL / minari-style offline RL libraries expect."""
        return dict(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            timeouts=self.timeouts,
            next_observations=self.next_observations,
        )

    @classmethod
    def from_d4rl(cls, d: Dict[str, Any], env_ids: Optional[np.ndarray] = None) -> "Dataset":
        n = len(d["observations"])
        nxt = d.get("next_observations")
        if nxt is None:  # D4RL files without next observations: shift, and repeat the last row of every episode
            nxt = np.roll(np.asarray(d["observations"]), -1, axis=0)
        return cls(
            d["observations"],
            d["actions"],
            d["rewards"],
            d["terminals"],
            d.get("timeouts", np.zeros(n, bool)),
            nxt,
            np.zeros(n, np.int32) if env_ids is None else env_ids,
        )

    def save(self, path: Any) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{n: getattr(self, n) for n in _ARRAYS}, meta=np.array(json.dumps(self.meta, default=float)))
        return p if p.suffix == ".npz" else p.with_suffix(p.suffix + ".npz")

    @classmethod
    def load(cls, path: Any) -> "Dataset":
        with np.load(Path(path), allow_pickle=False) as z:
            missing = [n for n in _ARRAYS if n not in z]
            if missing:
                raise ValueError(f"not an Aevion dataset (missing {missing})")
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
            return cls(**{n: z[n] for n in _ARRAYS}, meta=meta)


def collect(
    env: Any,
    policy: Callable[[Any], Any],
    steps: int,
    labeler: Optional[Callable[[Any], Any]] = None,
    beta: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
    reset: bool = True,
    reset_seed: Optional[int] = None,
) -> Dataset:
    """Run ``steps`` control steps in every sub-environment of ``env`` and record the transitions.

    ``policy(obs)`` chooses the executed action. If ``labeler`` is given, the **stored** action is the labeler's action for the visited observation (DAgger's
    relabelling); with probability ``beta`` per environment and step the executed action is the labeler's as well (the mixture policy of DAgger's early
    iterations). ``noise`` adds Gaussian exploration noise to the executed action.
    """
    B = env.B
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=reset_seed if reset_seed is not None else seed) if reset else env._obs()
    n = env.n
    O: List[np.ndarray] = []
    A: List[np.ndarray] = []
    R: List[np.ndarray] = []
    T: List[np.ndarray] = []
    U: List[np.ndarray] = []
    N: List[np.ndarray] = []
    E: List[np.ndarray] = []
    ids = np.arange(n, dtype=np.int32)
    for _ in range(int(steps)):
        a_pol = np.asarray(B.to_numpy(policy(obs)), float).reshape(n, -1)
        a_lab = np.asarray(B.to_numpy(labeler(obs)), float).reshape(n, -1) if labeler is not None else a_pol
        use_lab = rng.random(n) < beta
        a_exec = np.where(use_lab[:, None], a_lab, a_pol)
        if noise > 0:
            a_exec = a_exec + rng.normal(0.0, noise, a_exec.shape)
        a_exec = np.clip(a_exec, -1.0, 1.0)
        nxt, r, term, trunc, info = env.step(B.asarray(a_exec))
        O.append(np.asarray(B.to_numpy(obs), np.float32))
        A.append(np.clip(a_lab if labeler is not None else a_exec, -1.0, 1.0).astype(np.float32))
        R.append(np.asarray(B.to_numpy(r), np.float32))
        T.append(np.asarray(B.to_numpy(term)).astype(bool))
        U.append(np.asarray(B.to_numpy(trunc)).astype(bool))
        N.append(np.asarray(B.to_numpy(info["final_observation"]), np.float32))
        E.append(ids)
        obs = nxt
    meta = dict(env=env.info(), steps=int(steps), beta=float(beta), noise=float(noise), seed=int(seed), labeled=labeler is not None)
    return Dataset(
        np.concatenate(O), np.concatenate(A), np.concatenate(R), np.concatenate(T), np.concatenate(U), np.concatenate(N), np.concatenate(E), meta
    )
