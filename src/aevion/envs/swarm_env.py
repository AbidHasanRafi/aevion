"""Cooperative multi-agent environment: ``n_envs`` independent teams of ``n_agents`` quadrotors, with a PettingZoo-style wrapper.

:class:`VecSwarmEnv` simulates all agents of all teams in one batch (the dynamics of :class:`~aevion.envs.vec_env.VecQuadEnv`, driven in
velocity mode). Episodes are per team: they end when any member crashes or collides, or time runs out. Observations include a
neighbour list (the ``k`` nearest agents within ``comm_range``). Tasks:

``formation``   fly the ring/grid/line formation to a random centre; local reward ``exp(−‖p_i − slot_i‖)``.
``rendezvous``  gather at the centroid; local reward ``exp(−‖p_i − c‖ / 2)``.
``coverage``    spread over a square region; the team reward is the decrease of the Lloyd locational cost (altitude is held).

Observations and rewards are NumPy arrays of shape ``(n_envs, n_agents, ...)``. The dynamics run on the chosen backend (including
CUDA), but the neighbour lists and coverage cost are NumPy, so this environment is not tensor-resident like :class:`VecQuadEnv`.
:class:`ParallelSwarmEnv` is the dictionary-keyed API of PettingZoo's ``ParallelEnv`` (it subclasses the real class when
``pettingzoo`` is installed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core import rotation as rot
from ..multiagent.coverage import CoverageDomain, lloyd
from ..multiagent.formations import make_formation
from .tasks import Task
from .vec_env import EnvConfig, VecQuadEnv

__all__ = ["SwarmEnvConfig", "VecSwarmEnv", "ParallelSwarmEnv", "parallel_env", "make_swarm_sb3_vec_env", "SWARM_TASKS"]

SWARM_TASKS = ("formation", "rendezvous", "coverage")


@dataclass
class SwarmEnvConfig:
    n_agents: int = 4
    task: str = "formation"
    vehicle: object = "generic_1kg"
    formation: str = "ring"
    spacing: float = 2.0
    control_dt: float = 0.05
    physics_dt: float = 0.01
    max_episode_s: float = 15.0
    k_neighbors: int = 3
    comm_range: float = math.inf
    arena: float = 6.0  # half-width of the region the team operates in [m]
    altitude: Tuple[float, float] = (2.0, 4.0)
    max_speed: float = 3.0
    separation: float = 0.9  # closer than this is penalised
    collision_radius: float = 0.3  # closer than this ends the episode
    crash_penalty: float = 10.0
    coverage_resolution: float = 0.5
    backend: str = "numpy"
    dtype: Optional[str] = None
    seed: int = 0


class _PassiveTask(Task):
    """The inner environment only integrates the dynamics: goals, rewards and resets are the team's business."""

    name = "passive"
    allows_ground = True

    def reset(self, env, mask):
        pass

    def start_state(self, env, n):
        start = getattr(env, "swarm_start", None)
        return env.B.zeros(n, 3) + (env.B.asarray(start) if start is not None else 0.0)

    def start_euler(self, env, n):
        e = getattr(env, "swarm_euler", None)
        return env.B.zeros(n, 3) + (env.B.asarray(e) if e is not None else 0.0)

    def reward(self, env, action, prev_action):
        return env.B.zeros(env.n)

    def success(self, env):
        return env.B.zeros(env.n) > 1.0


class VecSwarmEnv:
    obs_per_neighbor = 7  # relative position (3), relative velocity (3), presence flag

    def __init__(self, n_envs: int = 8, config: Optional[SwarmEnvConfig] = None, **kw: Any) -> None:
        cfg = config or SwarmEnvConfig()
        for k, v in kw.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown SwarmEnvConfig field {k!r}")
            setattr(cfg, k, v)
        if cfg.task not in SWARM_TASKS:
            raise ValueError(f"task must be one of {SWARM_TASKS}")
        if cfg.n_agents < 2:
            raise ValueError("a swarm needs at least two agents")
        if cfg.k_neighbors < 1 or cfg.arena <= 0 or cfg.max_speed <= 0:
            raise ValueError("k_neighbors, arena and max_speed must be positive")
        self.cfg, self.E, self.A = cfg, int(n_envs), int(cfg.n_agents)
        inner_cfg = EnvConfig(
            vehicle=cfg.vehicle,
            task=_PassiveTask(),
            action_mode="velocity",
            control_dt=cfg.control_dt,
            physics_dt=cfg.physics_dt,
            max_episode_s=1e9,
            bounds=(-1e4, -1e4, -1e4, 1e4, 1e4, 1e4),
            crash_on_ground=False,
            max_speed=cfg.max_speed,
            backend=cfg.backend,
            dtype=cfg.dtype,
            seed=cfg.seed,
        )
        self.inner = VecQuadEnv(self.E * self.A, inner_cfg)
        self.inner.auto_reset = False
        self.B = self.inner.B
        self.rng = np.random.default_rng(cfg.seed + 101)
        self.max_steps = round(cfg.max_episode_s / self.inner.ctrl_dt)
        self.act_dim = 4
        self.obs_dim = 18 + 3 + cfg.k_neighbors * self.obs_per_neighbor
        self.offsets = make_formation(cfg.formation, self.A, cfg.spacing)
        self.offsets = self.offsets - self.offsets.mean(axis=0)
        # a lattice of sites over the square for coverage, and a reference cost for "good enough" (best of a few Lloyd runs)
        self.domain = CoverageDomain([-cfg.arena, -cfg.arena, cfg.arena, cfg.arena], cfg.coverage_resolution)
        self.cover_ref = (
            min(lloyd(np.random.default_rng(s).uniform(-cfg.arena, cfg.arena, (self.A, 2)), self.domain, 80).cost[-1] for s in range(4))
            if cfg.task == "coverage"
            else 1.0
        )
        self.cover_ref = float(self.cover_ref)
        self.rendezvous_radius = 0.6 * cfg.separation * math.sqrt(self.A)  # a ring of A agents that keep their distance has about this radius
        self.centre = np.zeros((self.E, 3))
        self.slots = np.zeros((self.E, self.A, 3))
        self.alt = np.zeros((self.E, self.A))
        self.steps = np.zeros(self.E)
        self.ep_return = np.zeros(self.E)
        self.prev_cost = np.zeros(self.E)
        self.prev_action = np.zeros((self.E, self.A, 4))
        self.reset()

    # ----------------------------------------------------------------------------------------- helpers
    def _np(self, x: Any) -> np.ndarray:
        return np.asarray(self.B.to_numpy(x), float)

    def _state(self) -> Dict[str, np.ndarray]:
        s = self.inner.state
        shape = (self.E, self.A)
        R = rot.to_matrix(self.B, s.quat)
        return dict(
            pos=self._np(s.pos).reshape(*shape, 3),
            vel=self._np(s.vel).reshape(*shape, 3),
            R=self._np(R).reshape(*shape, 9),
            omega=self._np(s.omega).reshape(*shape, 3),
            up=self._np(R)[:, 2, 2].reshape(shape),
        )

    def _cost(self, pos: np.ndarray) -> np.ndarray:
        """Normalised locational cost of every team (needs the coverage domain)."""
        pts = self.domain.points  # (P, 2)
        d2 = ((pts[None, :, None, :] - pos[:, None, :, :2]) ** 2).sum(-1).min(-1)  # (E, P)
        return (d2 * self.domain.weights[None]).sum(1) / self.cover_ref

    # ------------------------------------------------------------------------------------------ reset
    def _sample_teams(self, mask: np.ndarray) -> np.ndarray:
        cfg, rng, E, A = self.cfg, self.rng, self.E, self.A
        starts = np.zeros((E, A, 3))
        for e in np.nonzero(mask)[0]:
            lattice = make_formation("grid", A, 1.6)
            lattice = lattice - lattice.mean(axis=0)
            c0 = np.array([*rng.uniform(-0.4, 0.4, 2) * cfg.arena, rng.uniform(*cfg.altitude)])
            pts = c0 + lattice + rng.uniform(-0.25, 0.25, (A, 3)) * [1, 1, 0.5]
            starts[e] = pts[rng.permutation(A)]
            self.alt[e] = np.clip(starts[e, :, 2], cfg.altitude[0], cfg.altitude[1])
            if cfg.task == "formation":
                yaw = rng.uniform(0, 2 * math.pi)
                Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
                self.centre[e] = [*rng.uniform(-0.55, 0.55, 2) * cfg.arena, rng.uniform(*cfg.altitude)]
                self.slots[e] = self.centre[e] + self.offsets @ Rz.T
            else:
                self.centre[e] = c0
                self.slots[e] = c0
        return starts

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed + 101)
            self.inner.reset(seed=seed)
        self._reset_teams(np.ones(self.E, bool))
        return self._obs()

    def _reset_teams(self, mask: np.ndarray) -> None:
        starts = self._sample_teams(mask)
        keep = getattr(self.inner, "swarm_start", None)
        full = starts.reshape(self.E * self.A, 3) if keep is None else np.where(np.repeat(mask, self.A)[:, None], starts.reshape(-1, 3), keep)
        self.inner.swarm_start = full
        eul_new = self.rng.uniform(-0.1, 0.1, (self.E * self.A, 3)) * [1.0, 1.0, 0.0]
        keep_e = getattr(self.inner, "swarm_euler", None)
        self.inner.swarm_euler = eul_new if keep_e is None else np.where(np.repeat(mask, self.A)[:, None], eul_new, keep_e)
        self.inner._reset_mask(self.B.asarray(np.repeat(mask, self.A).astype(float)) > 0.5)
        self.steps[mask] = 0.0
        self.ep_return[mask] = 0.0
        self.prev_action[mask] = 0.0
        self.inner.prev_action = self.inner.prev_action * 0.0
        if self.cfg.task == "coverage":
            self.prev_cost[mask] = self._cost(self._state()["pos"])[mask]

    # ------------------------------------------------------------------------------------------ observations
    def _neighbours(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        cfg, E, A, k = self.cfg, self.E, self.A, self.cfg.k_neighbors
        rel = pos[:, None, :, :] - pos[:, :, None, :]  # (E, i, j, 3): from i to j
        dist = np.linalg.norm(rel, axis=-1)
        dist[:, np.arange(A), np.arange(A)] = np.inf
        dist = np.where(dist <= cfg.comm_range, dist, np.inf)
        order = np.argsort(dist, axis=-1, kind="stable")[..., :k]
        sel_d = np.take_along_axis(dist, order, axis=-1)
        present = np.isfinite(sel_d)
        gather = lambda x: np.take_along_axis(x, order[..., None], axis=2)  # noqa: E731
        rp = gather(rel) / cfg.arena
        rv = gather(vel[:, None, :, :] - vel[:, :, None, :]) / cfg.max_speed
        out = np.concatenate([rp, rv, present[..., None].astype(float)], axis=-1) * present[..., None]
        if out.shape[2] < k:  # fewer other agents than k
            pad = np.zeros((E, A, k - out.shape[2], self.obs_per_neighbor))
            out = np.concatenate([out, pad], axis=2)
        return out.reshape(E, A, k * self.obs_per_neighbor)

    def _obs(self) -> np.ndarray:
        st = self._state()
        cfg = self.cfg
        goal = (self.slots - st["pos"]) / cfg.arena if cfg.task == "formation" else np.zeros_like(st["pos"])
        return np.concatenate(
            [st["pos"] / cfg.arena, st["vel"] / cfg.max_speed, st["R"], st["omega"] / 5.0, goal, self._neighbours(st["pos"], st["vel"])], axis=-1
        )

    # ---------------------------------------------------------------------------------------------- step
    def step(self, actions: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        cfg, E, A = self.cfg, self.E, self.A
        a = np.clip(np.asarray(actions, float).reshape(E, A, 4), -1.0, 1.0)
        if cfg.task == "coverage":  # the policy steers in the plane; altitude is held
            pos_z = self._state()["pos"][..., 2]
            a = a.copy()
            a[..., 2] = np.clip(1.5 * (self.alt - pos_z) / cfg.max_speed, -1.0, 1.0)
        _, _, _, _, inner_info = self.inner.step(self.B.asarray(a.reshape(E * A, 4)))
        self.steps += 1.0
        st = self._state()
        pos, vel = st["pos"], st["vel"]

        d = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1)
        d[:, np.arange(A), np.arange(A)] = np.inf
        near = d.min(axis=2)  # (E, A) distance to the closest other agent
        min_sep = near.min(axis=1)
        inner_crash = self._np(inner_info["crashed"]).reshape(E, A) > 0.5
        out_of_arena = (np.abs(pos[..., :2]) > 3.0 * cfg.arena).any(-1) | (pos[..., 2] < 0.1) | (pos[..., 2] > 30.0)
        crashed_agent = inner_crash | out_of_arena | ~np.isfinite(pos).all(-1) | (st["up"] < -0.2)
        collided = min_sep < cfg.collision_radius
        crashed = crashed_agent.any(1) | collided

        close_pen = (near < cfg.separation).astype(float)
        act_pen = 0.01 * np.linalg.norm(a, axis=-1)
        success = np.zeros(E, bool)
        if cfg.task == "formation":
            err = np.linalg.norm(pos - self.slots, axis=-1)
            reward = np.exp(-err) + 0.2 * np.exp(-np.linalg.norm(vel, axis=-1)) - close_pen - act_pen
            success = (err < 0.3).all(axis=1)
        elif cfg.task == "rendezvous":
            c = pos.mean(axis=1, keepdims=True)
            dist = np.linalg.norm(pos - c, axis=-1)
            reward = np.exp(-dist / 2.0) - close_pen - act_pen
            success = (dist < self.rendezvous_radius).all(axis=1)
        else:
            cost = self._cost(pos)
            gain = 10.0 * (self.prev_cost - cost)
            self.prev_cost = cost
            reward = np.broadcast_to(gain[:, None], (E, A)) - close_pen - act_pen
            success = cost < 1.5
        reward = reward - cfg.crash_penalty * crashed[:, None]
        terminated = crashed
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated
        self.ep_return += reward.mean(axis=1)
        obs = self._obs()
        info: Dict[str, Any] = dict(
            success=success,
            crashed=crashed,
            collided=collided,
            min_separation=min_sep,
            done=done,
            truncated=truncated,
            final_observation=obs.copy(),
            final_return=self.ep_return.copy(),
            episode_length=self.steps.copy(),
        )
        if cfg.task == "coverage":
            info["coverage_cost"] = self._cost(pos)
        self.prev_action = a
        if done.any():
            self._reset_teams(done)
            obs = self._obs()
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------------------------- misc
    def spaces(self) -> Tuple[int, int]:
        return self.obs_dim, self.act_dim

    def info(self) -> Dict[str, Any]:
        return dict(
            n_envs=self.E,
            n_agents=self.A,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            task=self.cfg.task,
            control_dt=self.inner.ctrl_dt,
            backend=self.B.name,
        )

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ PettingZoo-style API
try:  # pragma: no cover - exercised only when pettingzoo is installed
    from pettingzoo import ParallelEnv as _ParallelBase  # type: ignore[import-not-found]

    HAVE_PETTINGZOO = True
except Exception:  # pragma: no cover
    HAVE_PETTINGZOO = False

    class _ParallelBase:  # type: ignore[no-redef]
        """Stand-in with the attributes of ``pettingzoo.ParallelEnv`` so the API is usable (and testable) without the package."""

        metadata: Dict[str, Any] = {}


class ParallelSwarmEnv(_ParallelBase):
    """Dictionary-keyed multi-agent API: ``reset() -> (obs, infos)``, ``step(actions) -> (obs, rewards, terminations, truncations, infos)``.

    One team (``n_envs = 1``). Agents are ``"agent_0" ... "agent_{n-1}"``; ``agents`` empties when the episode ends, as PettingZoo requires, and
    ``reset()`` refills it. Termination is a team event, so every agent receives the same ``terminations`` / ``truncations`` flag.
    """

    metadata = {"name": "aevion_swarm_v0", "render_modes": [], "is_parallelizable": True}

    def __init__(self, config: Optional[SwarmEnvConfig] = None, **kw: Any) -> None:
        self.venv = VecSwarmEnv(1, config, **kw)
        n = self.venv.A
        self.possible_agents: List[str] = [f"agent_{i}" for i in range(n)]
        self.agents: List[str] = []
        self.max_num_agents = self.num_agents = n
        from gymnasium import spaces

        self._obs_space = spaces.Box(-np.inf, np.inf, (self.venv.obs_dim,), dtype=np.float32)
        self._act_space = spaces.Box(-1.0, 1.0, (self.venv.act_dim,), dtype=np.float32)

    def observation_space(self, agent: str) -> Any:
        return self._obs_space

    def action_space(self, agent: str) -> Any:
        return self._act_space

    def _dict(self, arr: np.ndarray) -> Dict[str, Any]:
        return {a: arr[i] for i, a in enumerate(self.possible_agents)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        obs = self.venv.reset(seed=seed)[0].astype(np.float32)
        self.agents = list(self.possible_agents)
        return self._dict(obs), {a: {} for a in self.agents}

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        missing = [a for a in self.agents if a not in actions]
        if missing:
            raise KeyError(f"no action for agent(s) {missing}")
        act = np.stack([np.asarray(actions[a], float).reshape(4) for a in self.possible_agents])[None]
        obs, rew, term, trunc, info = self.venv.step(act)
        done = bool(term[0] or trunc[0])
        o = (info["final_observation"] if done else obs)[0].astype(np.float32)
        agents = list(self.agents)
        infos = {a: dict(success=bool(info["success"][0]), min_separation=float(info["min_separation"][0])) for a in agents}
        out = (
            {a: o[i] for i, a in enumerate(self.possible_agents) if a in agents},
            {a: float(rew[0, i]) for i, a in enumerate(self.possible_agents) if a in agents},
            {a: bool(term[0]) for a in agents},
            {a: bool(trunc[0]) for a in agents},
            infos,
        )
        if done:
            self.agents = []
        return out

    def state(self) -> np.ndarray:
        return self.venv._obs()[0].reshape(-1).astype(np.float32)

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.venv.close()


def parallel_env(**kw: Any) -> ParallelSwarmEnv:
    """Factory in the PettingZoo style (``aevion.envs.swarm_env.parallel_env(task="rendezvous")``)."""
    return ParallelSwarmEnv(**kw)


def make_swarm_sb3_vec_env(n_envs: int = 4, config: Optional[SwarmEnvConfig] = None, **kw: Any):
    """Parameter-sharing view for Stable-Baselines3: every agent of every team is one environment of a ``VecEnv`` of size ``n_envs · n_agents``."""
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import VecEnv

    venv = VecSwarmEnv(n_envs, config, **kw)
    N, od, ad = venv.E * venv.A, venv.obs_dim, venv.act_dim

    class _SB3(VecEnv):
        def __init__(self) -> None:
            super().__init__(N, spaces.Box(-np.inf, np.inf, (od,), dtype=np.float32), spaces.Box(-1.0, 1.0, (ad,), dtype=np.float32))
            self.venv = venv
            self._actions: Optional[np.ndarray] = None
            self._ret = np.zeros(N)
            self._len = np.zeros(N)

        def reset(self):
            self._ret[:] = 0
            self._len[:] = 0
            return venv.reset().reshape(N, od).astype(np.float32)

        def step_async(self, actions):
            self._actions = np.asarray(actions, np.float32)

        def step_wait(self):
            obs, rew, term, trunc, info = venv.step(self._actions)
            r = rew.reshape(N).astype(np.float64)
            done = np.repeat(term | trunc, venv.A)
            tr = np.repeat(trunc & ~term, venv.A)
            succ = np.repeat(info["success"], venv.A)
            fo = info["final_observation"].reshape(N, od)
            self._ret += r
            self._len += 1
            infos: List[Dict[str, Any]] = [dict() for _ in range(N)]
            for i in np.nonzero(done)[0]:
                infos[i].update({"terminal_observation": fo[i].astype(np.float32), "TimeLimit.truncated": bool(tr[i]), "is_success": bool(succ[i])})
                infos[i]["episode"] = {"r": float(self._ret[i]), "l": int(self._len[i])}
                self._ret[i] = 0
                self._len[i] = 0
            return obs.reshape(N, od).astype(np.float32), r, done, infos

        def close(self):
            pass

        def get_attr(self, attr_name, indices=None):
            return [getattr(venv, attr_name, None)] * N

        def set_attr(self, attr_name, value, indices=None):
            pass

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return [getattr(venv, method_name)(*args, **kwargs)] * N

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * N

        def seed(self, seed=None):
            venv.reset(seed=seed or 0)
            return [seed] * N

    return _SB3()
