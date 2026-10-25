"""Task definitions for the vectorised environment: targets, rewards and success criteria.

A task owns the *goal* part of an episode. The environment exposes tensors ``env.target``
(N,3), ``env.target_vel`` (N,3) and ``env.target_yaw`` (N,) that the task fills in. Everything
is written against the backend interface, so tasks run on CPU and CUDA unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


def _uniform(B, gen, n, lo, hi):
    lo, hi = B.asarray(lo), B.asarray(hi)
    return lo + (hi - lo) * B.rand((n, len(lo)), gen)


class Task:
    name = "task"
    allows_ground = False  # whether touching the ground is acceptable (landing tasks)
    extra_obs_dim = 0  # observation features the task appends (see ``extra_obs``)

    def extra_obs(self, env):
        """``(N, extra_obs_dim)`` task-specific observation features (e.g. radio link quality)."""
        raise NotImplementedError

    def reset(self, env, mask):
        raise NotImplementedError

    def update(self, env):
        pass

    def reward(self, env, action, prev_action):
        raise NotImplementedError

    def success(self, env):
        raise NotImplementedError

    def start_state(self, env, n):
        """Initial positions (n,3) for *all* envs; the env keeps those selected by its reset mask."""
        raise NotImplementedError

    def start_velocity(self, env, n):
        return env.B.zeros(n, 3)

    def post_reset(self, env, mask):
        """Called after the new initial state exists (e.g. to initialise progress trackers)."""

    def to_dict(self):
        return dict(name=self.name)


@dataclass
class HoverTask(Task):
    """Fly to a random point and hold it."""

    name: str = "hover"
    target_box: Tuple = ((-1.5, -1.5, 1.0), (1.5, 1.5, 3.0))
    start_box: Tuple = ((-1.0, -1.0, 0.8), (1.0, 1.0, 2.0))
    success_radius: float = 0.15
    w_pos: float = 1.0
    w_rate: float = 0.05

    def reset(self, env, mask):
        B, n = env.B, env.n
        new = _uniform(B, env.gen, n, *self.target_box)
        env.target = B.where(mask[:, None], new, env.target)
        z = new * 0.0
        env.target_vel = B.where(mask[:, None], z, env.target_vel)
        env.target_yaw = B.where(mask, B.zeros(n), env.target_yaw)

    def start_state(self, env, n):
        return _uniform(env.B, env.gen, n, *self.start_box)

    def reward(self, env, action, prev_action):
        B, s = env.B, env.state
        e = B.norm(s.pos - env.target, axis=-1)
        v = B.norm(s.vel - env.target_vel, axis=-1)
        w = B.norm(s.omega, axis=-1)
        da = B.norm(action - prev_action, axis=-1)
        return self.w_pos * B.exp(-2.0 * e) + 0.2 * B.exp(-v) - self.w_rate * 0.1 * w - 0.01 * da

    def success(self, env):
        B = env.B
        return B.norm(env.state.pos - env.target, axis=-1) < self.success_radius

    def to_dict(self):
        return dict(name=self.name, target_box=[list(b) for b in self.target_box], success_radius=self.success_radius)


@dataclass
class WaypointTask(Task):
    """Visit ``n_waypoints`` random goals in sequence; a goal is reached within ``radius``."""

    name: str = "waypoints"
    n_waypoints: int = 4
    box: Tuple = ((-4.0, -4.0, 1.0), (4.0, 4.0, 3.5))
    radius: float = 0.35
    start_box: Tuple = ((-0.5, -0.5, 0.8), (0.5, 0.5, 1.5))
    bonus: float = 5.0

    def reset(self, env, mask):
        B, n = env.B, env.n
        wp = B.stack([_uniform(B, env.gen, n, *self.box) for _ in range(self.n_waypoints)], 1)  # (N,K,3)
        if not hasattr(env, "_wp"):
            env._wp = wp
            env._wp_idx = B.zeros(n)
            env._prev_dist = B.zeros(n)
        env._wp = B.where(mask[:, None, None], wp, env._wp)
        env._wp_idx = B.where(mask, B.zeros(n), env._wp_idx)
        self._set_target(env)
        env.target_vel = env.target * 0.0
        env.target_yaw = B.zeros(n)

    def post_reset(self, env, mask):
        env._prev_dist = env.B.where(mask, env.B.norm(env.state.pos - env.target, axis=-1), env._prev_dist)

    def _set_target(self, env):
        B, n = env.B, env.n
        idx = B.clip(env._wp_idx, 0.0, self.n_waypoints - 1.0)
        if B.name == "numpy":
            sel = env._wp[np.arange(n), idx.astype(int)]
        else:
            import torch

            sel = env._wp[torch.arange(n, device=env._wp.device), idx.long()]
        env.target = sel

    def start_state(self, env, n):
        return _uniform(env.B, env.gen, n, *self.start_box)

    def update(self, env):
        B = env.B
        d = B.norm(env.state.pos - env.target, axis=-1)
        reached = d < self.radius
        env._reached_now = reached
        env._wp_idx = B.where(reached, env._wp_idx + 1.0, env._wp_idx)
        env._wp_idx = B.minimum(env._wp_idx, B.full(env._wp_idx.shape, float(self.n_waypoints)))
        self._set_target(env)

    def reward(self, env, action, prev_action):
        B = env.B
        d = B.norm(env.state.pos - env.target, axis=-1)
        prog = env._prev_dist - d
        env._prev_dist = d
        bonus = self.bonus * getattr(env, "_reached_now", d * 0.0) * 1.0
        return 2.0 * prog + bonus + 0.05 * B.exp(-d) - 0.01 * B.norm(action - prev_action, axis=-1)

    def success(self, env):
        return env._wp_idx >= self.n_waypoints

    def to_dict(self):
        return dict(name=self.name, n_waypoints=self.n_waypoints, radius=self.radius)


@dataclass
class TrackingTask(Task):
    """Track a moving reference (circle or figure-eight) with randomised size, rate and phase."""

    name: str = "tracking"
    kind: str = "circle"  # "circle" | "figure8"
    radius: Tuple = (0.8, 2.0)
    omega: Tuple = (0.4, 1.1)
    height: Tuple = (1.5, 2.5)
    success_radius: float = 0.3

    def reset(self, env, mask):
        B, n = env.B, env.n

        def r(lo, hi):
            return lo + (hi - lo) * B.rand((n,), env.gen)

        if not hasattr(env, "_trk"):
            env._trk = dict(r=r(*self.radius), w=r(*self.omega), h=r(*self.height), ph=r(0.0, 2 * math.pi), t0=B.zeros(n))
        for k, v in (("r", r(*self.radius)), ("w", r(*self.omega)), ("h", r(*self.height)), ("ph", r(0.0, 2 * math.pi))):
            env._trk[k] = B.where(mask, v, env._trk[k])
        env._trk["t0"] = B.where(mask, B.zeros(n) + env.sim_time, env._trk["t0"])
        self._eval(env)

    def _eval(self, env):
        B = env.B
        t = env.sim_time - env._trk["t0"]
        r, w, h, ph = env._trk["r"], env._trk["w"], env._trk["h"], env._trk["ph"]
        th = w * t + ph
        if self.kind == "circle":
            pos = B.stack([r * B.cos(th), r * B.sin(th), h], -1)
            vel = B.stack([-r * w * B.sin(th), r * w * B.cos(th), th * 0.0], -1)
        else:
            pos = B.stack([r * B.sin(th), 0.5 * r * B.sin(2 * th), h], -1)
            vel = B.stack([r * w * B.cos(th), r * w * B.cos(2 * th), th * 0.0], -1)
        env.target, env.target_vel = pos, vel
        env.target_yaw = th * 0.0

    def start_state(self, env, n):
        # start on the reference path at its initial phase (targets are sampled before the state)
        B = env.B
        return env.target + (B.rand((n, 3), env.gen) - 0.5) * 0.2

    def start_velocity(self, env, n):
        return env.target_vel

    def update(self, env):
        self._eval(env)

    def reward(self, env, action, prev_action):
        B, s = env.B, env.state
        e = B.norm(s.pos - env.target, axis=-1)
        ev = B.norm(s.vel - env.target_vel, axis=-1)
        return B.exp(-3.0 * e) + 0.2 * B.exp(-ev) - 0.01 * B.norm(action - prev_action, axis=-1)

    def success(self, env):
        return env.B.norm(env.state.pos - env.target, axis=-1) < self.success_radius

    def to_dict(self):
        return dict(name=self.name, kind=self.kind, radius=list(self.radius), omega=list(self.omega))


@dataclass
class NavigationTask(Task):
    """Reach a goal through a world with obstacles using LiDAR observations."""

    name: str = "navigation"
    start_x: Tuple = (-12.0, -8.0)
    goal_x: Tuple = (8.0, 12.0)
    half_width: float = 10.0
    altitude: Tuple = (1.5, 3.5)
    clearance: float = 1.5
    radius: float = 0.8
    allows_ground = False

    def reset(self, env, mask):
        B, n = env.B, env.n
        goal = self._sample(env, self.goal_x, n)
        env.target = B.where(mask[:, None], goal, env.target)
        env.target_vel = env.target * 0.0
        env.target_yaw = B.zeros(n)
        if not hasattr(env, "_prev_dist"):
            env._prev_dist = B.zeros(n)

    def post_reset(self, env, mask):
        env._prev_dist = env.B.where(mask, env.B.norm(env.state.pos - env.target, axis=-1), env._prev_dist)

    def _sample(self, env, xr, n):
        B = env.B
        rng = np.random.default_rng(int(B.to_numpy(B.rand((1,), env.gen))[0] * 2**31))
        pts = np.stack(
            [rng.uniform(xr[0], xr[1], n), rng.uniform(-self.half_width, self.half_width, n), rng.uniform(self.altitude[0], self.altitude[1], n)], -1
        )
        w = env.world
        if w.has_obstacles:
            for _ in range(20):
                bad = w.distance_to_nearest(pts) < self.clearance
                if not bad.any():
                    break
                k = int(bad.sum())
                pts[bad] = np.stack(
                    [
                        rng.uniform(xr[0], xr[1], k),
                        rng.uniform(-self.half_width, self.half_width, k),
                        rng.uniform(self.altitude[0], self.altitude[1], k),
                    ],
                    -1,
                )
        return B.asarray(pts)

    def start_state(self, env, n):
        return self._sample(env, self.start_x, n)

    def reward(self, env, action, prev_action):
        B = env.B
        d = B.norm(env.state.pos - env.target, axis=-1)
        prog = env._prev_dist - d
        env._prev_dist = d
        reached = d < self.radius
        return 3.0 * prog + 20.0 * reached * 1.0 + 0.02 - 0.01 * B.norm(action - prev_action, axis=-1)

    def success(self, env):
        return env.B.norm(env.state.pos - env.target, axis=-1) < self.radius

    def terminate_on_success(self):
        return True

    def to_dict(self):
        return dict(name=self.name, radius=self.radius)


@dataclass
class ConnectedNavigationTask(Task):
    """Fly to a goal **while staying connected**: communication-aware path planning as a learning problem.

    The agent observes six extra link features (normalised SINR and rate, unit vector and distance to its serving cell) and is
    rewarded for progress towards the goal *and* for link quality, with a penalty for outage. In a world with buildings the
    best path is not the straight one: shadowed areas make the detour worth it. The radio map comes from an
    :class:`~aevion.wireless.mac_network.NrCellularNetwork` (``network`` accepts an instance or its keyword dictionary) and is
    evaluated statelessly with :meth:`~aevion.wireless.mac_network.NrCellularNetwork.snapshot` every ``link_every`` control steps.

    The default boxes are outdoor-scale; use ``ConnectedNavigationTask.env_kwargs()`` for matching ``EnvConfig`` fields.
    """

    name: str = "connected_navigation"
    network: Any = None
    start_box: Tuple = ((-150.0, -60.0, 40.0), (-120.0, 60.0, 80.0))
    goal_box: Tuple = ((120.0, -60.0, 40.0), (150.0, 60.0, 80.0))
    radius: float = 6.0
    w_link: float = 0.15
    w_outage: float = 0.5
    outage_db: float = -6.0
    link_every: int = 5
    shadowing: bool = True
    extra_obs_dim = 6
    _network_obj: Any = field(default=None, init=False, repr=False)
    allows_ground = False

    @staticmethod
    def env_kwargs() -> dict:
        """``EnvConfig`` fields that give the task enough room, speed and time (outdoor scale)."""
        return dict(
            bounds=(-200.0, -120.0, 5.0, 200.0, 120.0, 120.0),
            max_speed=15.0,
            max_step=10.0,
            obs_scales=(100.0, 15.0, 5.0),
            max_episode_s=40.0,
            crash_on_ground=True,
        )

    def _net(self, env):
        if self._network_obj is None:
            from ..wireless.mac_network import NrCellularNetwork

            cfg = self.network
            if isinstance(cfg, NrCellularNetwork):
                net = cfg
            else:
                cfg = dict(cfg or {})
                cfg.setdefault("sites", [{"pos": [-40.0, 0.0, 25.0], "sectors": 1}, {"pos": [40.0, 0.0, 25.0], "sectors": 1}])
                cfg.setdefault("seed", 0)
                net = NrCellularNetwork(world=env.world, **cfg)
            self._network_obj = net
        return self._network_obj

    def _link(self, env):
        B = env.B
        snap = self._net(env).snapshot(B.to_numpy(env.state.pos), shadowing=self.shadowing)
        env._link = snap
        env._link_quality = B.asarray(np.clip((snap["sinr_db"] - self.outage_db) / 30.0, 0.0, 1.0))
        env._link_outage = B.asarray((snap["sinr_db"] < self.outage_db).astype(float))

    def reset(self, env, mask):
        B, n = env.B, env.n
        goal = _uniform(B, env.gen, n, *self.goal_box)
        env.target = B.where(mask[:, None], goal, env.target)
        env.target_vel = env.target * 0.0
        env.target_yaw = B.zeros(n)
        if not hasattr(env, "_prev_dist"):
            env._prev_dist = B.zeros(n)

    def post_reset(self, env, mask):
        env._prev_dist = env.B.where(mask, env.B.norm(env.state.pos - env.target, axis=-1), env._prev_dist)
        self._link(env)

    def start_state(self, env, n):
        return _uniform(env.B, env.gen, n, *self.start_box)

    def update(self, env):
        if int(env.B.to_numpy(env.steps)[0]) % self.link_every == 0 or not hasattr(env, "_link"):
            self._link(env)

    def extra_obs(self, env):
        B = env.B
        if not hasattr(env, "_link"):
            self._link(env)
        s = env._link
        feats = np.concatenate(
            [
                ((s["sinr_db"] - 10.0) / 20.0)[:, None],
                (np.log1p(s["rate_bps"] / 1e6) / 5.0)[:, None],
                s["direction"],
                (np.linalg.norm(self._net(env).cell_pos[s["serving"]] - B.to_numpy(env.state.pos), axis=1) / 200.0)[:, None],
            ],
            axis=1,
        )
        return B.asarray(feats)

    def reward(self, env, action, prev_action):
        B = env.B
        d = B.norm(env.state.pos - env.target, axis=-1)
        prog = env._prev_dist - d
        env._prev_dist = d
        reached = d < self.radius
        return (
            0.1 * prog
            + 20.0 * reached * 1.0
            + self.w_link * env._link_quality
            - self.w_outage * env._link_outage
            - 0.01 * B.norm(action - prev_action, axis=-1)
        )

    def success(self, env):
        return env.B.norm(env.state.pos - env.target, axis=-1) < self.radius

    def terminate_on_success(self):
        return True

    def to_dict(self):
        return dict(
            name=self.name, radius=self.radius, w_link=self.w_link, w_outage=self.w_outage, outage_db=self.outage_db, link_every=self.link_every
        )


TASKS = {
    "hover": HoverTask,
    "waypoints": WaypointTask,
    "tracking": TrackingTask,
    "navigation": NavigationTask,
    "connected_navigation": ConnectedNavigationTask,
}


from . import tasks_ext  # noqa: E402,F401  (registers the extended tasks in TASKS)


def make_task(task):
    if isinstance(task, Task):
        return task
    if task in TASKS:
        return TASKS[task]()
    raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
