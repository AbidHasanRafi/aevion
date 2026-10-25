"""Extended tasks: gate racing, landing on a moving platform, tracking an evasive target, structure inspection and area coverage.

They follow the :class:`~aevion.envs.tasks.Task` protocol (goals in ``env.target`` / ``env.target_vel``, extra observation features, reward, success).
Episode bookkeeping is kept in NumPy arrays and converted at the boundary, as :class:`~aevion.envs.tasks.ConnectedNavigationTask` does, so the tasks
run on every backend. Tasks that can fail in task-specific ways (landing off the platform, flying into the structure) implement ``extra_crash(env)``,
which the environment ORs into its crash flag.

Difficulty can be changed while running (``task.set_difficulty(level)`` with ``level ∈ [0, 1]``) which is what :class:`~aevion.ml.curriculum.Curriculum`
drives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .tasks import TASKS, Task

__all__ = ["GateRacingTask", "LandingTask", "TargetTrackingTask", "InspectionTask", "AreaCoverageTask"]


def _rng(env: Any) -> np.random.Generator:
    B = env.B
    return np.random.default_rng(int(B.to_numpy(B.rand((1,), env.gen))[0] * 2**31))


def _mask(env: Any, mask: Any) -> np.ndarray:
    return np.asarray(env.B.to_numpy(mask)).astype(bool)


def _put(env: Any, name: str, new: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Store ``new`` rows where ``m`` is true (creating the array on first use)."""
    old = getattr(env, name, None)
    cur = new.copy() if old is None else np.where(m.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)
    setattr(env, name, cur)
    return cur


def _np(env: Any, x: Any) -> np.ndarray:
    return np.asarray(env.B.to_numpy(x), float)


def _bool(env: Any, a: np.ndarray) -> Any:
    return env.B.asarray(np.asarray(a, float)) > 0.5


@dataclass
class GateRacingTask(Task):
    """Fly through ``n_gates`` circular gates in order; a gate counts when the path crosses its plane in the travel direction within ``radius`` of its centre."""

    name: str = "gate_racing"
    n_gates: int = 4
    radius: float = 0.8
    box: Tuple = ((-4.0, -4.0, 1.0), (4.0, 4.0, 3.0))
    extra_obs_dim = 9
    bonus: float = 10.0
    miss_penalty: float = 2.0
    scale: float = 5.0

    def set_difficulty(self, level: float) -> None:
        level = float(np.clip(level, 0.0, 1.0))
        self.radius = 1.4 - 0.9 * level  # narrower gates are harder

    def reset(self, env, mask):
        B, n, K = env.B, env.n, self.n_gates
        m, rng = _mask(env, mask), _rng(env)
        lo, hi = np.asarray(self.box[0]), np.asarray(self.box[1])
        gates = np.zeros((n, K, 3))
        rad = 0.5 * float(min(hi[0] - lo[0], hi[1] - lo[1]))  # a loop of gates around the origin: smooth turns, gates a few metres apart
        for e in range(n):
            th = rng.uniform(0, 2 * math.pi) + 2 * math.pi * np.arange(K) / K + rng.uniform(-0.25, 0.25, K) * (2 * math.pi / K)
            r = rng.uniform(0.6, 1.0, K) * rad
            gates[e] = np.stack([r * np.cos(th), r * np.sin(th), rng.uniform(lo[2], hi[2], K)], axis=1)
        nxt = np.roll(gates, -1, axis=1)
        prv = np.roll(gates, 1, axis=1)
        d = nxt - prv
        norm = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        _put(env, "_gates", gates, m)
        _put(env, "_gnorm", norm, m)
        _put(env, "_gidx", np.zeros(n), m)
        _put(env, "_gprev", np.zeros((n, 3)), m)
        _put(env, "_gmiss", np.zeros(n), np.ones(n, bool))
        _put(env, "_gpass", np.zeros(n), np.ones(n, bool))
        self._set_target(env)
        env.target_vel = env.target * 0.0
        env.target_yaw = B.zeros(n)

    def _current(self, env):
        n, K = env.n, self.n_gates
        idx = np.clip(env._gidx.astype(int), 0, K - 1)
        return env._gates[np.arange(n), idx], env._gnorm[np.arange(n), idx], idx

    def _set_target(self, env):
        c, _, _ = self._current(env)
        env.target = env.B.asarray(c)

    def start_state(self, env, n):
        c, nrm, _ = self._current(env)
        return env.B.asarray(c - 2.0 * nrm * np.array([1, 1, 0.0]))  # two metres upstream of the first gate, at its height

    def post_reset(self, env, mask):
        m = _mask(env, mask)
        env._gprev = np.where(m[:, None], _np(env, env.state.pos), env._gprev)
        c, _, _ = self._current(env)
        env._prev_dist = env.B.where(mask, env.B.norm(env.state.pos - env.target, axis=-1), getattr(env, "_prev_dist", env.B.zeros(env.n)))

    def update(self, env):
        _n, K = env.n, self.n_gates
        pos = _np(env, env.state.pos)
        c, nrm, idx = self._current(env)
        active = env._gidx < K
        s_prev = ((env._gprev - c) * nrm).sum(-1)
        s_now = ((pos - c) * nrm).sum(-1)
        crossed = active & (s_prev < 0) & (s_now >= 0)
        t = np.where(crossed, -s_prev / np.maximum(s_now - s_prev, 1e-9), 0.0)
        hit = env._gprev + t[:, None] * (pos - env._gprev)
        lateral = np.linalg.norm(hit - c - ((hit - c) * nrm).sum(-1, keepdims=True) * nrm, axis=-1)
        passed = crossed & (lateral < self.radius)
        env._gpass = passed.astype(float)
        env._gmiss = (crossed & ~passed).astype(float)
        env._gidx = env._gidx + passed
        env._gprev = pos
        self._set_target(env)

    def extra_obs(self, env):
        n, K = env.n, self.n_gates
        pos = _np(env, env.state.pos)
        idx = np.clip(env._gidx.astype(int), 0, K - 1)
        nxt = np.clip(idx + 1, 0, K - 1)
        c = env._gates[np.arange(n), idx]
        c2 = env._gates[np.arange(n), nxt]
        nrm = env._gnorm[np.arange(n), idx]
        return env.B.asarray(np.concatenate([(c - pos) / self.scale, nrm, (c2 - pos) / self.scale], axis=1))

    def reward(self, env, action, prev_action):
        B = env.B
        d = B.norm(env.state.pos - env.target, axis=-1)
        prog = env._prev_dist - d
        env._prev_dist = d
        # a new gate resets the progress reference so the jump to the next target is not scored as lost progress
        jump = env.B.asarray(env._gpass) > 0.5
        env._prev_dist = B.where(jump, B.norm(env.state.pos - env.target, axis=-1), env._prev_dist)
        prog = B.where(jump, prog * 0.0, prog)
        return (
            2.0 * prog
            + self.bonus * env.B.asarray(env._gpass)
            - self.miss_penalty * env.B.asarray(env._gmiss)
            - 0.01 * B.norm(action - prev_action, axis=-1)
        )

    def success(self, env):
        return env.B.asarray((env._gidx >= self.n_gates).astype(float)) > 0.5

    def terminate_on_success(self):
        return True

    def to_dict(self):
        return dict(name=self.name, n_gates=self.n_gates, radius=self.radius)


@dataclass
class LandingTask(Task):
    """Land on a platform that drifts across the ground: touchdown inside ``zone`` at a speed below ``max_touch_speed`` succeeds, anything else fails."""

    name: str = "landing"
    zone: float = 0.6
    platform_speed: Tuple = (0.0, 0.8)
    start_height: Tuple = (2.5, 4.0)
    start_offset: float = 2.0
    max_touch_speed: float = 1.2
    touch_margin: float = 0.15
    allows_ground = True
    extra_obs_dim = 3

    def set_difficulty(self, level: float) -> None:
        level = float(np.clip(level, 0.0, 1.0))
        self.platform_speed = (0.0, 1.2 * level)

    def reset(self, env, mask):
        B, n = env.B, env.n
        m, rng = _mask(env, mask), _rng(env)
        ang = rng.uniform(0, 2 * math.pi, n)
        spd = rng.uniform(self.platform_speed[0], self.platform_speed[1], n)
        _put(env, "_pf_p0", np.stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n)], 1), m)
        _put(env, "_pf_v", np.stack([spd * np.cos(ang), spd * np.sin(ang)], 1), m)
        _put(env, "_pf_t0", np.zeros(n) + env.sim_time, m)
        _put(env, "_pf_off", rng.uniform(-self.start_offset, self.start_offset, (n, 2)), m)
        _put(env, "_pf_h", rng.uniform(self.start_height[0], self.start_height[1], n), m)
        self._eval(env)
        env.target_yaw = B.zeros(n)

    def _platform(self, env):
        t = env.sim_time - env._pf_t0
        return env._pf_p0 + env._pf_v * t[:, None]

    def _eval(self, env):
        p = self._platform(env)
        env.target = env.B.asarray(np.concatenate([p, np.zeros((env.n, 1))], axis=1))
        env.target_vel = env.B.asarray(np.concatenate([env._pf_v, np.zeros((env.n, 1))], axis=1))

    def start_state(self, env, n):
        p = self._platform(env)
        return env.B.asarray(np.concatenate([p + env._pf_off, env._pf_h[:, None]], axis=1))

    def start_velocity(self, env, n):
        return env.target_vel

    def update(self, env):
        self._eval(env)

    def _landed(self, env):
        s = env.state
        pos, vel = _np(env, s.pos), _np(env, s.vel)
        contact = (
            pos[:, 2] <= getattr(env.sim, "ground_offset", 0.05) + self.touch_margin
        )  # velocity-commanded vehicles settle just above the contact height
        xy = np.linalg.norm(pos[:, :2] - _np(env, env.target)[:, :2], axis=1)
        rel_speed = np.linalg.norm(vel - _np(env, env.target_vel), axis=1)
        return contact, xy, rel_speed

    def success(self, env):
        contact, xy, rel = self._landed(env)
        return _bool(env, contact & (xy < self.zone) & (rel < self.max_touch_speed))

    def extra_crash(self, env):
        contact, xy, rel = self._landed(env)
        return _bool(env, contact & ((xy >= self.zone) | (rel >= self.max_touch_speed)))

    def terminate_on_success(self):
        return True

    def extra_obs(self, env):
        return env.B.asarray(_np(env, env.target_vel) / 2.0)

    def reward(self, env, action, prev_action):
        B, s = env.B, env.state
        xy = B.norm(s.pos[:, :2] - env.target[:, :2], axis=-1)
        align = B.exp(-xy)
        z = s.pos[:, 2]
        rel_v = B.norm(s.vel - env.target_vel, axis=-1)
        low = B.exp(-z)  # reward being low only while aligned
        landed = B.asarray(_np(env, self.success(env)))
        return align + 0.6 * align * low - 0.15 * rel_v * B.exp(-z) - 0.01 * B.norm(action - prev_action, axis=-1) + 20.0 * landed

    def to_dict(self):
        return dict(name=self.name, zone=self.zone, platform_speed=list(self.platform_speed))


@dataclass
class TargetTrackingTask(Task):
    """Stay with a target whose velocity is a bounded random process (unpredictable, unlike the scripted circle and figure-eight of ``tracking``)."""

    name: str = "target_tracking"
    max_speed: float = 1.2
    agility: float = 1.5  # std of the (correlated) random acceleration [m/s^2]
    box: Tuple = ((-4.0, -4.0, 1.2), (4.0, 4.0, 3.0))
    success_radius: float = 0.6

    def set_difficulty(self, level: float) -> None:
        level = float(np.clip(level, 0.0, 1.0))
        self.max_speed = 0.3 + 1.7 * level

    def reset(self, env, mask):
        B, n = env.B, env.n
        m, rng = _mask(env, mask), _rng(env)
        lo, hi = np.asarray(self.box[0]), np.asarray(self.box[1])
        _put(env, "_tt_pos", (lo + hi) / 2 + rng.uniform(-0.35, 0.35, (n, 3)) * (hi - lo), m)
        _put(env, "_tt_vel", np.zeros((n, 3)), m)
        _put(env, "_tt_acc", np.zeros((n, 3)), m)
        env.target = B.asarray(env._tt_pos)
        env.target_vel = B.asarray(env._tt_vel)
        env.target_yaw = B.zeros(n)

    def start_state(self, env, n):
        return env.target + (env.B.rand((n, 3), env.gen) - 0.5) * 0.6

    def update(self, env):
        n = env.n
        dt = env.ctrl_dt
        rng = _rng(env)
        lo, hi = np.asarray(self.box[0]), np.asarray(self.box[1])
        margin = 0.8
        inner_lo, inner_hi = lo + margin, hi - margin
        pull = -(env._tt_pos - np.clip(env._tt_pos, inner_lo, inner_hi)) * 2.0  # a soft wall: the target turns back smoothly near the edge of its box
        k = 0.92  # the random acceleration is correlated over about a second, so the target manoeuvres rather than jitters
        env._tt_acc = k * env._tt_acc + math.sqrt(1 - k * k) * self.agility * rng.normal(size=(n, 3)) * np.array([1.0, 1.0, 0.4])
        v = env._tt_vel * 0.98 + (env._tt_acc + pull) * dt
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, self.max_speed / np.maximum(sp, 1e-9))
        p = np.clip(env._tt_pos + v * dt, lo, hi)
        env._tt_pos, env._tt_vel = p, v
        env.target, env.target_vel = env.B.asarray(p), env.B.asarray(v)

    def reward(self, env, action, prev_action):
        B, s = env.B, env.state
        e = B.norm(s.pos - env.target, axis=-1)
        ev = B.norm(s.vel - env.target_vel, axis=-1)
        return B.exp(-2.0 * e) + 0.2 * B.exp(-ev) - 0.01 * B.norm(action - prev_action, axis=-1)

    def success(self, env):
        return env.B.norm(env.state.pos - env.target, axis=-1) < self.success_radius

    def to_dict(self):
        return dict(name=self.name, max_speed=self.max_speed, agility=self.agility)


@dataclass
class InspectionTask(Task):
    """Inspect points on the surface of a vertical cylinder: a point counts when the vehicle comes within ``sense`` of it; flying into the cylinder crashes."""

    name: str = "inspection"
    structure_radius: float = 1.5
    structure_height: float = 3.0
    n_ring: int = 8
    n_levels: int = 3
    sense: float = 0.9
    clearance: float = 0.3
    extra_obs_dim = 4
    scale: float = 6.0

    def set_difficulty(self, level: float) -> None:
        level = float(np.clip(level, 0.0, 1.0))
        self.sense = 1.6 - 0.9 * level

    @property
    def n_points(self) -> int:
        return self.n_ring * self.n_levels

    def _surface(self) -> np.ndarray:
        a = 2 * math.pi * np.arange(self.n_ring) / self.n_ring
        z = (np.arange(self.n_levels) + 0.5) / self.n_levels * self.structure_height
        pts = [[self.structure_radius * math.cos(t), self.structure_radius * math.sin(t), h] for h in z for t in a]
        return np.asarray(pts)

    def reset(self, env, mask):
        B, n = env.B, env.n
        m = _mask(env, mask)
        _put(env, "_ins_seen", np.zeros((n, self.n_points), bool), m)
        _put(env, "_ins_new", np.zeros(n), np.ones(n, bool))
        self._retarget(env)
        env.target_vel = env.target * 0.0
        env.target_yaw = B.zeros(n)

    def _retarget(self, env):
        pts = self._surface()
        pos = _np(env, env.state.pos) if hasattr(env, "state") else np.zeros((env.n, 3))
        d = np.linalg.norm(pts[None] - pos[:, None], axis=-1)
        d = np.where(env._ins_seen, np.inf, d)
        k = np.argmin(d, axis=1)
        tgt = pts[k] * np.array([1.35, 1.35, 1.0])  # aim slightly off the wall
        tgt = np.where(np.isfinite(d.min(axis=1))[:, None], tgt, pos)
        env.target = env.B.asarray(tgt)

    def start_state(self, env, n):
        rng = _rng(env)
        a = rng.uniform(0, 2 * math.pi, n)
        r = self.structure_radius + rng.uniform(2.0, 3.0, n)
        return env.B.asarray(np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(1.0, 2.5, n)], 1))

    def update(self, env):
        pts = self._surface()
        pos = _np(env, env.state.pos)
        d = np.linalg.norm(pts[None] - pos[:, None], axis=-1)
        newly = (d < self.sense) & ~env._ins_seen
        env._ins_new = newly.sum(axis=1).astype(float)
        env._ins_seen = env._ins_seen | newly
        self._retarget(env)

    def extra_obs(self, env):
        pos = _np(env, env.state.pos)
        rel = (_np(env, env.target) - pos) / self.scale
        frac = env._ins_seen.mean(axis=1, keepdims=True)
        return env.B.asarray(np.concatenate([rel, frac], axis=1))

    def extra_crash(self, env):
        pos = _np(env, env.state.pos)
        inside = (np.linalg.norm(pos[:, :2], axis=1) < self.structure_radius + self.clearance) & (pos[:, 2] < self.structure_height)
        return _bool(env, inside)

    def reward(self, env, action, prev_action):
        B = env.B
        d = B.norm(env.state.pos - env.target, axis=-1)
        return 3.0 * env.B.asarray(env._ins_new) + 0.05 * B.exp(-d) - 0.01 * B.norm(action - prev_action, axis=-1)

    def success(self, env):
        return env.B.asarray(env._ins_seen.all(axis=1).astype(float)) > 0.5

    def terminate_on_success(self):
        return True

    def to_dict(self):
        return dict(name=self.name, structure_radius=self.structure_radius, n_points=self.n_points, sense=self.sense)


@dataclass
class AreaCoverageTask(Task):
    """Visit every cell of a ``size × size`` metre area on a ``cells × cells`` grid; a cell is seen when the vehicle passes within ``footprint`` of its centre."""

    name: str = "area_coverage"
    size: float = 6.0
    cells: int = 6
    footprint: float = 1.0
    altitude: float = 2.0
    done_fraction: float = 0.95
    extra_obs_dim = 4

    def set_difficulty(self, level: float) -> None:
        level = float(np.clip(level, 0.0, 1.0))
        self.footprint = 1.8 - 0.9 * level

    def _centres(self) -> np.ndarray:
        e = (np.arange(self.cells) + 0.5) / self.cells * self.size - self.size / 2
        gx, gy = np.meshgrid(e, e, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel()], 1)

    def reset(self, env, mask):
        B, n = env.B, env.n
        m = _mask(env, mask)
        _put(env, "_cov_seen", np.zeros((n, self.cells**2), bool), m)
        _put(env, "_cov_new", np.zeros(n), np.ones(n, bool))
        self._retarget(env)
        env.target_vel = env.target * 0.0
        env.target_yaw = B.zeros(n)

    def _retarget(self, env):
        c = self._centres()
        pos = _np(env, env.state.pos) if hasattr(env, "state") else np.zeros((env.n, 3))
        d = np.linalg.norm(c[None] - pos[:, None, :2], axis=-1)
        d = np.where(env._cov_seen, np.inf, d)
        k = np.argmin(d, axis=1)
        tgt = np.concatenate([c[k], np.full((env.n, 1), self.altitude)], axis=1)
        none_left = ~np.isfinite(d.min(axis=1))
        tgt = np.where(none_left[:, None], np.concatenate([pos[:, :2], np.full((env.n, 1), self.altitude)], axis=1), tgt)
        env.target = env.B.asarray(tgt)

    def start_state(self, env, n):
        rng = _rng(env)
        return env.B.asarray(np.concatenate([rng.uniform(-self.size / 2, self.size / 2, (n, 2)), np.full((n, 1), self.altitude)], axis=1))

    def update(self, env):
        c = self._centres()
        pos = _np(env, env.state.pos)
        d = np.linalg.norm(c[None] - pos[:, None, :2], axis=-1)
        newly = (d < self.footprint) & ~env._cov_seen
        env._cov_new = newly.sum(axis=1).astype(float)
        env._cov_seen = env._cov_seen | newly
        self._retarget(env)

    def extra_obs(self, env):
        pos = _np(env, env.state.pos)
        rel = (_np(env, env.target) - pos) / self.size
        return env.B.asarray(np.concatenate([rel, env._cov_seen.mean(axis=1, keepdims=True)], axis=1))

    def reward(self, env, action, prev_action):
        B = env.B
        alt = env.state.pos[:, 2] - self.altitude
        return 5.0 * env.B.asarray(env._cov_new) / self.cells**2 * self.cells - 0.1 * alt * alt - 0.01 * B.norm(action - prev_action, axis=-1) + 0.02

    def success(self, env):
        return env.B.asarray((env._cov_seen.mean(axis=1) >= self.done_fraction).astype(float)) > 0.5

    def terminate_on_success(self):
        return True

    def to_dict(self):
        return dict(name=self.name, size=self.size, cells=self.cells, footprint=self.footprint)


TASKS.update(
    {
        "gate_racing": GateRacingTask,
        "landing": LandingTask,
        "target_tracking": TargetTrackingTask,
        "inspection": InspectionTask,
        "area_coverage": AreaCoverageTask,
    }
)
