"""Tensor-native vectorised quadrotor environment.

``VecQuadEnv`` simulates ``n_envs`` independent drones in one batch. Observations,
actions, rewards and done-flags are tensors on the chosen backend (NumPy arrays on CPU,
or torch tensors on CPU/CUDA), so a GPU policy can interact with the simulator without
any host round-trips. Finished environments auto-reset; the terminal observation is kept in
``info["final_observation"]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from ..control.allocation import Allocator
from ..control.controllers import CascadedPID, GeometricController, RateController
from ..control.reference import Reference
from ..core import rotation as rot
from ..core.backend import get_backend
from ..dynamics.multirotor import Multirotor, State
from ..environments.wind import RandomConstantWind
from ..environments.world import World
from ..sensors.ranging import Lidar
from ..vehicles.spec import get_vehicle
from .domain_rand import DomainRandomization, ParamRandomizer
from .tasks import make_task

ACTION_MODES = ("rotor", "thrust_rates", "velocity", "position")


@dataclass
class EnvConfig:
    vehicle: object = "generic_1kg"
    task: object = "hover"
    action_mode: str = "thrust_rates"
    control_dt: float = 0.02  # policy step [s]
    physics_dt: float = 0.005
    max_episode_s: float = 10.0
    backend: str = "numpy"  # "numpy" | "torch" | "cuda"
    dtype: Optional[str] = None
    lidar: bool = False
    lidar_channels: int = 1
    lidar_samples: int = 16
    lidar_range: float = 12.0
    domain_randomization: Optional[DomainRandomization] = None
    bounds: tuple = (-12.0, -12.0, 0.0, 12.0, 12.0, 8.0)
    crash_penalty: float = 10.0
    crash_on_ground: bool = True
    max_rate: tuple = (6.0, 6.0, 3.0)  # rad/s for "thrust_rates"
    max_speed: float = 4.0  # m/s for "velocity"
    max_step: float = 1.0  # m for "position"
    obs_scales: tuple = (5.0, 5.0, 5.0)  # position, velocity, angular rate
    seed: int = 0
    wind_max: float = 0.0  # shortcut: random constant wind magnitude (if no DR config)


class VecQuadEnv:
    state: State  # set by the first reset() in __init__

    """Batched drone environment with a Gym-like (but tensor-valued) API."""

    def __init__(self, n_envs: int = 64, config: Optional[EnvConfig] = None, world: Optional[World] = None, **kw):
        cfg = config or EnvConfig()
        for k, v in kw.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown EnvConfig field {k!r}")
            setattr(cfg, k, v)
        if cfg.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        self.cfg, self.n = cfg, int(n_envs)
        self.spec = get_vehicle(cfg.vehicle)
        self.task = make_task(cfg.task)
        self.B = get_backend(cfg.backend, cfg.dtype)
        B = self.B

        dr = cfg.domain_randomization
        wind = RandomConstantWind(dr.wind_speed if dr else cfg.wind_max) if (dr or cfg.wind_max) else None
        self.world = world if world is not None else World(wind=wind)
        if world is not None and wind is not None and type(self.world.wind).__name__ == "NoWind":
            self.world.wind = wind
        self.sim = Multirotor(self.spec, n=self.n, backend=B, dt=cfg.physics_dt, world=self.world, seed=cfg.seed)
        self.dr = ParamRandomizer(B, self.sim.params, dr) if dr else None
        self.gen = B.make_rng(cfg.seed + 7)
        self.decimation = max(1, round(cfg.control_dt / cfg.physics_dt))
        self.max_steps = round(cfg.max_episode_s / cfg.control_dt)
        self.ctrl_dt = self.decimation * cfg.physics_dt

        self.alloc = Allocator(self.spec, B)
        self.rate_ctrl = RateController(self.sim)
        self.vel_ctrl = CascadedPID(self.sim)
        self.pos_ctrl = GeometricController(self.sim)
        self.hover_w = self.sim.hover_speed()
        self.m, self.g = self.spec.mass, self.world.gravity.g
        self.tw = self.spec.thrust_to_weight(self.g)

        self.lidar = None
        if cfg.lidar:
            self.lidar = Lidar(
                channels=cfg.lidar_channels,
                samples=cfg.lidar_samples,
                v_fov=(0.0, 0.0) if cfg.lidar_channels == 1 else (-15.0, 15.0),
                max_range=cfg.lidar_range,
                return_points=False,
                dropout_prob=0.0,
            )
            self.lidar.bind(self.sim, cfg.seed)

        self.act_dim = self.spec.n_rotors if cfg.action_mode == "rotor" else 4
        self.obs_dim = 3 + 3 + 9 + 3 + self.act_dim + (self.lidar.n_rays if self.lidar else 0) + self.task.extra_obs_dim

        self.target = B.zeros(self.n, 3)
        self.target_vel = B.zeros(self.n, 3)
        self.target_yaw = B.zeros(self.n)
        self.steps = B.zeros(self.n)
        self.ep_return = B.zeros(self.n)
        self.sim_time = 0.0
        self.prev_action = B.zeros(self.n, self.act_dim)
        self.recorder = None
        self.swarm_start: Any = None  # set by VecSwarmEnv: initial positions and attitudes it chooses for the team
        self.swarm_euler: Any = None
        self.auto_reset = True  # wrappers that manage episodes themselves (e.g. team episodes) switch this off
        self.reset()

    # ------------------------------------------------------------- reset
    def _initial_state(self, mask) -> State:
        B, n = self.B, self.n
        pos0 = self.task.start_state(self, n)
        start_euler = getattr(self.task, "start_euler", None)  # tasks may fix the initial attitude (backend-independent reproducibility)
        if start_euler is not None:
            eul = start_euler(self, n)
        else:
            eul = B.stack(
                [(B.rand((n,), self.gen) - 0.5) * 0.2, (B.rand((n,), self.gen) - 0.5) * 0.2, (B.rand((n,), self.gen) - 0.5) * 2 * math.pi * 0.0], -1
            )
        q = rot.from_euler(B, eul[:, 0], eul[:, 1], eul[:, 2])
        new = State(
            pos=pos0,
            vel=self.task.start_velocity(self, n),
            quat=q,
            omega=B.zeros(n, 3),
            rotor=B.ones(n, self.spec.n_rotors) * self.hover_w,
            soc=B.ones(n),
            energy=B.zeros(n),
            t=self.sim_time,
        )
        return new

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.gen = self.B.make_rng(seed)
            self.world.reset(self.B, self.n, seed)
        mask = self.B.ones(self.n) > 0.0
        self._reset_mask(mask, first=True)
        return self._obs()

    def _reset_mask(self, mask, first=False):
        B, n = self.B, self.n
        if self.dr is not None:
            self.dr.apply(mask, self.gen)
        if hasattr(self.world.wind, "resample"):
            self.world.wind.resample(mask)
        self.task.reset(self, mask)  # goals first: some tasks start the vehicle relative to its goal
        new = self._initial_state(mask)
        if first:
            self.state = new
        else:
            s = self.state
            sel = lambda a, b: B.where(mask.reshape(n, *([1] * (a.ndim - 1))), a, b)  # noqa: E731
            self.state = State(
                pos=sel(new.pos, s.pos),
                vel=sel(new.vel, s.vel),
                quat=sel(new.quat, s.quat),
                omega=sel(new.omega, s.omega),
                rotor=sel(new.rotor, s.rotor),
                soc=sel(new.soc, s.soc),
                energy=sel(new.energy, s.energy),
                t=s.t,
            )
        self.task.post_reset(self, mask)
        zero = B.zeros(n)
        self.steps = B.where(mask, zero, self.steps)
        self.ep_return = B.where(mask, zero, self.ep_return)
        self.prev_action = B.where(mask[:, None], B.zeros(n, self.act_dim), self.prev_action)
        self.rate_ctrl.reset(n)
        self.vel_ctrl.reset(n)
        self.pos_ctrl.reset(n)

    # -------------------------------------------------------------- action
    def _command(self, action, state):
        """Map a normalised action in [-1, 1] to rotor-speed commands for the current physics state."""
        B, mode, spec = self.B, self.cfg.action_mode, self.spec
        if mode == "rotor":
            lo, hi = spec.w_min, spec.w_max
            w = B.where(action >= 0.0, self.hover_w + action * (hi - self.hover_w), self.hover_w + action * (self.hover_w - lo))
            return w
        if mode == "thrust_rates":
            a0 = action[:, 0]
            T = self.m * self.g * B.where(a0 >= 0.0, 1.0 + a0 * (self.tw - 1.0), 1.0 + a0)
            rate_des = action[:, 1:4] * B.asarray(self.cfg.max_rate)
            Jw = B.einsum("ij,nj->ni", self.rate_ctrl.J, state.omega)
            err = rate_des - state.omega
            tau = B.einsum("ij,nj->ni", self.rate_ctrl.J, self.rate_ctrl.kp * err) + B.cross(state.omega, Jw)
            return self.alloc.to_rotor_speeds(T, tau)
        if mode == "velocity":
            v_des = action[:, 0:3] * self.cfg.max_speed
            yaw_rate = action[:, 3] * 2.0
            ref = Reference(pos=state.pos, vel=v_des, yaw=B.zeros(self.n), yaw_rate=yaw_rate)
            # velocity-tracking cascade, yaw fixed at zero for a clean interface
            return self.vel_ctrl.act_velocity(state, v_des, B.zeros(self.n), self.sim.dt)
        # position: relative setpoint
        ref = Reference(pos=state.pos + action[:, 0:3] * self.cfg.max_step, yaw=action[:, 3] * math.pi)
        return self.pos_ctrl.act(state, ref, self.sim.dt)

    # ----------------------------------------------------------------- step
    def step(self, action):
        B, n, cfg = self.B, self.n, self.cfg
        action = B.clip(B.asarray(action).reshape(n, self.act_dim), -1.0, 1.0)
        if self.dr is not None and self.dr.cfg.action_noise > 0:
            action = B.clip(action + B.randn(action.shape, self.gen) * self.dr.cfg.action_noise, -1.0, 1.0)
        for _ in range(self.decimation):
            self.state = self.sim.step(self.state, self._command(action, self.state))
        self.sim_time = self.state.t
        self.steps = self.steps + 1.0
        self.task.update(self)
        s = self.state

        reward = self.task.reward(self, action, self.prev_action)
        R = rot.to_matrix(B, s.quat)
        flipped = R[:, 2, 2] < -0.2
        lo = B.asarray(cfg.bounds[:3])
        hi = B.asarray(cfg.bounds[3:])
        oob = ((s.pos < lo) | (s.pos > hi)).any(-1)
        if cfg.crash_on_ground and not self.task.allows_ground:
            ground = s.contact & (self.steps > 5.0)
        else:
            ground = s.impact_speed > 3.0
        crashed = flipped | oob | ground | s.collided | ~B.isfinite(s.pos).all(-1)
        extra_crash = getattr(self.task, "extra_crash", None)  # tasks with their own failure modes (landing off the platform, hitting a structure)
        if extra_crash is not None:
            crashed = crashed | extra_crash(self)
        reward = reward - cfg.crash_penalty * crashed
        success = self.task.success(self)
        terminated = crashed | (success & bool(getattr(self.task, "terminate_on_success", lambda: False)()))
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated
        self.ep_return = self.ep_return + reward

        info = dict(
            success=success,
            crashed=crashed,
            done=done,
            truncated=truncated,
            pos_error=B.norm(s.pos - self.target, axis=-1),
            episode_return=self.ep_return,
            episode_length=self.steps,
        )
        obs = self._obs(action)
        info["final_observation"] = obs
        info["final_return"] = self.ep_return
        self.prev_action = action
        if self.recorder is not None:
            self.recorder.record(self, action, reward)
        if self.auto_reset and bool(done.any()):
            self._reset_mask(done)
            obs = self._obs(self.prev_action)
        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------- observations
    def _obs(self, action=None):
        B, s, cfg = self.B, self.state, self.cfg
        ps, vs, ws = cfg.obs_scales
        R = rot.to_matrix(B, s.quat).reshape(self.n, 9)
        a = self.prev_action if action is None else action
        parts = [(self.target - s.pos) / ps, (s.vel - self.target_vel) / vs, R, s.omega / ws, a]
        if self.lidar is not None:
            r = self.lidar.measure(s)["ranges"].reshape(self.n, -1)
            rng = cfg.lidar_range
            parts.append(1.0 - B.minimum(B.where(B.isfinite(r), r, B.full(r.shape, rng)), B.full(r.shape, rng)) / rng)
        if self.task.extra_obs_dim:
            parts.append(self.task.extra_obs(self))
        obs = B.cat(parts, axis=-1)
        if self.dr is not None and self.dr.cfg.obs_noise > 0:
            obs = obs + B.randn(obs.shape, self.gen) * self.dr.cfg.obs_noise
        return obs

    # -------------------------------------------------------------- helpers
    def hover_action(self):
        return self.B.zeros(self.n, self.act_dim)

    def spaces(self):
        """Return ``(obs_dim, act_dim)``; Gymnasium spaces are built by the wrappers."""
        return self.obs_dim, self.act_dim

    def seed(self, seed):
        self.gen = self.B.make_rng(seed)

    def close(self):
        pass

    def info(self) -> dict:
        return dict(
            n_envs=self.n,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            backend=self.B.name,
            device=self.B.device,
            action_mode=self.cfg.action_mode,
            task=self.task.to_dict(),
            control_dt=self.ctrl_dt,
            vehicle=self.spec.name,
        )
