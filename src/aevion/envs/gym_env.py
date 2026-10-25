"""Gymnasium and Stable-Baselines3 adapters around :class:`VecQuadEnv`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .vec_env import EnvConfig, VecQuadEnv

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAVE_GYM = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    HAVE_GYM = False


def _to_np(B, x, dtype=np.float32):
    return np.asarray(B.to_numpy(x), dtype=dtype)


if HAVE_GYM:

    class QuadrotorEnv(gym.Env):
        """Single-drone Gymnasium environment (CPU/NumPy interface, any backend inside).

        Observations and actions are ``float32`` NumPy arrays. Pass any :class:`EnvConfig`
        field as a keyword, e.g. ``QuadrotorEnv(task="waypoints", action_mode="velocity")``.
        """

        metadata = {"render_modes": []}

        def __init__(self, config: Optional[EnvConfig] = None, world=None, **kw):
            super().__init__()
            self.venv = VecQuadEnv(1, config, world=world, **kw)
            od, ad = self.venv.spaces()
            self.observation_space = spaces.Box(-np.inf, np.inf, (od,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (ad,), dtype=np.float32)
            self._needs_reset = False

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            obs = self.venv.reset(seed=seed)
            self._needs_reset = False
            return _to_np(self.venv.B, obs[0]), {}

        def step(self, action):
            B = self.venv.B
            obs, r, term, trunc, info = self.venv.step(B.asarray(np.asarray(action, dtype=np.float32)[None]))
            done = bool(term[0]) or bool(trunc[0])
            o = info["final_observation"] if done else obs
            out_info = dict(success=bool(info["success"][0]), crashed=bool(info["crashed"][0]), pos_error=float(B.to_numpy(info["pos_error"])[0]))
            return _to_np(B, o[0]), float(B.to_numpy(r)[0]), bool(term[0]), bool(trunc[0]), out_info

        @property
        def unwrapped_sim(self):
            return self.venv.sim

    class QuadVectorEnv(gym.vector.VectorEnv):
        """Gymnasium ``VectorEnv`` over a :class:`VecQuadEnv` (``num_envs`` drones simulated in one batch).

        Finished sub-environments reset inside the same ``step`` call (Gymnasium's ``SAME_STEP`` auto-reset mode): the returned observation is the first of
        the new episode and the last observation of the old one is in ``info["final_obs"]`` (with the mask ``info["_final_obs"]``), as the Gymnasium
        documentation prescribes.
        """

        metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP, "render_modes": []}

        def __init__(self, num_envs: int = 16, config: Optional[EnvConfig] = None, world=None, **kw):
            self.venv = VecQuadEnv(num_envs, config, world=world, **kw)
            od, ad = self.venv.spaces()
            self.num_envs = int(num_envs)
            self.single_observation_space = spaces.Box(-np.inf, np.inf, (od,), dtype=np.float32)
            self.single_action_space = spaces.Box(-1.0, 1.0, (ad,), dtype=np.float32)
            self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
            self.render_mode = None
            self.closed = False

        def reset(self, *, seed: Optional[int] = None, options=None):
            return _to_np(self.venv.B, self.venv.reset(seed=seed)), {}

        def step(self, actions):
            B = self.venv.B
            obs, r, term, trunc, info = self.venv.step(B.asarray(np.asarray(actions, dtype=np.float32)))
            term_np, trunc_np = _to_np(B, term, bool), _to_np(B, trunc, bool)
            done = term_np | trunc_np
            out: Dict[str, Any] = dict(success=_to_np(B, info["success"], bool))
            if done.any():
                fo = _to_np(B, info["final_observation"])
                final = np.empty(self.num_envs, dtype=object)
                for i in np.nonzero(done)[0]:
                    final[i] = fo[i]
                out["final_obs"], out["_final_obs"] = final, done
            return _to_np(B, obs), _to_np(B, r), term_np, trunc_np, out

        def close_extras(self, **kwargs):
            pass

    def register_envs():
        for name, task in (("Hover", "hover"), ("Waypoints", "waypoints"), ("Tracking", "tracking"), ("Navigation", "navigation")):
            try:
                gym.register(id=f"Aevion-{name}-v0", entry_point="aevion.envs.gym_env:QuadrotorEnv", kwargs=dict(task=task))
            except Exception:  # already registered
                pass

    register_envs()


def make_sb3_vec_env(n_envs: int = 16, config: Optional[EnvConfig] = None, world=None, **kw):
    """Wrap a :class:`VecQuadEnv` as a Stable-Baselines3 ``VecEnv`` (observations as NumPy)."""
    from stable_baselines3.common.vec_env import VecEnv

    venv = VecQuadEnv(n_envs, config, world=world, **kw)
    od, ad = venv.spaces()

    class _SB3(VecEnv):
        def __init__(self):
            super().__init__(venv.n, spaces.Box(-np.inf, np.inf, (od,), dtype=np.float32), spaces.Box(-1.0, 1.0, (ad,), dtype=np.float32))
            self.venv = venv
            self._actions = None
            self._ep_ret = np.zeros(venv.n)
            self._ep_len = np.zeros(venv.n)

        def reset(self):
            self._ep_ret[:] = 0
            self._ep_len[:] = 0
            return _to_np(venv.B, venv.reset())

        def step_async(self, actions):
            self._actions = np.asarray(actions, dtype=np.float32)

        def step_wait(self):
            B = venv.B
            obs, r, term, trunc, info = venv.step(B.asarray(self._actions))
            r_np, term_np, trunc_np = _to_np(B, r, np.float64), _to_np(B, term, bool), _to_np(B, trunc, bool)
            done = term_np | trunc_np
            fo = _to_np(B, info["final_observation"])
            succ = _to_np(B, info["success"], bool)
            self._ep_ret += r_np
            self._ep_len += 1
            infos: List[Dict[str, Any]] = [dict() for _ in range(venv.n)]
            for i in np.nonzero(done)[0]:
                infos[i]["terminal_observation"] = fo[i]
                infos[i]["TimeLimit.truncated"] = bool(trunc_np[i] and not term_np[i])
                infos[i]["episode"] = {"r": float(self._ep_ret[i]), "l": int(self._ep_len[i])}
                infos[i]["is_success"] = bool(succ[i])
                self._ep_ret[i] = 0
                self._ep_len[i] = 0
            return _to_np(B, obs), r_np, done, infos

        def close(self):
            pass

        def get_attr(self, attr_name, indices=None):
            return [getattr(venv, attr_name, None)] * venv.n

        def set_attr(self, attr_name, value, indices=None):
            pass

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return [getattr(venv, method_name)(*args, **kwargs)] * venv.n

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * venv.n

        def get_images(self):
            return [None] * venv.n

        def seed(self, seed=None):
            venv.seed(seed or 0)
            return [seed] * venv.n

    return _SB3()
