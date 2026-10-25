"""Domain randomisation of plant parameters, wind and observation noise.

Ranges are *relative* (``0.1`` means ±10 %) unless noted. Parameters are re-sampled
per-environment whenever that environment resets, so a batch always contains a spread of
different vehicles and weather: the standard recipe for robust sim-to-real policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DomainRandomization:
    mass: float = 0.0
    inertia: float = 0.0
    kT: float = 0.0  # thrust coefficient
    kQ: float = 0.0
    tau: float = 0.0  # motor time constant
    drag: float = 0.0
    w_max: float = 0.0
    com: float = 0.0  # absolute centre-of-mass offset half-range [m]
    wind_speed: float = 0.0  # absolute max constant wind [m/s]
    obs_noise: float = 0.0  # absolute std added to normalised observations
    action_noise: float = 0.0
    rotor_failure_prob: float = 0.0  # per-episode probability that one random rotor is dead

    @classmethod
    def mild(cls):
        return cls(mass=0.1, inertia=0.1, kT=0.05, kQ=0.05, tau=0.2, drag=0.2, w_max=0.05, com=0.005, wind_speed=2.0, obs_noise=0.01)

    @classmethod
    def strong(cls):
        return cls(
            mass=0.25, inertia=0.25, kT=0.15, kQ=0.15, tau=0.4, drag=0.5, w_max=0.1, com=0.015, wind_speed=6.0, obs_noise=0.03, action_noise=0.02
        )

    def to_dict(self):
        return asdict(self)


class ParamRandomizer:
    """Holds nominal parameter tensors and writes randomised copies into a :class:`BatchParams`."""

    def __init__(self, B, params, cfg: DomainRandomization):
        self.B, self.p, self.cfg = B, params, cfg
        self.nom = {
            k: B.clone(getattr(params, k))
            for k in (
                "mass",
                "inertia",
                "inertia_inv",
                "kT",
                "kQ",
                "tau_up",
                "tau_down",
                "drag_lin",
                "drag_quad",
                "w_max",
                "com_offset",
                "rotor_health",
            )
        }

    def apply(self, mask, gen):
        """Re-sample parameters for environments where ``mask`` is True."""
        B, p, c, n = self.B, self.p, self.cfg, self.p.n

        def fac(r, shape):
            return 1.0 + r * (2.0 * B.rand(shape, gen) - 1.0)

        def sel(new, old):
            m = mask.reshape(n, *([1] * (new.ndim - 1)))
            return B.where(m, new, old)

        nom = self.nom
        k = fac(c.mass, (n, 1))
        p.mass = sel(nom["mass"] * k, p.mass)
        k = fac(c.inertia, (n, 1, 1))
        p.inertia = sel(nom["inertia"] * k, p.inertia)
        p.inertia_inv = sel(nom["inertia_inv"] / k, p.inertia_inv)
        p.kT = sel(nom["kT"] * fac(c.kT, (n, 1)), p.kT)
        p.kQ = sel(nom["kQ"] * fac(c.kQ, (n, 1)), p.kQ)
        k = fac(c.tau, (n, 1))
        p.tau_up = sel(nom["tau_up"] * k, p.tau_up)
        p.tau_down = sel(nom["tau_down"] * k, p.tau_down)
        k = fac(c.drag, (n, 1))
        p.drag_lin = sel(nom["drag_lin"] * k, p.drag_lin)
        p.drag_quad = sel(nom["drag_quad"] * k, p.drag_quad)
        p.w_max = sel(nom["w_max"] * fac(c.w_max, (n, 1)), p.w_max)
        p.com_offset = sel(c.com * (2.0 * B.rand((n, 3), gen) - 1.0), p.com_offset)
        health = nom["rotor_health"] * 1.0
        if c.rotor_failure_prob > 0:
            M = health.shape[1]
            fail = B.rand((n, 1), gen) < c.rotor_failure_prob
            idx = B.rand((n, 1), gen) * M
            cols = B.asarray(np.arange(M))[None, :]
            if B.name == "numpy":
                which = np.floor(idx)
            else:
                which = idx.floor()
            dead = (cols == which) & fail
            health = health * (1.0 - dead * 1.0)
        p.rotor_health = sel(health, p.rotor_health)
