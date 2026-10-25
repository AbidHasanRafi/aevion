"""Blockage of directional (mmWave / sub-THz) links.

Three mechanisms:

* **random static blockers** (buildings, trees, vehicles): the LoS survival probability of a link of length ``r`` is
  ``exp(-beta r)`` with ``beta = 2 lambda R / pi`` for blockers of density ``lambda`` and radius ``R`` (Bai & Heath, IEEE TWC 2014),
  with a height correction when blockers are lower than the transmitter;
* **time-varying blockage**: a two-state Markov process (clear / blocked) with exponential holding times that adds a
  fixed attenuation while blocked (typical 15-30 dB at mmWave);
* **self-blockage**: the airframe shadows the antenna for directions inside a cone in the body frame.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

__all__ = ["BlockageProcess", "blockage_probability", "blockage_rate", "self_blockage_loss_db"]


def blockage_rate(density_per_m2: float, blocker_radius_m: float) -> float:
    """``beta = 2 lambda R / pi`` [1/m] for randomly placed blockers (Boolean model)."""
    return 2.0 * density_per_m2 * blocker_radius_m / math.pi


def blockage_probability(
    distance_m: Any, density_per_m2: float, blocker_radius_m: float, h_tx: float = 25.0, h_rx: float = 1.5, h_blocker: float = 1.7
) -> np.ndarray:
    """Probability that a link is blocked: ``1 - exp(-beta r_eff)``.

    Blockers lower than the transmitter only block the part of the path where the ray is below their height, so the
    effective length is ``r * clip((h_blocker - h_rx) / (h_tx - h_rx), 0, 1)``.
    """
    frac = 1.0 if h_tx <= h_rx else float(np.clip((h_blocker - h_rx) / (h_tx - h_rx), 0.0, 1.0))
    return 1.0 - np.exp(-blockage_rate(density_per_m2, blocker_radius_m) * np.asarray(distance_m, float) * frac)


class BlockageProcess:
    """Two-state Markov blockage per link: clear -> blocked at rate ``arrival_rate_hz``, blocked -> clear after ``mean_duration_s``."""

    def __init__(
        self,
        n_links: int,
        arrival_rate_hz: float = 0.1,
        mean_duration_s: float = 0.5,
        attenuation_db: float = 20.0,
        rng: Optional[np.random.Generator] = None,
    ):
        if arrival_rate_hz < 0 or mean_duration_s <= 0:
            raise ValueError("arrival rate must be >= 0 and mean duration > 0")
        self.n, self.lam, self.mu, self.att = n_links, arrival_rate_hz, 1.0 / mean_duration_s, attenuation_db
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        p = self.stationary_blocked_probability
        self.blocked = self.rng.random(self.n) < p  # start in steady state

    @property
    def stationary_blocked_probability(self) -> float:
        return self.lam / (self.lam + self.mu) if self.lam > 0 else 0.0

    def step(self, dt: float) -> np.ndarray:
        """Advance by ``dt`` and return the attenuation [dB] of every link (0 when clear)."""
        u = self.rng.random(self.n)
        p_on = 1.0 - math.exp(-self.lam * dt)
        p_off = 1.0 - math.exp(-self.mu * dt)
        self.blocked = np.where(self.blocked, u >= p_off, u < p_on)
        return np.where(self.blocked, self.att, 0.0)


def self_blockage_loss_db(
    direction_body: Any, shadow_half_angle_deg: float = 40.0, axis_body: Any = (0.0, 0.0, -1.0), loss_db: float = 15.0, roll_off_deg: float = 10.0
) -> np.ndarray:
    """Airframe shadowing: ``loss_db`` for directions within ``shadow_half_angle_deg`` of ``axis_body`` (default: straight down
    through the battery/frame) with a smooth raised-cosine edge of width ``roll_off_deg``. ``direction_body`` is ``(..., 3)``."""
    d = np.asarray(direction_body, float)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    ax = np.asarray(axis_body, float)
    ax = ax / np.linalg.norm(ax)
    ang = np.degrees(np.arccos(np.clip(d @ ax, -1.0, 1.0)))
    t = np.clip((ang - shadow_half_angle_deg) / max(roll_off_deg, 1e-9), 0.0, 1.0)
    return loss_db * 0.5 * (1.0 + np.cos(math.pi * t))
