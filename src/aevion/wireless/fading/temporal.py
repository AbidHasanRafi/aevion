"""Time-correlated small-scale fading for moving terminals.

:class:`JakesProcess` is a sum-of-phasors (Clarke/Jakes) simulator,
``h(t) = (1 / sqrt(N)) sum_n exp(j (2 pi f_d cos(alpha_n) t + phi_n))``, with arrival angles ``alpha_n = 2 pi (n + 1/2) / N + theta``
evenly spaced around the circle (random common offset ``theta``) and independent uniform phases ``phi_n``. For even ``N`` the angles
form opposite pairs, so the autocorrelation is real and equals ``(1/N) sum_n cos(2 pi f_d tau cos(alpha_n))``, which approximates the
Bessel ``J0(2 pi f_d tau)`` closely for ``N >= 16`` over the first lobes. The independent phases make ``h`` circularly Gaussian by the
central limit theorem (Rayleigh envelope, exponential power). Oscillator phases integrate the instantaneous Doppler, so the process
stays consistent when the UAV changes speed.

:class:`FadingProcess` adds a Rician LoS component per link on top of it.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

__all__ = ["FadingProcess", "JakesProcess"]


class JakesProcess:
    """Complex Rayleigh fading (unit mean power) for ``n_links`` independent links; ``n_osc`` must be even and at least 8."""

    def __init__(self, n_links: int, n_osc: int = 32, rng: Optional[np.random.Generator] = None):
        if n_links < 1 or n_osc < 8 or n_osc % 2:
            raise ValueError("need n_links >= 1 and an even n_osc >= 8")
        self.n_links, self.n_osc = n_links, n_osc
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        m, n = self.n_osc, self.n_links
        theta = self.rng.uniform(0.0, 2.0 * math.pi / m, size=(n, 1))
        k = np.arange(m)[None, :]
        self._cos_alpha = np.cos(2.0 * math.pi * (k + 0.5) / m + theta)  # (L, M)
        self._phase = self.rng.uniform(-math.pi, math.pi, size=(n, m))

    def current(self) -> np.ndarray:
        """The complex gain at the present phase (no time advance)."""
        return np.exp(1j * self._phase).sum(axis=1) / math.sqrt(self.n_osc)

    def step(self, f_d_hz: Any, dt: float) -> np.ndarray:
        """Advance every link by ``dt`` with its current maximum Doppler ``f_d_hz`` (scalar or ``(L,)``); returns complex ``h``."""
        f_d = np.broadcast_to(np.asarray(f_d_hz, float), (self.n_links,))[:, None]
        self._phase = self._phase + 2.0 * math.pi * f_d * dt * self._cos_alpha
        return self.current()


class FadingProcess:
    """Per-link fading *power* gains (unit mean): Rician with elevation/LoS dependent K, Rayleigh when not LoS.

    The scattered part is a :class:`JakesProcess` at the link's maximum Doppler; the LoS phasor rotates at
    ``los_doppler_hz`` (the Doppler along the actual line of sight, ``f_d cos(angle)``).
    """

    def __init__(self, n_links: int, rng: Optional[np.random.Generator] = None, n_osc: int = 32):
        self.n_links = n_links
        self.scatter = JakesProcess(n_links, n_osc, rng)
        self._los_phase = self.scatter.rng.uniform(-math.pi, math.pi, size=n_links)

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self.scatter.reset(rng)
        self._los_phase = self.scatter.rng.uniform(-math.pi, math.pi, size=self.n_links)

    def step(self, los: Any, k_db: Any, f_d_hz: Any, dt: float, los_doppler_hz: Any = None) -> np.ndarray:
        k = np.where(np.asarray(los, bool), 10.0 ** (np.asarray(k_db, float) / 10.0), 0.0)
        h_s = self.scatter.step(f_d_hz, dt)
        fd_los = np.asarray(f_d_hz if los_doppler_hz is None else los_doppler_hz, float)
        self._los_phase = self._los_phase + 2.0 * math.pi * fd_los * dt
        h = np.sqrt(k / (k + 1.0)) * np.exp(1j * self._los_phase) + np.sqrt(1.0 / (k + 1.0)) * h_s
        return np.abs(h) ** 2

    def sample(self, k_db: Any, f_d_hz: Any, dt: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """``n`` consecutive power-gain samples for all links, LoS everywhere: ``(gains (n, L), times (n,))``."""
        out = np.empty((n, self.n_links))
        los = np.ones(self.n_links, bool)
        for i in range(n):
            out[i] = self.step(los, k_db, f_d_hz, dt)
        return out, np.arange(1, n + 1) * dt
