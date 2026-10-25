"""Event camera (dynamic vision sensor): asynchronous brightness-change events instead of frames.

Each pixel keeps a reference log-intensity and emits an event whenever the current log-intensity differs from it by the contrast
threshold ``C`` (positive polarity for brighter, negative for darker); the reference then moves by ``C``. Between two renders the
intensity change is assumed linear, so ``k = floor(|dL| / C)`` events are emitted at interpolated crossing times. The sensor adds
per-pixel threshold mismatch, a refractory period and Poisson background-activity noise.

The scene is rendered by the :class:`~aevion.sensors.ranging.Camera` machinery (lens models, sun, weather and rolling shutter apply).
``rate`` is the render rate; event timestamps are interpolated within each interval.

Fidelity: linear intensity interpolation between renders (no sub-frame scene motion), no bandwidth limit or pixel latency, no hot pixels.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .ranging import Camera

__all__ = ["EventCamera"]

EPS = 0.02  # intensity offset inside the logarithm (avoids log(0) in dark pixels)


class EventCamera(Camera):
    """Output ``events`` ``(N, E, 4)`` rows ``[x, y, t, polarity]`` (``NaN`` padded), ``n_events`` ``(N,)`` and ``frame`` ``(N, H, W)``
    (signed event count per pixel over the interval)."""

    name = "event"

    def __init__(
        self,
        rate: float = 500.0,
        width: int = 64,
        height: int = 48,
        contrast_threshold: float = 0.2,
        threshold_noise: float = 0.03,
        refractory_s: float = 5e-4,
        background_rate_hz: float = 0.1,
        max_events: int = 50_000,
        **kw: Any,
    ):
        kw.pop("modes", None)
        kw.setdefault("pixel_noise", 0.0)
        super().__init__(rate=rate, width=width, height=height, modes=("rgb",), **kw)
        if contrast_threshold <= 0 or threshold_noise < 0 or refractory_s < 0 or background_rate_hz < 0 or max_events < 1:
            raise ValueError("need contrast_threshold > 0, non-negative noise/refractory/background rates and max_events >= 1")
        self.C, self.C_noise, self.refractory = float(contrast_threshold), float(threshold_noise), float(refractory_s)
        self.bg_rate, self.max_events = float(background_rate_hz), int(max_events)

    def on_bind(self) -> None:
        n, hw = self.n, self.H * self.W
        self._ref = None
        self._t_prev = 0.0
        self._last_event = np.full((n, hw), -1e9)
        mismatch = self.B.to_numpy(self.noise((n, hw), self.C_noise))
        self.C_pix = np.maximum(self.C * (1.0 + mismatch), 0.2 * self.C)  # fixed per-pixel threshold mismatch
        self._rng = np.random.default_rng(int(self.B.to_numpy(self.B.rand((1,), self.gen))[0] * 2**31))

    def measure(self, state: Any) -> dict:
        rgb = self.B.to_numpy(super().measure(state)["rgb"]).astype(float) / 255.0
        n, hw = self.n, self.H * self.W
        logi = np.log(rgb.mean(axis=-1).reshape(n, hw) + EPS)
        t = float(state.t)
        empty = dict(
            events=self.B.asarray(np.full((n, 0, 4), np.nan)),
            n_events=self.B.asarray(np.zeros(n)),
            frame=self.B.asarray(np.zeros((n, self.H, self.W))),
            t=t,
        )
        if self._ref is None:
            self._ref, self._t_prev = logi.copy(), t
            return empty
        dt = max(t - self._t_prev, 1e-9)
        diff = logi - self._ref
        k = np.floor(np.abs(diff) / self.C_pix).astype(int)
        if self.refractory > 0:  # a pixel cannot fire faster than its refractory period
            k = np.minimum(k, np.maximum(np.floor(dt / self.refractory).astype(int), 1))
        sign = np.sign(diff)
        events = []
        counts = np.zeros(n, int)
        frame = np.zeros((n, hw))
        for i in range(n):
            px = np.nonzero(k[i])[0]
            rows = []
            for p in px:
                kk = int(k[i, p])
                frac = (np.arange(1, kk + 1) * self.C_pix[i, p]) / max(
                    abs(diff[i, p]), 1e-12
                )  # fraction of the interval at which each threshold is crossed
                ts = self._t_prev + np.clip(frac, 0.0, 1.0) * dt
                rows.append(np.column_stack([np.full(kk, p % self.W), np.full(kk, p // self.W), ts, np.full(kk, sign[i, p])]))
                frame[i, p] = sign[i, p] * kk
            if self.bg_rate > 0:  # background activity: random pixels fire spuriously
                m = self._rng.poisson(self.bg_rate * hw * dt)
                if m:
                    pp = self._rng.integers(0, hw, m)
                    rows.append(
                        np.column_stack([pp % self.W, pp // self.W, self._t_prev + self._rng.random(m) * dt, self._rng.choice([-1.0, 1.0], m)])
                    )
                    np.add.at(frame[i], pp, rows[-1][:, 3])
            ev = np.vstack(rows) if rows else np.zeros((0, 4))
            ev = ev[np.argsort(ev[:, 2], kind="stable")][: self.max_events]
            events.append(ev)
            counts[i] = len(ev)
        width = int(max(counts.max(), 0))
        padded = np.full((n, width, 4), np.nan)
        for i, ev in enumerate(events):
            padded[i, : len(ev)] = ev
        self._ref = self._ref + np.sign(diff) * k * self.C_pix  # the reference follows by whole thresholds
        self._t_prev = t
        return dict(
            events=self.B.asarray(padded), n_events=self.B.asarray(counts.astype(float)), frame=self.B.asarray(frame.reshape(n, self.H, self.W)), t=t
        )

    def record_payload(self, reading: dict) -> Any:
        return self.B.to_numpy(reading["frame"]).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "contrast_threshold": self.C, "refractory_s": self.refractory, "background_rate_hz": self.bg_rate}
