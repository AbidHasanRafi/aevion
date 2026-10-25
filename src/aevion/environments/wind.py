"""Wind models returning the world-frame air velocity ``w`` (m/s) at each vehicle.

Models are stateful objects with ``reset(B, n, seed)`` and
``sample(B, pos, vel, t, dt) -> (N, 3)``. Combine them with :class:`CompositeWind`.
Everything is batched over ``N`` vehicles.
"""

from __future__ import annotations

import math

import numpy as np

FT = 0.3048


class Wind:
    def reset(self, B, n, seed=None):
        self.B, self.n = B, n
        self.gen = B.make_rng(0 if seed is None else seed)

    def bind_world(self, world):
        """Called by :meth:`World.reset` before ``reset``: lets a wind read the world's geometry (e.g. buildings)."""

    def sample(self, B, pos, vel, t, dt):
        return B.zeros(pos.shape[0], 3)

    def to_dict(self):
        return dict(type=type(self).__name__)


class NoWind(Wind):
    pass


class ConstantWind(Wind):
    """Uniform wind. ``velocity`` is ``(3,)`` or per-env ``(N, 3)``."""

    def __init__(self, velocity=(0.0, 0.0, 0.0)):
        self.velocity = np.asarray(velocity, dtype=float)

    def sample(self, B, pos, vel, t, dt):
        return B.ones(pos.shape[0], 3) * B.asarray(self.velocity)

    def to_dict(self):
        return dict(type="ConstantWind", velocity=self.velocity.tolist())


class WindShear(Wind):
    """Atmospheric boundary-layer profile.

    ``profile="power"``: ``u(z) = u_ref (z / z_ref)^alpha`` (alpha≈0.14 open terrain).
    ``profile="log"``:   ``u(z) = u_ref ln(z/z0) / ln(z_ref/z0)``.
    """

    def __init__(self, speed_ref=5.0, height_ref=10.0, direction_deg=0.0, profile="power", alpha=0.14, z0=0.03):
        self.speed_ref, self.height_ref = float(speed_ref), float(height_ref)
        self.direction = math.radians(direction_deg)
        self.profile, self.alpha, self.z0 = profile, float(alpha), float(z0)

    def sample(self, B, pos, vel, t, dt):
        z = B.maximum(pos[..., 2], 0.5 * B.ones(1))
        if self.profile == "power":
            mag = self.speed_ref * (z / self.height_ref) ** self.alpha
        else:
            zz = B.maximum(z, 1.5 * self.z0 * B.ones(1))
            mag = self.speed_ref * _log(B, zz / self.z0) / math.log(self.height_ref / self.z0)
        return B.stack([mag * math.cos(self.direction), mag * math.sin(self.direction), mag * 0.0], -1)

    def to_dict(self):
        return dict(
            type="WindShear",
            speed_ref=self.speed_ref,
            height_ref=self.height_ref,
            direction_deg=math.degrees(self.direction),
            profile=self.profile,
            alpha=self.alpha,
            z0=self.z0,
        )


def _log(B, x):
    if B.name == "numpy":
        return np.log(x)
    import torch

    return torch.log(x)


class Gust(Wind):
    """Discrete '1 - cos' gust (e.g. FAR/CS-25 style): ``0.5 A (1 - cos(2π (t-t0)/T))``."""

    def __init__(self, amplitude=(3.0, 0.0, 0.0), t_start=2.0, duration=2.0):
        self.amplitude = np.asarray(amplitude, dtype=float)
        self.t_start, self.duration = float(t_start), float(duration)

    def sample(self, B, pos, vel, t, dt):
        s = (t - self.t_start) / self.duration
        k = 0.5 * (1.0 - math.cos(2 * math.pi * s)) if 0.0 <= s <= 1.0 else 0.0
        return B.ones(pos.shape[0], 3) * B.asarray(self.amplitude * k)

    def to_dict(self):
        return dict(type="Gust", amplitude=self.amplitude.tolist(), t_start=self.t_start, duration=self.duration)


class DrydenTurbulence(Wind):
    """Dryden continuous turbulence, low-altitude MIL-F-8785C parameterisation.

    ``w20`` is the mean wind speed at 20 ft (6.096 m) [m/s]; intensity scales with it
    (``sigma_w = 0.1 w20``). ``severity`` multiplies all intensities. The shaping filters
    depend on the vehicle airspeed and altitude and are advanced each ``sample`` call.
    """

    def __init__(self, w20=5.0, severity=1.0, min_altitude=3.0, max_altitude=300.0):
        self.w20, self.severity = float(w20), float(severity)
        self.zmin, self.zmax = min_altitude, max_altitude

    def reset(self, B, n, seed=None):
        super().reset(B, n, seed)
        self.xu = B.zeros(n)
        self.xv1, self.xv2 = B.zeros(n), B.zeros(n)
        self.xw1, self.xw2 = B.zeros(n), B.zeros(n)

    def sample(self, B, pos, vel, t, dt):
        n = pos.shape[0]
        h = B.clip(pos[..., 2], self.zmin, self.zmax)
        hft = h / FT
        base = 0.177 + 0.000823 * hft
        Lu = (hft / base**1.2) * FT
        Lv = Lu
        Lw = h
        sw = 0.1 * self.w20 * self.severity
        su = sw / base**0.4
        sv = su
        V = B.maximum(B.norm(vel, axis=-1), 1.0 * B.ones(1))  # airspeed floor avoids singularities at hover
        # The MIL-F-8785C shaping filters have unit output variance (times sigma^2) when driven by white
        # noise of two-sided spectral density pi (u) and 2*pi (v, w); a discrete sample of such noise
        # has standard deviation sqrt(S0 / dt).
        inv_sqrt_dt = 1.0 / math.sqrt(dt)
        ru = B.randn((n,), self.gen) * (math.sqrt(math.pi) * inv_sqrt_dt)
        rv = B.randn((n,), self.gen) * (math.sqrt(2 * math.pi) * inv_sqrt_dt)
        rw = B.randn((n,), self.gen) * (math.sqrt(2 * math.pi) * inv_sqrt_dt)

        # u: H = s_u sqrt(2 L/(pi V)) / (1 + (L/V) s)
        tu = Lu / V
        a = B.minimum(dt / tu, 1.0 * B.ones(1))
        self.xu = self.xu + a * (ru - self.xu)
        u = su * B.sqrt(2 * Lu / (math.pi * V)) * self.xu

        # v, w: H = s sqrt(L/(pi V)) (1 + sqrt3 tau s)/(1 + tau s)^2, tau = 2L/V
        def second_order(x1, x2, r, L, s):
            tau = 2 * L / V
            a = B.minimum(dt / tau, 1.0 * B.ones(1))
            x1n = x1 + a * (r - x1)
            x2n = x2 + a * (x1 - x2)
            out = s * B.sqrt(L / (math.pi * V)) * (x2 + math.sqrt(3.0) * (x1 - x2))
            return x1n, x2n, out

        self.xv1, self.xv2, v = second_order(self.xv1, self.xv2, rv, Lv, sv)
        self.xw1, self.xw2, w = second_order(self.xw1, self.xw2, rw, Lw, sw)
        return B.stack([u, v, w], -1)

    def to_dict(self):
        return dict(type="DrydenTurbulence", w20=self.w20, severity=self.severity)


class CompositeWind(Wind):
    def __init__(self, *models):
        self.models = list(models)

    def reset(self, B, n, seed=None):
        super().reset(B, n, seed)
        for i, m in enumerate(self.models):
            m.reset(B, n, None if seed is None else seed + 101 * (i + 1))

    def bind_world(self, world):
        for m in self.models:
            m.bind_world(world)

    def sample(self, B, pos, vel, t, dt):
        out = B.zeros(pos.shape[0], 3)
        for m in self.models:
            out = out + m.sample(B, pos, vel, t, dt)
        return out

    def to_dict(self):
        return dict(type="CompositeWind", models=[m.to_dict() for m in self.models])


def wind_from_dict(d):
    t = d["type"]
    if t == "NoWind":
        return NoWind()
    if t == "ConstantWind":
        return ConstantWind(d["velocity"])
    if t == "WindShear":
        return WindShear(d["speed_ref"], d["height_ref"], d["direction_deg"], d["profile"], d["alpha"], d["z0"])
    if t == "Gust":
        return Gust(d["amplitude"], d["t_start"], d["duration"])
    if t == "DrydenTurbulence":
        return DrydenTurbulence(d["w20"], d["severity"])
    if t == "CompositeWind":
        return CompositeWind(*[wind_from_dict(m) for m in d["models"]])
    if t == "RandomConstantWind":
        return RandomConstantWind(d["max_speed"], d["vertical_frac"])
    if t in ("LogLawWind", "GriddedWind", "UrbanWind"):
        from . import wind_fields as WF

        if t == "GriddedWind":
            path = d["file"]
            loader = WF.GriddedWind.from_csv if str(path).lower().endswith(".csv") else WF.GriddedWind.from_npz
            return loader(path, **{k: d[k] for k in ("outside", "scale", "loop") if k in d})
        return getattr(WF, t)(**{k: v for k, v in d.items() if k != "type"})
    raise ValueError(f"unknown wind type {t!r}")


class RandomConstantWind(Wind):
    """Per-environment constant wind that can be re-sampled for a subset of environments.

    Used for domain randomisation: speed ~ U(0, ``max_speed``), uniform direction in the
    horizontal plane, optional vertical component ``vertical_frac * speed``.
    """

    def __init__(self, max_speed=5.0, vertical_frac=0.0):
        self.max_speed, self.vertical_frac = float(max_speed), float(vertical_frac)

    def reset(self, B, n, seed=None):
        super().reset(B, n, seed)
        self.w = B.zeros(n, 3)
        self.resample(B.ones(n) > 0.0)

    def resample(self, mask):
        B, n = self.B, self.n
        if self.max_speed <= 0.0:
            return
        sp = B.rand((n,), self.gen) * self.max_speed
        th = B.rand((n,), self.gen) * (2 * math.pi)
        vz = (B.rand((n,), self.gen) * 2.0 - 1.0) * self.vertical_frac * sp
        new = B.stack([sp * B.cos(th), sp * B.sin(th), vz], -1)
        self.w = B.where(mask[:, None], new, self.w)

    def sample(self, B, pos, vel, t, dt):
        return self.w

    def to_dict(self):
        return dict(type="RandomConstantWind", max_speed=self.max_speed, vertical_frac=self.vertical_frac)
