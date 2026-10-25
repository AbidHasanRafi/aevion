"""Exteroceptive sensors built on ray casting: rangefinder, LiDAR, depth/RGB/segmentation camera."""

from __future__ import annotations

import math

import numpy as np

from ..core import rotation as rot
from .base import Sensor
from .raycast import cast_rays


def _rotate_dirs(B, quat, dirs_body):
    """(N,4) quats and (R,3) body-frame directions -> (N,R,3) world directions."""
    Rm = rot.to_matrix(B, quat)
    return B.einsum("nij,rj->nri", Rm, dirs_body)


def _origins(B, state, mount):
    Rm = rot.to_matrix(B, state.quat)
    return state.pos + B.einsum("nij,j->ni", Rm, B.asarray(mount))


def _cast_batch(sim, origins, dirs_world, max_range):
    """origins (N,3), dirs (N,R,3) -> RayHits with t (N,R), normal (N,R,3), prim (N,R)."""
    B = sim.B
    n, r, _ = dirs_world.shape
    o = B.ones(n, r, 3) * origins[:, None, :]
    h = cast_rays(sim.world, o.reshape(n * r, 3), dirs_world.reshape(n * r, 3), max_range, backend=B)
    t = h.t.reshape(n, r)
    nm = h.normal.reshape(n, r, 3)
    pr = h.prim.reshape(n, r)
    return t, nm, pr


class RangeFinder(Sensor):
    """Single-beam range sensor (default: downward-facing). Returns ``range`` and ``valid``."""

    name = "rangefinder"

    def __init__(self, rate=30.0, max_range=20.0, min_range=0.05, direction=(0.0, 0.0, -1.0), noise_std=0.01, noise_rel=0.002, **kw):
        super().__init__(rate=rate, **kw)
        self.max_range, self.min_range = max_range, min_range
        self.dir = np.asarray(direction, float) / np.linalg.norm(direction)
        self.noise_std, self.noise_rel = noise_std, noise_rel
        if self.extrinsics is not None:
            self.dir = self.extrinsics.rotate(self.dir)

    def measure(self, state):
        B = self.B
        d = _rotate_dirs(B, state.quat, B.asarray(self.dir)[None, :])
        t, _, prim = _cast_batch(self.sim, state.pos, d, self.max_range)
        t = t[:, 0]
        valid = (prim[:, 0] != -1) & (t >= self.min_range)
        rng = t + self.noise((self.n,), 1.0) * (self.noise_std + self.noise_rel * t)
        rng = B.where(valid, rng, B.full((self.n,), self.max_range))
        return dict(range=rng, valid=valid, t=state.t)


class Lidar(Sensor):
    """Multi-beam spinning or solid-state LiDAR.

    ``channels`` beams spread over ``v_fov`` degrees, ``samples`` azimuth bins over ``h_fov``.
    Output ``ranges`` has shape ``(N, channels, samples)``; misses are ``inf``. ``points`` is the
    body-frame point cloud (N, channels*samples, 3) with NaN for misses and ``intensity`` follows
    the Lambertian incidence angle.
    """

    name = "lidar"

    def __init__(
        self,
        rate=10.0,
        channels=16,
        samples=360,
        v_fov=(-15.0, 15.0),
        h_fov=360.0,
        max_range=60.0,
        min_range=0.3,
        noise_std=0.02,
        noise_rel=0.001,
        dropout_prob=0.002,
        mount=(0.0, 0.0, 0.05),
        return_points=True,
        pattern=None,
        motion_distortion=False,
        doppler=False,
        doppler_noise=0.03,
        **kw,
    ):
        super().__init__(rate=rate, **kw)
        self.pattern = None
        self.motion_distortion, self.doppler, self.doppler_noise = bool(motion_distortion), bool(doppler), float(doppler_noise)
        if pattern is not None:  # a named / configured scan pattern replaces the channels x samples grid
            from .scan_patterns import pattern_from_spec

            self.pattern = pattern_from_spec(pattern)
            channels, samples = self.pattern.grid_shape if self.pattern.grid_shape else (1, self.pattern.n_rays)
        self.channels, self.samples = channels, samples
        self.v_fov, self.h_fov = v_fov, h_fov
        self.max_range, self.min_range = max_range, min_range
        self.noise_std, self.noise_rel, self.dropout_prob = noise_std, noise_rel, dropout_prob
        self.mount = np.asarray(mount, float)
        self.return_points = return_points
        el = np.deg2rad(np.linspace(v_fov[0], v_fov[1], channels)) if channels > 1 else np.array([np.deg2rad(sum(v_fov) / 2)])
        full = abs(h_fov - 360.0) < 1e-6
        az = np.deg2rad(np.linspace(-h_fov / 2, h_fov / 2, samples, endpoint=not full))
        E, A = np.meshgrid(el, az, indexing="ij")
        self.dirs_body = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        if self.pattern is not None:
            self.dirs_body = self.pattern.directions(0)
            self.v_fov = (-90.0, 90.0)
        if self.extrinsics is not None:  # sensor-frame scan pattern rotated/translated into the body frame
            self.dirs_body = self.extrinsics.rotate(self.dirs_body)
            self.mount = self.extrinsics.t.copy()
        self._scan = 0
        self._t_off = self.pattern.time_offsets(1.0 / rate) if self.pattern is not None else np.arange(channels * samples) % samples / samples / rate

    @property
    def n_rays(self):
        return self.channels * self.samples

    def on_bind(self):
        self._scan = 0

    def measure(self, state):
        B = self.B
        if self.pattern is not None and not self.pattern.static:  # e.g. the rosette: every scan points somewhere new
            d = self.pattern.directions(self._scan)
            self.dirs_body = self.extrinsics.rotate(d) if self.extrinsics is not None else d
        self._scan += 1
        db = B.asarray(self.dirs_body)
        if self.motion_distortion:  # each beam leaves from the pose the vehicle has when it is fired (constant velocity / turn rate)
            dt_i = B.asarray(self._t_off)
            d_b = (
                db[None, :, :]
                + B.cross(state.omega[:, None, :] * B.ones(1, db.shape[0], 1), db[None, :, :] * B.ones(self.n, 1, 1)) * dt_i[None, :, None]
            )
            Rm = rot.to_matrix(B, state.quat)
            dw = B.einsum("nij,nrj->nri", Rm, d_b)
            org = _origins(B, state, self.mount)[:, None, :] + state.vel[:, None, :] * dt_i[None, :, None]
            n_, r_, _ = dw.shape
            h = cast_rays(self.sim.world, org.reshape(n_ * r_, 3), dw.reshape(n_ * r_, 3), self.max_range, backend=B)
            t, nrm, prim = h.t.reshape(n_, r_), h.normal.reshape(n_, r_, 3), h.prim.reshape(n_, r_)
        else:
            dw = _rotate_dirs(B, state.quat, db)
            org = _origins(B, state, self.mount)
            t, nrm, prim = _cast_batch(self.sim, org, dw, self.max_range)
        hit = (prim != -1) & (t >= self.min_range)
        if self.dropout_prob > 0:
            hit = hit & (B.rand(t.shape, self.gen) >= self.dropout_prob)
        tn = t + self.noise(t.shape, 1.0) * (self.noise_std + self.noise_rel * t)
        inf = B.full(t.shape, math.inf)
        ranges = B.where(hit, tn, inf)
        cosinc = B.clip(-B.sum(nrm * dw, axis=-1), 0.0, 1.0)
        intensity = B.where(hit, cosinc, B.zeros(*t.shape))
        shape = (self.n, self.channels, self.samples) if (self.pattern is None or self.pattern.grid_shape) else (self.n, -1)
        out = dict(ranges=ranges.reshape(shape), intensity=intensity.reshape(shape), t=state.t)
        if self.doppler:  # FMCW LiDAR measures the radial velocity of every return (positive = closing)
            world = self.sim.world
            vs = world.primitive_velocity(B.to_numpy(prim), float(state.t)) if getattr(world, "actors", None) else 0.0
            vr = B.sum(dw * B.asarray(B.to_numpy(state.vel)[:, None, :] - vs), axis=-1)
            vr = vr + self.noise(vr.shape, self.doppler_noise)
            out["radial_velocity"] = B.where(hit, vr, B.full(vr.shape, math.nan)).reshape(shape)
        if self.return_points:
            r0 = B.where(hit, ranges, ranges * 0.0)
            pts = (db[None] if db.ndim == 2 else db) * r0[..., None] + B.asarray(self.mount)
            out["points"] = B.where(hit[..., None], pts, B.full(pts.shape, math.nan))
        return out

    def to_dict(self):
        return {
            **super().to_dict(),
            "channels": self.channels,
            "samples": self.samples,
            "v_fov": list(self.v_fov),
            "h_fov": self.h_fov,
            "max_range": self.max_range,
            **({"pattern": self.pattern.to_dict()} if self.pattern is not None else {}),
        }


class Camera(Sensor):
    """Camera rendered by ray casting.

    ``modes`` may contain ``"depth"`` (metric z-depth, ``inf`` on miss), ``"rgb"`` (uint8, lit by the sun with shadows, sky, distance fog),
    ``"seg"`` (primitive id) and ``"thermal"`` (apparent temperature in degrees C). Resolution is intentionally modest (RL-sized); cost scales
    with ``width * height`` (times ``blur_samples``).

    Beyond the ideal pinhole:

    * ``lens``: ``"pinhole"`` (with Brown-Conrady ``distortion = (k1, k2, p1, p2, k3)``), ``"equidistant"``, ``"equisolid"``,
      ``"stereographic"``, ``"orthographic"`` or a :mod:`~aevion.sensors.camera_models` lens; fisheyes may exceed 180 degrees;
    * ``rolling_shutter_s``: rows are exposed one after another over this readout time, so a moving camera skews the picture;
    * ``exposure_s`` and ``blur_samples``: the RGB image averages ``blur_samples`` poses spread over the exposure (motion blur);
    * ``exposure_ev`` (gain of 2^EV), ``vignetting`` (cos^4 fall-off strength in [0, 1]);
    * ``use_environment``: take the sun from the world's origin and UTC time (``Camera.SUN`` otherwise), and fog from its weather.
    """

    name = "camera"

    SUN = np.array([0.45, 0.25, 0.85])

    def __init__(
        self,
        rate=30.0,
        width=64,
        height=48,
        fov_deg=90.0,
        near=0.1,
        far=60.0,
        mount=(0.1, 0.0, 0.0),
        pitch_deg=0.0,
        modes=("depth",),
        shadows=True,
        fog_distance=120.0,
        depth_noise=(0.0015, 0.0025),
        pixel_noise=0.01,
        lens="pinhole",
        distortion=None,
        rolling_shutter_s=0.0,
        exposure_s=0.0,
        blur_samples=1,
        exposure_ev=0.0,
        vignetting=0.0,
        use_environment=True,
        thermal_noise_k=0.05,
        **kw,
    ):
        super().__init__(rate=rate, **kw)
        from .camera_models import make_lens

        self.W, self.H, self.fov = int(width), int(height), fov_deg
        self.near, self.far = near, far
        self.mount = np.asarray(mount, float)
        self.pitch = math.radians(pitch_deg)
        self.modes = tuple(modes)
        bad = set(self.modes) - {"depth", "rgb", "seg", "thermal"}
        if bad:
            raise ValueError(f"unknown camera mode(s) {sorted(bad)}; use depth, rgb, seg, thermal")
        if rolling_shutter_s < 0 or exposure_s < 0 or blur_samples < 1 or not 0.0 <= vignetting <= 1.0:
            raise ValueError("need rolling_shutter_s >= 0, exposure_s >= 0, blur_samples >= 1 and vignetting in [0, 1]")
        self.shadows, self.fog = shadows, fog_distance
        self.depth_noise, self.pixel_noise = depth_noise, pixel_noise
        self.lens = make_lens(lens, self.W, self.H, fov_deg, tuple(distortion) if distortion is not None else None)
        self.rolling_s, self.exposure_s, self.blur_samples = float(rolling_shutter_s), float(exposure_s), int(blur_samples)
        self.exposure_ev, self.vignetting, self.use_env, self.thermal_noise = (
            float(exposure_ev),
            float(vignetting),
            bool(use_environment),
            float(thermal_noise_k),
        )
        f = self.lens.f
        self.fx = self.fy = f
        d, self.pixel_ok = self.lens.unproject()
        self.fwd_comp = d[:, 0].copy() if self.lens.kind == "pinhole" else np.ones(len(d))  # range -> z-depth (a fisheye reports range)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)  # positive pitch looks down (rotation about +y by +p)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        self.dirs_body = d @ Ry.T
        self._off_axis = np.clip(d[:, 0], 0.0, 1.0)  # cos of the angle off the optical axis (vignetting)
        if self.extrinsics is not None:
            self.dirs_body = self.extrinsics.rotate(self.dirs_body)
            self.mount = self.extrinsics.t.copy()
        self._row_t = (np.repeat(np.arange(self.H), self.W) + 0.5) / self.H * self.rolling_s  # exposure start of each pixel's row

    @property
    def intrinsics(self):
        return self.lens.intrinsics

    def _palette(self):
        return self.sim.world.prim_colors()

    # ------------------------------------------------------------------------------------- environment
    def _environment(self, t):
        """Sun direction (ENU), daylight factor in [0, 1], cloud cover and the fog distance for time ``t``."""
        w = self.sim.world
        sun, day, cloud, fog = self.SUN / np.linalg.norm(self.SUN), 1.0, 0.0, self.fog
        if self.use_env:
            if w.weather is not None:
                fog = 1.0 / (1.0 / self.fog + w.weather.extinction())
                cloud = w.weather.cloud_cover
            if w.utc is not None and w.origin is not None:
                from ..environments.solar import _to_unix, sun_direction_enu, sun_position

                p = sun_position(w.origin.lat, w.origin.lon, float(_to_unix(w.utc)) + float(t))
                sun = sun_direction_enu(p["azimuth_deg"], p["elevation_deg"])
                el = float(p["elevation_deg"])
                day = float(np.clip((el + 6.0) / 16.0, 0.0, 1.0))  # civil twilight fades to dark night
                day = day * day * (3.0 - 2.0 * day)
        return dict(sun=np.asarray(sun, float), day=day, cloud=cloud, fog=fog)

    # --------------------------------------------------------------------------------------------- casting
    def _cast(self, state, dt_ray):
        """Cast every pixel's ray from the pose at ``dt_ray`` seconds (scalar or per ray) after the nominal time; returns (org, dw, t, nrm, prim)."""
        B = self.B
        db = B.asarray(self.dirs_body)
        moving = np.any(np.asarray(dt_ray) != 0.0)
        if not moving:
            dw = _rotate_dirs(B, state.quat, db)
            org = _origins(B, state, self.mount)
            t, nrm, prim = _cast_batch(self.sim, org, dw, self.far)
            return org, dw, t, nrm, prim
        dt = B.asarray(np.broadcast_to(np.asarray(dt_ray, float), (db.shape[0],)).copy())
        d_b = db[None, :, :] + B.cross(state.omega[:, None, :] * B.ones(1, db.shape[0], 1), db[None, :, :] * B.ones(self.n, 1, 1)) * dt[None, :, None]
        dw = B.einsum("nij,nrj->nri", rot.to_matrix(B, state.quat), d_b)
        org = _origins(B, state, self.mount)[:, None, :] + state.vel[:, None, :] * dt[None, :, None]
        n_, r_, _ = dw.shape
        h = cast_rays(self.sim.world, org.reshape(n_ * r_, 3), dw.reshape(n_ * r_, 3), self.far, backend=B)
        return org, dw, h.t.reshape(n_, r_), h.normal.reshape(n_, r_, 3), h.prim.reshape(n_, r_)

    def measure(self, state):
        B = self.B
        env = self._environment(state.t)
        rolling = self.rolling_s > 0.0
        S = self.blur_samples
        shape = (self.n, self.H, self.W)
        rgb_acc = th_acc = None
        out = dict(t=state.t)
        okmask = B.asarray(self.pixel_ok.astype(float))[None, :] > 0.5
        for k in range(S):
            blur_off = 0.0 if S == 1 else (-0.5 + (k + 0.5) / S) * self.exposure_s
            dt_ray = (self._row_t + blur_off) if rolling else blur_off
            org, dw, t, nrm, prim = self._cast(state, dt_ray)
            hit = (prim != -1) & (t >= self.near) & okmask
            if k == S // 2:  # geometry channels come from the middle of the exposure
                if "depth" in self.modes:
                    z = t * B.asarray(self.fwd_comp)[None, :]
                    z = z + self.noise(z.shape, 1.0) * (self.depth_noise[0] + self.depth_noise[1] * z * z)
                    out["depth"] = B.where(hit, z, B.full(z.shape, math.inf)).reshape(shape)
                if "seg" in self.modes:
                    out["seg"] = B.where(hit, prim.double() if B.name == "torch" else prim, B.full(prim.shape, -1.0)).reshape(shape)
            if "rgb" in self.modes:
                img = self._to_numpy_shade(org, dw, t, nrm, prim, env).astype(float)
                rgb_acc = img if rgb_acc is None else rgb_acc + img
            if "thermal" in self.modes:
                th = self._thermal(B.to_numpy(org), B.to_numpy(dw), B.to_numpy(t), B.to_numpy(nrm), B.to_numpy(prim), env)
                th_acc = th if th_acc is None else th_acc + th
        if "rgb" in self.modes:
            assert rgb_acc is not None
            rgb = rgb_acc / S
            if self.exposure_ev:
                rgb = rgb * 2.0**self.exposure_ev
            if self.vignetting > 0.0:
                rgb = rgb * (1.0 - self.vignetting + self.vignetting * self._off_axis**4)[None, :, None]
            rgb = np.where(self.pixel_ok[None, :, None], rgb, 0.0)
            out["rgb"] = (
                B.asarray(np.clip(rgb, 0, 255).astype(np.uint8).reshape(self.n, self.H, self.W, 3), dtype=None)
                if B.name == "torch"
                else np.clip(rgb, 0, 255).astype(np.uint8).reshape(self.n, self.H, self.W, 3)
            )
        if "thermal" in self.modes:
            assert th_acc is not None
            th = th_acc / S + B.to_numpy(self.noise((self.n, self.H * self.W), self.thermal_noise))
            out["thermal"] = B.asarray(np.where(self.pixel_ok[None, :], th, np.nan).reshape(shape))
        return out

    def _to_numpy_shade(self, org, dw, t, nrm, prim, env):
        B = self.B
        return self._shade(B.to_numpy(org), B.to_numpy(dw), B.to_numpy(t), B.to_numpy(nrm), B.to_numpy(prim), env)

    # ---------------------------------------------------------------------------------------- shading
    def _shade(self, org, dw, t, nrm, prim, env=None):
        world = self.sim.world
        env = env or self._environment(0.0)
        n, r, _ = dw.shape
        sun = env["sun"]
        hit = (prim != -1) & (t >= self.near) & (t <= self.far)
        p = org[:, None, :] + dw * t[..., None] if org.ndim == 2 else org + dw * t[..., None]

        pal = self._palette()
        base = np.zeros((n, r, 3))
        obst = prim >= 0
        if len(pal):
            base[obst] = pal[np.clip(prim[obst], 0, len(pal) - 1)]
        gnd = prim == -2
        if gnd.any():
            gx = np.floor(p[..., 0] / 2.0) + np.floor(p[..., 1] / 2.0)
            chk = (gx % 2 == 0)[..., None]
            gc = np.where(chk, np.array([0.30, 0.43, 0.29]), np.array([0.34, 0.48, 0.32]))
            if world.heightmap is not None:
                hh = np.clip(p[..., 2] / max(world.heightmap.max(), 1e-3), 0, 1)[..., None]
                gc = gc * (0.7 + 0.5 * hh) + np.array([0.12, 0.08, 0.02]) * hh
            base[gnd] = gc[gnd]

        day, cloud = env["day"], env["cloud"]
        diff = np.clip((nrm * sun).sum(-1), 0.0, 1.0)
        if self.shadows and day > 0.0 and hit.any():
            idx = np.nonzero(hit)
            so = p[idx] + nrm[idx] * 1e-3
            sd = np.broadcast_to(sun, so.shape)
            sh = cast_rays(world, so, sd, 400.0)
            lit = np.ones_like(diff)
            lit[idx] = (sh.prim == -1).astype(float)
            diff = diff * lit
        light = 0.38 * (0.12 + 0.88 * day) + 0.62 * day * (1.0 - 0.7 * cloud) * diff
        color = base * light[..., None]

        # fog + sky
        el = np.clip(dw[..., 2], -1, 1)[..., None]
        horizon, zenith = np.array([0.80, 0.87, 0.95]), np.array([0.28, 0.52, 0.86])
        dim = 0.04 + 0.96 * day
        horizon, zenith = horizon * dim, zenith * dim
        if cloud > 0.0:
            grey = np.array([0.72, 0.74, 0.78]) * dim
            horizon, zenith = horizon * (1 - cloud) + grey * cloud, zenith * (1 - cloud) + grey * cloud
        sky = horizon + (zenith - horizon) * np.clip(el, 0, 1) ** 0.6
        fogf = 1.0 - np.exp(-np.where(hit, t, self.far)[..., None] / env["fog"])
        color = color * (1 - fogf) + horizon * fogf
        color = np.where(hit[..., None], color, sky)
        if self.pixel_noise > 0:
            color = color + self.sim.B.to_numpy(self.noise((n, r, 3), self.pixel_noise))
        return (np.clip(color, 0, 1) * 255).astype(np.uint8)

    def _thermal(self, org, dw, t, nrm, prim, env):
        """Apparent temperature [C] per pixel: surface temperature (ground and walls warmed by the sun, actors warm) attenuated by the air."""
        world = self.sim.world
        w = world.weather
        amb = w.temperature_c if w is not None else 15.0
        sun_h = max(float(env["sun"][2]), 0.0) * env["day"] * (1.0 - 0.7 * env["cloud"])
        hit = (prim != -1) & (t >= self.near) & (t <= self.far)
        temp = np.full(t.shape, amb)
        gnd = prim == -2
        temp = np.where(gnd, amb + 14.0 * sun_h, temp)
        lit = np.clip((nrm * env["sun"]).sum(-1), 0.0, 1.0)
        temp = np.where((prim >= 0) & ~gnd, amb + 3.0 + 9.0 * sun_h * lit, temp)
        if getattr(world, "actors", None):
            k, s_ = len(world.boxes), len(world.spheres)
            base = {"boxes": 0, "spheres": k, "cylinders": k + s_}
            for actor, array, row in world.actors:
                temp = np.where(prim == base[array] + row, getattr(actor, "temperature_c", None) or amb + 8.0, temp)
        el = np.clip(dw[..., 2], 0.0, 1.0)
        sky = amb - 10.0 - 35.0 * el  # effective sky temperature: cold at the zenith, near ambient at the horizon
        surface = np.where(hit, temp, sky)
        beta = w.extinction() if w is not None else 0.0
        tau = np.exp(-beta * np.where(hit, t, self.far))
        return surface * tau + amb * (1.0 - tau)

    def to_dict(self):
        d = {
            **super().to_dict(),
            "width": self.W,
            "height": self.H,
            "fov_deg": self.fov,
            "modes": list(self.modes),
            "far": self.far,
            "lens": self.lens.to_dict(),
        }
        if self.rolling_s:
            d["rolling_shutter_s"] = self.rolling_s
        if self.exposure_s:
            d.update(exposure_s=self.exposure_s, blur_samples=self.blur_samples)
        return d
