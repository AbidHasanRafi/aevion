"""State estimation: Mahony attitude filter and a position/velocity Kalman filter.

Both are batched and backend-agnostic. :class:`Estimator` combines them and exposes an
:class:`EstimatedState` that quacks like the true simulator state, so any controller can be
run on estimates instead of ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import rotation as rot


@dataclass
class EstimatedState:
    pos: object
    vel: object
    quat: object
    omega: object
    t: float = 0.0


class MahonyAHRS:
    """Nonlinear complementary filter (Mahony et al., 2008) with gyro-bias estimation.

    Accelerometer correction is gated by ``|f| ≈ g`` so aggressive manoeuvres do not corrupt tilt;
    an optional magnetometer corrects heading only.
    """

    def __init__(self, B, n, kp=1.5, ki=0.02, kp_mag=0.8, g=9.80665, declination_deg=0.0):
        self.B, self.n = B, n
        self.kp, self.ki, self.kp_mag, self.g = kp, ki, kp_mag, g
        self.dec = math.radians(declination_deg)
        self.reset()

    def reset(self, quat=None):
        B = self.B
        self.q = rot.identity(B, self.n) if quat is None else B.clone(quat)
        self.bias = B.zeros(self.n, 3)
        self.initialised = quat is not None

    def init_from_accel(self, accel, mag=None):
        """Set roll/pitch from gravity (and yaw from the magnetometer if given)."""
        B = self.B
        ax, ay, az = accel[..., 0], accel[..., 1], accel[..., 2]
        roll = B.atan2(ay, az)
        pitch = B.atan2(-ax, B.sqrt(ay * ay + az * az))
        yaw = B.zeros(self.n)
        q = rot.from_euler(B, roll, pitch, yaw)
        if mag is not None:
            mw = rot.rotate(B, q, mag)
            yaw = -B.atan2(mw[..., 0], mw[..., 1]) + self.dec
            q = rot.from_euler(B, roll, pitch, yaw)
        self.q, self.initialised = q, True

    def update(self, gyro, accel, dt, mag=None, acc_world=None):
        """One filter step. If ``acc_world`` (a low-passed world-frame linear acceleration estimate,
        e.g. from differentiated GPS velocity) is given, tilt is corrected against the thrust-axis
        direction ``normalize(acc_world + g e3)``, which stays valid in manoeuvres. Otherwise the
        accelerometer direction is used (valid only when ``|f| ≈ g``, i.e. near hover)."""
        B = self.B
        if not self.initialised:
            self.init_from_accel(accel, mag)
        R = rot.to_matrix(B, self.q)
        if acc_world is None:
            v_est = R[..., 2, :]  # world-up expressed in body = third row of R (R^T e3)
            fn = B.norm(accel, axis=-1, keepdims=True)
            a_n = accel / B.maximum(fn, 1e-6 * B.ones(1))
            gate = B.exp(-((fn[..., 0] - self.g) ** 2) / (2 * 1.5**2))[..., None]
        else:
            b3 = acc_world + B.asarray([0.0, 0.0, self.g])
            b3 = b3 / B.norm(b3, axis=-1, keepdims=True)
            # measured: the thrust axis is e3 in the body frame; estimated: b3 mapped into the body via R
            a_n = B.zeros(self.n, 3) + B.asarray([0.0, 0.0, 1.0])
            v_est = B.einsum("nji,nj->ni", R, b3)
            gate = 1.0
        e = B.cross(a_n, v_est) * gate
        if mag is not None:
            mw = rot.rotate(B, self.q, mag)
            eps = B.atan2(mw[..., 0], mw[..., 1]) - self.dec  # heading error about world z
            ez = B.stack([eps * 0.0, eps * 0.0, eps], -1)
            e = e + (self.kp_mag / max(self.kp, 1e-9)) * rot.rotate_inv(B, self.q, ez)
        self.bias = self.bias - self.ki * e * dt
        w = gyro - self.bias + self.kp * e
        self.q = rot.integrate(B, self.q, w, dt)
        return self.q, gyro - self.bias


class PosVelKF:
    """9-state Kalman filter: position, velocity, accelerometer bias (world-frame propagation).

    Predict with IMU specific force rotated by the attitude estimate; correct with GPS
    (position + velocity) and barometer (altitude).
    """

    def __init__(
        self,
        B,
        n,
        g=9.80665,
        accel_noise=0.15,
        accel_bias_walk=2e-3,
        gps_pos_std=1.0,
        gps_vel_std=0.15,
        baro_std=0.6,
        init_pos_std=2.0,
        init_vel_std=1.0,
    ):
        self.B, self.n, self.g = B, n, g
        self.qa, self.qb = accel_noise, accel_bias_walk
        self.gps_pos_std, self.gps_vel_std, self.baro_std = gps_pos_std, gps_vel_std, baro_std
        self.init = (init_pos_std, init_vel_std)
        self.reset()

    def reset(self, pos=None, vel=None):
        B, n = self.B, self.n
        self.x = B.zeros(n, 9)
        if pos is not None:
            self.x = B.cat([pos, vel if vel is not None else B.zeros(n, 3), B.zeros(n, 3)], axis=-1)
        p0, v0 = self.init
        d = B.asarray([p0**2] * 3 + [v0**2] * 3 + [0.05**2] * 3)
        self.P = B.ones(n, 1, 1) * (B.eye(9) * d[None, :])

    def predict(self, quat, accel, dt):
        B = self.B
        R = rot.to_matrix(B, quat)
        ba = self.x[..., 6:9]
        a_w = B.einsum("nij,nj->ni", R, accel - ba) + B.asarray([0.0, 0.0, -self.g])
        p, v = self.x[..., 0:3], self.x[..., 3:6]
        self.x = B.cat([p + v * dt + 0.5 * a_w * dt * dt, v + a_w * dt, ba], axis=-1)
        n = self.n
        I3 = B.ones(n, 1, 1) * B.eye(3)
        Z = B.zeros(n, 3, 3)
        F = B.cat(
            [
                B.cat([I3, I3 * dt, -0.5 * R * dt * dt], axis=-1),
                B.cat([Z, I3, -R * dt], axis=-1),
                B.cat([Z, Z, I3], axis=-1),
            ],
            axis=-2,
        )
        qd = [0.25 * dt**4 * self.qa**2] * 3 + [(self.qa**2) * dt] * 3 + [(self.qb**2) * dt] * 3
        Q = B.eye(9) * B.asarray(qd)[None, :]
        self.P = B.einsum("nij,njk,nlk->nil", F, self.P, F) + Q[None]

    def _update(self, H, z, Rn):
        B = self.B
        S = B.einsum("ij,njk,lk->nil", H, self.P, H) + Rn[None]
        K = B.einsum("nij,lj,nlk->nik", self.P, H, B.inv(S))
        innov = z - B.einsum("ij,nj->ni", H, self.x)
        self.x = self.x + B.einsum("nij,nj->ni", K, innov)
        KH = B.einsum("nij,jk->nik", K, H)
        I = B.eye(9)
        IKH = I[None] - KH
        self.P = B.einsum("nij,njk,nlk->nil", IKH, self.P, IKH) + B.einsum("nij,jk,nlk->nil", K, Rn, K)

    def update_gps(self, pos, vel):
        B = self.B
        H = B.cat([B.eye(6), B.zeros(6, 3)], axis=1)
        Rn = B.eye(6) * B.asarray([self.gps_pos_std**2] * 3 + [self.gps_vel_std**2] * 3)[None, :]
        self._update(H, B.cat([pos, vel], axis=-1), Rn)

    def update_baro(self, altitude):
        B = self.B
        H = B.asarray(np.eye(9)[2:3])
        Rn = B.eye(1) * (self.baro_std**2)
        self._update(H, altitude[..., None], Rn)

    @property
    def pos(self):
        return self.x[..., 0:3]

    @property
    def vel(self):
        return self.x[..., 3:6]


class Estimator:
    """Sensor-fusion pipeline fed by a :class:`~aevion.sensors.SensorSuite` output dict."""

    def __init__(self, sim, kp=1.0, ki=0.01, use_mag=True, gps_aided=True, acc_cutoff_hz=0.3, **kf_kw):
        self.B, self.n = sim.B, sim.n
        g = sim.world.gravity.g
        self.att = MahonyAHRS(self.B, self.n, kp=kp, ki=ki, g=g)
        self.kf = PosVelKF(self.B, self.n, g=g, **kf_kw)
        self.use_mag = use_mag
        self._t_imu = None
        self.gyro = self.B.zeros(self.n, 3)
        self.started = False
        self.gps_aided = gps_aided
        self.acc_cutoff = acc_cutoff_hz
        self._v_prev = None
        self._t_gps = None
        self.acc_lp = None

    def reset(self, true_state=None):
        self.att.reset()
        self.kf.reset(None if true_state is None else true_state.pos, None if true_state is None else true_state.vel)
        self._t_imu = None
        self._v_prev, self._t_gps, self.acc_lp = None, None, None
        self.started = true_state is not None
        if true_state is not None:
            self.att.reset(true_state.quat)

    def update(self, readings: dict, fresh: dict) -> EstimatedState:
        imu = readings.get("imu")
        if imu is not None and fresh.get("imu", True):
            t = imu["t"]
            dt = (t - self._t_imu) if self._t_imu is not None else 1.0 / 200.0
            self._t_imu = t
            mag = readings["mag"]["field"] if (self.use_mag and readings.get("mag") is not None) else None
            q, w = self.att.update(imu["gyro"], imu["accel"], dt, mag, self.acc_lp if self.gps_aided else None)
            self.gyro = w
            self.kf.predict(q, imu["accel"], dt)
        gps = readings.get("gps")
        if (
            gps is not None
            and fresh.get("gps", False)
            and ("fix" not in gps or float(gps["fix"].min()) > 0.5)
            and math.isfinite(float(gps["pos"].sum()) + float(gps["vel"].sum()))
        ):  # no fix or a corrupted (NaN) reading: skip the update
            if not self.started:
                self.kf.reset(gps["pos"], gps["vel"])
                self.started = True
            self.kf.update_gps(gps["pos"], gps["vel"])
            if self._v_prev is not None and gps["t"] > self._t_gps:
                dtg = gps["t"] - self._t_gps
                a_raw = (gps["vel"] - self._v_prev) / dtg
                alpha = 1.0 - math.exp(-2 * math.pi * self.acc_cutoff * dtg)
                self.acc_lp = a_raw * alpha if self.acc_lp is None else self.acc_lp + alpha * (a_raw - self.acc_lp)
            self._v_prev, self._t_gps = gps["vel"], gps["t"]
        baro = readings.get("baro")
        if baro is not None and fresh.get("baro", False) and self.started:
            self.kf.update_baro(baro["altitude"])
        return EstimatedState(pos=self.kf.pos, vel=self.kf.vel, quat=self.att.q, omega=self.gyro, t=imu["t"] if imu else 0.0)
