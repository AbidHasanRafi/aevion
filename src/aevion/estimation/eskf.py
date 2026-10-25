"""Error-state (multiplicative) extended Kalman filter for a strapdown IMU with aiding sensors.

Nominal state ``x = (p, v, q, b_a, b_g)`` (world ENU, body FLU, ``q`` body to world); 15-dimensional error state
``δx = (δp, δv, δθ, δb_a, δb_g)`` with the attitude error in the **body** frame, ``q_true = q ⊗ exp(δθ)`` (Solà, *Quaternion kinematics
for the error-state Kalman filter*, 2017).

Process model (IMU as input; ``a`` and ``ω`` are bias-corrected specific force and rate)::

    ṗ = v,   v̇ = R a + g,   q̇ = ½ q ⊗ ω,   ḃ_a = w_a,   ḃ_g = w_g
    δṗ = δv
    δv̇ = -R [a]× δθ - R δb_a
    δθ̇ = -[ω]× δθ - δb_g

Aiding updates (batched, all optional): GNSS position and velocity, barometric altitude, magnetometer heading, single-beam range to a
flat ground, optical flow and a full pose (visual odometry). Each update is gated on the Mahalanobis distance of its innovation and
records its normalised innovation squared (NIS).

**Delayed measurements.** A measurement stamped ``t_m`` older than the newest IMU sample is fused at ``t_m``: the filter rewinds to
its snapshot before that time, applies the update, then re-propagates the stored IMU samples and later measurements.
``max_delay_s`` bounds the history kept.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..core import rotation as rot

__all__ = ["ESKF", "STATE_DIM", "ERR_SLICES", "ESKFEstimator"]

STATE_DIM = 15
ERR_SLICES = {"p": slice(0, 3), "v": slice(3, 6), "theta": slice(6, 9), "ba": slice(9, 12), "bg": slice(12, 15)}
_TOL = 1e-6


def _chi2_gate(prob: float, dof: int) -> float:
    from scipy.stats import chi2

    return float(chi2.ppf(prob, dof))


def _sym(P: Any) -> Any:
    return 0.5 * (P + P.swapaxes(-1, -2))


class _Snapshot:
    __slots__ = ("P", "ba", "bg", "p", "q", "v")

    def __init__(self, f: "ESKF") -> None:
        B = f.B
        self.p, self.v, self.q, self.ba, self.bg, self.P = (B.clone(f.p), B.clone(f.v), B.clone(f.q), B.clone(f.ba), B.clone(f.bg), B.clone(f.P))

    def restore(self, f: "ESKF") -> None:
        B = f.B
        f.p, f.v, f.q, f.ba, f.bg, f.P = B.clone(self.p), B.clone(self.v), B.clone(self.q), B.clone(self.ba), B.clone(self.bg), B.clone(self.P)


class _Step:
    """One IMU sample and the filter state right after it (and after the measurements stamped inside the step)."""

    __slots__ = ("accel", "dt", "gyro", "snap", "t")

    def __init__(self, t: float, gyro: Any, accel: Any, dt: float, snap: _Snapshot) -> None:
        self.t, self.gyro, self.accel, self.dt, self.snap = t, gyro, accel, dt, snap


class ESKF:
    """Batched 15-state error-state Kalman filter (``n`` independent vehicles).

    Parameters are the IMU noise *densities* (accelerometer ``accel_noise`` [m/s²/√Hz], gyro ``gyro_noise`` [rad/s/√Hz]),
    bias random-walk densities, and the initial 1-σ uncertainties. ``gate_prob`` is the probability mass of the innovation
    chi-square distribution inside which an update is accepted (``None`` disables gating).
    """

    def __init__(
        self,
        B: Any,
        n: int,
        g: float = 9.80665,
        accel_noise: float = 0.02,
        gyro_noise: float = 0.002,
        accel_bias_walk: float = 3e-4,
        gyro_bias_walk: float = 2e-5,
        init_pos_std: float = 1.0,
        init_vel_std: float = 0.5,
        init_att_std_deg: float = 3.0,
        init_ba_std: float = 0.05,
        init_bg_std: float = 0.005,
        gate_prob: Optional[float] = 0.9999,
        max_delay_s: float = 0.5,
    ) -> None:
        self.B, self.n, self.g = B, int(n), float(g)
        self.sa, self.sg, self.sba, self.sbg = float(accel_noise), float(gyro_noise), float(accel_bias_walk), float(gyro_bias_walk)
        self.init_std = (init_pos_std, init_vel_std, math.radians(init_att_std_deg), init_ba_std, init_bg_std)
        self.gate_prob = gate_prob
        self.max_delay_s = float(max_delay_s)
        self.g_w = B.asarray([0.0, 0.0, -self.g])
        self.mag_ref: Any = None
        self.ground_z = 0.0
        self.reset()

    # ------------------------------------------------------------------------------------------ set-up
    def reset(self, pos: Any = None, vel: Any = None, quat: Any = None) -> None:
        B, n = self.B, self.n
        row = lambda x, k: B.clone(B.ones(n, k) * B.asarray(x)) if B.asarray(x).ndim == 1 else B.clone(B.asarray(x))  # noqa: E731  (one row for all)
        self.p = B.zeros(n, 3) if pos is None else row(pos, 3)
        self.v = B.zeros(n, 3) if vel is None else row(vel, 3)
        self.q = rot.identity(B, n) if quat is None else row(quat, 4)
        self.ba, self.bg = B.zeros(n, 3), B.zeros(n, 3)
        sp, sv, sa_, sba_, sbg_ = self.init_std
        d = B.asarray([sp**2] * 3 + [sv**2] * 3 + [sa_**2] * 3 + [sba_**2] * 3 + [sbg_**2] * 3)
        self.P = B.ones(n, 1, 1) * (B.eye(STATE_DIM) * d[None, :])
        self.t: Optional[float] = None
        self._hist: Deque[_Step] = deque()
        self._meas: List[Tuple[float, str, tuple]] = []
        self._replaying = False
        self.nis: Dict[str, Any] = {}  # latest NIS per update kind, (n,) each
        self.nis_log: Dict[str, List[np.ndarray]] = {}
        self.rejected: Dict[str, int] = {}

    def attitude_from_accel(self, accel: Any, yaw: float = 0.0) -> None:
        """Roll and pitch from the specific force (valid near hover), heading ``yaw``."""
        B = self.B
        ax, ay, az = accel[..., 0], accel[..., 1], accel[..., 2]
        roll = B.atan2(ay, az)
        pitch = B.atan2(-ax, B.sqrt(ay * ay + az * az))
        self.q = rot.from_euler(B, roll, pitch, roll * 0.0 + yaw)

    # ------------------------------------------------------------------------------------------ propagation
    def transition(self, R: Any, a: Any, w: Any, dt: float) -> Any:
        """Discrete error-state transition ``F = exp(A dt)`` (second order) for body-to-world ``R``, bias-corrected ``a`` and ``w``."""
        B, n = self.B, self.n
        A = B.zeros(n, STATE_DIM, STATE_DIM)
        A[:, 0:3, 3:6] = B.eye(3)
        A[:, 3:6, 6:9] = -B.einsum("nij,njk->nik", R, rot.hat(B, a))
        A[:, 3:6, 9:12] = -R
        A[:, 6:9, 6:9] = -rot.hat(B, w)
        A[:, 6:9, 12:15] = -B.eye(3)
        return B.eye(STATE_DIM)[None] + A * dt + 0.5 * B.einsum("nij,njk->nik", A, A) * (dt * dt)

    def _propagate(self, gyro: Any, accel: Any, dt: float) -> None:
        B = self.B
        w = gyro - self.bg
        a = accel - self.ba
        R = rot.to_matrix(B, self.q)
        R_mid = rot.to_matrix(B, rot.integrate(B, self.q, w, 0.5 * dt))  # attitude half-way through the step: second-order accurate
        a_w = B.einsum("nij,nj->ni", R_mid, a) + self.g_w
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        F = self.transition(R, a, w, dt)
        qd = [1e-12] * 3 + [self.sa**2 * dt] * 3 + [self.sg**2 * dt] * 3 + [self.sba**2 * dt] * 3 + [self.sbg**2 * dt] * 3
        Q = B.eye(STATE_DIM)[None] * B.asarray(qd)[None, None, :]
        self.P = _sym(B.einsum("nij,njk,nlk->nil", F, self.P, F) + Q)
        self.q = rot.integrate(B, self.q, w, dt)

    def predict(self, gyro: Any, accel: Any, dt: float, t: Optional[float] = None) -> None:
        """Propagate with one IMU sample covering ``dt`` seconds and ending at time ``t``."""
        self._propagate(gyro, accel, dt)
        self.t = (0.0 if self.t is None else self.t) + dt if t is None else float(t)
        if self.max_delay_s > 0.0:
            self._hist.append(_Step(self.t, self.B.clone(gyro), self.B.clone(accel), dt, _Snapshot(self)))
            while len(self._hist) > 1 and self.t - self._hist[0].t > self.max_delay_s:
                self._hist.popleft()
            t0 = self._hist[0].t
            self._meas = [m for m in self._meas if m[0] > t0]

    # ------------------------------------------------------------------------------------------ generic correction
    def _correct(self, kind: str, H: Any, y: Any, var: Any) -> None:
        """Apply one linearised update ``y = H δx + noise``; ``var`` is the noise variance, ``(m,)`` or ``(n, m)``."""
        B, n = self.B, self.n
        m = H.shape[-2]
        v = var if var.ndim == 2 else B.ones(n, 1) * var[None, :]
        Rn = B.eye(m)[None] * v[:, :, None]
        PHt = B.einsum("nij,nkj->nik", self.P, H)
        S = B.einsum("nij,njk->nik", H, PHt) + Rn
        Sinv = B.inv(S)
        nis = B.einsum("ni,nij,nj->n", y, Sinv, y)
        ok = B.isfinite(nis)
        if self.gate_prob is not None:
            ok = ok & (nis <= _chi2_gate(self.gate_prob, m))
        zero = self.p[:, 0] * 0.0  # a NaN-free zero of the right type: a rejected NaN innovation must not poison the gain
        okf = B.where(ok, zero + 1.0, zero)
        if not self._replaying:
            nn = np.array(B.to_numpy(nis), copy=True)
            self.nis[kind] = nn
            self.nis_log.setdefault(kind, []).append(nn)
            self.rejected[kind] = self.rejected.get(kind, 0) + int((~np.asarray(B.to_numpy(ok), bool)).sum())
        K = B.einsum("nij,njk->nik", PHt, Sinv) * okf[:, None, None]
        dx = B.einsum("nij,nj->ni", K, B.where(ok[:, None], y, B.zeros_like(y)))
        IKH = B.eye(STATE_DIM)[None] - B.einsum("nij,njk->nik", K, H)
        P = B.einsum("nij,njk,nlk->nil", IKH, self.P, IKH) + B.einsum("nij,njk,nlk->nil", K, Rn, K)
        # inject into the nominal state, then reset the error (rotation block of the reset Jacobian is I - ½[δθ]×)
        self.p = self.p + dx[:, 0:3]
        self.v = self.v + dx[:, 3:6]
        dth = dx[:, 6:9]
        self.q = rot.qnormalize(B, rot.qmul(B, self.q, rot.from_rotvec(B, dth)))
        self.ba = self.ba + dx[:, 9:12]
        self.bg = self.bg + dx[:, 12:15]
        G = B.ones(n, 1, 1) * B.eye(STATE_DIM)[None]
        G[:, 6:9, 6:9] = B.eye(3)[None] - 0.5 * rot.hat(B, dth)
        self.P = _sym(B.einsum("nij,njk,nlk->nil", G, P, G))

    # ------------------------------------------------------------------------------------------ measurement scheduling
    def _fuse(self, t: Optional[float], kind: str, args: tuple) -> None:
        """Apply now, or - when the stamp lies in the past - rewind, apply at that time, and replay."""
        if t is None or self.t is None or not self._hist or t >= self.t - _TOL:
            self._apply(kind, args)
            if self._hist and self.t is not None:
                self._hist[-1].snap = _Snapshot(self)
                self._meas.append((self.t if (t is None or t > self.t) else t, kind, args))
            return
        hist = list(self._hist)
        idx = next(i for i, s in enumerate(hist) if s.t >= t - _TOL)  # the step that ends at or after t_m
        if idx == 0:  # older than the stored history: there is no state to rewind to
            self.rejected["too_old"] = self.rejected.get("too_old", 0) + 1
            return
        self._meas.append((t, kind, args))
        self._meas.sort(key=lambda m: m[0])
        hist[idx - 1].snap.restore(self)
        self._replaying = True
        try:
            for j in range(idx, len(hist)):
                s = hist[j]
                self._propagate(s.gyro, s.accel, s.dt)
                lo = hist[j - 1].t
                for mt, mk, ma in self._meas:
                    if lo + _TOL < mt <= s.t + _TOL:
                        self._apply(mk, ma)
                s.snap = _Snapshot(self)
        finally:
            self._replaying = False

    def _apply(self, kind: str, args: tuple) -> None:
        getattr(self, "_do_" + kind)(*args)

    # ------------------------------------------------------------------------------------------ public updates
    def update_position(self, pos: Any, std: Any = 1.0, t: Optional[float] = None) -> None:
        self._fuse(t, "position", (self.B.asarray(pos), std))

    def update_velocity(self, vel: Any, std: Any = 0.15, t: Optional[float] = None) -> None:
        self._fuse(t, "velocity", (self.B.asarray(vel), std))

    def update_gnss(self, pos: Any, vel: Any = None, pos_std: Any = 1.0, vel_std: Any = 0.15, t: Optional[float] = None) -> None:
        """Position (and velocity) fix; fused together so one NIS covers the whole fix."""
        self._fuse(t, "gnss", (self.B.asarray(pos), None if vel is None else self.B.asarray(vel), pos_std, vel_std))

    def update_baro(self, altitude: Any, std: float = 0.6, t: Optional[float] = None) -> None:
        self._fuse(t, "baro", (self.B.asarray(altitude), std))

    def update_mag_heading(self, field_body: Any, std_deg: float = 4.0, t: Optional[float] = None) -> None:
        """Heading from a body-frame magnetic field against ``mag_ref`` (learned from the first reading when unset)."""
        self._fuse(t, "mag", (self.B.asarray(field_body), math.radians(std_deg)))

    def update_range(self, rng: Any, valid: Any = None, std: float = 0.05, direction: Any = (0.0, 0.0, -1.0), t: Optional[float] = None) -> None:
        """Single-beam range to flat ground at ``ground_z`` along the body direction ``direction``."""
        self._fuse(t, "range", (self.B.asarray(rng), None if valid is None else self.B.asarray(valid), std, tuple(float(x) for x in direction)))

    def update_flow(
        self, flow: Any, rng: Any, valid: Any = None, std: float = 0.03, normal: Any = (0.0, 0.0, -1.0), t: Optional[float] = None
    ) -> None:
        """Gyro-compensated optical flow ``-(n × v_b)/ρ`` (first two components, rad/s) with the measured range ``rng``."""
        v = None if valid is None else self.B.asarray(valid)
        self._fuse(t, "flow", (self.B.asarray(flow), self.B.asarray(rng), v, std, tuple(float(x) for x in normal)))

    def update_pose(self, pos: Any, quat: Any = None, pos_std: float = 0.05, att_std_deg: float = 1.0, t: Optional[float] = None) -> None:
        """Full pose from a visual-odometry or motion-capture source."""
        q = None if quat is None else self.B.asarray(quat)
        self._fuse(t, "pose", (self.B.asarray(pos), q, pos_std, math.radians(att_std_deg)))

    # ------------------------------------------------------------------------------------------ update implementations
    def _H(self, m: int) -> Any:
        return self.B.zeros(self.n, m, STATE_DIM)

    def _var(self, std: Any, m: int, is_var: bool = False) -> Any:
        """Variance vector of length ``m`` (or ``(n, m)``) from a scalar / per-axis / per-vehicle standard deviation."""
        B = self.B
        s = B.asarray(std)
        v = s if is_var else s * s
        if v.ndim == 0:
            return B.ones(m) * v
        if v.ndim == 1 and v.shape[0] == m:
            return v
        if v.ndim == 1 and v.shape[0] == self.n:
            return B.ones(self.n, m) * v[:, None]
        return v

    def _do_position(self, pos: Any, std: Any) -> None:
        H = self._H(3)
        H[:, :, 0:3] = self.B.eye(3)[None]
        self._correct("position", H, pos - self.p, self._var(std, 3))

    def _do_velocity(self, vel: Any, std: Any) -> None:
        H = self._H(3)
        H[:, :, 3:6] = self.B.eye(3)[None]
        self._correct("velocity", H, vel - self.v, self._var(std, 3))

    def _do_gnss(self, pos: Any, vel: Any, pos_std: Any, vel_std: Any) -> None:
        B = self.B
        if vel is None:
            return self._do_position(pos, pos_std)
        H = self._H(6)
        H[:, 0:3, 0:3] = B.eye(3)[None]
        H[:, 3:6, 3:6] = B.eye(3)[None]
        y = B.cat([pos - self.p, vel - self.v], axis=-1)
        self._correct("gnss", H, y, B.cat([self._var(pos_std, 3), self._var(vel_std, 3)], axis=-1))

    def _do_baro(self, alt: Any, std: float) -> None:
        H = self._H(1)
        H[:, 0, 2] = 1.0
        a = alt if alt.ndim == 1 else alt[..., 0]
        self._correct("baro", H, (a - self.p[:, 2])[:, None], self._var(std, 1))

    def _do_mag(self, m_b: Any, std: float) -> None:
        """Heading from the magnetometer. The residual is the horizontal angle from the measured field (rotated into the world
        by the current attitude) to the reference; its Jacobian includes the tilt coupling ``-m_z m_xy / |m_xy|²`` that a
        steeply inclined field produces, so roll and pitch errors do not masquerade as heading errors."""
        B = self.B
        R = rot.to_matrix(B, self.q)
        h = B.einsum("nij,nj->ni", R, m_b)
        if self.mag_ref is None:
            self.mag_ref = B.clone(h)  # learned reference: heading is then relative to the heading at the first reading
            return
        ref = self.mag_ref
        cross = h[:, 0] * ref[:, 1] - h[:, 1] * ref[:, 0]
        dot = h[:, 0] * ref[:, 0] + h[:, 1] * ref[:, 1]
        hxy2 = ref[:, 0] ** 2 + ref[:, 1] ** 2
        usable = (hxy2 > 1e-6) & ((h[:, 0] ** 2 + h[:, 1] ** 2) > 1e-6)
        y = B.where(usable, B.atan2(cross, dot), cross * 0.0)  # world-yaw error of the estimate (truth - estimate)
        safe = B.maximum(hxy2, hxy2 * 0.0 + 1e-6)
        hw = B.stack([-ref[:, 2] * ref[:, 0] / safe, -ref[:, 2] * ref[:, 1] / safe, hxy2 * 0.0 + 1.0], axis=-1)
        H = self._H(1)
        H[:, 0, 6:9] = B.einsum("ni,nij->nj", hw, R)  # δφ = h_w^T (R δθ)
        H = H * B.where(usable, y * 0.0 + 1.0, y * 0.0)[:, None, None]
        self._correct("mag", H, y[:, None], self._var(std * std, 1, is_var=True))

    def _do_range(self, rng: Any, valid: Any, std: float, direction: tuple) -> None:
        B, n = self.B, self.n
        d = np.asarray(direction, float)
        d_b = B.asarray(d / np.linalg.norm(d))
        R = rot.to_matrix(B, self.q)
        r2 = R[:, 2, :]  # third row of R
        dz = B.einsum("ni,i->n", r2, d_b)  # vertical component of the beam direction in the world
        h_alt = self.p[:, 2] - self.ground_z
        good = dz < -0.2  # pointing at the ground with a usable incidence
        if valid is not None:
            good = good & (valid > 0.5 if valid.dtype != bool else valid)
        one = dz * 0.0 + 1.0
        gf = B.where(good, one, one * 0.0)
        den = B.where(good, dz, one * -1.0)
        pred = -h_alt / den
        H = self._H(1)
        H[:, 0, 2] = -1.0 / den
        H[:, 0, 6:9] = (h_alt / den**2)[:, None] * B.cross(B.ones(n, 3) * d_b, r2)
        H = H * gf[:, None, None]
        self._correct("range", H, ((rng - pred) * gf)[:, None], self._var(std, 1))

    def _do_flow(self, flow: Any, rng: Any, valid: Any, std: float, normal: tuple) -> None:
        B, n = self.B, self.n
        nb = B.asarray(list(normal))
        R = rot.to_matrix(B, self.q)
        Rt = R.swapaxes(-1, -2)
        vb = B.einsum("nij,nj->ni", Rt, self.v)
        rho = B.maximum(rng, rng * 0.0 + 0.05)
        nx = rot.hat(B, B.ones(n, 3) * nb)
        h3 = -B.einsum("nij,nj->ni", nx, vb) / rho[:, None]
        H3 = self._H(3)
        H3[:, :, 3:6] = -B.einsum("nij,njk->nik", nx, Rt) / rho[:, None, None]
        H3[:, :, 6:9] = -B.einsum("nij,njk->nik", nx, rot.hat(B, vb)) / rho[:, None, None]
        H, y = H3[:, 0:2, :], flow[..., 0:2] - h3[:, 0:2]
        if valid is not None:
            w = B.where(valid, rho * 0.0 + 1.0, rho * 0.0)
            H, y = H * w[:, None, None], y * w[:, None]
        self._correct("flow", H, y, self._var(std, 2))

    def _do_pose(self, pos: Any, quat: Any, pos_std: float, att_std: float) -> None:
        B = self.B
        if quat is None:
            H = self._H(3)
            H[:, :, 0:3] = B.eye(3)[None]
            return self._correct("pose", H, pos - self.p, self._var(pos_std, 3))
        H = self._H(6)
        H[:, 0:3, 0:3] = B.eye(3)[None]
        H[:, 3:6, 6:9] = B.eye(3)[None]
        y = B.cat([pos - self.p, self._attitude_error(quat)], axis=-1)
        self._correct("pose", H, y, B.cat([self._var(pos_std, 3), self._var(att_std, 3)], axis=-1))

    def _attitude_error(self, quat: Any) -> Any:
        """Body-frame rotation vector ``δθ`` with ``q_true = q ⊗ exp(δθ)``."""
        B = self.B
        dq = rot.qmul(B, rot.qconj(B, self.q), B.asarray(quat))
        s = B.where(dq[:, 0] < 0, dq[:, 0] * 0.0 - 1.0, dq[:, 0] * 0.0 + 1.0)
        v = dq[:, 1:4] * s[:, None]
        sn = B.norm(v, axis=-1)
        ang = 2.0 * B.atan2(sn, B.abs(dq[:, 0]))
        k = B.where(sn < 1e-9, sn * 0.0 + 2.0, ang / B.maximum(sn, sn * 0.0 + 1e-12))
        return v * k[:, None]

    # ------------------------------------------------------------------------------------------ outputs
    @property
    def pos(self) -> Any:
        return self.p

    @property
    def vel(self) -> Any:
        return self.v

    def std(self) -> Any:
        """1-σ of every error state, ``(n, 15)``."""
        B = self.B
        d = B.stack([self.P[:, i, i] for i in range(STATE_DIM)], axis=-1)
        return B.sqrt(B.maximum(d, d * 0.0))

    def error_to(self, pos: Any, vel: Any, quat: Any, ba: Any = None, bg: Any = None) -> Any:
        """True error state ``δx`` (truth minus estimate, attitude error in the body frame), ``(n, 15)``."""
        B = self.B
        zero = self.p * 0.0
        return B.cat(
            [
                B.asarray(pos) - self.p,
                B.asarray(vel) - self.v,
                self._attitude_error(quat),
                zero if ba is None else B.asarray(ba) - self.ba,
                zero if bg is None else B.asarray(bg) - self.bg,
            ],
            axis=-1,
        )

    def nees(self, pos: Any, vel: Any, quat: Any, ba: Any = None, bg: Any = None, parts: Tuple[str, ...] = ("p", "v", "theta")) -> Any:
        """Normalised estimation error squared on the chosen blocks, ``(n,)``; ~ chi-square with ``3 len(parts)`` dof when consistent."""
        B = self.B
        e = self.error_to(pos, vel, quat, ba, bg)
        idx = np.concatenate([np.arange(STATE_DIM)[ERR_SLICES[k]] for k in parts])
        ei = e[:, idx]
        Pi = self.P[:, idx][:, :, idx]
        return B.einsum("ni,nij,nj->n", ei, B.inv(Pi), ei)


# ======================================================================================================== engine wrapper
def _kind_of(r: Dict[str, Any]) -> Optional[str]:
    if "gyro" in r and "accel" in r:
        return "imu"
    if "pos" in r and "vel" in r:
        return "gnss"
    if "altitude" in r:
        return "baro"
    if "field" in r:
        return "mag"
    if "flow_comp" in r:
        return "flow"
    if "range" in r and "valid" in r:
        return "range"
    return None


class ESKFEstimator:
    """Registry-facing estimator: an :class:`ESKF` fed by a :class:`~aevion.sensors.SensorSuite` reading dictionary.

    Sensors are recognised by their output keys (so any name works): ``gyro``+``accel`` is the IMU, ``pos``+``vel`` a GNSS
    receiver, ``altitude`` a barometer, ``field`` a magnetometer, ``range``+``valid`` a rangefinder and ``flow_comp`` an
    optical-flow sensor. Measurements are fused at their *sample time*, so sensor latency is compensated automatically (up to
    ``max_delay_s``). Position is initialised from the first GNSS fix, roll and pitch from the first accelerometer sample, and
    heading from the magnetometer reference (``mag_ref``, an ENU field vector) or, when none is given, taken as zero at start.
    """

    def __init__(
        self,
        sim: Any,
        use_mag: bool = True,
        use_baro: bool = True,
        use_range: bool = True,
        use_flow: bool = True,
        gnss_pos_std: float = 1.0,
        gnss_vel_std: float = 0.15,
        baro_std: float = 0.6,
        mag_std_deg: float = 4.0,
        range_std: float = 0.05,
        flow_std: float = 0.03,
        mag_ref: Any = None,
        ground_z: float = 0.0,
        **kw: Any,
    ) -> None:
        self.B, self.n = sim.B, sim.n
        kw.setdefault("g", sim.world.gravity.g)
        self.filter = ESKF(self.B, self.n, **kw)
        self.use = dict(mag=use_mag, baro=use_baro, range=use_range, flow=use_flow)
        self.std = dict(gnss_pos=gnss_pos_std, gnss_vel=gnss_vel_std, baro=baro_std, mag=mag_std_deg, range=range_std, flow=flow_std)
        self._mag_ref_cfg = mag_ref
        self.filter.ground_z = float(ground_z)
        self.reset(None)

    @property
    def kf(self) -> ESKF:
        return self.filter

    def position_std(self) -> Any:
        """1-σ of the position estimate, ``(n, 3)`` (recorded by the engine as ``est_pos_std``)."""
        return self.filter.std()[:, 0:3]

    def reset(self, true_state: Any = None) -> None:
        f, B = self.filter, self.B
        f.reset()
        f.mag_ref = None if self._mag_ref_cfg is None else B.ones(self.n, 3) * B.asarray([float(x) for x in self._mag_ref_cfg])
        self._t_imu: Optional[float] = None
        self._att_ready = False
        self.started = False
        self.gyro = B.zeros(self.n, 3)
        self._stamp = {"gnss": -1.0, "baro": -1.0, "mag": -1.0, "range": -1.0, "flow": -1.0}
        if true_state is not None:
            f.reset(true_state.pos, true_state.vel, true_state.quat)
            self._att_ready = self.started = True

    def _estimate(self) -> Any:
        from .filters import EstimatedState

        f = self.filter
        return EstimatedState(pos=f.p, vel=f.v, quat=f.q, omega=self.gyro, t=self._t_imu or 0.0)

    def update(self, readings: Dict[str, Any], fresh: Dict[str, bool]) -> Any:
        f, B = self.filter, self.B
        got: Dict[str, Tuple[Dict[str, Any], bool]] = {}
        for name, r in readings.items():
            k = _kind_of(r) if r is not None else None
            if k is not None and k not in got:
                got[k] = (r, bool(fresh.get(name, False)))
        imu, imu_new = got.get("imu", (None, False))
        if imu is not None and imu_new:
            t = float(imu["t"])
            if not self._att_ready:
                f.attitude_from_accel(imu["accel"])
                self._att_ready, self._t_imu = True, t
            dt = (t - self._t_imu) if self._t_imu is not None else 1.0 / 200.0
            self._t_imu = t
            self.gyro = imu["gyro"] - f.bg
            if self.started and dt > 0.0:
                f.predict(imu["gyro"], imu["accel"], dt, t)
        g, g_new = got.get("gnss", (None, False))
        if g is not None and g_new and self._stamp["gnss"] < float(g["t"]):
            finite = math.isfinite(float(g["pos"].sum()) + float(g["vel"].sum()))
            if finite and ("fix" not in g or float(g["fix"].min()) > 0.5):  # no fix or a corrupted (NaN) reading: skip it
                self._stamp["gnss"] = float(g["t"])
                if not self.started:
                    f.p, f.v, f.t = B.clone(g["pos"]), B.clone(g["vel"]), float(g["t"])
                    self.started = True
                else:
                    f.update_gnss(g["pos"], g["vel"], self.std["gnss_pos"], self.std["gnss_vel"], t=float(g["t"]))
        if not self.started:
            return self._estimate()
        for kind in ("baro", "mag", "range", "flow"):
            r, new = got.get(kind, (None, False))
            if r is None or not new or not self.use[kind] or self._stamp[kind] >= float(r["t"]):
                continue
            self._stamp[kind], t = float(r["t"]), float(r["t"])
            if kind == "baro":
                f.update_baro(r["altitude"], self.std["baro"], t=t)
            elif kind == "mag" and math.isfinite(float(r["field"].sum())):
                f.update_mag_heading(r["field"], self.std["mag"], t=t)
            elif kind == "range":
                f.update_range(r["range"], r["valid"], self.std["range"], t=t)
            elif kind == "flow" and "range" in r:
                f.update_flow(r["flow_comp"], r["range"], r["valid"], self.std["flow"], t=t)
        return self._estimate()
