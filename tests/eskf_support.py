"""A truth model and sensor simulators for exercising the ESKF with statistics (Monte Carlo over the batch axis)."""

import math

import numpy as np

from aevion.core import rotation as rot
from aevion.core.backend import NumpyBackend
from aevion.estimation.eskf import ESKF

B = NumpyBackend()
G = 9.80665
M_REF = np.array([18.0, 4.0, -43.0])  # a mid-latitude field, ENU [uT]


def inputs(t):
    """Smooth world-frame acceleration (what the vehicle does) and body angular rate (how it tilts, within ~20 degrees)."""
    a = np.array([0.5 * math.sin(0.7 * t), 0.4 * math.cos(0.5 * t), 0.3 * math.sin(0.3 * t)])
    w = np.array([0.25 * math.sin(1.0 * t), 0.2 * math.cos(0.8 * t), 0.3 * math.sin(0.2 * t)])
    return a, w


class Truth:
    """True state of ``n`` vehicles: the translation follows the commanded acceleration, the attitude the angular rate."""

    def __init__(self, n, p0, v0, q0, sub=4):
        self.n, self.sub = n, sub
        self.p, self.v, self.q = p0.copy(), v0.copy(), q0.copy()

    def step(self, a, w, dt):
        """Advance one step; returns the specific force (body) an ideal accelerometer measures over it, ``(n, 3)``."""
        h = dt / self.sub
        fb = 0.0
        for _ in range(self.sub):
            qm = rot.integrate(B, self.q, np.tile(w, (self.n, 1)), 0.5 * h)
            fb = fb + B.einsum("nji,j->ni", rot.to_matrix(B, qm), a + np.array([0.0, 0.0, G])) / self.sub
            self.p = self.p + self.v * h + 0.5 * a * h * h
            self.v = self.v + a * h
            self.q = rot.integrate(B, self.q, np.tile(w, (self.n, 1)), h)
        return fb


def monte_carlo(
    n=200,
    duration=20.0,
    dt=0.01,
    seed=0,
    sensors=("gnss", "baro", "mag"),
    delay_s=0.0,
    filt=None,
    flush=False,
    flow_std=0.03,
    range_std=0.05,
    mag_std_deg=1.5,
    **kw,
):
    """Run ``n`` independent vehicles through the same manoeuvre with independent noise and initial errors.

    Initial errors and true biases are drawn from the filter's own prior, as consistency tests require. Returns a dict with
    the NEES history ``(T, n)`` of the position / velocity / attitude block (``nees``), of each block, the filter, and truth.
    """
    rng = np.random.default_rng(seed)
    f = filt or ESKF(B, n, max_delay_s=max(0.5, delay_s + 0.2), **kw)
    sp, sv, sth, sba, sbg = f.init_std
    p0 = np.tile([0.0, 0.0, 40.0], (n, 1))
    v0 = np.tile([1.0, 0.5, 0.0], (n, 1))
    q0 = rot.from_euler(B, np.full(n, 0.05), np.full(n, -0.04), np.full(n, 0.6))
    truth = Truth(n, p0, v0, q0)
    ba = rng.normal(0.0, sba, (n, 3))
    bg = rng.normal(0.0, sbg, (n, 3))
    # the filter starts with an error drawn from its prior covariance
    f.reset(
        p0 - rng.normal(0.0, sp, (n, 3)),
        v0 - rng.normal(0.0, sv, (n, 3)),
        rot.qmul(B, q0, rot.from_rotvec(B, -rng.normal(0.0, sth, (n, 3)))),
    )
    f.mag_ref = np.tile(M_REF, (n, 1))
    steps = int(duration / dt)
    out = dict(nees=[], nees_p=[], nees_v=[], nees_th=[], t=[], bias_err=[])
    pending = []  # (deliver_time, call, args)
    t = 0.0
    rates = dict(gnss=5.0, baro=20.0, mag=20.0, range=20.0, flow=25.0, pose=10.0)
    for k in range(steps):
        ab, wb = inputs(t + 0.5 * dt)
        fb = truth.step(ab, wb, dt)
        ba += rng.normal(0.0, f.sba * math.sqrt(dt), (n, 3))
        bg += rng.normal(0.0, f.sbg * math.sqrt(dt), (n, 3))
        acc_m = fb + ba + rng.normal(0.0, f.sa / math.sqrt(dt), (n, 3))
        gyr_m = wb + bg + rng.normal(0.0, f.sg / math.sqrt(dt), (n, 3))
        t += dt
        f.predict(gyr_m, acc_m, dt, t)
        due = lambda name, tt=t: abs((tt / (1.0 / rates[name])) - round(tt / (1.0 / rates[name]))) < 0.5 * dt * rates[name]
        R = rot.to_matrix(B, truth.q)
        if "gnss" in sensors and due("gnss"):
            pend = (
                f.update_gnss,
                (truth.p + rng.normal(0, 1.0, (n, 3)), truth.v + rng.normal(0, 0.15, (n, 3)), 1.0, 0.15),
                dict(t=t),
            )
            pending.append((t + delay_s, *pend))
        if "baro" in sensors and due("baro"):
            pending.append((t + delay_s, f.update_baro, (truth.p[:, 2] + rng.normal(0, 0.6, n), 0.6), dict(t=t)))
        if "mag" in sensors and due("mag"):
            m_b = B.einsum("nji,j->ni", R, M_REF) + rng.normal(0, 0.5, (n, 3))
            pending.append((t + delay_s, f.update_mag_heading, (m_b, mag_std_deg), dict(t=t)))
        if "range" in sensors and due("range"):
            r2 = R[:, 2, :]
            true_r = truth.p[:, 2] / r2[:, 2]
            pending.append((t + delay_s, f.update_range, (true_r + rng.normal(0, range_std, n), None, range_std), dict(t=t)))
        if "flow" in sensors and due("flow"):
            vb = B.einsum("nji,nj->ni", R, truth.v)
            rho = truth.p[:, 2] / R[:, 2, 2]
            fl = np.stack([-vb[:, 1], vb[:, 0]], -1) / rho[:, None]
            pending.append((t + delay_s, f.update_flow, (fl + rng.normal(0, flow_std, (n, 2)), rho, None, flow_std), dict(t=t)))
        if "pose" in sensors and due("pose"):
            qn = rot.qmul(B, truth.q, rot.from_rotvec(B, rng.normal(0.0, math.radians(1.0), (n, 3))))
            pending.append((t + delay_s, f.update_pose, (truth.p + rng.normal(0, 0.05, (n, 3)), qn, 0.05, 1.0), dict(t=t)))
        ready = [m for m in pending if m[0] <= t + 1e-9]
        pending = [m for m in pending if m[0] > t + 1e-9]
        for _, fn, args, kws in ready:
            fn(*args, **kws)
        if k % 10 == 9:
            out["nees"].append(f.nees(truth.p, truth.v, truth.q))
            out["nees_p"].append(f.nees(truth.p, truth.v, truth.q, parts=("p",)))
            out["nees_v"].append(f.nees(truth.p, truth.v, truth.q, parts=("v",)))
            out["nees_th"].append(f.nees(truth.p, truth.v, truth.q, parts=("theta",)))
            out["t"].append(t)
    if flush:  # deliver whatever is still in flight, so runs with different delays can be compared at the same instant
        for _, fn, args, kws in pending:
            fn(*args, **kws)
    res = {k: np.array(v) for k, v in out.items()}
    res.update(filter=f, truth=truth, ba=ba, bg=bg)
    return res
