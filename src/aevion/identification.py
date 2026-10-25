"""System identification: fit a vehicle's physical parameters to a flight log.

Given a log with the vehicle state (position, velocity, attitude, body rates) and its rotor inputs, :func:`identify` finds the
multipliers of selected parameters (thrust coefficient, drag, inertia, motor time constant, ...) for which the simulation reproduces
the log. The method is **multiple shooting**: the log is cut into windows of about a second, each simulated forward from the recorded
state at its start with the recorded inputs, and the mismatch along the window is the residual. Short windows keep the problem
well-conditioned (a long open-loop prediction of an unstable vehicle is hopeless), and all windows run as one batch. Multipliers
come from bounded nonlinear least squares in log-space (so they stay positive); the Gauss-Newton covariance ``σ² (JᵀJ)⁻¹`` gives
their standard errors.

**What can be identified.** From a flight with known mass: the thrust coefficient ``kT``, the roll/pitch inertia ``Jxy`` (given ``kT``),
the drag (``drag_lin``, ``drag_quad``) and the motor time constant. The yaw channel only fixes the ratio ``kQ / Jz``, so fit one
of them. Mass and ``kT`` appear only as ``kT / m``, so supply the mass. :func:`identify` reports the correlation matrix to expose such
degeneracies (a correlation near ±1 means the pair is not separable from this log).

Inputs are the rotor-speed commands when the log has them (``cmd``), otherwise the measured speeds (``rotor``, with the motor lag
excluded from the fit).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import rotation as rot
from .core.backend import NumpyBackend

__all__ = ["identify", "IdentificationResult", "PARAMETERS", "simulate_windows", "load_csv_log"]

B = NumpyBackend()


def _inertia(s: Any, jx: float = 1.0, jy: float = 1.0, jz: float = 1.0) -> Dict[str, Any]:
    J = np.array(np.asarray(s.inertia_matrix), float)
    J[0, 0] *= jx
    J[1, 1] *= jy
    J[2, 2] *= jz
    return dict(inertia=tuple(J.reshape(-1)))


def _tau(s: Any, f: float) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(motor_tau=s.motor_tau * f)
    if s.motor_tau_down:
        out["motor_tau_down"] = s.motor_tau_down * f
    return out


# name -> (description, function(spec, multiplier) -> the VehicleSpec fields it changes)
PARAMETERS: Dict[str, Tuple[str, Callable[[Any, float], Dict[str, Any]]]] = {
    "mass": ("vehicle mass", lambda s, f: dict(mass=s.mass * f)),
    "kT": ("rotor thrust coefficient", lambda s, f: dict(kT=s.kT * f)),
    "kQ": ("rotor drag-torque coefficient", lambda s, f: dict(kQ=s.kQ * f)),
    "Jxy": ("roll and pitch inertia", lambda s, f: _inertia(s, f, f, 1.0)),
    "Jz": ("yaw inertia", lambda s, f: _inertia(s, 1.0, 1.0, f)),
    "tau": ("motor time constant", _tau),
    "drag_lin": ("linear body drag (all axes)", lambda s, f: dict(drag_lin=tuple(np.asarray(s.drag_lin, float) * f))),
    "drag_quad": ("quadratic body drag (all axes)", lambda s, f: dict(drag_quad=tuple(np.asarray(s.drag_quad, float) * f))),
    "ang_damp": ("angular damping", lambda s, f: dict(angular_damping=tuple(np.asarray(s.angular_damping, float) * f))),
}


def apply_factors(spec: Any, factors: Dict[str, float]) -> Any:
    """A copy of ``spec`` with the named parameters multiplied by the given factors."""
    changes: Dict[str, Any] = {}
    for name, f in factors.items():
        changes.update(PARAMETERS[name][1](spec, float(f)))
    return spec.copy(**changes) if changes else copy.deepcopy(spec)


@dataclass
class IdentificationResult:
    """Fitted multipliers and the values they imply, with their uncertainty."""

    names: List[str]
    factors: np.ndarray  # multipliers of the nominal values
    values: Dict[str, Any]  # name -> identified value (scalar or array)
    stderr: np.ndarray  # one standard error of each factor (relative, since the fit is in log space)
    correlation: np.ndarray
    cost: float
    rms: float  # root-mean-square weighted residual
    n_windows: int
    n_residuals: int
    success: bool
    message: str
    initial_rms: float = float("nan")

    def report(self) -> str:
        lines = [
            f"identification: {self.n_windows} windows, {self.n_residuals} residuals, rms {self.initial_rms:.3g} -> {self.rms:.3g} ({self.message})"
        ]
        for n, f, s in zip(self.names, self.factors, self.stderr):
            lines.append(f"  {n:<10} x{f:7.4f}  (+/- {100 * s:.2f} %)")
        worst = (
            np.unravel_index(np.argmax(np.abs(self.correlation - np.eye(len(self.names)))), self.correlation.shape) if len(self.names) > 1 else None
        )
        if worst is not None and abs(self.correlation[worst]) > 0.9:
            lines.append(
                f"  warning: {self.names[worst[0]]} and {self.names[worst[1]]} are strongly correlated ({self.correlation[worst]:+.2f}): not separable from this log"
            )
        return "\n".join(lines)


def _log_arrays(log: Any, vehicle: int) -> Dict[str, np.ndarray]:
    get = lambda k: np.asarray(log[k])  # noqa: E731
    out = {k: get(k) for k in ("t", "pos", "vel", "quat", "omega")}
    for k in ("pos", "vel", "quat", "omega"):
        if out[k].ndim == 3:
            out[k] = out[k][:, vehicle]
    for k in ("rotor", "cmd"):
        try:
            v = get(k)
            out[k] = v[:, vehicle] if v.ndim == 3 else v
        except KeyError:
            pass
    if "cmd" not in out and "rotor" not in out:
        raise ValueError("the log needs rotor commands ('cmd') or measured rotor speeds ('rotor') as inputs")
    return out


def _windows(t: np.ndarray, window_s: float, stride_s: float) -> List[Tuple[int, int]]:
    dt = float(np.median(np.diff(t)))
    w = max(round(window_s / dt), 2)
    s = max(round(stride_s / dt), 1)
    return [(i, i + w) for i in range(0, len(t) - w, s)]


def simulate_windows(
    spec: Any,
    log: Any,
    factors: Dict[str, float],
    windows: Optional[Sequence[Tuple[int, int]]] = None,
    window_s: float = 1.0,
    stride_s: float = 0.5,
    vehicle: int = 0,
    world: Any = None,
    use_measured_rotors: bool = False,
) -> Dict[str, np.ndarray]:
    """Simulate every window from its recorded initial state; returns the predicted ``pos, vel, quat, omega`` as ``(W, K+1, ...)``."""
    from .dynamics.multirotor import Multirotor

    d = _log_arrays(log, vehicle)
    t = d["t"]
    win = list(windows) if windows is not None else _windows(t, window_s, stride_s)
    W = len(win)
    K = win[0][1] - win[0][0]
    dt = float(np.median(np.diff(t)))
    use_meas = use_measured_rotors or "cmd" not in d
    spec2 = apply_factors(spec, factors)
    if use_meas:  # inputs are the measured rotor speeds: take the motor lag out so they act directly (2 ms keeps the step cheap)
        spec2 = spec2.copy(motor_tau=2e-3, motor_tau_down=None)
    sim = Multirotor(spec2, n=W, dt=dt, world=world, kernel="auto", verify_kernel=False)
    inp = d["rotor"] if use_meas else d["cmd"]
    i0 = np.array([a for a, _ in win])
    st = sim.reset(pos=d["pos"][i0], vel=d["vel"][i0], omega=d["omega"][i0], hover_rotors=True)
    st.quat = d["quat"][i0].copy()
    st.rotor = (d["rotor"][i0] if "rotor" in d else inp[i0]).astype(float).copy()
    out = {k: np.empty((W, K + 1) + v.shape[1:]) for k, v in (("pos", d["pos"]), ("vel", d["vel"]), ("quat", d["quat"]), ("omega", d["omega"]))}
    for key in ("pos", "vel", "quat", "omega"):
        out[key][:, 0] = getattr(st, key)
    for j in range(K):
        cmd = inp[i0 + j]
        st = sim.step(st, cmd)
        for key in ("pos", "vel", "quat", "omega"):
            out[key][:, j + 1] = getattr(st, key)
    return out


def identify(
    spec: Any,
    log: Any,
    free: Sequence[str] = ("kT", "kQ", "Jxy", "drag_lin", "tau"),
    window_s: float = 1.0,
    stride_s: float = 0.5,
    vehicle: int = 0,
    world: Any = None,
    noise: Optional[Dict[str, float]] = None,
    bounds: Tuple[float, float] = (0.2, 5.0),
    x0: Optional[Dict[str, float]] = None,
    use_measured_rotors: bool = False,
    max_nfev: int = 60,
) -> IdentificationResult:
    """Fit multipliers of ``free`` parameters of ``spec`` so the simulation reproduces ``log`` (see the module documentation).

    ``log`` is a :class:`~aevion.datasets.archive.RunArchive` or a dictionary with ``t, pos, vel, quat, omega`` and ``cmd`` (rotor
    speed commands) and/or ``rotor``. ``noise`` gives the 1-σ measurement noise of ``pos, vel, att`` (rad) and ``omega`` used to
    weight the residuals (defaults are typical of a good state estimate). ``bounds`` limit each multiplier.
    """
    from scipy.optimize import least_squares

    for n in free:
        if n not in PARAMETERS:
            raise ValueError(f"cannot identify {n!r}; choose from {sorted(PARAMETERS)}")
    d = _log_arrays(log, vehicle)
    win = _windows(d["t"], window_s, stride_s)
    if len(win) < 3:
        raise ValueError("the log is too short for the requested windows")
    sig = dict(pos=0.02, vel=0.03, att=0.01, omega=0.02)
    sig.update(noise or {})
    free = list(free)
    ref = d

    def residuals(x: np.ndarray) -> np.ndarray:
        f = {n: float(np.exp(v)) for n, v in zip(free, x)}
        sim = simulate_windows(spec, log, f, win, vehicle=vehicle, world=world, use_measured_rotors=use_measured_rotors)
        idx = np.array([[a + j for j in range(win[0][1] - win[0][0] + 1)] for a, _ in win])
        idx = np.minimum(idx, len(ref["t"]) - 1)
        rp = (sim["pos"] - ref["pos"][idx]) / sig["pos"]
        rv = (sim["vel"] - ref["vel"][idx]) / sig["vel"]
        ro = (sim["omega"] - ref["omega"][idx]) / sig["omega"]
        qd = rot.qmul(B, rot.qconj(B, ref["quat"][idx]), sim["quat"])
        sgn = np.where(qd[..., :1] < 0, -1.0, 1.0)
        ra = 2.0 * (qd[..., 1:4] * sgn) / sig["att"]
        r = np.concatenate([rp.reshape(-1), rv.reshape(-1), ro.reshape(-1), ra.reshape(-1)])
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

    start = np.log(np.array([(x0 or {}).get(n, 1.0) for n in free]))
    lo, hi = np.log(bounds[0]) * np.ones(len(free)), np.log(bounds[1]) * np.ones(len(free))
    r0 = residuals(start)
    res = least_squares(residuals, start, bounds=(lo, hi), x_scale=0.1, max_nfev=max_nfev, diff_step=1e-3, method="trf")
    n_res = len(res.fun)
    dof = max(n_res - len(free), 1)
    s2 = float(res.fun @ res.fun) / dof
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((len(free), len(free)), np.nan)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = cov / np.outer(np.where(se > 0, se, 1.0), np.where(se > 0, se, 1.0))
    factors = np.exp(res.x)
    nominal = _nominal_values(spec)
    values = {n: _scaled(nominal[n], f) for n, f in zip(free, factors)}
    return IdentificationResult(
        free,
        factors,
        values,
        se,
        corr,
        float(res.cost),
        float(np.sqrt(np.mean(res.fun**2))),
        len(win),
        n_res,
        bool(res.success),
        res.message,
        float(np.sqrt(np.mean(r0**2))),
    )


def _nominal_values(spec: Any) -> Dict[str, Any]:
    return dict(
        mass=spec.mass,
        kT=spec.kT,
        kQ=spec.kQ,
        Jxy=float(np.asarray(spec.inertia_matrix)[0, 0]),
        Jz=float(np.asarray(spec.inertia_matrix)[2, 2]),
        tau=spec.motor_tau,
        drag_lin=np.asarray(spec.drag_lin, float),
        drag_quad=np.asarray(spec.drag_quad, float),
        ang_damp=np.asarray(spec.angular_damping, float),
    )


def _scaled(v: Any, f: float) -> Any:
    return v * f if isinstance(v, np.ndarray) else float(v) * float(f)


def load_csv_log(path: Any, columns: Optional[Dict[str, Any]] = None, delimiter: str = ",") -> Dict[str, np.ndarray]:
    """Read a CSV flight log into the dictionary :func:`identify` expects.

    ``columns`` maps channel -> column name(s): ``{"t": "time", "pos": ["x", "y", "z"], "vel": ["vx", "vy", "vz"], "quat": ["qw",
    "qx", "qy", "qz"], "omega": ["p", "q", "r"], "cmd": ["m1", "m2", "m3", "m4"]}`` (the defaults assume exactly these names).
    Arrays come back with a vehicle axis of length one, ``(T, 1, k)``.
    """
    import csv

    cols = columns or {
        "t": "t",
        "pos": ["x", "y", "z"],
        "vel": ["vx", "vy", "vz"],
        "quat": ["qw", "qx", "qy", "qz"],
        "omega": ["p", "q", "r"],
        "cmd": ["m1", "m2", "m3", "m4"],
    }
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=delimiter))
    if not rows:
        raise ValueError("the CSV file has no rows")
    out: Dict[str, np.ndarray] = {}
    for ch, names in cols.items():
        nm = [names] if isinstance(names, str) else list(names)
        missing = [n for n in nm if n not in rows[0]]
        if missing:
            if ch in ("cmd", "rotor"):
                continue
            raise ValueError(f"column(s) {missing} for channel {ch!r} not found; available: {sorted(rows[0])}")
        a = np.array([[float(r[n]) for n in nm] for r in rows])
        out[ch] = a[:, 0] if ch == "t" else a[:, None, :]
    return out
