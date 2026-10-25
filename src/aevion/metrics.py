"""Flight and research metrics computed from a :class:`~aevion.datasets.RunArchive`."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _per_vehicle(archive, fn):
    return np.array([fn(i) for i in range(archive.n_vehicles)])


def tracking_errors(archive, skip: float = 0.0) -> dict:
    """Position-error time series and summary stats against ``ref_pos`` (per vehicle)."""
    pos, ref = archive["pos"], archive.get("ref_pos")
    if ref is None:
        raise KeyError("archive has no 'ref_pos' channel")
    mask = archive.t >= archive.t[0] + skip
    e = (pos - ref)[mask]
    e3 = np.linalg.norm(e, axis=-1)
    exy = np.linalg.norm(e[..., :2], axis=-1)
    ez = np.abs(e[..., 2])
    return dict(
        rmse=np.sqrt((e3**2).mean(0)),
        rmse_xy=np.sqrt((exy**2).mean(0)),
        rmse_z=np.sqrt((ez**2).mean(0)),
        mean=e3.mean(0),
        max=e3.max(0),
        final=e3[-1],
        series=np.linalg.norm(pos - ref, axis=-1),
    )


def flight_metrics(archive, skip: float = 0.0) -> Dict[str, np.ndarray]:
    """Per-vehicle metrics as arrays of length N."""
    t = archive.t
    pos, vel = archive["pos"], archive["vel"]
    dt = np.gradient(t) if len(t) > 1 else np.ones(1)
    m: Dict[str, np.ndarray] = {}
    if "ref_pos" in archive:
        te = tracking_errors(archive, skip)
        m.update(rmse_pos=te["rmse"], rmse_xy=te["rmse_xy"], rmse_z=te["rmse_z"], mean_err=te["mean"], max_err=te["max"], final_err=te["final"])
        half = tracking_errors(archive, skip=(t[-1] - t[0]) / 2)
        m["rmse_steady"] = half["rmse"]
    speed = np.linalg.norm(vel, axis=-1)
    m["path_length"] = (np.linalg.norm(np.diff(pos, axis=0), axis=-1)).sum(0)
    m["mean_speed"], m["max_speed"] = speed.mean(0), speed.max(0)
    e = archive["euler"]
    m["max_tilt_deg"] = np.degrees(np.maximum(np.abs(e[..., 0]), np.abs(e[..., 1]))).max(0)
    m["max_rate_deg_s"] = np.degrees(np.linalg.norm(archive["omega"], axis=-1)).max(0)
    acc = archive["acc"]
    m["max_acc"] = np.linalg.norm(acc, axis=-1).max(0)
    if len(t) > 3:
        jerk = np.gradient(acc, t, axis=0)
        m["rms_jerk"] = np.sqrt((np.linalg.norm(jerk, axis=-1) ** 2).mean(0))
    p = archive["power"]
    m["energy_wh"] = (p * dt[:, None]).sum(0) / 3600.0
    m["mean_power_w"] = p.mean(0)
    m["soc_end"] = archive["soc"][-1]
    rot = archive["rotor"]
    m["rotor_rms_dev"] = np.sqrt(((rot - rot.mean(0)) ** 2).mean((0, 2)))
    spec = archive.spec
    if spec is not None:
        m["rotor_saturation_frac"] = (rot > 0.98 * spec.w_max).any(-1).mean(0)
        m["thrust_margin"] = 1.0 - (archive["thrust"] / spec.max_thrust()).max(0)
    if "collided" in archive:
        m["collisions"] = (np.diff(archive["collided"].astype(int), axis=0, prepend=0) > 0).sum(0)
    if "contact" in archive:
        c = archive["contact"].astype(bool)
        m["ground_contact_frac"] = c.mean(0)
        late = t > t[0] + 1.0
        m["crashed"] = (c & late[:, None]).any(0).astype(float)
    if archive.n_vehicles > 1:
        from .multiagent import min_separation

        m["min_separation"] = np.full(archive.n_vehicles, float(min_separation(pos).min()))
    if "link_sinr_db" in archive:
        from .wireless.metrics import link_metrics

        m.update(link_metrics(archive))
    world = archive.world
    if world is not None and world.has_obstacles:
        clr = np.stack([world.distance_to_nearest(pos[:, i]) for i in range(archive.n_vehicles)], 1)
        m["min_clearance"] = clr.min(0)
    return m


def aggregate(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Mean over vehicles of each metric."""
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def step_response_metrics(t: np.ndarray, y: np.ndarray, target: float, y0: float = 0.0, band: float = 0.02) -> dict:
    """Classical step metrics: rise time (10-90 %), overshoot, settling time (±``band``), steady-state error."""
    t, y = np.asarray(t), np.asarray(y)
    span = target - y0
    if abs(span) < 1e-12:
        raise ValueError("zero step")
    z = (y - y0) / span
    idx10 = np.argmax(z >= 0.1) if (z >= 0.1).any() else None
    idx90 = np.argmax(z >= 0.9) if (z >= 0.9).any() else None
    rise = float(t[idx90] - t[idx10]) if idx10 is not None and idx90 is not None else float("nan")
    overshoot = float(max(z.max() - 1.0, 0.0) * 100)
    outside = np.nonzero(np.abs(z - 1.0) > band)[0]
    settle = float(t[outside[-1] + 1] - t[0]) if len(outside) and outside[-1] + 1 < len(t) else (0.0 if not len(outside) else float("nan"))
    return dict(rise_time=rise, overshoot_pct=overshoot, settling_time=settle, steady_state_error=float(abs(z[-1] - 1.0) * abs(span)))


def success_rate(flags) -> float:
    return float(np.mean(np.asarray(flags, dtype=float)))
