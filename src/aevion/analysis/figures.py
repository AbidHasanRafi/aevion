"""Research-grade figures from a :class:`~aevion.datasets.RunArchive`.

Every function returns a Matplotlib ``Figure`` sized for a journal column (``columns="single"``)
or full page width (``"double"``). Call :func:`aevion.analysis.style.use_style` once (the
functions also apply it themselves through a context manager).
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as MCircle
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import style as S

C_ACT, C_REF, C_B, C_G, C_P, C_Y = "#D55E00", "#000000", "#0072B2", "#009E73", "#CC79A7", "#E69F00"
RPM = 60.0 / (2 * math.pi)


def _with_style(fn):
    def wrapper(*a, font_size: float = 8.0, venue: str = "ieee", **kw):
        with S.style(font_size, venue):
            return fn(*a, venue=venue, **kw)

    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


def _ref(archive, v):
    r = archive.get("ref_pos")
    return None if r is None else r[:, v]


def _speed_collection(x, y, speed, cmap="viridis", lw=1.6, vmin=None, vmax=None):
    pts = np.stack([x, y], -1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw, capstyle="round")  # type: ignore[arg-type]
    lc.set_array(0.5 * (speed[:-1] + speed[1:]))
    if vmin is not None:
        lc.set_clim(vmin, vmax)
    return lc


def legend_top(ax, ncol=None, **kw):
    """Legend above the axes (does not cover data), a common convention for multi-curve panels."""
    h, l = ax.get_legend_handles_labels()
    if not h:
        return None
    kw.setdefault("fontsize", 6.5)
    return ax.legend(
        h,
        l,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol or len(h),
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0.15,
        handletextpad=0.4,
        **kw,
    )


# ---------------------------------------------------------------------------- scene drawing
def _draw_world_2d(ax, world, color="#B8B8B8", ec="#555555"):
    if world is None:
        return
    for b in world.boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[3] - b[0], b[4] - b[1], fc=color, ec=ec, lw=0.5, zorder=1))
    for s in world.spheres:
        ax.add_patch(MCircle((s[0], s[1]), s[3], fc=color, ec=ec, lw=0.5, zorder=1))
    for c in world.cylinders:
        ax.add_patch(MCircle((c[0], c[1]), c[4], fc=color, ec=ec, lw=0.5, zorder=1))


def _box_edges(b):
    x0, y0, z0, x1, y1, z1 = b
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [[c[i], c[j]] for i, j in idx]


def _draw_world_3d(ax, world, lo, hi, max_objects=80):
    """Draw obstacle wireframes that intersect the region of interest ``lo..hi`` (clipped in height)."""
    if world is None:
        return
    segs = []
    zmax = hi[2]

    def near(xmin, ymin, xmax, ymax):
        return not (xmax < lo[0] - 1.0 or xmin > hi[0] + 1.0 or ymax < lo[1] - 1.0 or ymin > hi[1] + 1.0)

    n = 0
    for bx in world.boxes:
        if near(bx[0], bx[1], bx[3], bx[4]) and n < max_objects:
            bb = bx.copy()
            bb[5] = min(bb[5], zmax)
            segs += _box_edges(bb)
            n += 1
    th = np.linspace(0, 2 * np.pi, 20)
    for c in world.cylinders:
        if near(c[0] - c[4], c[1] - c[4], c[0] + c[4], c[1] + c[4]) and n < max_objects:
            z1 = min(c[3], zmax)
            for z in (c[2], z1):
                ring = np.stack([c[0] + c[4] * np.cos(th), c[1] + c[4] * np.sin(th), np.full_like(th, z)], -1)
                segs += [[ring[i], ring[i + 1]] for i in range(len(ring) - 1)]
            for k in (0, 5, 10, 15):
                x_, y_ = c[0] + c[4] * np.cos(th[k]), c[1] + c[4] * np.sin(th[k])
                segs.append([[x_, y_, c[2]], [x_, y_, z1]])
            n += 1
    for sph in world.spheres:
        if near(sph[0] - sph[3], sph[1] - sph[3], sph[0] + sph[3], sph[1] + sph[3]) and n < max_objects:
            ring = np.stack([sph[0] + sph[3] * np.cos(th), sph[1] + sph[3] * np.sin(th), np.full_like(th, sph[2])], -1)
            segs += [[ring[i], ring[i + 1]] for i in range(len(ring) - 1)]
            n += 1
    if segs:
        ax.add_collection3d(Line3DCollection(segs, colors="#A5A5A5", linewidths=0.4))


# ---------------------------------------------------------------------------------- trajectory
def _plot_traj_3d(ax, archive, vehicle, color_by_speed=True, show_world=True, cbar=True, fig=None):
    pos = archive["pos"][:, vehicle]
    speed = np.linalg.norm(archive["vel"][:, vehicle], axis=-1)
    ref = _ref(archive, vehicle)
    lo, hi = np.minimum(pos.min(0), pos[0]), np.maximum(pos.max(0), pos[0])
    if ref is not None:
        lo, hi = np.minimum(lo, ref.min(0)), np.maximum(hi, ref.max(0))
    pad = 0.12 * np.maximum(hi - lo, 0.8)
    lo, hi = lo - pad, hi + pad
    lo[2] = min(lo[2], 0.0)
    if show_world:
        _draw_world_3d(ax, archive.world, lo, hi)
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color=C_REF, ls=(0, (4, 2)), lw=0.9, label="Reference")
    pts = pos.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap="viridis", linewidth=1.6)
    lc.set_array(0.5 * (speed[:-1] + speed[1:]))
    ax.add_collection3d(lc)
    ax.scatter(*pos[0], marker="o", s=14, color=C_G, edgecolor="k", linewidth=0.5, zorder=5, label="Start")
    ax.scatter(*pos[-1], marker="s", s=14, color=C_ACT, edgecolor="k", linewidth=0.5, zorder=5, label="End")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    span = np.maximum(hi - lo, 1e-3)
    ax.set_box_aspect(tuple(span / span.max() * np.array([1.0, 1.0, 1.0])), zoom=1.0)
    ax.set_xlabel("$x$ (m)", labelpad=-6)
    ax.set_ylabel("$y$ (m)", labelpad=-6)
    ax.set_zlabel("$z$ (m)", labelpad=-8)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    ax.tick_params(pad=-3, labelsize=6)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor("#BBBBBB")
        a._axinfo["grid"].update(linewidth=0.3, color="#CCCCCC")
    ax.view_init(elev=24, azim=-128)
    if cbar and fig is not None:
        cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.0, shrink=0.7)
        cb.set_label("Speed (m/s)")
        cb.outline.set_linewidth(0.6)
    return lc


@_with_style
def fig_trajectory_3d(archive, vehicle=0, columns="single", venue="ieee", legend=True):
    fig = plt.figure(figsize=S.figsize(columns, 0.85, venue))
    ax = fig.add_subplot(projection="3d")
    _plot_traj_3d(ax, archive, vehicle, fig=fig)
    if legend:
        ax.legend(loc="upper left", fontsize=6.5, borderaxespad=0.2)
    return fig


def _plot_topdown(ax, archive, vehicle, cbar_fig=None, all_vehicles=False):
    _draw_world_2d(ax, archive.world)
    vs = range(archive.n_vehicles) if all_vehicles else [vehicle]
    lc = None
    for v in vs:
        pos = archive["pos"][:, v]
        sp = np.linalg.norm(archive["vel"][:, v], axis=-1)
        ref = _ref(archive, v)
        if ref is not None and v == vehicle:
            ax.plot(ref[:, 0], ref[:, 1], color=C_REF, ls=(0, (4, 2)), lw=0.9, label="Reference", zorder=3)
        lc = _speed_collection(pos[:, 0], pos[:, 1], sp)
        ax.add_collection(lc)
        ax.plot(*pos[0, :2], "o", ms=3.5, color=C_G, mec="k", mew=0.5, zorder=5, label="Start" if v == vehicle else None)
        ax.plot(*pos[-1, :2], "s", ms=3.5, color=C_ACT, mec="k", mew=0.5, zorder=5, label="End" if v == vehicle else None)
    allp = archive["pos"][:, list(vs)]
    lo, hi = allp.reshape(-1, 3).min(0)[:2], allp.reshape(-1, 3).max(0)[:2]
    ref = archive.get("ref_pos")
    if ref is not None:
        lo, hi = np.minimum(lo, ref.reshape(-1, 3).min(0)[:2]), np.maximum(hi, ref.reshape(-1, 3).max(0)[:2])
    pad = 0.1 * np.maximum(hi - lo, 1.0)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    S.finish_axes(ax)
    if cbar_fig is not None and lc is not None:
        cb = cbar_fig.colorbar(lc, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("Speed (m/s)")
        cb.outline.set_linewidth(0.6)
    return lc


@_with_style
def fig_topdown(archive, vehicle=0, columns="single", venue="ieee", all_vehicles=False):
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.9, venue))
    _plot_topdown(ax, archive, vehicle, fig, all_vehicles)
    ax.legend(loc="best", fontsize=6.5)
    return fig


# ------------------------------------------------------------------------------ time series
def _plot_tracking(axes, archive, vehicle):
    t = archive.t - archive.t[0]
    pos = archive["pos"][:, vehicle]
    ref = _ref(archive, vehicle)
    for i, (ax, lab) in enumerate(zip(axes, ("$x$", "$y$", "$z$"))):
        if ref is not None:
            ax.plot(t, ref[:, i], color=C_REF, ls=(0, (4, 2)), lw=0.9, label="Reference")
        ax.plot(t, pos[:, i], color=C_ACT, lw=1.1, ls="-", label="Actual")
        ax.set_ylabel(f"{lab} (m)")
        S.finish_axes(ax)
    axes[-1].set_xlabel("Time (s)")


@_with_style
def fig_tracking(archive, vehicle=0, columns="single", venue="ieee"):
    has_ref = archive.get("ref_pos") is not None
    n = 4 if has_ref else 3
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=S.figsize(columns, 1.25 if has_ref else 0.95, venue))
    _plot_tracking(axes[:3], archive, vehicle)
    legend_top(axes[0])
    if has_ref:
        t = archive.t - archive.t[0]
        e = archive["pos"][:, vehicle] - archive["ref_pos"][:, vehicle]
        axes[3].plot(t, np.linalg.norm(e, axis=-1) * 100, color=C_B, lw=1.0)
        axes[3].set_ylabel("$\\|e\\|$ (cm)")
        S.finish_axes(axes[3])
        for a in axes[:3]:
            a.set_xlabel("")
    axes[-1].set_xlabel("Time (s)")
    S.panel_labels(axes, fontsize=7)
    return fig


def _plot_attitude(ax, ax2, archive, vehicle):
    t = archive.t - archive.t[0]
    e = np.degrees(archive["euler"][:, vehicle])
    for i, (lab, c) in enumerate(zip(("Roll $\\phi$", "Pitch $\\theta$", "Yaw $\\psi$"), (C_ACT, C_B, C_G))):
        ax.plot(t, e[:, i], color=c, lw=1.0, ls=("-", (0, (5, 1.5)), (0, (3, 1, 1, 1)))[i], label=lab)
    ax.set_ylabel("Angle (deg)")
    legend_top(ax)
    w = np.degrees(archive["omega"][:, vehicle])
    for i, (lab, c) in enumerate(zip(("$p$", "$q$", "$r$"), (C_ACT, C_B, C_G))):
        ax2.plot(t, w[:, i], color=c, lw=1.0, ls=("-", (0, (5, 1.5)), (0, (3, 1, 1, 1)))[i], label=lab)
    ax2.set_ylabel("Rate (deg/s)")
    ax2.set_xlabel("Time (s)")
    legend_top(ax2)
    S.finish_axes(ax)
    S.finish_axes(ax2)


@_with_style
def fig_attitude(archive, vehicle=0, columns="single", venue="ieee"):
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 0.95, venue))
    _plot_attitude(a, b, archive, vehicle)
    a.set_xlabel("")
    S.panel_labels([a, b], fontsize=7)
    return fig


def _plot_rotors(ax, archive, vehicle, spec=None):
    t = archive.t - archive.t[0]
    r = archive["rotor"][:, vehicle] * RPM / 1000.0
    for m in range(r.shape[1]):
        ax.plot(t, r[:, m], lw=0.9, label=f"M{m + 1}")
    if spec is not None and r.max() > 0.85 * spec.w_max * RPM / 1000:
        ax.axhline(spec.w_max * RPM / 1000, color="k", lw=0.7, ls=":")
        ax.text(t[-1], spec.w_max * RPM / 1000, " $\\omega_{max}$", va="bottom", ha="right", fontsize=6)
    ax.set_ylabel("Rotor speed (krpm)")
    ax.set_xlabel("Time (s)")
    legend_top(ax, ncol=min(r.shape[1], 8))
    S.finish_axes(ax)


@_with_style
def fig_rotors(archive, vehicle=0, columns="single", venue="ieee"):
    t = archive.t - archive.t[0]
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 1.0, venue))
    _plot_rotors(a, archive, vehicle, archive.spec)
    a.set_xlabel("")
    b.plot(t, archive["thrust"][:, vehicle], color=C_B, lw=1.0, label="Thrust")
    spec = archive.spec
    if spec is not None:
        b.axhline(spec.mass * 9.80665, color="k", ls=":", lw=0.7)
        b.text(t[0], spec.mass * 9.80665, " $mg$", va="bottom", fontsize=6)
    b.set_ylabel("Thrust (N)")
    b.set_xlabel("Time (s)")
    S.finish_axes(b)
    S.panel_labels([a, b], fontsize=7)
    return fig


def _plot_energy(ax, ax2, archive, vehicle):
    t = archive.t - archive.t[0]
    p = archive["power"][:, vehicle]
    ax.plot(t, p, color=C_ACT, lw=1.0)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time (s)")
    S.finish_axes(ax)
    e_wh = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))]) / 3600.0
    ax2.plot(t, e_wh, color=C_B, lw=1.0, label="Energy")
    ax2.set_ylabel("Energy (Wh)")
    ax2.set_xlabel("Time (s)")
    twin = ax2.twinx()
    twin.plot(t, archive["soc"][:, vehicle] * 100, color=C_G, lw=1.0, ls=(0, (5, 1.5)), label="SoC")
    twin.set_ylabel("SoC (\\%)" if mpl.rcParams.get("text.usetex") else "SoC (%)")
    twin.tick_params(direction="in")
    S.finish_axes(ax2)
    lines = ax2.get_lines() + twin.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=6)


@_with_style
def fig_energy(archive, vehicle=0, columns="single", venue="ieee"):
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 1.0, venue))
    _plot_energy(a, b, archive, vehicle)
    a.set_xlabel("")
    S.panel_labels([a, b], fontsize=7)
    return fig


# --------------------------------------------------------------------------------- statistics
def _ecdf(x):
    x = np.sort(np.asarray(x).ravel())
    return x, np.arange(1, len(x) + 1) / len(x)


@_with_style
def fig_error_distribution(archive, vehicle=0, columns="single", venue="ieee"):
    """Tracking-error histogram with CDF and percentile markers."""
    ref = _ref(archive, vehicle)
    if ref is None:
        raise ValueError("archive has no reference trajectory")
    e = np.linalg.norm(archive["pos"][:, vehicle] - ref, axis=-1) * 100
    fig, (a, b) = plt.subplots(1, 2, figsize=S.figsize("double" if columns == "double" else "single", 0.45 if columns == "single" else 0.3, venue))
    a.hist(e, bins=30, color=C_B, edgecolor="k", linewidth=0.4, alpha=0.85, density=True)
    a.axvline(np.sqrt((e**2).mean()), color=C_ACT, lw=1.0, label=f"RMSE {np.sqrt((e**2).mean()):.2f} cm")
    a.set_xlabel("$\\|e\\|$ (cm)")
    a.set_ylabel("Density")
    a.legend(fontsize=6, loc="upper right")
    x, y = _ecdf(e)
    b.plot(x, y, color=C_ACT, lw=1.1)
    for q, ls in ((0.5, ":"), (0.95, "--")):
        v = np.quantile(e, q)
        b.axhline(q, color="k", lw=0.5, ls=ls)
        b.plot([v], [q], "o", ms=3, color="k")
        b.text(v, q - 0.07, f" P{int(q * 100)}={v:.2f}", fontsize=6, ha="left")
    b.set_xlabel("$\\|e\\|$ (cm)")
    b.set_ylabel("CDF")
    b.set_ylim(0, 1.02)
    S.finish_axes(a)
    S.finish_axes(b)
    S.panel_labels([a, b], fontsize=7)
    return fig


def _plot_psd(ax, archive, vehicle, channel):
    from scipy.signal import welch

    t = archive.t
    fs = 1.0 / np.median(np.diff(t))
    if channel == "error":
        x = archive["pos"][:, vehicle] - archive["ref_pos"][:, vehicle]
        unit = "m$^2$/Hz"
    elif channel == "acc":
        x = archive["acc"][:, vehicle]
        unit = "(m/s$^2$)$^2$/Hz"
    else:
        x = archive["omega"][:, vehicle]
        unit = "(rad/s)$^2$/Hz"
    nper = min(len(x), max(64, len(x) // 4))
    for i, (lab, c) in enumerate(zip(("$x$", "$y$", "$z$"), (C_ACT, C_B, C_G))):
        f, P = welch(x[:, i] - x[:, i].mean(), fs=fs, nperseg=nper)
        ax.semilogy(f[1:], P[1:], color=c, lw=1.0, ls=("-", (0, (5, 1.5)), (0, (3, 1, 1, 1)))[i], label=lab)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"PSD ({unit})")
    legend_top(ax)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())


@_with_style
def fig_psd(archive, vehicle=0, channel="error", columns="single", venue="ieee"):
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.7, venue))
    _plot_psd(ax, archive, vehicle, channel)
    return fig


@_with_style
def fig_phase_portrait(archive, vehicle=0, columns="single", venue="ieee"):
    """Position-velocity phase portraits for the three axes."""
    fig, axes = plt.subplots(1, 3, figsize=S.figsize("double", 0.32, venue))
    pos, vel = archive["pos"][:, vehicle], archive["vel"][:, vehicle]
    t = archive.t - archive.t[0]
    for i, (ax, lab) in enumerate(zip(axes, "xyz")):
        sc = ax.scatter(pos[:, i], vel[:, i], c=t, s=2, cmap="viridis")
        ax.set_xlabel(f"${lab}$ (m)")
        ax.set_ylabel(f"$\\dot{{{lab}}}$ (m/s)")
        S.finish_axes(ax)
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Time (s)")
    cb.outline.set_linewidth(0.6)
    S.panel_labels(axes, fontsize=7)
    return fig


@_with_style
def fig_wind(archive, vehicle=0, columns="single", venue="ieee"):
    t = archive.t - archive.t[0]
    w = archive["wind"][:, vehicle]
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 0.95, venue))
    for i, (lab, c) in enumerate(zip(("$w_x$", "$w_y$", "$w_z$"), (C_ACT, C_B, C_G))):
        a.plot(t, w[:, i], color=c, lw=0.9, ls=("-", (0, (5, 1.5)), (0, (3, 1, 1, 1)))[i], label=lab)
    a.set_ylabel("Wind (m/s)")
    legend_top(a)
    b.plot(t, np.linalg.norm(w, axis=-1), color=C_REF, lw=1.0)
    b.set_ylabel("$\\|w\\|$ (m/s)")
    b.set_xlabel("Time (s)")
    S.finish_axes(a)
    S.finish_axes(b)
    S.panel_labels([a, b], fontsize=7)
    return fig


# ------------------------------------------------------------------------------------ sensors
@_with_style
def fig_lidar_scan(archive, frame=-1, vehicle=0, columns="single", venue="ieee"):
    """Polar plot of one LiDAR frame (channel-averaged rings) against the vehicle pose."""
    if "sensors/lidar" not in archive:
        raise KeyError("archive has no recorded lidar stream (record_sensors=['lidar'])")
    scan = archive["sensors/lidar"][frame][vehicle]
    ch, na = scan.shape
    az = np.linspace(-np.pi, np.pi, na, endpoint=False)
    fig = plt.figure(figsize=S.figsize(columns, 1.0, venue))
    ax: Any = fig.add_subplot(projection="polar")
    cm = plt.get_cmap("viridis")
    for k in range(ch):
        r = scan[k]
        ok = np.isfinite(r)
        ax.scatter(az[ok], r[ok], s=1.5, color=cm(k / max(ch - 1, 1)), linewidths=0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.grid(True, lw=0.3)
    ax.tick_params(labelsize=6)
    ax.set_title("LiDAR range returns (m)", fontsize=7, pad=8)
    return fig


@_with_style
def fig_imu(archive, vehicle=0, columns="single", venue="ieee"):
    if "sensors/imu" not in archive:
        raise KeyError("archive has no recorded imu stream (record_sensors=['imu'])")
    t = archive["sensors/imu/t"]
    x = archive["sensors/imu"][:, vehicle]
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 1.0, venue))
    for i, (lab, c) in enumerate(zip(("$f_x$", "$f_y$", "$f_z$"), (C_ACT, C_B, C_G))):
        a.plot(t, x[:, i], color=c, lw=0.6, label=lab)
    a.set_ylabel("Accelerometer (m/s$^2$)")
    legend_top(a)
    for i, (lab, c) in enumerate(zip(("$\\omega_x$", "$\\omega_y$", "$\\omega_z$"), (C_ACT, C_B, C_G))):
        b.plot(t, x[:, 3 + i], color=c, lw=0.6, label=lab)
    b.set_ylabel("Gyroscope (rad/s)")
    b.set_xlabel("Time (s)")
    legend_top(b)
    S.finish_axes(a)
    S.finish_axes(b)
    S.panel_labels([a, b], fontsize=7)
    return fig


@_with_style
def fig_estimation(archive, vehicle=0, columns="single", venue="ieee"):
    """Estimator error against ground truth (position and attitude)."""
    if "est_pos" not in archive:
        raise KeyError("archive has no estimator output (Scenario(use_estimator=True))")
    from ..core import rotation as rot
    from ..core.backend import get_backend

    B = get_backend("numpy")
    t = archive.t - archive.t[0]
    ep = archive["est_pos"][:, vehicle] - archive["pos"][:, vehicle]
    att = np.degrees(rot.angle_between(B, archive["est_quat"][:, vehicle], archive["quat"][:, vehicle]))
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 0.95, venue))
    for i, (lab, c) in enumerate(zip(("$x$", "$y$", "$z$"), (C_ACT, C_B, C_G))):
        a.plot(t, ep[:, i], color=c, lw=0.9, label=lab)
    a.set_ylabel("Position error (m)")
    legend_top(a)
    b.plot(t, att, color=C_P, lw=1.0)
    b.set_ylabel("Attitude error (deg)")
    b.set_xlabel("Time (s)")
    S.finish_axes(a)
    S.finish_axes(b)
    S.panel_labels([a, b], fontsize=7)
    return fig


# ---------------------------------------------------------------------------------- overview
@_with_style
def fig_overview(archive, vehicle=0, venue="ieee", columns="double"):
    """Six-panel dashboard: 3-D path, top view, position error, attitude, rotor speeds, power/energy."""
    fig = plt.figure(figsize=S.figsize("double", 0.62, venue, height=4.4 if venue == "ieee" else 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.0, 1.0])
    a = fig.add_subplot(gs[0, 0], projection="3d")
    _plot_traj_3d(a, archive, vehicle, cbar=False, fig=fig)
    b = fig.add_subplot(gs[0, 1])
    _plot_topdown(b, archive, vehicle, fig)
    c = fig.add_subplot(gs[0, 2])
    t = archive.t - archive.t[0]
    ref = _ref(archive, vehicle)
    if ref is not None:
        e = archive["pos"][:, vehicle] - ref
        c.plot(t, e[:, 0] * 100, color=C_ACT, lw=0.9, label="$e_x$")
        c.plot(t, e[:, 1] * 100, color=C_B, lw=0.9, ls=(0, (5, 1.5)), label="$e_y$")
        c.plot(t, e[:, 2] * 100, color=C_G, lw=0.9, ls=(0, (3, 1, 1, 1)), label="$e_z$")
        c.set_ylabel("Position error (cm)")
        legend_top(c)
    else:
        for i, lab in enumerate("xyz"):
            c.plot(t, archive["pos"][:, vehicle, i], lw=0.9, label=f"${lab}$")
        c.set_ylabel("Position (m)")
        legend_top(c)
    c.set_xlabel("Time (s)")
    S.finish_axes(c)
    d = fig.add_subplot(gs[1, 0])
    e_ = np.degrees(archive["euler"][:, vehicle])
    for i, (lab, col) in enumerate(zip(("$\\phi$", "$\\theta$", "$\\psi$"), (C_ACT, C_B, C_G))):
        d.plot(t, e_[:, i], color=col, lw=0.9, ls=("-", (0, (5, 1.5)), (0, (3, 1, 1, 1)))[i], label=lab)
    d.set_ylabel("Attitude (deg)")
    d.set_xlabel("Time (s)")
    legend_top(d)
    S.finish_axes(d)
    f = fig.add_subplot(gs[1, 1])
    _plot_rotors(f, archive, vehicle, archive.spec)
    g = fig.add_subplot(gs[1, 2])
    p = archive["power"][:, vehicle]
    g.plot(t, p, color=C_ACT, lw=0.9)
    g.set_ylabel("Power (W)")
    g.set_xlabel("Time (s)")
    tw = g.twinx()
    tw.plot(t, archive["soc"][:, vehicle] * 100, color=C_B, lw=0.9, ls=(0, (5, 1.5)))
    tw.set_ylabel("SoC (%)")
    tw.tick_params(direction="in")
    S.finish_axes(g)
    S.panel_labels([a, b, c, d, f, g], fontsize=7)
    return fig


@_with_style
def fig_link(archive, vehicle=0, columns="single", venue="ieee"):
    """SINR and throughput along the flight, with LoS/NLoS shading and the outage threshold."""
    if "link_sinr_db" not in archive:
        raise KeyError("archive has no wireless channels (Scenario(wireless=...))")
    t = archive.t - archive.t[0]
    sinr = archive["link_sinr_db"][:, vehicle]
    los = archive["link_los"][:, vehicle] > 0.5
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 1.0, venue))
    a.plot(t, sinr, color=C_B, lw=1.0)
    thr = (archive.meta.get("wireless") or {}).get("outage_db")
    if thr is not None:
        a.axhline(thr, color="k", lw=0.7, ls=":")
        a.text(t[-1], thr, " outage", va="bottom", ha="right", fontsize=6)
    a.fill_between(t, a.get_ylim()[0], a.get_ylim()[1], where=~los, color="#999999", alpha=0.25, linewidth=0, label="NLoS")
    a.set_ylabel("SINR (dB)")
    legend_top(a)
    b.plot(t, archive["link_rate_mbps"][:, vehicle], color=C_ACT, lw=1.0)
    b.set_ylabel("Throughput (Mbit/s)")
    b.set_xlabel("Time (s)")
    S.finish_axes(a)
    S.finish_axes(b)
    S.panel_labels([a, b], fontsize=7)
    return fig


def fig_coverage_map(network, altitude, extent, resolution=5.0, world=None, columns="single", venue="ieee", path=None, vmin=None, vmax=None):
    """Median-SINR coverage map at a fixed altitude with base stations, obstacles and (optionally) a flight path."""
    with S.style(8.0, venue):
        xs, ys, sinr = network.coverage_map(altitude, extent, resolution)
        fig, ax = plt.subplots(figsize=S.figsize(columns, 0.9, venue))
        im = ax.pcolormesh(xs, ys, sinr, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
        _draw_world_2d(ax, world or network.world, color="#FFFFFF", ec="#222222")
        sites = [b.pos for b in network.bs] if hasattr(network, "bs") else network.cell_pos
        for pos in sites:
            ax.plot(pos[0], pos[1], marker="^", ms=6, color="white", mec="k", mew=0.8, zorder=5)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], color="white", lw=1.4, zorder=4)
            ax.plot(path[:, 0], path[:, 1], color="k", lw=0.6, zorder=4, ls="--")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(f"SINR at {altitude:g} m (dB)")
        cb.outline.set_linewidth(0.6)
        S.finish_axes(ax)
        return fig


@_with_style
def fig_link_stack(archive, vehicle=0, columns="single", venue="ieee"):
    """Protocol-stack view of the radio link: SINR with outage, goodput with MCS, handovers and failures, Doppler."""
    if "link_sinr_db" not in archive:
        raise KeyError("archive has no wireless channels (add a network to the configuration)")
    t = archive.t - archive.t[0]
    v = vehicle
    sinr = archive["link_sinr_db"][:, v]
    rate = archive["link_rate_mbps"][:, v]
    serving = archive["link_serving"][:, v]
    thr = (archive.meta.get("wireless") or {}).get("outage_db", -6.0)
    out = archive["link_outage"][:, v] > 0.5 if "link_outage" in archive else sinr < thr
    rows = 4 if "link_mcs" in archive else 3
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=S.figsize(columns, 1.55 if rows == 4 else 1.25, venue))
    a = axes[0]
    a.plot(t, sinr, color=C_B, lw=0.9)
    a.axhline(thr, color="k", lw=0.6, ls=":")
    _shade(a, t, out, "#C0C0C0")
    a.set_ylabel("SINR (dB)")
    b = axes[1]
    b.plot(t, rate, color=C_ACT, lw=0.9)
    b.set_ylabel("Rate (Mbit/s)")
    if "link_mcs" in archive:
        b2 = b.twinx()
        b2.step(t, archive["link_mcs"][:, v], where="post", color=C_G, lw=0.7)
        b2.set_ylabel("MCS")
        b2.tick_params(direction="in")
        c = axes[2]
        if "link_bler" in archive:
            c.plot(t, np.maximum(archive["link_bler"][:, v], 1e-4), color=C_P, lw=0.8)
            c.set_yscale("log")
            c.set_ylim(1e-4, 1.2)
            c.axhline(0.1, color="k", lw=0.5, ls=":")
        c.set_ylabel("BLER")
    d = axes[-1]
    d.step(t, serving, where="post", color=C_REF, lw=0.9)
    d.set_ylabel("Serving cell")
    d.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    if "link_n_handovers" in archive:
        ho = np.nonzero(np.diff(archive["link_n_handovers"][:, v]) > 0)[0] + 1
        for h in ho:
            d.axvline(t[h], color=C_ACT, lw=0.7, ls="--")
    if "link_n_rlf" in archive:
        rl = np.nonzero(np.diff(archive["link_n_rlf"][:, v]) > 0)[0] + 1
        d.plot(t[rl], serving[rl], "x", color="#D00000", ms=4, mew=1.0)
    d.set_xlabel("Time (s)")
    for ax in axes:
        S.finish_axes(ax)
    S.panel_labels(list(axes), fontsize=7)
    return fig


def _shade(ax, t, mask, color):
    """Light vertical bands where ``mask`` is true."""
    m = np.asarray(mask, bool)
    if not m.any():
        return
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    for s0, e0 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        ax.axvspan(t[s0], t[min(e0, len(t) - 1)], color=color, alpha=0.3, lw=0)


def fig_rem(rem, network=None, path=None, field="sinr_db", columns="single", venue="ieee", vmin=None, vmax=None, world=None):
    """Radio environment map (:class:`~aevion.wireless.maps.RadioEnvironmentMap`) with cell sites, cell borders and an optional flight path."""
    labels = dict(sinr_db="SINR (dB)", rsrp_dbm="RSRP (dBm)", rate_bps="Rate (Mbit/s)")
    with S.style(8.0, venue):
        fig, ax = plt.subplots(figsize=S.figsize(columns, 0.9, venue))
        data = getattr(rem, field) / (1e6 if field == "rate_bps" else 1.0)
        im = ax.pcolormesh(rem.xs, rem.ys, data, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
        ax.contour(rem.xs, rem.ys, rem.serving, levels=np.arange(rem.serving.max() + 1) + 0.5, colors="white", linewidths=0.5, linestyles="--")
        if world is not None:
            _draw_world_2d(ax, world, color="#FFFFFF", ec="#222222")
        if network is not None:
            ax.plot(network.cell_pos[:, 0], network.cell_pos[:, 1], "^", ms=6, color="white", mec="k", mew=0.8, zorder=5)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], color="white", lw=1.4, zorder=4)
            ax.plot(path[:, 0], path[:, 1], color="k", lw=0.6, zorder=4, ls="--")
        ax.set_xlim(rem.xs[0], rem.xs[-1])
        ax.set_ylim(rem.ys[0], rem.ys[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(f"{labels.get(field, field)} at {rem.altitude:g} m")
        cb.outline.set_linewidth(0.6)
        S.finish_axes(ax)
        return fig


def fig_outage_cdf(archive, columns="single", venue="ieee", vehicle=None):
    """Empirical CDF of the SINR of every vehicle with the outage threshold marked: the link-quality distribution at a glance."""
    if "link_sinr_db" not in archive:
        raise KeyError("archive has no wireless channels")
    thr = (archive.meta.get("wireless") or {}).get("outage_db", -6.0)
    with S.style(8.0, venue):
        fig, ax = plt.subplots(figsize=S.figsize(columns, 0.75, venue))
        s = archive["link_sinr_db"]
        vs = range(s.shape[1]) if vehicle is None else [vehicle]
        for v in vs:
            x = np.sort(s[:, v])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=0.9, label=f"UAV {v}" if len(vs) <= 6 else None)
        ax.axvline(thr, color="k", lw=0.6, ls=":")
        ax.set_xlabel("SINR (dB)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1)
        if len(vs) <= 6:
            legend_top(ax)
        S.finish_axes(ax)
        return fig


ALL_RUN_FIGURES = {
    "overview": fig_overview,
    "trajectory_3d": fig_trajectory_3d,
    "topdown": fig_topdown,
    "tracking": fig_tracking,
    "attitude": fig_attitude,
    "rotors": fig_rotors,
    "energy": fig_energy,
    "psd_error": fig_psd,
    "phase": fig_phase_portrait,
    "wind": fig_wind,
    "error_distribution": fig_error_distribution,
    "lidar_scan": fig_lidar_scan,
    "imu": fig_imu,
    "estimation": fig_estimation,
    "link": fig_link,
    "link_stack": fig_link_stack,
    "outage_cdf": fig_outage_cdf,
}
