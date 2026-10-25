"""Figures for method comparison: box/strip plots with significance, CI bars, CDFs, heatmaps,
learning curves and Pareto scatter. They take an :class:`~aevion.experiments.ExperimentResult`
(or plain dicts of arrays) and return Matplotlib figures in the research-grade style.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from . import stats as ST
from . import style as S
from .figures import _with_style, legend_top

PALETTE = S.OKABE_ITO[1:] + [S.OKABE_ITO[0]]
HATCHES = ["", "//", "\\\\", "xx", "..", "++", "--"]


def _colors(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _label(metric: str) -> str:
    table = {
        "rmse_pos": "RMSE (m)",
        "rmse_steady": "Steady-state RMSE (m)",
        "max_err": "Max error (m)",
        "energy_wh": "Energy (Wh)",
        "mean_power_w": "Mean power (W)",
        "max_tilt_deg": "Max tilt (deg)",
        "path_length": "Path length (m)",
        "wall_s": "Wall time (s)",
        "success_rate": "Success rate",
        "crash_rate": "Crash rate",
        "min_clearance": "Min. clearance (m)",
        "rms_jerk": "RMS jerk (m/s$^3$)",
    }
    return table.get(metric, metric.replace("_", " "))


def _sig_bar(ax, x1, x2, y, p, h):
    txt = ST.significance_stars(p)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="k", lw=0.6, ls="-")
    ax.text((x1 + x2) / 2, y + h * 0.9, txt, ha="center", va="bottom", fontsize=6)


@_with_style
def fig_box(
    samples: Dict[str, Sequence[float]],
    ylabel: str = "",
    columns: str = "single",
    venue: str = "ieee",
    baseline: Optional[str] = None,
    show_points: bool = True,
    log: bool = False,
    sig: bool = True,
):
    """Box plot (median, IQR, 1.5 IQR whiskers) with jittered data points and Holm-corrected significance bars."""
    names = list(samples)
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.78, venue))
    data = [np.asarray(samples[n], float) for n in names]
    cols = _colors(len(names))
    bp = ax.boxplot(
        data,
        positions=range(len(names)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="k", lw=1.0),
        whiskerprops=dict(lw=0.7),
        capprops=dict(lw=0.7),
        boxprops=dict(lw=0.7),
    )
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    if show_points:
        for i, (d, c) in enumerate(zip(data, cols)):
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(d)), d, s=6, color=c, edgecolor="k", linewidth=0.3, zorder=3)
    for i, d in enumerate(data):
        ax.plot(i, d.mean(), marker="D", ms=3.5, color="white", mec="k", mew=0.6, zorder=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20 if max(len(n) for n in names) > 7 else 0, ha="right" if max(len(n) for n in names) > 7 else "center")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if sig and len(names) > 1:
        rows = ST.pairwise_table(samples, baseline)
        top = max(d.max() for d in data)
        lo = min(d.min() for d in data)
        span = (top - lo) if not log else top
        h = 0.05 * span
        for k, r in enumerate(rows):
            if r["t_p_holm"] < 0.05:
                i, j = names.index(r["a"]), names.index(r["b"])
                _sig_bar(ax, i, j, top + (0.06 + 0.12 * k) * span, r["t_p_holm"], h)
    S.finish_axes(ax)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    return fig


@_with_style
def fig_bars_ci(summary: Sequence[dict], ylabel: str = "", columns: str = "single", venue: str = "ieee", log: bool = False):
    """Grouped bars (mean ± 95 % CI). ``summary`` rows come from ``ExperimentResult.summary()``."""
    methods = list(dict.fromkeys(r["method"] for r in summary))
    conds = list(dict.fromkeys(r["condition"] for r in summary))
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.75, venue))
    w = 0.8 / len(methods)
    cols = _colors(len(methods))
    for i, m in enumerate(methods):
        xs, ys, lo, hi = [], [], [], []
        for j, c in enumerate(conds):
            r = next((r for r in summary if r["method"] == m and r["condition"] == c), None)
            if r:
                xs.append(j + (i - (len(methods) - 1) / 2) * w)
                ys.append(r["mean"])
                lo.append(r["mean"] - r["ci_lo"])
                hi.append(r["ci_hi"] - r["mean"])
        ax.bar(xs, ys, w * 0.92, color=cols[i], edgecolor="k", linewidth=0.5, hatch=HATCHES[i % len(HATCHES)], label=m, alpha=0.9)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="k", elinewidth=0.7, capsize=1.8, capthick=0.7)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    legend_top(ax, ncol=len(methods))
    S.finish_axes(ax)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    return fig


@_with_style
def fig_cdf(samples: Dict[str, Sequence[float]], xlabel: str = "", columns: str = "single", venue: str = "ieee", log_x: bool = False):
    """Empirical CDFs; the curve that is further left is better for error metrics."""
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.72, venue))
    for (name, d), c in zip(samples.items(), _colors(len(samples))):
        x = np.sort(np.asarray(d, float))
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, label=name, lw=1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    if log_x:
        ax.set_xscale("log")
    for q in (0.5, 0.95):
        ax.axhline(q, color="#888", lw=0.4, ls=":")
    legend_top(ax)
    S.finish_axes(ax)
    return fig


@_with_style
def fig_heatmap(
    matrix: np.ndarray,
    xlabels: Sequence,
    ylabels: Sequence,
    cbar_label: str = "",
    columns: str = "single",
    venue: str = "ieee",
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = "{:.2f}",
    xlabel: str = "",
    ylabel: str = "",
):
    """Annotated heatmap (e.g. error across wind × mass sweeps)."""
    M = np.asarray(matrix, float)
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.8, venue))
    im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels([f"{x:g}" if isinstance(x, (int, float)) else x for x in xlabels])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels([f"{y:g}" if isinstance(y, (int, float)) else y for y in ylabels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.6)
    if annotate and M.size <= 100:
        thr = 0.5 * (np.nanmin(M) + np.nanmax(M))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=5.5, color="white" if M[i, j] < thr else "black")
    return fig


@_with_style
def fig_learning_curves(log, ylabel: str = "Episode return", columns: str = "single", venue: str = "ieee", xlabel: str = "Environment steps"):
    """Mean ± 95 % CI learning curves from a :class:`~aevion.experiments.TrainingLog`."""
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.72, venue))
    for m, c in zip(log.runs, _colors(len(log.runs))):
        x, mean, lo, hi = log.aggregate(m)
        ax.plot(x, mean, color=c, label=f"{m} (n={len(log.runs[m])})", lw=1.2)
        ax.fill_between(x, lo, hi, color=c, alpha=S.BAND_ALPHA, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"))
    legend_top(ax)
    S.finish_axes(ax)
    return fig


@_with_style
def fig_pareto(
    points: Dict[str, tuple], xlabel: str, ylabel: str, columns: str = "single", venue: str = "ieee", errors: Optional[Dict[str, tuple]] = None
):
    """Trade-off scatter (e.g. tracking error vs energy) with the Pareto front highlighted."""
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.75, venue))
    names = list(points)
    P = np.array([points[n] for n in names], float)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, n in enumerate(names):
        ax.scatter(
            P[i][0], P[i][1], marker=markers[i % len(markers)], s=26, color=_colors(len(names))[i], edgecolor="k", linewidth=0.5, zorder=3, label=n
        )
        if errors and n in errors:
            ax.errorbar(P[i][0], P[i][1], xerr=errors[n][0], yerr=errors[n][1], fmt="none", ecolor="k", elinewidth=0.5, capsize=1.5, zorder=2)
    front = [i for i in range(len(P)) if not any((P[j] <= P[i]).all() and (P[j] < P[i]).any() for j in range(len(P)))]
    if len(front) > 1:
        F = P[front][np.argsort(P[front][:, 0])]
        ax.step(F[:, 0], F[:, 1], where="post", color="k", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend_top(ax)
    S.finish_axes(ax)
    return fig


@_with_style
def fig_line_ci(
    x: Sequence[float],
    series: Dict[str, np.ndarray],
    xlabel: str = "",
    ylabel: str = "",
    columns: str = "single",
    venue: str = "ieee",
    log_y: bool = False,
):
    """Metric vs a swept parameter, mean ± 95 % CI over seeds. ``series[name]`` has shape ``(len(x), n_seeds)``."""
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.72, venue))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (name, Y) in enumerate(series.items()):
        Y = np.asarray(Y, float)
        stats = np.array([ST.mean_ci(Y[k]) for k in range(len(x))])
        c = _colors(len(series))[i]
        ax.plot(x, stats[:, 0], color=c, marker=markers[i % 6], ms=3, label=name)
        ax.fill_between(x, stats[:, 1], stats[:, 2], color=c, alpha=S.BAND_ALPHA, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    legend_top(ax)
    S.finish_axes(ax)
    return fig


def experiment_figures(result, metric: str = "rmse_pos", condition: Optional[str] = None, baseline: Optional[str] = None) -> Dict[str, plt.Figure]:
    """Standard figure set for an :class:`ExperimentResult`: box, bars-with-CI and CDF."""
    conds = result.conditions
    cond = condition or conds[0]
    samples = {m: result.values(metric, m, cond) for m in result.methods}
    out = {
        f"box_{metric}": fig_box(samples, _label(metric), baseline=baseline),
        f"cdf_{metric}": fig_cdf(samples, _label(metric)),
    }
    if len(conds) > 0:
        out[f"bars_{metric}"] = fig_bars_ci(result.summary(metric), _label(metric))
    return out
