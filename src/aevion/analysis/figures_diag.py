"""Diagnostic figures for control, estimation and classification studies: Bode, pole-zero, Allan deviation, NEES consistency, ROC and ridgeline plots.

They share the research-grade style of the other figures (journal column widths, boxed axes, inward ticks, colour-blind-safe palette) and take a ``venue``
(``ieee``, ``acm``, ``elsevier``, ``springer``, ``plain``) that sets the width and typeface. The numbers behind them are in :mod:`aevion.analysis.diagnostics`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from . import diagnostics as D
from . import style as S
from .figures import _with_style, legend_top

__all__ = ["fig_bode", "fig_pzmap", "fig_allan", "fig_nees", "fig_roc", "fig_ridgeline", "fig_step_response"]

PAL = S.OKABE_ITO[1:] + [S.OKABE_ITO[0]]


@_with_style
def fig_bode(
    systems: Any,
    w: Optional[np.ndarray] = None,
    columns: str = "single",
    venue: str = "ieee",
    show_margins: bool = True,
    labels: Optional[Sequence[str]] = None,
):
    """Bode magnitude [dB] and phase [deg]; ``systems`` is one system or a dict ``{name: system}`` (transfer function ``(num, den)``, ``(A, B, C, D)`` or scipy LTI).

    With ``show_margins`` the gain crossover, phase margin and gain margin of the first system are marked and written in the legend.
    """
    items = systems if isinstance(systems, dict) else {(labels[0] if labels else "L(s)"): systems}
    if w is None:
        w = np.logspace(-2, 3, 800)
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=S.figsize(columns, 1.15, venue))
    for i, (name, sys_) in enumerate(items.items()):
        H = D.freqresp(sys_, w)
        a.semilogx(w, 20 * np.log10(np.maximum(np.abs(H), 1e-30)), color=PAL[i % len(PAL)], label=name if len(items) > 1 else None)
        b.semilogx(w, np.degrees(np.unwrap(np.angle(H))), color=PAL[i % len(PAL)])
    if show_margins:
        m = D.margins(next(iter(items.values())), w[0], w[-1])
        if np.isfinite(m["w_gc"]):
            a.axvline(m["w_gc"], color="0.5", lw=0.6, ls=":")
            b.axvline(m["w_gc"], color="0.5", lw=0.6, ls=":")
            b.plot([m["w_gc"]], [m["phase_margin_deg"] - 180.0], "o", color="k", ms=3, label=f"PM = {m['phase_margin_deg']:.0f}°")
        if np.isfinite(m["w_pc"]):
            a.plot([m["w_pc"]], [-20 * np.log10(m["gain_margin"])], "s", color="k", ms=3, label=f"GM = {20 * np.log10(m['gain_margin']):.1f} dB")
    a.axhline(0.0, color="0.4", lw=0.6)
    b.axhline(-180.0, color="0.4", lw=0.6)
    a.set_ylabel("Magnitude (dB)")
    b.set_ylabel("Phase (deg)")
    b.set_xlabel("Frequency (rad/s)")
    if len(items) > 1 or show_margins:
        a.legend(loc="best")
        b.legend(loc="best")
    for ax in (a, b):
        S.finish_axes(ax)
    return fig


@_with_style
def fig_pzmap(systems: Any, discrete: bool = False, columns: str = "single", venue: str = "ieee"):
    """Poles (×) and zeros (○) in the complex plane with the stability boundary: the imaginary axis (continuous time) or the unit circle (``discrete=True``).

    Unstable poles are drawn filled and in the accent colour; ``systems`` is one system or a dict of systems.
    """
    items = systems if isinstance(systems, dict) else {"": systems}
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.95, venue))
    parts: list = []
    for i, (name, sys_) in enumerate(items.items()):
        p, z = D.poles_zeros(sys_)
        col = PAL[i % len(PAL)]
        unstable = (np.abs(p) > 1 + 1e-9) if discrete else (p.real > 1e-9)
        ax.plot(p.real[~unstable], p.imag[~unstable], "x", color=col, ms=5, mew=1.1, label=(name or "poles") if len(items) > 1 or name else "poles")
        if unstable.any():
            ax.plot(p.real[unstable], p.imag[unstable], "X", color=S.OKABE_ITO[1], ms=5, mfc=S.OKABE_ITO[1], label="unstable pole")
        if len(z):
            ax.plot(z.real, z.imag, "o", mfc="none", color=col, ms=5, mew=0.9, label="zeros" if i == 0 else None)
        parts += [p, z]
    allv = np.concatenate([np.atleast_1d(v) for v in parts]) if parts else np.zeros(1)
    r = max(1.2 * float(np.max(np.abs(allv))) if allv.size else 1.0, 1.0)
    if discrete:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="0.4", lw=0.7)
        r = max(r, 1.2)
    else:
        ax.axvline(0.0, color="0.4", lw=0.7)
    ax.axhline(0.0, color="0.7", lw=0.5)
    if discrete:
        ax.set_xlim(-r, r)
    else:
        ax.set_xlim(min(float(allv.real.min()), 0.0) * 1.2 - 0.1 * r, max(float(allv.real.max()), 0.0) * 1.2 + 0.1 * r)
    ax.set_ylim(-r, r)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.legend(loc="best")
    S.finish_axes(ax)
    return fig


@_with_style
def fig_allan(
    taus: np.ndarray, adev: np.ndarray, fit: Optional[Dict[str, float]] = None, columns: str = "single", venue: str = "ieee", label: str = "measured"
):
    """Allan deviation on log-log axes with the slope guides of the noise types (−½ angle/velocity random walk, 0 bias instability, +½ rate random walk).

    ``fit`` may hold ``N`` (angle random walk: ``σ(τ) = N / √τ``), ``B`` (bias instability: ``0.664 B`` at the plateau) and ``K`` (``K √(τ/3)``), drawn as dashed asymptotes.
    """
    taus, adev = np.asarray(taus, float), np.asarray(adev, float)
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.85, venue))
    ax.loglog(taus, adev, "-", color=PAL[2], label=label)
    t = np.geomspace(taus.min(), taus.max(), 100)
    for key, curve, name in (
        ("N", lambda v: v / np.sqrt(t), "$N/\\sqrt{\\tau}$"),
        ("B", lambda v: np.full_like(t, 0.664 * v), "$0.664\\,B$"),
        ("K", lambda v: v * np.sqrt(t / 3.0), "$K\\sqrt{\\tau/3}$"),
    ):
        if fit and key in fit and fit[key] > 0:
            ax.loglog(t, curve(fit[key]), "--", lw=0.8, label=name)
    ax.set_xlabel(r"Averaging time $\tau$ (s)")
    ax.set_ylabel("Allan deviation")
    ax.legend(loc="best")
    ax.grid(True, which="major", alpha=0.5)
    return fig


@_with_style
def fig_nees(
    t: np.ndarray,
    nees: np.ndarray,
    dof: int,
    n_runs: int = 1,
    alpha: float = 0.05,
    columns: str = "single",
    venue: str = "ieee",
    ylabel: str = "NEES",
):
    """Normalised estimation error squared against time with the two-sided χ² acceptance band; the title-free legend reports the fraction of samples inside it.

    ``nees`` is ``(T,)`` (one run or already averaged over ``n_runs``) or ``(runs, T)`` (averaged here). A consistent filter stays inside the band about ``1 − α`` of the time.
    """
    from ..estimation.consistency import nees_interval

    nees = np.asarray(nees, float)
    if nees.ndim == 2:
        n_runs = nees.shape[0]
        nees = nees.mean(axis=0)
    lo, hi = nees_interval(dof, n_runs, alpha)
    inside = float(np.mean((nees >= lo) & (nees <= hi)))
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.7, venue))
    ax.axhspan(lo, hi, color=S.OKABE_ITO[3], alpha=0.18, lw=0, label=f"{100 * (1 - alpha):.0f} % χ² band")
    ax.axhline(dof, color="0.3", lw=0.7, ls="--", label=f"dof = {dof}")
    ax.plot(t, nees, color=PAL[2], label=f"{ylabel} ({100 * inside:.0f} % inside)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(hi * 2.0, float(nees.max()) * 1.05))
    ax.legend(loc="upper right")
    S.finish_axes(ax)
    return fig


@_with_style
def fig_roc(curves: Dict[str, Tuple[Any, Any]], columns: str = "single", venue: str = "ieee", ci: bool = False):
    """ROC curves for ``{name: (y_true, score)}`` with the chance diagonal; the legend carries the AUC (with a bootstrap interval when ``ci``)."""
    fig, ax = plt.subplots(figsize=S.figsize(columns, 1.0, venue))
    for i, (name, (y, sc)) in enumerate(curves.items()):
        fpr, tpr, _ = D.roc_curve(y, sc)
        if ci:
            a, lo, hi = D.auc_ci(y, sc)
            lab = f"{name} (AUC {a:.3f} [{lo:.3f}, {hi:.3f}])"
        else:
            lab = f"{name} (AUC {D.auc(y, sc):.3f})"
        ax.plot(fpr, tpr, color=PAL[i % len(PAL)], drawstyle="steps-post", label=lab)
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.7, ls="--", label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(loc="lower right")
    S.finish_axes(ax)
    return fig


@_with_style
def fig_ridgeline(samples: Dict[str, Any], xlabel: str = "", columns: str = "single", venue: str = "ieee", overlap: float = 0.55, bw: Any = "scott"):
    """Stacked kernel-density curves (one ridge per group, drawn bottom to top) with each group's median marked; groups share one x axis."""
    names = list(samples)
    if not names:
        raise ValueError("no samples to draw")
    lo = min(float(np.min(v)) for v in samples.values())
    hi = max(float(np.max(v)) for v in samples.values())
    pad = 0.1 * (hi - lo or 1.0)
    grid = np.linspace(lo - pad, hi + pad, 400)
    dens = {k: D.kde_curve(v, grid, bw)[1] for k, v in samples.items()}
    peak = max(d.max() for d in dens.values())
    1.0 / (1.0 + overlap * (len(names) - 1)) if len(names) > 1 else 1.0
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.35 + 0.18 * len(names), venue))
    for i, k in enumerate(reversed(names)):  # first group on top
        base = i * (1.0 - overlap)
        y = dens[k] / peak
        col = PAL[(len(names) - 1 - i) % len(PAL)]
        ax.fill_between(grid, base, base + y, color=col, alpha=0.75, lw=0, zorder=len(names) - i)
        ax.plot(grid, base + y, color="k", lw=0.6, ls="-", zorder=len(names) - i + 0.5)
        med = float(np.median(samples[k]))
        ax.plot([med, med], [base, base + float(np.interp(med, grid, y))], color="k", lw=0.8, ls="-", zorder=len(names) - i + 0.6)
        ax.axhline(base, color="0.6", lw=0.4, zorder=0)
    ax.set_yticks([i * (1.0 - overlap) for i in range(len(names))])
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(top=False, right=False)
    return fig


@_with_style
def fig_step_response(t: np.ndarray, y: np.ndarray, target: float, y0: float = 0.0, band: float = 0.02, columns: str = "single", venue: str = "ieee"):
    """Step response with the ±``band`` settling band, and rise time, overshoot and settling time (from :func:`~aevion.metrics.step_response_metrics`) in the legend."""
    from ..metrics import step_response_metrics

    m = step_response_metrics(np.asarray(t, float), np.asarray(y, float), target, y0, band)
    fig, ax = plt.subplots(figsize=S.figsize(columns, 0.7, venue))
    ax.plot(t, y, color=PAL[2], label="response")
    ax.axhline(target, color="0.3", lw=0.7, ls="--", label="target")
    span = abs(target - y0)
    ax.axhspan(target - band * span, target + band * span, color="0.5", alpha=0.2, lw=0, label=f"±{100 * band:.0f} % band")
    ax.plot([], [], " ", label=f"rise {m['rise_time']:.2f} s, overshoot {m['overshoot_pct']:.1f} %, settling {m['settling_time']:.2f} s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    legend_top(ax, ncol=1)
    S.finish_axes(ax)
    return fig
