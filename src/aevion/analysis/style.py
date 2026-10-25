"""Research-grade figure style: journal column sizes, serif text, boxed axes.

``use_style()`` configures Matplotlib globally; ``style()`` is a context manager. Figures have all four spines,
inward major and minor ticks on every side, Times-family text and STIX math, and are sized to journal columns
so they can go into a paper unchanged.
"""

from __future__ import annotations

import contextlib
import string
from typing import Any, Dict, Optional

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator

# Okabe–Ito palette: distinguishable under all common colour-vision deficiencies
OKABE_ITO = ["#000000", "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A7A7A"]
# Slightly darker series for lines on white, light companions for bands/fills
BAND_ALPHA = 0.22

SERIF_PREFERENCE = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "TeX Gyre Termes",
    "Liberation Serif",
    "FreeSerif",
    "STIXGeneral",
    "DejaVu Serif",
]

SANS_PREFERENCE = ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "FreeSans", "DejaVu Sans"]

# Journal templates. Widths are the printed column widths in inches (IEEE 3.5 / 7.16, ACM sigconf 3.33 / 7.0, Elsevier 90 mm / 140 mm / 190 mm,
# Springer 84 mm / 174 mm); ``family`` is the typeface class the publisher's figure guidelines ask for and ``fonts`` are tried before the generic lists.
VENUES: Dict[str, Dict[str, Any]] = {
    "ieee": dict(single=3.5, double=7.16, family="serif", fonts=[], size=8.0, label="IEEE (Transactions / conference)"),
    "acm": dict(single=3.33, double=7.0, family="serif", fonts=["Linux Libertine O", "Libertinus Serif"], size=8.0, label="ACM (acmart sigconf)"),
    "elsevier": dict(single=3.54, one_half=5.51, double=7.48, family="sans-serif", fonts=[], size=8.0, label="Elsevier (1, 1.5 and 2 column widths)"),
    "springer": dict(single=3.31, double=6.85, family="serif", fonts=[], size=8.0, label="Springer (single and double column)"),
    "plain": dict(single=3.4, double=6.8, family="serif", fonts=[], size=8.0, label="generic two-column paper"),
}
# column widths in inches (kept for code that reads this table)
WIDTHS = {k: {c: v[c] for c in ("single", "one_half", "double") if c in v} for k, v in VENUES.items()}
GOLDEN = 0.618


def resolve_font(venue: str = "ieee") -> str:
    """First installed typeface of the venue's preference list (the generic serif / sans lists follow the venue's own choices)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    info = VENUES.get(venue, VENUES["ieee"])
    generic = SANS_PREFERENCE if info["family"] == "sans-serif" else SERIF_PREFERENCE
    for name in list(info["fonts"]) + generic:
        if name in available:
            return name
    return info["family"]


def figsize(columns: str = "single", aspect: float = GOLDEN, venue: str = "ieee", height: Optional[float] = None):
    if venue not in WIDTHS:
        raise ValueError(f"unknown venue {venue!r}; choose from {sorted(WIDTHS)}")
    if columns not in WIDTHS[venue]:
        raise ValueError(f"venue {venue!r} has no {columns!r} width; it has {sorted(WIDTHS[venue])}")
    w = WIDTHS[venue][columns]
    return (w, height if height is not None else w * aspect)


def rc(font_size: float = 8.0, venue: str = "ieee", dpi: int = 600) -> dict:
    from cycler import cycler

    font = resolve_font(venue)
    info = VENUES.get(venue, VENUES["ieee"])
    sans = info["family"] == "sans-serif"
    linestyles = ["-", (0, (5, 1.5)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1)), "-", (0, (5, 1.5)), "-"]
    return {
        "font.family": info["family"],
        "font.serif": [font] + SERIF_PREFERENCE if not sans else SERIF_PREFERENCE,
        "font.sans-serif": [font] + SANS_PREFERENCE if sans else SANS_PREFERENCE,
        "font.size": font_size,
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "axes.titleweight": "normal",
        "axes.titlepad": 4,
        "axes.labelpad": 2,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "axes.prop_cycle": cycler(color=OKABE_ITO) + cycler(linestyle=linestyles),
        "grid.color": "#B0B0B0",
        "grid.linestyle": (0, (1, 2)),
        "grid.linewidth": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.6,
        "legend.fontsize": font_size - 1,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.borderpad": 0.35,
        "legend.handlelength": 2.2,
        "legend.labelspacing": 0.3,
        "patch.linewidth": 0.6,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.h_pad": 0.03,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "image.cmap": "viridis",
    }


def use_style(font_size: float = 8.0, venue: str = "ieee", dpi: int = 600):
    """Apply the research-grade figure style globally."""
    mpl.rcParams.update(rc(font_size, venue, dpi))


@contextlib.contextmanager
def style(font_size: float = 8.0, venue: str = "ieee", dpi: int = 600):
    with mpl.rc_context(rc(font_size, venue, dpi)):
        yield


def finish_axes(ax, minor: bool = True, grid: bool = False):
    """Finishing touches: minor ticks on both axes, optional light grid."""
    if minor and ax.name != "3d":
        if ax.get_xscale() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if ax.get_yscale() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid:
        ax.grid(True, which="major")
    return ax


def panel_labels(axes, fmt: str = "({})", loc=(0.02, 0.97), fontsize: Optional[float] = None, bold: bool = True, box: bool = True):
    """Add (a), (b), … labels inside the top-left corner of each axes."""
    axes = list(axes) if hasattr(axes, "__iter__") else [axes]
    flat = []
    for a in axes:
        flat.extend(list(a.flat) if hasattr(a, "flat") else [a])
    for i, ax in enumerate(flat):
        put = ax.text2D if ax.name == "3d" else ax.text
        put(
            loc[0],
            loc[1],
            fmt.format(string.ascii_lowercase[i]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize or mpl.rcParams["font.size"],
            fontweight="bold" if bold else "normal",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85) if box else None,
            zorder=10,
        )


def save(fig, path, formats=("pdf", "png"), **kw):
    """Save a figure to ``path`` (extension ignored) in each format; returns the list of written files."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        q = p.with_suffix("." + f)
        fig.savefig(q, format=f, **kw)
        out.append(q)
    return out


def describe() -> dict:
    return dict(font=resolve_font(), palette=OKABE_ITO)
