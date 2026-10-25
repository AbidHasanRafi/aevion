"""Journal templates: figure sizes, typefaces and the LaTeX wrapper each publisher expects.

``venue_summary`` tells you the printed column widths and typeface class of a venue (the same table :func:`aevion.analysis.style.figsize` uses), and
``figure_latex`` writes the ``figure`` environment with the right placement, the physical width of the figure and, for ACM, the mandatory ``\\Description``
(alternative text). The widths come from the publishers' author guidelines as of writing (IEEE, ACM ``acmart`` sigconf, Elsevier, Springer): check them
against the template you submit to, since publishers revise them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import style as S

__all__ = ["available_venues", "venue_summary", "figure_latex", "latex_escape"]

_PLACEMENT = {"ieee": "t", "acm": "t", "elsevier": "htbp", "springer": "ht", "plain": "t"}


def available_venues() -> List[str]:
    return sorted(S.VENUES)


def venue_summary(venue: str) -> Dict[str, Any]:
    if venue not in S.VENUES:
        raise ValueError(f"unknown venue {venue!r}; choose from {available_venues()}")
    v = S.VENUES[venue]
    widths = {k: v[k] for k in ("single", "one_half", "double") if k in v}
    return dict(
        venue=venue,
        label=v["label"],
        widths_in=widths,
        widths_mm={k: round(w * 25.4, 1) for k, w in widths.items()},
        family=v["family"],
        font=S.resolve_font(venue),
        size_pt=v["size"],
    )


def latex_escape(text: str) -> str:
    """Escape the characters LaTeX treats specially in running text."""
    table = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(table.get(c, c) for c in text)


def figure_latex(
    path: str,
    caption: str,
    label: str,
    venue: str = "ieee",
    columns: str = "single",
    description: Optional[str] = None,
    placement: Optional[str] = None,
) -> str:
    """A ``figure`` (or ``figure*`` for the double-column width of two-column venues) that includes ``path`` at the venue's printed width.

    Captions are placed below the figure. ACM requires alternative text: ``description`` (defaulting to the caption) is written with ``\\Description``.
    """
    if columns not in S.WIDTHS.get(venue, {}):
        raise ValueError(f"venue {venue!r} has no {columns!r} width; it has {sorted(S.WIDTHS.get(venue, {}))}")
    if not label or any(c in label for c in " {}\\"):
        raise ValueError("a LaTeX label must be non-empty and contain no spaces, braces or backslashes")
    width = S.WIDTHS[venue][columns]
    wide = columns == "double" and venue in ("ieee", "acm", "springer", "plain")
    env = "figure*" if wide else "figure"
    pl = placement or (_PLACEMENT[venue] if not wide else "!t")
    lines = [f"\\begin{{{env}}}[{pl}]", "  \\centering", f"  \\includegraphics[width={width:.2f}in]{{{path}}}"]
    if venue == "acm":
        lines.append(f"  \\Description{{{latex_escape(description or caption)}}}")
    lines += [f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", f"\\end{{{env}}}"]
    return "\n".join(lines)
