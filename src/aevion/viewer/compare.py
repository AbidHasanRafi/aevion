"""Compare runs side by side: the first run is the main vehicle, the others are translucent ghosts driven by the same clock.

The viewer shows each ghost's distance to the main run over time and the instant they diverge. Runs may differ in length, sampling and
number of vehicles (ghosts hold their last pose when they end).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from .export import export_html

__all__ = ["compare_html", "label_runs"]


def label_runs(archives: Sequence[Any], labels: Optional[Sequence[str]] = None) -> List[Tuple[str, Any]]:
    """Pair every archive with a unique, human-readable label (``labels`` or the scenario name, numbered when repeated)."""
    if labels is not None and len(labels) != len(archives):
        raise ValueError(f"{len(labels)} labels for {len(archives)} runs")
    seen: dict = {}
    out = []
    for i, a in enumerate(archives):
        base = labels[i] if labels else str(a.scenario.get("name") or f"run {i + 1}")
        seen[base] = seen.get(base, 0) + 1
        out.append((base if seen[base] == 1 else f"{base} ({seen[base]})", a))
    return out


def compare_html(archives: Sequence[Any], path: Union[str, Path], labels: Optional[Sequence[str]] = None, main: int = 0, **options: Any) -> Path:
    """Write one HTML file that replays ``archives[main]`` with every other run as a ghost."""
    if len(archives) < 2:
        raise ValueError("comparison needs at least two runs")
    if not 0 <= main < len(archives):
        raise IndexError(f"main={main} is not one of the {len(archives)} runs")
    labelled = label_runs(archives, labels)
    primary = labelled[main]
    ghosts = [r for i, r in enumerate(labelled) if i != main]
    options.setdefault("title", f"{primary[0]} vs {', '.join(g[0] for g in ghosts)}")
    return export_html(primary[1], path, runs=ghosts, **options)
