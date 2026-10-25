"""Show a run inside a Jupyter / VS Code notebook.

>>> from aevion.viewer import show        # doctest: +SKIP
>>> show(archive)                         # doctest: +SKIP

``mode="inline"`` embeds the single-file viewer in the notebook (works on remote kernels, offline and in exported notebooks; adds ~1.5 MB per
view). ``mode="server"`` starts a local viewer server and embeds an iframe to it (small notebooks; the browser must be able to reach the
kernel's localhost).
"""

from __future__ import annotations

import html as _html
from typing import Any, Optional

from .export import build_payload, render_html

__all__ = ["show", "ViewerCell"]


class ViewerCell:
    """Rich-display object: ``_repr_html_`` returns the iframe, so it renders when it is the last expression of a cell."""

    def __init__(self, html: str, server: Any = None):
        self.html, self.server = html, server

    def _repr_html_(self) -> str:
        return self.html

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None


def show(
    archive: Any,
    *,
    height: int = 620,
    mode: str = "inline",
    title: Optional[str] = None,
    theme: Optional[str] = None,
    max_frames: int = 1200,
    display: bool = True,
    **payload_options: Any,
) -> ViewerCell:
    """Embed the interactive viewer for ``archive``. Returns the :class:`ViewerCell` (also displayed when IPython is available)."""
    if mode not in ("inline", "server"):
        raise ValueError("mode must be 'inline' or 'server'")
    if mode == "inline":
        payload = build_payload(archive, max_frames=max_frames, title=title, **payload_options)
        page = render_html(payload, {"theme": theme} if theme else None)
        cell = ViewerCell(
            f'<iframe srcdoc="{_html.escape(page, quote=True)}" width="100%" height="{int(height)}" style="border:0;border-radius:8px" allowfullscreen></iframe>'
        )
    else:
        from .server import ViewerServer

        srv = ViewerServer(archive, embed=True, title=title, theme=theme).start()
        cell = ViewerCell(
            f'<iframe src="{srv.url}" width="100%" height="{int(height)}" style="border:0;border-radius:8px" allowfullscreen></iframe>', srv
        )
    if display:
        try:
            from IPython.display import display as _display  # type: ignore[import-not-found]

            _display(cell)
        except ImportError:
            pass
    return cell
