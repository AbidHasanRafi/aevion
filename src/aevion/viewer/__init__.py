"""The Aevion viewer: interactive 3-D replay, live streaming, comparison, video export and a scene editor.

The browser app lives in ``web/`` (plain ES modules and a vendored three.js, no build step); this package prepares its data and
serves it:

====================  ==========================================================================================
``export_html``       one self-contained HTML file for a run (works offline)
``serve``             local server for long runs (an overview plus detail windows fetched on demand)
``serve_live``        run a simulation and watch it live in the browser
``compare_html``      replay one run with others as ghosts and a difference timeline
``render_video``      MP4/GIF/PNG-sequence through a headless browser; ``screenshot`` for one frame
``show``              embed the viewer in a Jupyter / VS Code notebook
``open_editor``       the in-browser scene editor that exports an Aevion configuration
====================  ==========================================================================================

Heavier submodules are imported on first use, so ``import aevion.viewer`` stays cheap.
"""

from __future__ import annotations

import importlib
from typing import Any

from .export import build_payload, export_html
from .payload import FORMAT

_LAZY = {
    "ViewerServer": "server",
    "serve": "server",
    "LiveHub": "live",
    "LiveStreamHook": "live",
    "serve_live": "live",
    "compare_html": "compare",
    "render_video": "video",
    "render_frames": "video",
    "screenshot": "video",
    "show": "jupyter",
    "open_editor": "editor",
    "write_editor": "editor",
    "find_browser": "browser",
    "Browser": "browser",
}

__all__ = ["FORMAT", "build_payload", "export_html", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'aevion.viewer' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))
