"""The in-browser scene editor: a viewer page with no run loaded.

The editor builds a world (buildings, trees, spheres), a start position, waypoints and base stations by clicking, and downloads an Aevion
configuration (``scene.json``) that ``aevion run scene.json`` executes. The data model and the configuration export are the pure JavaScript
module ``web/src/data/editor.js``; this module only provides the empty page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .export import render_html
from .payload import FORMAT, b64

__all__ = ["empty_payload", "write_editor", "open_editor"]

_CHANNELS = {"pos": 3, "quat": 4, "vel": 3, "euler": 3, "rotor": 4, "thrust": 0, "power": 0, "soc": 0, "wind": 3}


def empty_payload(title: str = "Scene editor", half_extent: float = 50.0, altitude: float = 2.0) -> Dict[str, Any]:
    """A two-frame run with one hovering vehicle: just enough for the viewer to open with an empty, editable world."""
    T, N, M = 2, 1, 4
    shapes: Dict[str, Any] = {}
    payload: Dict[str, Any] = {"format": FORMAT, "t": b64([0.0, 1.0])}
    for name, k in _CHANNELS.items():
        shape = [T, N, k] if k else [T, N]
        arr = np.zeros(shape, np.float32)
        if name == "quat":
            arr[..., 0] = 1.0
        if name == "pos":
            arr[..., 2] = altitude
        if name == "soc":
            arr[:] = 1.0
        if name == "rotor":
            shape = [T, N, M]
            arr = np.zeros(shape, np.float32)
        shapes[name] = shape
        payload[name] = b64(arr)
    payload["meta"] = dict(
        title=title,
        n=N,
        T=T,
        M=M,
        shapes=shapes,
        events=[],
        duration=1.0,
        t0=0.0,
        t1=1.0,
        controller="",
        trajectory="",
        editor=True,
        vehicle=dict(
            name="editor", mass=1.0, rotors=[[0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0], [0, -0.2, 0]], spin=[1, -1, 1, -1], radius=0.1, w_max=700.0
        ),
    )
    h = float(half_extent)
    payload["world"] = dict(boxes=[], spheres=[], cylinders=[], box_colors=[], sphere_colors=[], cyl_colors=[], bounds=[-h, -h, 0.0, h, h, 4 * h / 5])
    payload["extras"] = {}
    return payload


def write_editor(path: Union[str, Path] = "scene_editor.html", **kw: Any) -> Path:
    """Write the single-file editor page."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(empty_payload(**kw), {"autoplay": False}), encoding="utf-8")
    return out


def open_editor(path: Optional[Union[str, Path]] = None, **kw: Any) -> Path:
    """Write the editor page and open it in the default browser."""
    import webbrowser

    out = write_editor(path or "scene_editor.html", **kw)
    webbrowser.open(out.resolve().as_uri())
    return out
