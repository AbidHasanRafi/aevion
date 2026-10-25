"""Export a :class:`RunArchive` as ONE self-contained, interactive 3-D HTML file.

The viewer is a set of ES modules (``web/src``) plus a vendored copy of three.js (``web/vendor``); nothing is downloaded, so the file works
offline and on air-gapped machines. For a single file the bundler

* rewrites every relative import of the app's modules to a bare ``@aevion/...`` specifier,
* stores the module sources (and the vendored three.js files) in one JSON block, and
* at load time turns each into a ``blob:`` URL, installs an import map pointing the specifiers at them, and imports the app.

The page runs on the viewer's own GPU, so the Python side needs no GPU, window system or 3-D library.
"""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from .payload import build_payload

__all__ = ["build_payload", "export_html", "web_root", "bundle_modules", "render_html", "THREE_VERSION"]

THREE_VERSION = "0.160.0"
VENDOR_SPECIFIERS = {
    "three": "three.module.min.js",
    "three/addons/controls/OrbitControls.js": "OrbitControls.js",
    "three/addons/loaders/GLTFLoader.js": "GLTFLoader.js",
    "three/addons/utils/BufferGeometryUtils.js": "BufferGeometryUtils.js",
}
_IMPORT = re.compile(r"""(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])(\.{1,2}/[^'"]+)\2""")
DEV_START, DEV_END = "<!--aevion:dev-->", "<!--/aevion:dev-->"


def web_root() -> Path:
    """The directory holding the viewer's web sources (``index.html``, ``src/``, ``css/``, ``vendor/``)."""
    return Path(__file__).parent / "web"


def _rewrite_imports(source: str, module: str) -> str:
    """Replace relative specifiers in ``source`` (the module at ``src/<module>``) with ``@aevion/<normalised path>``."""
    base = Path(module).parent

    def repl(m: "re.Match[str]") -> str:
        parts: list = []
        for seg in (base / m.group(3)).as_posix().split("/"):
            if seg == "..":
                if not parts:
                    raise ValueError(f"{module}: import {m.group(3)!r} escapes the source tree")
                parts.pop()
            elif seg not in ("", "."):
                parts.append(seg)
        return f"{m.group(1)}{m.group(2)}@aevion/{'/'.join(parts)}{m.group(2)}"

    return _IMPORT.sub(repl, source)


@lru_cache(maxsize=1)
def bundle_modules() -> Dict[str, str]:
    """``{specifier: source}`` for the app modules and the vendored three.js files."""
    root = web_root()
    mods: Dict[str, str] = {}
    src = root / "src"
    for p in sorted(src.rglob("*.js")):
        rel = p.relative_to(src).as_posix()
        mods[f"@aevion/{rel}"] = _rewrite_imports(p.read_text(encoding="utf-8"), rel)
    for spec, fname in VENDOR_SPECIFIERS.items():
        mods[spec] = (root / "vendor" / "three" / fname).read_text(encoding="utf-8")
    return mods


def _script_safe(text: str) -> str:
    """Make JSON safe inside a <script> element (no ``</script``, ``<!--`` or U+2028/9)."""
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


LOADER = r"""
(function () {
  var data = JSON.parse(document.getElementById('aevion-modules').textContent);
  var imports = {};
  Object.keys(data).forEach(function (k) { imports[k] = URL.createObjectURL(new Blob([data[k]], { type: 'text/javascript' })); });
  var map = document.createElement('script'); map.type = 'importmap'; map.textContent = JSON.stringify({ imports: imports });
  document.head.appendChild(map);
  window.__AEVION_BOOT__ = Object.assign({ mode: 'static', payload: JSON.parse(document.getElementById('aevion-payload').textContent) }, window.__AEVION_OPTIONS__ || {});
  window.__AEVION_NO_AUTOBOOT__ = true;
  import('@aevion/app.js').then(function (m) { return m.boot(window.__AEVION_BOOT__); }).catch(function (e) {
    console.error(e); var b = document.getElementById('banner'); if (b) { b.hidden = false; b.dataset.kind = 'error'; b.textContent = 'Viewer failed to start: ' + e.message; }
    document.body.dataset.error = String(e && e.message || e);
  });
})();
"""


def render_html(payload: dict, options: Optional[dict] = None) -> str:
    """The complete single-file page for ``payload`` (an ``aevion.view/2`` dict)."""
    root = web_root()
    page = (root / "index.html").read_text(encoding="utf-8")
    css = (root / "css" / "aevion.css").read_text(encoding="utf-8")
    start, end = page.index(DEV_START), page.index(DEV_END) + len(DEV_END)
    opts = f"<script>window.__AEVION_OPTIONS__={_script_safe(json.dumps(options or {}))};</script>\n" if options else ""
    block = (
        f"<style>\n{css}\n</style>\n"
        f'<script type="application/json" id="aevion-payload">{_script_safe(json.dumps(payload, separators=(",", ":"), allow_nan=False))}</script>\n'
        f'<script type="application/json" id="aevion-modules">{_script_safe(json.dumps(bundle_modules(), separators=(",", ":")))}</script>\n'
        f"{opts}<script>{LOADER}</script>"
    )
    title = str(payload["meta"].get("title", "Aevion run")).replace("&", "&amp;").replace("<", "&lt;")
    page = page[:start] + block + page[end:]
    return re.sub(r"<title>.*?</title>", lambda _m: f"<title>{title}</title>", page, count=1)


def _model_data_url(path: Union[str, Path]) -> str:
    p = Path(path)
    mime = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json"}.get(p.suffix.lower())
    if mime is None:
        raise ValueError("vehicle models must be .glb or .gltf files")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def export_html(
    archive: Any,
    path: Union[str, Path],
    offline: bool = True,
    title: Optional[str] = None,
    max_frames: int = 2400,
    *,
    runs: Optional[Sequence[Tuple[str, Any]]] = None,
    vehicle_model: Optional[Union[str, Path]] = None,
    theme: Optional[str] = None,
    options: Optional[dict] = None,
    **payload_options: Any,
) -> Path:
    """Write ``path`` (an ``.html`` file) containing the interactive replay.

    ``offline`` is accepted for backward compatibility: the page has always been self-contained since the viewer vendors three.js.
    ``options`` are extra boot options for the page (``manual``, ``maxFps``, ``autoplay``); ``runs`` adds comparison ghosts; ``vehicle_model`` embeds a glTF/GLB model; extra keyword arguments go to
    :func:`~aevion.viewer.payload.build_payload`.
    """
    payload = build_payload(archive, max_frames=max_frames, title=title, runs=runs, **payload_options)
    if vehicle_model is not None:
        veh = payload["meta"].get("vehicle") or {}
        veh["model"] = {"url": _model_data_url(vehicle_model)}
        payload["meta"]["vehicle"] = veh
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload, {**({"theme": theme} if theme else {}), **(options or {})} or None), encoding="utf-8")
    return out
