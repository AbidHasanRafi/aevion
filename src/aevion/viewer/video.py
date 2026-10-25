"""Render a replay to a video, an animated GIF, a PNG sequence or a single screenshot.

The frames are produced by the real viewer running in a headless Chromium-family browser, so a video looks exactly like the interactive
replay (same scene, layers and charts). Each frame is rendered at an explicit simulation time (``__aevion.renderFrame(t)``), so the output
is deterministic and independent of how fast the machine is.

* ``.gif``                  needs Pillow
* ``.mp4`` / ``.webm`` / ``.mkv``   need an ``ffmpeg`` executable on PATH (``AEVION_FFMPEG`` overrides)
* a path without suffix     writes a directory of ``frame_00001.png`` files
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

from .browser import Browser, BrowserError, Page
from .export import export_html

__all__ = ["render_video", "screenshot", "render_frames", "find_ffmpeg"]

MAX_FRAMES = 6000


def find_ffmpeg() -> Optional[str]:
    env = os.environ.get("AEVION_FFMPEG")
    if env:
        return env if Path(env).exists() or shutil.which(env) else None
    return shutil.which("ffmpeg")


@contextmanager
def _session(
    archive: Any,
    size: Tuple[int, int],
    camera: str,
    ui: bool,
    theme: str,
    layers: Optional[Dict[str, bool]],
    time_of_day: Optional[float],
    focus: int,
    follow: bool,
    scale: float,
    runs: Optional[Sequence[Tuple[str, Any]]],
    max_source_frames: int,
    payload_options: Dict[str, Any],
) -> Iterator[Page]:
    with tempfile.TemporaryDirectory(prefix="aevion-render-") as tmp:
        html = export_html(
            archive, Path(tmp) / "replay.html", max_frames=max_source_frames, runs=runs, theme=theme, options={"manual": True}, **payload_options
        )
        with Browser(width=size[0], height=size[1]) as b:
            page = b.new_page()
            page.resize(size[0], size[1], scale)
            page.goto(html.resolve().as_uri())
            page.wait_for("document.body.dataset.ready === '1' || document.body.dataset.error", timeout=60)
            err = page.evaluate("document.body.dataset.error || ''")
            if err:
                raise BrowserError(f"the viewer failed to start: {err}")
            js = [
                "const a = window.__aevion;",
                f"a.setCamera({camera!r});",
                f"a.setUI({str(bool(ui)).lower()});",
                f"a.fleet.setFocus({int(focus)});",
                "a.playback.setPlaying(false);",
            ]
            if follow:
                js.append("a.settings.set('followFocus', true);")
            if time_of_day is not None:
                js.append(f"a.settings.set('timeOfDay', {float(time_of_day)}); a.stage.setTimeOfDay({float(time_of_day)});")
            for lid, on in (layers or {}).items():
                js.append(f"{{ const L = a.layers.find((x) => x.id === {lid!r}); if (L && L.available) L.enabled = {str(bool(on)).lower()}; }}")
            page.evaluate("(() => {" + " ".join(js) + "})()", await_promise=False)
            page.evaluate("window.__aevion.renderFrame(0)")
            yield page


def render_frames(
    archive: Any,
    times: Sequence[float],
    *,
    size: Tuple[int, int] = (1280, 720),
    camera: str = "orbit",
    sweep_deg: float = 0.0,
    ui: bool = False,
    theme: str = "dark",
    layers: Optional[Dict[str, bool]] = None,
    time_of_day: Optional[float] = None,
    focus: int = 0,
    follow: bool = False,
    scale: float = 1.0,
    runs: Optional[Sequence[Tuple[str, Any]]] = None,
    max_source_frames: int = 4000,
    **payload_options: Any,
) -> Iterator[bytes]:
    """Yield one PNG (bytes) per requested simulation time (seconds from the start of the run)."""
    with _session(archive, size, camera, ui, theme, layers, time_of_day, focus, follow, scale, runs, max_source_frames, payload_options) as page:
        step = math.radians(sweep_deg) / max(len(times) - 1, 1) if sweep_deg else 0.0
        for i, t in enumerate(times):
            if step and i:
                page.evaluate(f"window.__aevion.orbitBy({step!r})", await_promise=False)
            page.evaluate(f"window.__aevion.renderFrame({float(t)!r})")
            yield page.screenshot()
        if page.errors:
            raise BrowserError("the viewer reported errors while rendering: " + "; ".join(page.errors[:3]))


def screenshot(
    archive: Any, path: Union[str, Path], *, t: Optional[float] = None, size: Tuple[int, int] = (1600, 900), ui: bool = True, **kw: Any
) -> Path:
    """Render one frame (at ``t`` seconds; the middle of the run by default) to a PNG file."""
    out = Path(path)
    dur = float(archive.t[-1] - archive.t[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    png = next(render_frames(archive, [dur / 2 if t is None else t], size=size, ui=ui, **kw))
    out.write_bytes(png)
    return out


def _encode_gif(frames: Iterator[bytes], path: Path, fps: float) -> None:
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("GIF export needs Pillow: pip install pillow") from e
    imgs = [Image.open(io.BytesIO(f)).convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000.0 / fps), loop=0, optimize=False)


def _encode_ffmpeg(frames: Iterator[bytes], path: Path, fps: float, crf: int) -> None:
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError("video export needs ffmpeg on PATH (or AEVION_FFMPEG); alternatively write a .gif or a PNG sequence")
    codec = (
        ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf + 10)]
        if path.suffix.lower() == ".webm"
        else ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    )
    cmd = [exe, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps:g}", "-i", "-", *codec, str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)  # nosec B603 (fixed argv, no shell)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for f in frames:
            proc.stdin.write(f)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:400]}")


def render_video(
    archive: Any,
    path: Union[str, Path],
    *,
    fps: float = 30.0,
    speed: float = 1.0,
    start: float = 0.0,
    stop: Optional[float] = None,
    size: Tuple[int, int] = (1280, 720),
    camera: str = "orbit",
    sweep_deg: float = 0.0,
    ui: bool = False,
    theme: str = "dark",
    crf: int = 20,
    progress: Union[bool, Callable[[int, int], None]] = False,
    **kw: Any,
) -> Path:
    """Render ``archive`` to ``path`` (format by suffix; see the module docstring).

    ``speed`` is simulated seconds per video second; ``sweep_deg`` turns the orbit camera by that angle over the clip; ``ui=True`` includes
    the charts, HUD and timeline. Further keyword arguments go to :func:`render_frames` (``layers``, ``focus``, ``time_of_day``, ``runs``…).
    """
    if fps <= 0 or speed <= 0:
        raise ValueError("fps and speed must be positive")
    dur = float(archive.t[-1] - archive.t[0])
    stop = dur if stop is None else min(stop, dur)
    if not 0 <= start < stop:
        raise ValueError(f"need 0 <= start < stop <= {dur:g} s")
    n = int((stop - start) * fps / speed) + 1
    if n > MAX_FRAMES:
        raise ValueError(f"{n} frames requested (limit {MAX_FRAMES}); raise speed or shorten the clip")
    times = [start + i * speed / fps for i in range(n)]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def counted(frames: Iterator[bytes]) -> Iterator[bytes]:
        for i, f in enumerate(frames):
            if progress is True:
                print(f"\r  frame {i + 1}/{n}", end="", flush=True)
            elif callable(progress):
                progress(i + 1, n)
            yield f
        if progress is True:
            print()

    frames = counted(render_frames(archive, times, size=size, camera=camera, sweep_deg=sweep_deg, ui=ui, theme=theme, **kw))
    suffix = out.suffix.lower()
    if suffix == ".gif":
        _encode_gif(frames, out, fps)
    elif suffix in (".mp4", ".webm", ".mkv", ".mov"):
        _encode_ffmpeg(frames, out, fps, crf)
    elif suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames, 1):
            (out / f"frame_{i:05d}.png").write_bytes(f)
    else:
        raise ValueError(f"unsupported output '{suffix}': use .gif, .mp4, .webm, .mkv or a directory name")
    return out
