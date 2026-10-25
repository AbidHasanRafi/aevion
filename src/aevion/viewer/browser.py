"""Drive a local Chromium-family browser (Chrome, Edge, Chromium) through the DevTools Protocol.

Used for headless screenshots and video export, and by the viewer's own browser tests. Nothing is downloaded: the browser must already be
installed (``AEVION_BROWSER`` can point to a specific executable). The protocol connection is a localhost WebSocket opened by
:mod:`aevion.viewer.ws`.
"""

from __future__ import annotations

import atexit
import base64
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import ws as _ws

__all__ = ["find_browser", "Browser", "Page", "BrowserError"]


_LIVE: "set[Browser]" = set()


@atexit.register
def _close_leftover_browsers() -> None:  # a crashed or interrupted run must not leave headless browsers burning CPU
    for b in list(_LIVE):
        try:
            b.__exit__()
        except Exception:  # best effort at interpreter exit
            pass


class BrowserError(RuntimeError):
    """No usable browser, the browser failed to start, or a DevTools command failed."""


def find_browser() -> Optional[str]:
    """Path of a Chromium-family browser, or ``None``. ``AEVION_BROWSER`` wins when set."""
    env = os.environ.get("AEVION_BROWSER")
    if env:
        return env if Path(env).exists() or shutil.which(env) else None
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    roots = [os.environ.get(k) for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    candidates = [Path(r) / sub for r in roots if r for sub in ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")]
    candidates += [
        Path(p)
        for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


class Page:
    """One browser tab controlled over the DevTools Protocol (synchronous)."""

    def __init__(self, sock: _ws.WebSocket, timeout: float = 30.0):
        self._ws, self._id, self.timeout = sock, 0, timeout
        self.target_id = ""
        self._browser: Optional["Browser"] = None
        self.console: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self._events: List[Dict[str, Any]] = []
        for domain in ("Page", "Runtime", "Log"):
            self.send(f"{domain}.enable")

    def send(self, method: str, **params: Any) -> Dict[str, Any]:
        self._id += 1
        mid = self._id
        self._ws.send_text(json.dumps({"id": mid, "method": method, "params": params}))
        deadline = time.monotonic() + self.timeout
        while True:
            self._ws.sock.settimeout(max(deadline - time.monotonic(), 0.01))
            try:
                msg = json.loads(self._ws.recv_text())
            except TimeoutError as e:
                raise BrowserError(f"DevTools command {method} timed out after {self.timeout:.0f} s") from e
            if msg.get("id") == mid:
                if "error" in msg:
                    raise BrowserError(f"{method}: {msg['error'].get('message')}")
                return msg.get("result", {})
            self._on_event(msg)

    def _on_event(self, msg: Dict[str, Any]) -> None:
        m, p = msg.get("method"), msg.get("params", {})
        if m == "Runtime.consoleAPICalled":
            text = " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", []))
            self.console.append({"type": p.get("type"), "text": text})
            if p.get("type") == "error":
                self.errors.append(text)
        elif m == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails", {})
            self.errors.append((d.get("exception") or {}).get("description") or d.get("text", "exception"))
        elif m == "Log.entryAdded" and p.get("entry", {}).get("level") == "error":
            self.errors.append(p["entry"].get("text", ""))
        self._events.append(msg)

    def pump(self, seconds: float = 0.0) -> None:
        """Read events for a while (so console output and exceptions are collected)."""
        end = time.monotonic() + seconds
        while True:
            self._ws.sock.settimeout(max(end - time.monotonic(), 0.01))
            try:
                self._on_event(json.loads(self._ws.recv_text()))
            except TimeoutError:
                return

    def goto(self, url: str, wait_load: bool = True) -> None:
        self.send("Page.navigate", url=url)
        if wait_load:
            self.wait_for("document.readyState === 'complete'")

    def evaluate(self, expression: str, await_promise: bool = True) -> Any:
        r = self.send("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=await_promise, userGesture=True)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise BrowserError("page script failed: " + ((d.get("exception") or {}).get("description") or d.get("text", "")))
        return r.get("result", {}).get("value")

    def wait_for(self, condition: str, timeout: float = 30.0, interval: float = 0.1) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.evaluate(f"Boolean({condition})", await_promise=False):
                return
            self.pump(interval)
        raise BrowserError(f"timed out after {timeout:.0f} s waiting for: {condition}" + (f"; page errors: {self.errors[:3]}" if self.errors else ""))

    def resize(self, width: int, height: int, scale: float = 1.0) -> None:
        self.send("Emulation.setDeviceMetricsOverride", width=int(width), height=int(height), deviceScaleFactor=scale, mobile=False)

    def screenshot(self, path: Optional[Path] = None, fmt: str = "png") -> bytes:
        data = base64.b64decode(self.send("Page.captureScreenshot", format=fmt, fromSurface=True)["data"])
        if path is not None:
            Path(path).write_bytes(data)
        return data

    def close(self) -> None:
        """Close the tab."""
        self._ws.close()
        browser = self._browser
        if browser is not None and self.target_id:
            try:
                browser._http("GET", f"/json/close/{self.target_id}")
            except OSError:
                pass


class Browser:
    """A headless browser process with a throw-away profile. Use as a context manager."""

    def __init__(
        self, executable: Optional[str] = None, width: int = 1280, height: int = 720, extra_args: Optional[List[str]] = None, timeout: float = 30.0
    ):
        self.exe = executable or find_browser()
        if not self.exe:
            raise BrowserError("no Chrome, Edge or Chromium found; install one or set AEVION_BROWSER to its path")
        self.width, self.height, self.timeout = width, height, timeout
        self.extra = list(extra_args or [])
        self.proc: Optional[subprocess.Popen] = None
        self.profile = ""
        self.port = 0

    def __enter__(self) -> "Browser":
        self.profile = tempfile.mkdtemp(prefix="aevion-browser-")
        _LIVE.add(self)
        args = [
            self.exe,
            "--headless=new",
            "--remote-debugging-port=0",
            f"--user-data-dir={self.profile}",
            f"--window-size={self.width},{self.height}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--hide-scrollbars",
            "--mute-audio",
            "--ignore-gpu-blocklist",
            "--use-angle=swiftshader",
            "--disable-features=CalculateNativeWinOcclusion",
            "--enable-unsafe-swiftshader",
            *self.extra,
            "about:blank",
        ]
        self.proc = subprocess.Popen([str(a) for a in args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603 (fixed argv, no shell)
        port_file = Path(self.profile) / "DevToolsActivePort"
        end = time.monotonic() + self.timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                raise BrowserError(f"the browser exited during start-up (code {self.proc.returncode})")
            try:  # the browser may still hold the file open while writing it (a sharing violation on Windows)
                text = port_file.read_text() if port_file.exists() else ""
            except OSError:
                text = ""
            if text.strip():
                self.port = int(text.split()[0])
                return self
            time.sleep(0.05)
        self.__exit__()
        raise BrowserError("the browser did not open its DevTools port in time")

    def _http(self, method: str, path: str) -> Any:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            conn.request(method, path)
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, body
        finally:
            conn.close()

    def new_page(self, url: str = "about:blank") -> Page:
        """Open a fresh tab (so state, console output and errors of earlier pages cannot leak into this one)."""
        status, body = self._http("PUT", "/json/new?about:blank")
        if status != 200:  # older browsers only accept GET here
            status, body = self._http("GET", "/json/new?about:blank")
        if status != 200:
            raise BrowserError("the browser would not open a new tab")
        target = json.loads(body)
        path = "/" + target["webSocketDebuggerUrl"].split("/", 3)[3]
        page = Page(_ws.connect("127.0.0.1", self.port, path, timeout=self.timeout), self.timeout)
        page.target_id = target.get("id", "")
        page._browser = self
        page.resize(self.width, self.height)
        if url != "about:blank":
            page.goto(url)
        return page

    def __exit__(self, *exc: Any) -> None:
        _LIVE.discard(self)
        proc = self.proc
        if proc is not None and proc.poll() is None:
            if sys.platform == "win32":  # Edge/Chrome spawn a process tree; kill exactly this one
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)  # nosec B603 B607
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for _ in range(10):
            try:
                shutil.rmtree(self.profile, ignore_errors=False)
                break
            except OSError:
                time.sleep(0.2)
        else:
            shutil.rmtree(self.profile, ignore_errors=True)
