"""A local web server for the viewer: static files, the chunked frame API and the live WebSocket.

Standard library only (``http.server``). It binds to ``127.0.0.1`` by default and guards against the usual local-server risks:

* the ``Host`` header must name this server (blocks DNS rebinding), and so must the ``Origin`` of WebSocket upgrades (blocks cross-site
  hijacking);
* static files come from three fixed directories of the packaged web app, with a path-containment check and a type allow-list;
* every request parameter is parsed strictly and bounded, and nothing the client sends reaches a shell, a file path or ``eval``.

Endpoints::

    GET /                 the viewer (dev shell, modules served as files)
    GET /api/meta         run metadata, layer data and channel list (JSON)
    GET /api/frames       ?names=a,b&start=&stop=&max=   binary frame bundle (see protocol.py)
    GET /api/health       {"ok": true}
    GET /ws               live stream (WebSocket) when the server has a live hub
"""

from __future__ import annotations

import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qs, urlsplit

from . import ws as _ws
from .export import web_root
from .payload import FrameSet, server_meta

__all__ = ["ViewerServer", "serve"]

STATIC_DIRS = ("src", "css", "vendor")
CONTENT_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
}
MAX_FRAMES_CAP = 20000
MAX_NAMES = 200


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:  # browsers closing tabs mid-response are normal, not errors
        import sys

        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


class _Bad(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _float_param(q: Dict[str, list], key: str) -> Optional[float]:
    if key not in q:
        return None
    try:
        v = float(q[key][0])
    except ValueError as e:
        raise _Bad(HTTPStatus.BAD_REQUEST, f"'{key}' must be a number") from e
    if not math.isfinite(v):
        raise _Bad(HTTPStatus.BAD_REQUEST, f"'{key}' must be finite")
    return v


class _Handler(BaseHTTPRequestHandler):
    server_version = "AevionViewer"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # -------------------------------------------------------------- plumbing
    @property
    def app(self) -> "ViewerServer":
        return self.server.app  # type: ignore[attr-defined, no-any-return]

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.app.verbose:
            super().log_message(fmt, *args)

    def _host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        return name.lower() in self.app.allowed_hosts

    def _send(self, status: HTTPStatus, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", self.app.csp)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send(status, json.dumps({"error": message}).encode("utf-8"), "application/json; charset=utf-8")

    # ---------------------------------------------------------------- routing
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        try:
            if not self._host_ok():
                raise _Bad(HTTPStatus.FORBIDDEN, "unexpected Host header")
            url = urlsplit(self.path)
            path = url.path
            if path in ("/", "/index.html"):
                self._index()
            elif path == "/api/health":
                self._send(HTTPStatus.OK, b'{"ok":true}', "application/json; charset=utf-8")
            elif path == "/api/meta":
                self._meta()
            elif path == "/api/frames":
                self._frames(parse_qs(url.query, max_num_fields=16))
            elif path == "/ws":
                self._websocket()
            elif path == "/favicon.ico":
                self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            else:
                self._static(path)
        except _Bad as e:
            self._error(e.status, e.message)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # never leak a stack trace to the client
            self.app.last_error = e
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

    def do_POST(self) -> None:
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "read-only server")

    do_PUT = do_DELETE = do_PATCH = do_POST

    # --------------------------------------------------------------- handlers
    def _index(self) -> None:
        page = (web_root() / "index.html").read_text(encoding="utf-8")
        boot = json.dumps(self.app.boot_options()).replace("<", "\\u003c")
        self._send(HTTPStatus.OK, page.replace("/*AEVION_BOOT*/{}", boot).encode("utf-8"), CONTENT_TYPES[".html"])

    def _meta(self) -> None:
        if self.app.fs is None:
            raise _Bad(HTTPStatus.NOT_FOUND, "this server streams a live run; there is no archive")
        self._send(HTTPStatus.OK, self.app.meta_bytes(), "application/json; charset=utf-8")

    def _frames(self, q: Dict[str, list]) -> None:
        fs = self.app.fs
        if fs is None:
            raise _Bad(HTTPStatus.NOT_FOUND, "this server streams a live run; there is no archive")
        names = [n for n in (q.get("names", [""])[0]).split(",") if n]
        if len(names) > MAX_NAMES:
            raise _Bad(HTTPStatus.BAD_REQUEST, "too many channels requested")
        unknown = [n for n in names if n not in set(fs.names)]
        if unknown:
            raise _Bad(HTTPStatus.BAD_REQUEST, f"unknown channel '{unknown[0][:60]}'")
        mx = _float_param(q, "max")
        maxf = int(min(max(mx if mx is not None else 4000, 2), MAX_FRAMES_CAP))
        body = fs.bundle(names or fs.names, _float_param(q, "start"), _float_param(q, "stop"), maxf)
        self._send(HTTPStatus.OK, body, "application/octet-stream")

    def _static(self, path: str) -> None:
        rel = path.lstrip("/")
        parts = Path(rel).parts
        if not parts or parts[0] not in STATIC_DIRS or ".." in parts or "\\" in rel or "\x00" in rel:
            raise _Bad(HTTPStatus.NOT_FOUND, "not found")
        root = web_root().resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            raise _Bad(HTTPStatus.NOT_FOUND, "not found")
        ctype = CONTENT_TYPES.get(target.suffix.lower())
        if ctype is None:
            raise _Bad(HTTPStatus.NOT_FOUND, "not found")
        self._send(HTTPStatus.OK, target.read_bytes(), ctype)

    def _websocket(self) -> None:
        hub = self.app.hub
        if hub is None:
            raise _Bad(HTTPStatus.NOT_FOUND, "no live stream on this server")
        h = self.headers
        origin = h.get("Origin")
        if origin and urlsplit(origin).hostname not in {x.strip("[]") for x in self.app.allowed_hosts}:
            raise _Bad(HTTPStatus.FORBIDDEN, "origin not allowed")
        key = h.get("Sec-WebSocket-Key")
        if h.get("Upgrade", "").lower() != "websocket" or not key or h.get("Sec-WebSocket-Version") != "13":
            raise _Bad(HTTPStatus.BAD_REQUEST, "expected a WebSocket upgrade")
        self.wfile.write(_ws.handshake_response(key))
        self.wfile.flush()
        self.close_connection = True
        sock = _ws.WebSocket(self.connection, client=False, max_size=1 << 16)
        self.connection.settimeout(self.app.ws_timeout)
        try:
            hub.serve(sock)
        finally:
            sock.close()


class ViewerServer:
    """Serve an archive (``archive=...``) and/or a live stream (``hub=...``) to browsers.

    >>> srv = ViewerServer(archive).start()        # doctest: +SKIP
    >>> srv.url                                    # doctest: +SKIP
    'http://127.0.0.1:53124/'
    """

    def __init__(
        self,
        archive: Any = None,
        *,
        hub: Any = None,
        runs: Optional[Sequence[Tuple[str, Any]]] = None,
        title: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        embed: bool = False,
        theme: Optional[str] = None,
        layer_options: Optional[dict] = None,
        extras: Any = "auto",
        verbose: bool = False,
        ws_timeout: float = 30.0,
        options: Optional[dict] = None,
    ):
        if archive is None and hub is None:
            raise ValueError("a viewer server needs an archive, a live hub, or both")
        self.archive, self.hub, self.runs, self.title = archive, hub, runs, title
        self.host, self.embed, self.theme, self.verbose, self.ws_timeout = host, embed, theme, verbose, ws_timeout
        self.options = dict(options or {})  # extra boot options for the page (manual, maxFps, autoplay)
        self.layer_options = dict(layer_options or {})
        self.fs = FrameSet(archive, extras=extras) if archive is not None else None
        self._meta: Optional[bytes] = None
        self._meta_lock = threading.Lock()
        self.last_error: Optional[BaseException] = None
        self.allowed_hosts: Set[str] = {"127.0.0.1", "localhost", "[::1]", host.lower()}
        self.httpd = _Server((host, port), _Handler)
        self.httpd.app = self  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        ancestors = "" if embed else "frame-ancestors 'none'; "
        self.csp = (
            f"default-src 'self'; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            f"connect-src 'self' ws://{host}:{self.port} ws://localhost:{self.port} ws://127.0.0.1:{self.port} data: blob:; worker-src blob:; "
            f"object-src 'none'; base-uri 'none'; {ancestors}".strip()
        )
        self._thread: Optional[threading.Thread] = None
        self._serving = False

    # ------------------------------------------------------------------ data
    def boot_options(self) -> dict:
        opts: Dict[str, Any] = {"mode": "server" if self.fs is not None else "live"}
        if self.theme:
            opts["theme"] = self.theme
        return {**opts, **self.options}

    def meta_bytes(self) -> bytes:
        with self._meta_lock:
            if self._meta is None:
                assert self.fs is not None
                self._meta = json.dumps(
                    server_meta(self.archive, self.fs, title=self.title, runs=self.runs, **self.layer_options), separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            return self._meta

    # -------------------------------------------------------------- lifecycle
    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "localhost") else self.host
        scheme = "http"  # local, cleartext by design: the server binds to loopback unless the user asks otherwise
        return f"{scheme}://{host}:{self.port}/"

    def start(self) -> "ViewerServer":
        self._serving = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="aevion-viewer", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._serving = True
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self._serving:
            self._serving = False
            self.httpd.shutdown()
        self.httpd.server_close()
        if self.hub is not None:
            self.hub.close()

    def __enter__(self) -> "ViewerServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def serve(archive: Any, *, port: int = 0, host: str = "127.0.0.1", open_browser: bool = False, block: bool = True, **kw: Any) -> ViewerServer:
    """Serve ``archive`` at ``http://localhost:<port>/`` (``block=False`` returns the running server)."""
    srv = ViewerServer(archive, host=host, port=port, **kw)
    if open_browser:
        import webbrowser

        webbrowser.open(srv.url)
    if block:
        print(f"Aevion viewer: {srv.url}  (Ctrl+C to stop)")
        srv.serve_forever()
    else:
        srv.start()
    return srv
