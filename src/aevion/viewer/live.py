"""Stream a running simulation to the browser.

:class:`LiveHub` fans messages out to WebSocket clients and keeps them, so a browser that connects late (or reloads) is caught up first.
:class:`LiveStreamHook` is an engine hook that turns each recorded step into a frame bundle. :func:`serve_live` wires both to a
:class:`~aevion.viewer.server.ViewerServer` and runs a simulation.

Message protocol (see ``web/src/data/source.js``)::

    text   {"type": "meta", "format": "aevion.view/2", "meta": {...}, "world": ..., "widths": {channel: k}, ...}   first
    binary frame bundle  {"type": "frames", "t0": ..}   zero or more, in time order
    text   {"type": "event", "event": {"type": "handover", "t": ..}}
    text   {"type": "end", "frames": n, "stop_reason": ...}                                                       last
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..engine.hooks import Hook
from . import ws as _ws
from .payload import FORMAT, _beams, _cameras, _environment, _extent, _geofences, _planned, _rem, _vehicle, _wireless, _world, json_safe
from .protocol import encode_bundle

__all__ = ["LiveHub", "LiveStreamHook", "serve_live", "viewer_row"]

_CORE = ("pos", "quat", "vel", "euler", "rotor", "thrust", "power", "soc", "wind")
_SKIP = set(_CORE) | {"t", "ref_pos"}


def viewer_row(row: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
    """Map one recorder row to the viewer's channel names (core channels, ``ref``/``err``, and ``x:`` extras), each of shape ``(N, ...)``."""
    out: Dict[str, np.ndarray] = {}
    for k in _CORE:
        if k in row:
            out[k] = np.asarray(row[k], np.float32)
    if "ref_pos" in row:
        ref = np.asarray(row["ref_pos"], np.float32)
        out["ref"] = ref
        out["err"] = np.linalg.norm(np.asarray(row["pos"]) - ref, axis=-1).astype(np.float32)
    for k, v in row.items():
        if k in _SKIP or "/" in k:
            continue
        a = np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == n and a.dtype.kind in "fiub":
            out[f"x:{k}"] = a.astype(np.float32)
    return out


class LiveHub:
    """Thread-safe broadcaster with a replay buffer for late joiners."""

    def __init__(self, retain_mb: float = 256.0):
        self._lock = threading.RLock()
        self._retained: List[Tuple[bool, Union[str, bytes]]] = []
        self._bytes = 0
        self._cap = int(retain_mb * 1e6)
        self._clients: Set[_ws.WebSocket] = set()
        self.finished = False
        self.closed = False
        self.dropped_retention = False

    # ------------------------------------------------------------ publishing
    def _publish(self, is_text: bool, payload: Union[str, bytes], retain: bool = True) -> None:
        with self._lock:
            if retain:
                if is_text or self._bytes + len(payload) <= self._cap:  # small control messages are never lost to the byte cap
                    self._retained.append((is_text, payload))
                    self._bytes += len(payload)
                else:
                    self.dropped_retention = True
            for c in list(self._clients):
                try:
                    c.send_text(payload) if is_text else c.send_binary(payload)  # type: ignore[arg-type]
                except (_ws.WebSocketClosed, OSError):
                    self._clients.discard(c)

    def publish_meta(self, message: Dict[str, Any]) -> None:
        self._publish(True, json.dumps(json_safe({"type": "meta", **message}), separators=(",", ":"), allow_nan=False))

    def publish_frames(self, bundle: bytes) -> None:
        self._publish(False, bundle)

    def publish_event(self, event: Dict[str, Any]) -> None:
        self._publish(True, json.dumps(json_safe({"type": "event", "event": event}), separators=(",", ":"), allow_nan=False))

    def finish(self, **info: Any) -> None:
        self._publish(True, json.dumps(json_safe({"type": "end", **info}), separators=(",", ":"), allow_nan=False))
        self.finished = True

    # -------------------------------------------------------------- clients
    @property
    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def serve(self, sock: _ws.WebSocket) -> None:
        """Catch ``sock`` up, then keep it subscribed until it closes (blocks; called from the server's handler thread)."""
        with self._lock:
            if self.closed:
                return
            for is_text, payload in self._retained:
                sock.send_text(payload) if is_text else sock.send_binary(payload)  # type: ignore[arg-type]
            self._clients.add(sock)
        try:
            while not self.closed:
                try:
                    sock.recv()  # clients only send pings/keep-alives; anything else is ignored
                except TimeoutError:
                    if self.finished:
                        break
                    continue
        except (_ws.WebSocketClosed, OSError):
            pass
        finally:
            with self._lock:
                self._clients.discard(sock)

    def close(self) -> None:
        with self._lock:
            self.closed = True
            for c in list(self._clients):
                c.close()
            self._clients.clear()


class LiveStreamHook(Hook):
    """Engine hook that publishes the simulation to a :class:`LiveHub` while it runs.

    ``every``     record one frame per this many physics steps (default: about 30 frames per simulated second)
    ``speed``     simulated seconds per wall-clock second (``None``: as fast as the machine allows)
    ``flush_s``   send accumulated frames at least this often (wall-clock) so the browser stays responsive
    """

    name = "live_stream"
    priority = 90

    def __init__(
        self, hub: LiveHub, every: Optional[int] = None, speed: Optional[float] = 1.0, flush_s: float = 0.1, chunk_frames: int = 64, rem: bool = False
    ):
        if speed is not None and speed <= 0:
            raise ValueError("speed must be positive (or None for unthrottled)")
        self.hub, self.every, self.speed, self.flush_s, self.chunk, self.rem = hub, every, speed, flush_s, chunk_frames, rem
        self._rows: List[Dict[str, np.ndarray]] = []
        self._times: List[float] = []
        self._k = 0
        self._t_first: Optional[float] = None
        self._wall0 = 0.0
        self._last_flush = 0.0
        self._frames = 0
        self._widths: Optional[Dict[str, int]] = None
        self._prev: Dict[str, np.ndarray] = {}
        self._recorder: Any = None

    # -------------------------------------------------------------- lifecycle
    def on_start(self, sim: Any) -> None:
        from ..datasets.archive import Recorder

        self._recorder = Recorder(every=1, collision_events=False)
        self._wall0 = self._last_flush = time.perf_counter()
        if self.every is None:
            self.every = max(1, round(1.0 / (30.0 * sim.dt)))

    def post_step(self, sim: Any) -> None:
        self._k += 1
        if (self._k - 1) % int(self.every or 1):
            return
        row = viewer_row(self._recorder._row(sim.B, sim.state, sim._extras()), sim.n)
        if self._widths is None:
            self._t_first = float(sim.state.t)
            self._widths = {k: int(np.prod(v.shape[1:])) if v.ndim > 1 else 1 for k, v in row.items()}
            self._send_meta(sim, row)
        t = float(sim.state.t) - float(self._t_first or 0.0)
        self._detect_radio_events(row, t)
        self._rows.append(row)
        self._times.append(t)
        now = time.perf_counter()
        if len(self._rows) >= self.chunk or now - self._last_flush >= self.flush_s:
            self._flush()
            self._last_flush = now
        if self.speed is not None:
            lag = t / self.speed - (now - self._wall0)
            if lag > 0:
                time.sleep(min(lag, 0.25))

    def on_event(self, sim: Any, event: Any) -> None:
        if self._t_first is None:
            return
        rec = event.as_record()
        rec["t"] = float(rec["t"]) - float(self._t_first)
        self.hub.publish_event(rec)

    def on_end(self, sim: Any, archive: Any = None) -> None:
        self._flush()
        self.hub.finish(frames=self._frames, stop_reason=getattr(sim, "stop_reason", None))

    # ------------------------------------------------------------------ parts
    def _flush(self) -> None:
        if not self._rows:
            return
        names = list(self._rows[0])
        arrays = {"t": np.asarray(self._times, np.float32) - np.float32(self._times[0])}
        for n in names:
            arrays[n] = np.stack([r[n] for r in self._rows])
        self.hub.publish_frames(encode_bundle({"type": "frames", "t0": self._times[0], "t1": self._times[-1], "frames": len(self._rows)}, arrays))
        self._frames += len(self._rows)
        self._rows, self._times = [], []

    def _detect_radio_events(self, row: Dict[str, np.ndarray], t: float) -> None:
        """Handovers (serving cell changes) and radio-link failures are derived here, as :func:`derive_events` does for archives."""
        s, r = row.get("x:link_serving"), row.get("x:link_rlf")
        prev_s, prev_r = self._prev.get("s"), self._prev.get("r")
        if s is not None and prev_s is not None and "x:link_hops" not in row:
            for v in np.nonzero(s != prev_s)[0]:
                self.hub.publish_event({"type": "handover", "t": t, "vehicle": int(v), "from": int(prev_s[v]), "to": int(s[v])})
        if r is not None:
            fail = r > 0.5
            was = prev_r > 0.5 if prev_r is not None else np.zeros_like(fail)
            for v in np.nonzero(fail & ~was)[0]:
                self.hub.publish_event({"type": "rlf", "t": t, "vehicle": int(v)})
        if s is not None:
            self._prev["s"] = s.copy()
        if r is not None:
            self._prev["r"] = r.copy()

    def _send_meta(self, sim: Any, row: Dict[str, np.ndarray]) -> None:
        cfg = sim.cfg.to_dict()
        net = sim.net
        wmeta = net.to_dict() if net is not None else None
        out: Dict[str, Any] = {
            "format": FORMAT,
            "meta": dict(
                title=cfg.get("name") or "Aevion live run",
                n=int(sim.n),
                T=0,
                M=int(sim.spec.rotor_positions.shape[0]),
                controller=cfg.get("controller", {}).get("type", "") if isinstance(cfg.get("controller"), dict) else str(cfg.get("controller", "")),
                trajectory=(cfg.get("trajectory") or {}).get("type", "") if isinstance(cfg.get("trajectory"), dict) else "",
                seed=cfg.get("seed"),
                config_hash=getattr(sim, "config_hash", None),
                events=[],
                vehicle=_vehicle(sim.spec),
                backend=sim.B.name,
                duration=float(cfg.get("duration", 0.0)),
                live=True,
                environment=_environment(sim.world),
            ),
            "world": _world(sim.world),
            "wireless": _wireless(wmeta, "x:link_hops" in row) if wmeta and "x:link_sinr_db" in row else None,
            "camera": _cameras(cfg),
            "planned": _planned(cfg),
            "geofences": _geofences(cfg),
            "widths": self._widths,
            "runs": [],
        }
        if out["wireless"] and net is not None:
            try:
                out["beams"] = _beams(net)
                if self.rem:
                    pos = np.asarray(row["pos"])
                    out["rem"] = _rem(net, _extent(pos, sim.world, 0.6), [float(pos[:, 2].mean())])
            except Exception:  # optional overlays never block the stream
                pass
        self.hub.publish_meta(out)


def serve_live(
    config: Any,
    *,
    port: int = 0,
    host: str = "127.0.0.1",
    speed: Optional[float] = 1.0,
    every: Optional[int] = None,
    open_browser: bool = False,
    wait_for_client: bool = True,
    keep_serving: bool = False,
    theme: Optional[str] = None,
    hooks: Any = (),
) -> Any:
    """Run the simulation described by ``config`` (anything :class:`~aevion.engine.Simulation` accepts) and watch it live.

    Returns the finished :class:`~aevion.datasets.RunArchive`. With ``wait_for_client`` the run starts when the first browser connects, so
    nothing happens unseen; with ``keep_serving`` the server stays up after the run (Ctrl+C to stop) so the finished run can be inspected.
    """
    from ..engine import Simulation
    from .server import ViewerServer

    hub = LiveHub()
    server = ViewerServer(hub=hub, host=host, port=port, theme=theme).start()
    print(f"Aevion live viewer: {server.url}")
    if open_browser:
        import webbrowser

        webbrowser.open(server.url)
    try:
        if wait_for_client:
            print("waiting for a browser to connect (Ctrl+C to cancel)…")
            while hub.n_clients == 0:
                time.sleep(0.1)
        sim = Simulation(config, hooks=[LiveStreamHook(hub, every=every, speed=speed), *hooks])
        archive = sim.run()
        if keep_serving:
            print("run finished; the viewer stays available (Ctrl+C to stop)")
            while True:
                time.sleep(0.5)
        return archive
    except KeyboardInterrupt:
        return None
    finally:
        time.sleep(0.2)  # let the last frames flush to slow clients
        server.stop()
