"""Byte transports for the bridges: an in-process loopback and UDP.

The bridges speak to an autopilot or a ground station over *some* byte pipe; this module is the only place in the bridge package
that touches the network. :class:`UdpTransport` binds and sends to the **loopback interface only** unless ``allow_remote=True``
is passed, so a simulation never listens on (or talks to) an external address by accident. :class:`LoopbackTransport` connects two
endpoints in one process and can run a callback when the other end sends, which makes closed-loop bridge tests deterministic
without any socket.
"""

from __future__ import annotations

import socket
from collections import deque
from typing import Callable, Deque, Optional, Tuple

__all__ = ["Transport", "LoopbackTransport", "UdpTransport"]

_LOCAL = ("127.0.0.1", "localhost", "::1")


class Transport:
    """Minimal byte pipe: ``send`` a frame, ``recv`` whatever has arrived (non-blocking), ``close``."""

    def send(self, data: bytes) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def recv(self, max_bytes: int = 65536) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LoopbackTransport(Transport):
    """One end of an in-memory pair (:meth:`pair`). ``on_receive`` (if set) is called with the bytes the *peer* sends."""

    def __init__(self) -> None:
        self._inbox: Deque[bytes] = deque()
        self.peer: Optional["LoopbackTransport"] = None
        self.on_receive: Optional[Callable[[bytes], None]] = None
        self.sent = 0

    @classmethod
    def pair(cls) -> Tuple["LoopbackTransport", "LoopbackTransport"]:
        a, b = cls(), cls()
        a.peer, b.peer = b, a
        return a, b

    def send(self, data: bytes) -> None:
        if self.peer is None:
            raise RuntimeError("this loopback end has no peer")
        self.sent += len(data)
        self.peer._inbox.append(bytes(data))
        if self.peer.on_receive is not None:
            self.peer.on_receive(bytes(data))

    def recv(self, max_bytes: int = 65536) -> bytes:
        out = b""
        while self._inbox and len(out) + len(self._inbox[0]) <= max_bytes:
            out += self._inbox.popleft()
        return out


class UdpTransport(Transport):
    """A non-blocking UDP endpoint: bound to ``(bind_host, local_port)``, sending to ``(remote_host, remote_port)``.

    Hosts other than the loopback addresses are refused unless ``allow_remote=True`` (an explicit decision to leave the machine).
    """

    def __init__(
        self,
        local_port: int,
        remote_port: Optional[int] = None,
        remote_host: str = "127.0.0.1",
        bind_host: str = "127.0.0.1",
        allow_remote: bool = False,
    ) -> None:
        for h in (bind_host, remote_host):
            if h not in _LOCAL and not allow_remote:
                raise ValueError(f"refusing to use non-loopback address {h!r}; pass allow_remote=True if the autopilot is on another machine")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((bind_host, int(local_port)))
        self.remote = (remote_host, int(remote_port)) if remote_port is not None else None
        self.last_sender: Optional[Tuple[str, int]] = None

    @property
    def port(self) -> int:
        return int(self.sock.getsockname()[1])

    def send(self, data: bytes) -> None:
        dest = self.remote or self.last_sender
        if dest is None:
            return  # nobody has talked to us yet and no remote was configured: nothing to answer
        self.sock.sendto(data, dest)

    def recv(self, max_bytes: int = 65536) -> bytes:
        out = b""
        while True:
            try:
                data, addr = self.sock.recvfrom(max_bytes)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            self.last_sender = addr
            out += data
        return out

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
