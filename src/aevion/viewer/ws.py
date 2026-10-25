"""A small RFC 6455 WebSocket implementation (client and server side), standard library only.

The viewer needs WebSockets in two places and nothing else: streaming a running simulation to the browser (server side) and talking the Chrome
DevTools Protocol to a headless browser for video export (client side). Both are local, so this is deliberately minimal: text/binary
messages, fragmentation, ping/pong and close; no extensions, no compression.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import struct
import threading
from typing import Optional, Tuple

__all__ = ["WebSocket", "WebSocketClosed", "accept_key", "connect", "handshake_response"]

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketClosed(ConnectionError):
    """The peer closed the connection (or it dropped)."""


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + GUID).encode("ascii"), usedforsecurity=False).digest()).decode("ascii")  # nosec B324 (mandated by RFC 6455; not used for security)


def handshake_response(client_key: str) -> bytes:
    return (
        f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept_key(client_key)}\r\n\r\n"
    ).encode("ascii")


class WebSocket:
    """A connected WebSocket. ``client=True`` masks outgoing frames as the RFC requires."""

    def __init__(self, sock: socket.socket, client: bool = False, max_size: int = 1 << 20):
        self.sock, self.client, self.max_size = sock, client, max_size
        self._lock = threading.Lock()
        self._buf = b""  # bytes read ahead during the handshake
        self.closed = False

    # ------------------------------------------------------------------ sending
    def _send(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        head = bytes([0x80 | opcode])
        mask_bit = 0x80 if self.client else 0
        if n < 126:
            head += bytes([mask_bit | n])
        elif n < 1 << 16:
            head += bytes([mask_bit | 126]) + struct.pack(">H", n)
        else:
            head += bytes([mask_bit | 127]) + struct.pack(">Q", n)
        if self.client:
            key = os.urandom(4)
            payload = _xor_mask(payload, key)
            head += key
        with self._lock:
            try:
                self.sock.sendall(head + payload)
            except OSError as e:
                self.closed = True
                raise WebSocketClosed(str(e)) from e

    def send_text(self, text: str) -> None:
        self._send(OP_TEXT, text.encode("utf-8"))

    def send_binary(self, data: bytes) -> None:
        self._send(OP_BIN, data)

    def close(self, code: int = 1000) -> None:
        if not self.closed:
            try:
                self._send(OP_CLOSE, struct.pack(">H", code))
            except WebSocketClosed:
                pass
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass

    # ---------------------------------------------------------------- receiving
    def _read(self, n: int) -> bytes:
        buf = bytearray(self._buf[:n])
        self._buf = self._buf[n:]
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except TimeoutError:
                raise
            except OSError as e:
                self.closed = True
                raise WebSocketClosed(str(e)) from e
            if not chunk:
                self.closed = True
                raise WebSocketClosed("connection closed")
            buf += chunk
        return bytes(buf)

    def _frame(self) -> Tuple[bool, int, bytes]:
        b0, b1 = self._read(2)
        fin, opcode, masked, n = bool(b0 & 0x80), b0 & 0x0F, bool(b1 & 0x80), b1 & 0x7F
        if n == 126:
            (n,) = struct.unpack(">H", self._read(2))
        elif n == 127:
            (n,) = struct.unpack(">Q", self._read(8))
        if n > self.max_size:
            self.close(1009)
            raise WebSocketClosed(f"frame of {n} bytes exceeds the limit of {self.max_size}")
        key = self._read(4) if masked else b""
        data = self._read(n)
        if masked:
            data = _xor_mask(data, key)
        return fin, opcode, data

    def recv(self) -> Tuple[int, bytes]:
        """Next complete data message as ``(opcode, payload)``; control frames are handled here. Raises :class:`WebSocketClosed`."""
        message, kind = bytearray(), OP_TEXT
        while True:
            fin, op, data = self._frame()
            if op == OP_PING:
                self._send(OP_PONG, data)
            elif op == OP_PONG:
                continue
            elif op == OP_CLOSE:
                self.close()
                raise WebSocketClosed("closed by peer")
            else:
                if op != OP_CONT:
                    kind, message = op, bytearray()
                message += data
                if len(message) > self.max_size:
                    self.close(1009)
                    raise WebSocketClosed("message exceeds the size limit")
                if fin:
                    return kind, bytes(message)

    def recv_text(self) -> str:
        return self.recv()[1].decode("utf-8")


def _xor_mask(data: bytes, key: bytes) -> bytes:
    if not data:
        return data
    n = len(data)
    mask = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(mask, "little")).to_bytes(n, "little")


def connect(host: str, port: int, path: str, timeout: float = 10.0, max_size: int = 64 << 20, origin: Optional[str] = None) -> WebSocket:
    """Open a client connection (used for the DevTools protocol of a local headless browser)."""
    sock = socket.create_connection((host, port), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n" + (f"Origin: {origin}\r\n" if origin else "") + "\r\n"
    )
    sock.sendall(req.encode("ascii"))
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = sock.recv(4096)
        if not chunk:
            raise WebSocketClosed("handshake failed: connection closed")
        resp += chunk
        if len(resp) > 65536:
            raise WebSocketClosed("handshake response too long")
    head, _, rest = resp.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    if " 101 " not in lines[0] or accept_key(key).lower() not in head.decode("latin-1").lower():
        raise WebSocketClosed(f"handshake rejected: {lines[0]}")
    ws = WebSocket(sock, client=True, max_size=max_size)
    ws._buf = rest  # bytes that arrived with the handshake belong to the first frame
    return ws
