"""Binary frame container shared by the viewer's HTTP API and WebSocket stream.

Layout (all integers little-endian)::

    [uint32 H][H bytes UTF-8 JSON header][zero padding to a multiple of 4][float32 arrays back to back]

The header is ``{"type": ..., "arrays": [{"name": ..., "shape": [...]}, ...], ...}``. The JavaScript twin lives in
``web/src/data/decode.js``; ``tests/test_viewer_protocol.py`` round-trips both through Node.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Mapping, Tuple

import numpy as np

__all__ = ["encode_bundle", "decode_bundle", "FormatError"]

MAX_HEADER_BYTES = 16 * 1024 * 1024


class FormatError(ValueError):
    """A bundle that does not follow the container layout."""


def encode_bundle(header: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> bytes:
    """Pack ``arrays`` (converted to float32) behind a JSON header that records their names and shapes."""
    names = list(arrays)
    flat = {n: np.ascontiguousarray(arrays[n], dtype="<f4") for n in names}
    head = json.dumps(
        {**header, "arrays": [{"name": n, "shape": list(flat[n].shape)} for n in names]}, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    pad = (-(4 + len(head))) % 4
    return b"".join([struct.pack("<I", len(head)), head, b"\0" * pad, *[flat[n].tobytes() for n in names]])


def decode_bundle(buf: bytes) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Inverse of :func:`encode_bundle` (used by tests and by Python clients of the live stream)."""
    if len(buf) < 4:
        raise FormatError("bundle too short")
    (hlen,) = struct.unpack_from("<I", buf, 0)
    if hlen > MAX_HEADER_BYTES or 4 + hlen > len(buf):
        raise FormatError("bundle header exceeds the buffer")
    header = json.loads(buf[4 : 4 + hlen].decode("utf-8"))
    off = (4 + hlen + 3) & ~3
    arrays: Dict[str, np.ndarray] = {}
    for a in header.get("arrays", []):
        shape = tuple(int(s) for s in a["shape"])
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if n < 0 or off + 4 * n > len(buf):
            raise FormatError(f"array {a['name']!r} does not fit in the bundle")
        arrays[a["name"]] = np.frombuffer(buf, dtype="<f4", count=n, offset=off).reshape(shape).copy()
        off += 4 * n
    return header, arrays
