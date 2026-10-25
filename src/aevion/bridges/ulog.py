"""Reading (and, for tests and exports, writing) PX4 **ULog** flight logs.

ULog is PX4's binary log format (https://docs.px4.io/main/en/dev_log/ulog_file_format.html): a header, a definition section
(flag bits, message *formats*, information, parameters) and a data section of timestamped messages grouped by *topic*. The reader
parses all of it with plain ``struct`` (no third-party package), including nested message types, arrays, multi-instance topics,
logged strings and dropouts, and tolerates truncated files (a log cut short by a crash is the common case worth reading).

>>> log = ULog("flight.ulg")              # doctest: +SKIP
>>> pos = log.data("vehicle_local_position")["x"]   # doctest: +SKIP

:meth:`ULog.to_archive` turns a PX4 flight into an Aevion archive in ENU/FLU axes from ``vehicle_local_position`` and
``vehicle_attitude`` (PX4 uses NED/FRD).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core import rotation as rot
from ..core.backend import NumpyBackend
from ..datasets.archive import RunArchive
from ..errors import ArchiveError

__all__ = ["ULog", "write_ulog", "matrix_to_quat", "ned_frd_to_enu_flu"]

MAGIC = b"\x55\x4c\x6f\x67\x01\x12\x35"
_PRIM = {
    "int8_t": "b",
    "uint8_t": "B",
    "int16_t": "h",
    "uint16_t": "H",
    "int32_t": "i",
    "uint32_t": "I",
    "int64_t": "q",
    "uint64_t": "Q",
    "float": "f",
    "double": "d",
    "bool": "?",
    "char": "c",
}
B = NumpyBackend()
MAX_FILE_BYTES = 512 * 1024 * 1024


@dataclass
class _Format:
    name: str
    fields: List[Tuple[str, str, int]]  # (type, name, array length or 0)


class ULog:
    """A parsed log: ``formats``, ``info``, ``parameters``, logged ``messages`` and per-topic data (``data(topic)``)."""

    def __init__(self, source: Any) -> None:
        raw = Path(source).read_bytes() if not isinstance(source, (bytes, bytearray)) else bytes(source)
        if len(raw) > MAX_FILE_BYTES:
            raise ArchiveError("the ULog file is larger than the safety limit")
        if raw[:7] != MAGIC:
            raise ArchiveError("not a ULog file (bad magic)")
        self.version = raw[7]
        self.start_us = struct.unpack("<Q", raw[8:16])[0]
        self.formats: Dict[str, _Format] = {}
        self.info: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = {}
        self.messages: List[Tuple[int, int, str]] = []  # (level, timestamp_us, text)
        self.dropouts: List[Tuple[int, int]] = []  # (timestamp_us, duration_ms)
        self.truncated = False
        self._subs: Dict[int, Tuple[str, int]] = {}  # msg_id -> (topic, multi_id)
        self._rows: Dict[Tuple[str, int], List[bytes]] = {}
        self._parse(raw)

    # ------------------------------------------------------------------------------------------ parsing
    def _parse(self, raw: bytes) -> None:
        i, n = 16, len(raw)
        while i + 3 <= n:
            size, mtype = struct.unpack("<HB", raw[i : i + 3])
            if i + 3 + size > n:
                self.truncated = True
                break
            body = raw[i + 3 : i + 3 + size]
            i += 3 + size
            t = chr(mtype)
            if t == "F":
                self._format(body)
            elif t in ("I", "M"):
                self._keyvalue(body, self.info, multi=(t == "M"))
            elif t == "P":
                self._keyvalue(body, self.parameters)
            elif t == "A":
                multi, msg_id = struct.unpack("<BH", body[:3])
                self._subs[msg_id] = (body[3:].decode("ascii", "replace"), multi)
                self._rows.setdefault(self._subs[msg_id], [])
            elif t == "D":
                (msg_id,) = struct.unpack("<H", body[:2])
                key = self._subs.get(msg_id)
                if key is not None:
                    self._rows[key].append(body[2:])
            elif t == "L":
                level, ts = struct.unpack("<BQ", body[:9])
                self.messages.append((level, ts, body[9:].decode("utf-8", "replace")))
            elif t == "C":
                level, tag, ts = struct.unpack("<BHQ", body[:11])
                self.messages.append((level, ts, body[11:].decode("utf-8", "replace")))
            elif t == "O":
                self.dropouts.append((0, struct.unpack("<H", body[:2])[0]))
            # 'B' (flag bits), 'R' (unsubscribe), 'S' (sync), 'Q' (default parameters) carry nothing the reader needs

    def _format(self, body: bytes) -> None:
        text = body.decode("ascii", "replace")
        name, _, rest = text.partition(":")
        fields = []
        for item in rest.split(";"):
            if not item.strip():
                continue
            typ, _, fname = item.strip().partition(" ")
            count = 0
            if typ.endswith("]"):
                typ, c = typ[:-1].split("[")
                count = int(c)
            fields.append((typ, fname, count))
        self.formats[name] = _Format(name, fields)

    def _keyvalue(self, body: bytes, into: Dict[str, Any], multi: bool = False) -> None:
        off = 1 if multi else 0
        klen = body[off]
        key = body[off + 1 : off + 1 + klen].decode("ascii", "replace")
        value = body[off + 1 + klen :]
        typ, _, name = key.partition(" ")
        try:
            if typ.startswith("char["):
                into[name] = value.decode("utf-8", "replace")
            elif typ in _PRIM:
                into[name] = struct.unpack("<" + _PRIM[typ], value[: struct.calcsize(_PRIM[typ])])[0]
            else:
                into[name] = value
        except struct.error:
            into[name] = value

    # ------------------------------------------------------------------------------------------ data
    def topics(self) -> List[Tuple[str, int]]:
        """``(topic, multi_id)`` for every subscribed topic that logged at least one message."""
        return sorted(k for k, v in self._rows.items() if v)

    def _layout(self, fmt: _Format, prefix: str = "") -> List[Tuple[str, str, int]]:
        """Flatten a (possibly nested) format into ``(column name, struct code, count)`` entries."""
        out: List[Tuple[str, str, int]] = []
        for typ, fname, count in fmt.fields:
            if fname.startswith("_padding"):
                out.append(("", f"{count or 1}x" if typ in ("uint8_t", "char") else "", 0))
                continue
            if typ in _PRIM:
                out.append((prefix + fname, _PRIM[typ], count))
            elif typ in self.formats:
                for k in range(count or 1):
                    sub = self._layout(self.formats[typ], f"{prefix}{fname}{f'[{k}]' if count else ''}.")
                    out.extend(sub)
            else:
                raise ArchiveError(f"unknown ULog type {typ!r}")
        return out

    def data(self, topic: str, multi_id: int = 0) -> Dict[str, np.ndarray]:
        """Columns of a topic as arrays (arrays of a field become 2-D columns, nested fields are named ``parent.child``)."""
        key = (topic, multi_id)
        if key not in self._rows or topic not in self.formats:
            raise KeyError(f"no logged data for topic {topic!r} (instance {multi_id})")
        layout = self._layout(self.formats[topic])
        parts = []
        for _name, code, count in layout:
            if not code:
                continue
            parts.append(code if code.endswith("x") else (f"{count}{code}" if count else code))
        st = struct.Struct("<" + "".join(parts))
        rows = []
        for r in self._rows[key]:
            if len(r) >= st.size:
                rows.append(st.unpack(r[: st.size]))
        cols: Dict[str, np.ndarray] = {}
        j = 0
        for name, code, count in layout:
            if not code or code.endswith("x"):
                continue
            w = count or 1
            arr = np.array([row[j : j + w] if count else row[j] for row in rows])
            if code == "c":
                arr = np.array([b"".join(row[j : j + w]).split(b"\x00", 1)[0].decode("utf-8", "replace") for row in rows])
            cols[name] = arr
            j += w
        return cols

    @property
    def duration_s(self) -> float:
        ts = [self.data(t, m)["timestamp"] for t, m in self.topics() if "timestamp" in self.formats[t].fields[0]]
        return float(max((a.max() for a in ts if len(a)), default=0) - self.start_us) * 1e-6

    # ------------------------------------------------------------------------------------------ conversion
    def to_archive(self, name: str = "px4") -> RunArchive:
        """Aevion archive (ENU/FLU) from ``vehicle_local_position`` and ``vehicle_attitude``."""
        pos = self.data("vehicle_local_position")
        t = pos["timestamp"] * 1e-6
        p_ned = np.stack([pos["x"], pos["y"], pos["z"]], axis=1)
        v_ned = np.stack([pos["vx"], pos["vy"], pos["vz"]], axis=1)
        ch: Dict[str, Any] = dict(
            t=t - t[0],
            pos=np.stack([p_ned[:, 1], p_ned[:, 0], -p_ned[:, 2]], 1)[:, None, :],
            vel=np.stack([v_ned[:, 1], v_ned[:, 0], -v_ned[:, 2]], 1)[:, None, :],
        )
        if self._rows.get(("vehicle_attitude", 0)):
            att = self.data("vehicle_attitude")
            ta = att["timestamp"] * 1e-6
            q = np.stack([np.interp(t, ta, att["q"][:, k]) for k in range(4)], axis=1)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            ch["quat"] = ned_frd_to_enu_flu(q)[:, None, :]
        return RunArchive(
            channels=ch,
            meta=dict(
                name=name,
                source="PX4 ULog",
                complete=not self.truncated,
                parameters=dict(self.parameters),
                info={k: v for k, v in self.info.items() if isinstance(v, (str, int, float))},
            ),
        )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion ``[w, x, y, z]`` of rotation matrices ``(..., 3, 3)`` (Shepperd's method, numerically stable)."""
    R = np.asarray(R, float)
    flat = R.reshape(-1, 3, 3)
    q = np.zeros((len(flat), 4))
    for i, m in enumerate(flat):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(R.shape[:-2] + (4,))


_M = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # NED -> ENU
_D = np.diag([1.0, -1.0, -1.0])  # FLU <-> FRD


def ned_frd_to_enu_flu(q: np.ndarray) -> np.ndarray:
    """Convert attitude quaternions (body FRD -> world NED, as PX4 logs them) to Aevion's (body FLU -> world ENU)."""
    q = np.asarray(q, float)
    R = rot.to_matrix(B, q)
    return matrix_to_quat(np.einsum("ij,njk,kl->nil", _M, R, _D))


# ======================================================================================================== writing
def write_ulog(
    path: Any,
    topics: Dict[str, Tuple[str, Dict[str, np.ndarray]]],
    parameters: Optional[Dict[str, float]] = None,
    info: Optional[Dict[str, str]] = None,
    start_us: int = 0,
) -> Path:
    """Write a small ULog file (used by the tests and to export simulated flights for PX4 tools).

    ``topics`` maps a topic name to ``(format, columns)`` with ``format`` like ``"uint64_t timestamp;float x;float[3] q"`` (as in
    the specification) and ``columns`` the arrays (one per field, arrays of fields as 2-D)."""
    out = bytearray(MAGIC + bytes([1]) + struct.pack("<Q", start_us))

    def msg(t: str, body: bytes) -> bytes:
        return struct.pack("<HB", len(body), ord(t)) + body

    out += msg("B", bytes(8) + bytes(8) + bytes(24))
    for name, (fmt, _) in topics.items():
        out += msg("F", f"{name}:{fmt}".encode())
    for k, v in (info or {}).items():
        key = f"char[{len(v)}] {k}".encode()
        out += msg("I", bytes([len(key)]) + key + v.encode())
    for pk, pv in (parameters or {}).items():
        key = f"float {pk}".encode()
        out += msg("P", bytes([len(key)]) + key + struct.pack("<f", float(pv)))
    rows: List[Tuple[int, int, bytes]] = []
    for mid, (name, (fmt, cols)) in enumerate(topics.items()):
        out += msg("A", struct.pack("<BH", 0, mid) + name.encode())
        fields = []
        for item in fmt.split(";"):
            if not item.strip():
                continue
            typ, _, fname = item.strip().partition(" ")
            count = 0
            if typ.endswith("]"):
                typ, c = typ[:-1].split("[")
                count = int(c)
            fields.append((typ, fname, count))
        n = len(next(iter(cols.values())))
        for r in range(n):
            payload = b""
            for typ, fname, count in fields:
                v = cols[fname][r]
                payload += struct.pack("<" + (f"{count}{_PRIM[typ]}" if count else _PRIM[typ]), *(list(v) if count else [v]))
            ts = int(cols["timestamp"][r]) if "timestamp" in cols else r
            rows.append((ts, mid, struct.pack("<H", mid) + payload))
    for _, _, body in sorted(rows, key=lambda x: (x[0], x[1])):
        out += msg("D", body)
    p = Path(path)
    p.write_bytes(bytes(out))
    return p
