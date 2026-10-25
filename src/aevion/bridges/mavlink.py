"""A self-contained MAVLink 1/2 codec for the messages a simulator and a ground station exchange.

A frame is ``STX | len | [flags] | seq | sysid | compid | msgid | payload | crc16 [| signature]``. The payload holds the message fields
in wire order (sorted by decreasing size, stable), little-endian; MAVLink 2 trims trailing zero bytes (keeping at least one) and
appends extension fields. The checksum is X.25 CRC-16 over the header, the payload and a per-message ``CRC_EXTRA`` byte derived from
the message definition, so ends with different layouts reject each other's packets.

``CRC_EXTRA`` is computed from the field definitions with the MAVLink generator's algorithm instead of a table of constants; the tests
compare the well-known values (HEARTBEAT 50, ATTITUDE 39, LOCAL_POSITION_NED 185, GLOBAL_POSITION_INT 104, SYSTEM_TIME 137, ...).

Only the common-dialect messages the bridges use are defined (add more with :func:`define`). Frames with an unknown message id are
skipped and counted, since their checksum cannot be verified.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

__all__ = ["Field", "MessageSpec", "Message", "MESSAGES", "BY_ID", "define", "crc_extra", "x25", "pack", "Parser", "MavlinkError", "STX_V1", "STX_V2"]

STX_V1, STX_V2 = 0xFE, 0xFD
SIGNATURE_LEN = 13
_SIZES = {
    "char": 1,
    "int8_t": 1,
    "uint8_t": 1,
    "int16_t": 2,
    "uint16_t": 2,
    "int32_t": 4,
    "uint32_t": 4,
    "int64_t": 8,
    "uint64_t": 8,
    "float": 4,
    "double": 8,
}
_FMT = {
    "char": "c",
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
}


class MavlinkError(ValueError):
    pass


@dataclass(frozen=True)
class Field:
    type: str
    name: str
    count: int = 0  # 0: scalar, N: array of N
    extension: bool = False

    @property
    def size(self) -> int:
        return _SIZES[self.type] * (self.count or 1)


@dataclass
class MessageSpec:
    name: str
    id: int
    fields: Tuple[Field, ...]

    def __post_init__(self) -> None:
        base = [f for f in self.fields if not f.extension]
        ext = [f for f in self.fields if f.extension]
        # wire order: base fields sorted by element size (descending, stable), then the extensions in declaration order
        self.wire: Tuple[Field, ...] = tuple(sorted(base, key=lambda f: -_SIZES[f.type])) + tuple(ext)
        self.crc_extra = crc_extra(self)
        self.max_len = sum(f.size for f in self.wire)
        self.base_len = sum(f.size for f in self.wire if not f.extension)
        fmt = "<" + "".join(
            (f"{f.count}{_FMT[f.type]}" if f.count else _FMT[f.type]) if f.type != "char" else (f"{f.count}s" if f.count else "c") for f in self.wire
        )
        self._struct = struct.Struct(fmt)


def x25(data: bytes, crc: int = 0xFFFF) -> int:
    """X.25 CRC-16 (the MAVLink checksum): ``tmp = b ^ (crc & 0xff); tmp ^= tmp << 4; crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)``."""
    for b in data:
        tmp = b ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def crc_extra(spec: MessageSpec) -> int:
    """The CRC_EXTRA of a message definition: CRC-16 of ``"NAME "`` then, for each non-extension field in wire order,
    ``"type name "`` (and the array length as one byte for arrays), folded to a byte."""
    s = spec.name + " "
    crc = x25(s.encode())
    for f in spec.wire:
        if f.extension:
            continue
        crc = x25(f"{f.type} ".encode(), crc)
        crc = x25(f"{f.name} ".encode(), crc)
        if f.count:
            crc = x25(bytes([f.count]), crc)
    return (crc & 0xFF) ^ (crc >> 8)


MESSAGES: Dict[str, MessageSpec] = {}
BY_ID: Dict[int, MessageSpec] = {}


def define(name: str, msg_id: int, *fields: Tuple[Any, ...]) -> MessageSpec:
    """Register a message: ``define("ATTITUDE", 30, ("uint32_t", "time_boot_ms"), ("float", "roll"), ...)``; a trailing ``"ext"`` marks
    an extension field and a list type ``("float[4]", "q")`` an array."""
    fs = []
    for f in fields:
        t, n = f[0], f[1]
        ext = len(f) > 2 and f[2] == "ext"
        count = 0
        if t.endswith("]"):
            t, c = t[:-1].split("[")
            count = int(c)
        if t not in _SIZES:
            raise MavlinkError(f"unknown field type {t!r}")
        fs.append(Field(t, n, count, ext))
    spec = MessageSpec(name, msg_id, tuple(fs))
    MESSAGES[name] = spec
    BY_ID[msg_id] = spec
    return spec


# ---- the common dialect, as far as the bridges need it (definitions as in MAVLink common.xml)
define(
    "HEARTBEAT",
    0,
    ("uint32_t", "custom_mode"),
    ("uint8_t", "type"),
    ("uint8_t", "autopilot"),
    ("uint8_t", "base_mode"),
    ("uint8_t", "system_status"),
    ("uint8_t", "mavlink_version"),
)
define(
    "SYS_STATUS",
    1,
    ("uint32_t", "onboard_control_sensors_present"),
    ("uint32_t", "onboard_control_sensors_enabled"),
    ("uint32_t", "onboard_control_sensors_health"),
    ("uint16_t", "load"),
    ("uint16_t", "voltage_battery"),
    ("int16_t", "current_battery"),
    ("int8_t", "battery_remaining"),
    ("uint16_t", "drop_rate_comm"),
    ("uint16_t", "errors_comm"),
    ("uint16_t", "errors_count1"),
    ("uint16_t", "errors_count2"),
    ("uint16_t", "errors_count3"),
    ("uint16_t", "errors_count4"),
)
define("SYSTEM_TIME", 2, ("uint64_t", "time_unix_usec"), ("uint32_t", "time_boot_ms"))
define(
    "ATTITUDE",
    30,
    ("uint32_t", "time_boot_ms"),
    ("float", "roll"),
    ("float", "pitch"),
    ("float", "yaw"),
    ("float", "rollspeed"),
    ("float", "pitchspeed"),
    ("float", "yawspeed"),
)
define(
    "ATTITUDE_QUATERNION",
    31,
    ("uint32_t", "time_boot_ms"),
    ("float", "q1"),
    ("float", "q2"),
    ("float", "q3"),
    ("float", "q4"),
    ("float", "rollspeed"),
    ("float", "pitchspeed"),
    ("float", "yawspeed"),
    ("float[4]", "repr_offset_q", "ext"),
)
define(
    "LOCAL_POSITION_NED",
    32,
    ("uint32_t", "time_boot_ms"),
    ("float", "x"),
    ("float", "y"),
    ("float", "z"),
    ("float", "vx"),
    ("float", "vy"),
    ("float", "vz"),
)
define(
    "GLOBAL_POSITION_INT",
    33,
    ("uint32_t", "time_boot_ms"),
    ("int32_t", "lat"),
    ("int32_t", "lon"),
    ("int32_t", "alt"),
    ("int32_t", "relative_alt"),
    ("int16_t", "vx"),
    ("int16_t", "vy"),
    ("int16_t", "vz"),
    ("uint16_t", "hdg"),
)
define(
    "COMMAND_LONG",
    76,
    ("uint8_t", "target_system"),
    ("uint8_t", "target_component"),
    ("uint16_t", "command"),
    ("uint8_t", "confirmation"),
    ("float", "param1"),
    ("float", "param2"),
    ("float", "param3"),
    ("float", "param4"),
    ("float", "param5"),
    ("float", "param6"),
    ("float", "param7"),
)
define(
    "SET_POSITION_TARGET_LOCAL_NED",
    84,
    ("uint32_t", "time_boot_ms"),
    ("uint8_t", "target_system"),
    ("uint8_t", "target_component"),
    ("uint8_t", "coordinate_frame"),
    ("uint16_t", "type_mask"),
    ("float", "x"),
    ("float", "y"),
    ("float", "z"),
    ("float", "vx"),
    ("float", "vy"),
    ("float", "vz"),
    ("float", "afx"),
    ("float", "afy"),
    ("float", "afz"),
    ("float", "yaw"),
    ("float", "yaw_rate"),
)
define("HIL_ACTUATOR_CONTROLS", 93, ("uint64_t", "time_usec"), ("float[16]", "controls"), ("uint8_t", "mode"), ("uint64_t", "flags"))
define(
    "HIGHRES_IMU",
    105,
    ("uint64_t", "time_usec"),
    ("float", "xacc"),
    ("float", "yacc"),
    ("float", "zacc"),
    ("float", "xgyro"),
    ("float", "ygyro"),
    ("float", "zgyro"),
    ("float", "xmag"),
    ("float", "ymag"),
    ("float", "zmag"),
    ("float", "abs_pressure"),
    ("float", "diff_pressure"),
    ("float", "pressure_alt"),
    ("float", "temperature"),
    ("uint16_t", "fields_updated"),
    ("uint8_t", "id", "ext"),
)
define(
    "HIL_SENSOR",
    107,
    ("uint64_t", "time_usec"),
    ("float", "xacc"),
    ("float", "yacc"),
    ("float", "zacc"),
    ("float", "xgyro"),
    ("float", "ygyro"),
    ("float", "zgyro"),
    ("float", "xmag"),
    ("float", "ymag"),
    ("float", "zmag"),
    ("float", "abs_pressure"),
    ("float", "diff_pressure"),
    ("float", "pressure_alt"),
    ("float", "temperature"),
    ("uint32_t", "fields_updated"),
    ("uint8_t", "id", "ext"),
)
define(
    "HIL_GPS",
    113,
    ("uint64_t", "time_usec"),
    ("uint8_t", "fix_type"),
    ("int32_t", "lat"),
    ("int32_t", "lon"),
    ("int32_t", "alt"),
    ("uint16_t", "eph"),
    ("uint16_t", "epv"),
    ("uint16_t", "vel"),
    ("int16_t", "vn"),
    ("int16_t", "ve"),
    ("int16_t", "vd"),
    ("uint16_t", "cog"),
    ("uint8_t", "satellites_visible"),
    ("uint8_t", "id", "ext"),
    ("uint16_t", "yaw", "ext"),
)
define(
    "HIL_STATE_QUATERNION",
    115,
    ("uint64_t", "time_usec"),
    ("float[4]", "attitude_quaternion"),
    ("float", "rollspeed"),
    ("float", "pitchspeed"),
    ("float", "yawspeed"),
    ("int32_t", "lat"),
    ("int32_t", "lon"),
    ("int32_t", "alt"),
    ("int16_t", "vx"),
    ("int16_t", "vy"),
    ("int16_t", "vz"),
    ("uint16_t", "ind_airspeed"),
    ("uint16_t", "true_airspeed"),
    ("int16_t", "xacc"),
    ("int16_t", "yacc"),
    ("int16_t", "zacc"),
)
define("STATUSTEXT", 253, ("uint8_t", "severity"), ("char[50]", "text"), ("uint16_t", "id", "ext"), ("uint8_t", "chunk_seq", "ext"))


@dataclass
class Message:
    """A decoded message: ``fields`` by name, plus the header (``sysid``, ``compid``, ``seq``)."""

    spec: MessageSpec
    fields: Dict[str, Any]
    sysid: int = 1
    compid: int = 1
    seq: int = 0

    @property
    def name(self) -> str:
        return self.spec.name

    def __getattr__(self, k: str) -> Any:
        try:
            return self.__dict__["fields"][k]
        except KeyError:
            raise AttributeError(k) from None

    def to_dict(self) -> Dict[str, Any]:
        return dict(name=self.name, sysid=self.sysid, compid=self.compid, seq=self.seq, **self.fields)


# ======================================================================================================== encode
def _default(f: Field) -> Any:
    if f.type == "char":
        return b"" if f.count else b"\x00"
    return [0.0 if f.type in ("float", "double") else 0] * f.count if f.count else (0.0 if f.type in ("float", "double") else 0)


def pack(name: str, sysid: int = 1, compid: int = 1, seq: int = 0, v2: bool = True, **values: Any) -> bytes:
    """Encode one frame. Missing fields default to zero; unknown names raise. MAVLink 2 trims trailing zeros of the payload."""
    spec = MESSAGES.get(name)
    if spec is None:
        raise MavlinkError(f"unknown message {name!r}")
    names = {f.name for f in spec.fields}
    bad = set(values) - names
    if bad:
        raise MavlinkError(f"{name} has no field(s) {sorted(bad)}")
    if not v2 and spec.id > 255:
        raise MavlinkError("MAVLink 1 only carries message ids below 256")
    args: List[Any] = []
    wire = [f for f in spec.wire if v2 or not f.extension]  # MAVLink 1 has no extension fields
    st = struct.Struct(
        "<"
        + "".join(
            (f"{f.count}{_FMT[f.type]}" if f.count else _FMT[f.type]) if f.type != "char" else (f"{f.count}s" if f.count else "c") for f in wire
        )
    )
    for f in wire:
        v = values.get(f.name, _default(f))
        if f.type == "char":
            if f.count:
                v = v.encode() if isinstance(v, str) else bytes(v)
                v = v[: f.count].ljust(f.count, b"\x00")
            else:
                v = v if isinstance(v, bytes) else bytes([int(v)]) if isinstance(v, int) else str(v).encode()[:1]
            args.append(v)
        elif f.count:
            vv = list(v) + [0] * (f.count - len(v))
            args.extend(vv[: f.count])
        else:
            args.append(v)
    try:
        payload = st.pack(*args)
    except struct.error as ex:
        raise MavlinkError(f"cannot encode {name}: {ex}") from ex
    if v2:
        stripped = payload.rstrip(b"\x00") or payload[:1]  # trailing zeros are not sent (at least one byte is)
        n = len(stripped)
        head = bytes([n, 0, 0, seq & 0xFF, sysid & 0xFF, compid & 0xFF, spec.id & 0xFF, (spec.id >> 8) & 0xFF, (spec.id >> 16) & 0xFF])
        crc = x25(head + stripped + bytes([spec.crc_extra]))
        return bytes([STX_V2]) + head + stripped + struct.pack("<H", crc)
    head = bytes([len(payload), seq & 0xFF, sysid & 0xFF, compid & 0xFF, spec.id & 0xFF])
    crc = x25(head + payload + bytes([spec.crc_extra]))
    return bytes([STX_V1]) + head + payload + struct.pack("<H", crc)


# ======================================================================================================== decode
def _decode(spec: MessageSpec, payload: bytes, sysid: int, compid: int, seq: int) -> Message:
    full = payload.ljust(spec.max_len, b"\x00")[: spec.max_len]  # MAVLink 2 trailing zeros (and absent extensions) are implied
    vals = spec._struct.unpack(full)
    out: Dict[str, Any] = {}
    i = 0
    for f in spec.wire:
        if f.type == "char":
            if f.count:
                out[f.name] = vals[i].split(b"\x00", 1)[0].decode("utf-8", "replace")
            else:
                out[f.name] = vals[i]
            i += 1
        elif f.count:
            out[f.name] = list(vals[i : i + f.count])
            i += f.count
        else:
            out[f.name] = vals[i]
            i += 1
    return Message(spec, out, sysid, compid, seq)


class Parser:
    """Incremental frame parser: ``feed(bytes)`` returns the messages completed so far.

    It resynchronises on garbage (looks for the next start byte), verifies the checksum with the message's CRC_EXTRA and counts
    what it dropped: ``bad_crc`` (checksum mismatch), ``unknown`` (a message id without a definition), ``skipped_bytes``.
    MAVLink 2 signatures are skipped (the signature is not verified).
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.bad_crc = 0
        self.unknown = 0
        self.skipped_bytes = 0
        self.good = 0

    def feed(self, data: bytes) -> List[Message]:
        self.buf.extend(data)
        out: List[Message] = []
        while True:
            m = self._next()
            if m is None:
                break
            if m is not False:
                out.append(m)  # type: ignore[arg-type]
        return out

    def _next(self) -> Any:
        b = self.buf
        while b and b[0] not in (STX_V1, STX_V2):
            del b[0]
            self.skipped_bytes += 1
        if not b:
            return None
        if b[0] == STX_V2:
            if len(b) < 10:
                return None
            n, incompat = b[1], b[2]
            total = 10 + n + 2 + (SIGNATURE_LEN if incompat & 1 else 0)
            if len(b) < total:
                return None
            sysid, compid, seq = b[5], b[6], b[4]
            msg_id = b[7] | (b[8] << 8) | (b[9] << 16)
            head, payload = bytes(b[1:10]), bytes(b[10 : 10 + n])
            crc_rx = b[10 + n] | (b[11 + n] << 8)
        else:
            if len(b) < 6:
                return None
            n = b[1]
            total = 6 + n + 2
            if len(b) < total:
                return None
            sysid, compid, seq, msg_id = b[3], b[4], b[2], b[5]
            head, payload = bytes(b[1:6]), bytes(b[6 : 6 + n])
            crc_rx = b[6 + n] | (b[7 + n] << 8)
        spec = BY_ID.get(msg_id)
        if spec is None:
            self.unknown += 1
            del b[0]  # the frame cannot be checked without its definition: resynchronise from the next byte
            return False
        if x25(head + payload + bytes([spec.crc_extra])) != crc_rx or n > spec.max_len:
            self.bad_crc += 1
            del b[0]
            return False
        del b[:total]
        self.good += 1
        return _decode(spec, payload, sysid, compid, seq)

    def parse(self, data: bytes) -> List[Message]:
        """All messages in ``data`` (a complete buffer)."""
        return self.feed(data)
