"""MAVLink telemetry logs (``.tlog``) in and out, and their conversion to and from Aevion run archives.

A ``.tlog`` (Mission Planner, QGroundControl) is a sequence of frames, each preceded by an 8-byte big-endian timestamp in microseconds.
:func:`read_tlog` yields ``(t_us, Message)`` pairs and :func:`write_tlog` writes them. :class:`TelemetryLog` collects the state
messages (``ATTITUDE``, ``ATTITUDE_QUATERNION``, ``LOCAL_POSITION_NED``, ``GLOBAL_POSITION_INT``) and
:meth:`TelemetryLog.to_archive` resamples them onto one time base, so a real flight can be replayed, analysed or compared with a
simulated one. :func:`archive_to_tlog` converts the other way.

Frames: MAVLink is NED/FRD, Aevion ENU/FLU. Vectors map ``(x, y, z) -> (y, x, -z)``; Euler angles keep roll, negate pitch and use
``yaw_enu = π/2 - yaw_ned``; body rates map ``(p, q, r) -> (p, -q, -r)``.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from ..core import rotation as rot
from ..core.backend import NumpyBackend
from ..datasets.archive import RunArchive
from ..errors import ArchiveError
from .mavlink import MESSAGES, SIGNATURE_LEN, STX_V1, STX_V2, Message, Parser, pack

__all__ = ["read_tlog", "write_tlog", "TelemetryLog", "archive_to_tlog", "ned_to_enu", "enu_to_ned"]

B = NumpyBackend()
EARTH_R = 6378137.0


def ned_to_enu(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    return ned_to_enu(v)  # the map is its own inverse


def write_tlog(path: Any, messages: Iterable[Tuple[float, Message | bytes]]) -> Path:
    """Write ``(t_seconds, message_or_frame)`` pairs; the timestamp is stored in microseconds, big-endian."""
    p = Path(path)
    with open(p, "wb") as fh:
        for t, m in messages:
            frame = m if isinstance(m, (bytes, bytearray)) else pack(m.name, m.sysid, m.compid, m.seq, **m.fields)
            fh.write(struct.pack(">Q", round(t * 1e6)) + bytes(frame))
    return p


def _frame_length(buf: bytes, i: int) -> int:
    """Length of the frame starting at ``buf[i]`` (0 when it cannot be determined)."""
    if i >= len(buf):
        return 0
    if buf[i] == STX_V2:
        if i + 3 > len(buf):
            return 0
        return 10 + buf[i + 1] + 2 + (SIGNATURE_LEN if buf[i + 2] & 1 else 0)
    if buf[i] == STX_V1:
        if i + 2 > len(buf):
            return 0
        return 6 + buf[i + 1] + 2
    return 0


def read_tlog(path: Any) -> Iterator[Tuple[float, Message]]:
    """Yield ``(t_seconds, Message)`` for every valid frame of a ``.tlog`` (frames with a bad checksum or an unknown message are skipped)."""
    data = Path(path).read_bytes()
    i = 0
    while i + 8 < len(data):
        t_us = struct.unpack(">Q", data[i : i + 8])[0]
        n = _frame_length(data, i + 8)
        if n == 0 or i + 8 + n > len(data):
            i += 1  # resynchronise inside damaged data
            continue
        msgs = Parser().feed(data[i + 8 : i + 8 + n])
        if msgs:
            yield t_us * 1e-6, msgs[0]
            i += 8 + n
        else:
            i += 1


@dataclass
class TelemetryLog:
    """State time series collected from MAVLink messages (in MAVLink's own NED/FRD conventions)."""

    attitude: List[Tuple[float, float, float, float, float, float, float]] = field(default_factory=list)  # t, roll, pitch, yaw, p, q, r
    local: List[Tuple[float, float, float, float, float, float, float]] = field(default_factory=list)  # t, x, y, z, vx, vy, vz (NED)
    global_: List[Tuple[float, float, float, float, float]] = field(default_factory=list)  # t, lat, lon, alt, rel_alt (deg, deg, m, m)
    heartbeats: int = 0
    other: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_messages(cls, msgs: Iterable[Tuple[float, Message]]) -> "TelemetryLog":
        log = cls()
        for _t, m in msgs:
            f = m.fields
            if m.name == "ATTITUDE":
                log.attitude.append((f["time_boot_ms"] * 1e-3, f["roll"], f["pitch"], f["yaw"], f["rollspeed"], f["pitchspeed"], f["yawspeed"]))
            elif m.name == "ATTITUDE_QUATERNION":
                w, x, y, z = f["q1"], f["q2"], f["q3"], f["q4"]
                roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
                pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                log.attitude.append((f["time_boot_ms"] * 1e-3, roll, pitch, yaw, f["rollspeed"], f["pitchspeed"], f["yawspeed"]))
            elif m.name == "LOCAL_POSITION_NED":
                log.local.append((f["time_boot_ms"] * 1e-3, f["x"], f["y"], f["z"], f["vx"], f["vy"], f["vz"]))
            elif m.name == "GLOBAL_POSITION_INT":
                log.global_.append((f["time_boot_ms"] * 1e-3, f["lat"] * 1e-7, f["lon"] * 1e-7, f["alt"] * 1e-3, f["relative_alt"] * 1e-3))
            elif m.name == "HEARTBEAT":
                log.heartbeats += 1
            else:
                log.other[m.name] = log.other.get(m.name, 0) + 1
        return log

    @classmethod
    def from_tlog(cls, path: Any) -> "TelemetryLog":
        return cls.from_messages(read_tlog(path))

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Time series in Aevion's ENU/FLU conventions on the time base of the position messages (attitude is interpolated)."""
        if not self.local:
            raise ArchiveError("the log has no LOCAL_POSITION_NED messages to take the time base from")
        L = np.array(self.local, float)
        t = L[:, 0]
        pos = ned_to_enu(L[:, 1:4])
        vel = ned_to_enu(L[:, 4:7])
        out: Dict[str, np.ndarray] = dict(t=t, pos=pos, vel=vel)
        if self.attitude:
            A = np.array(self.attitude, float)
            ta = A[:, 0]
            roll = np.interp(t, ta, A[:, 1])
            pitch = np.interp(t, ta, A[:, 2])
            yaw = np.interp(t, ta, np.unwrap(A[:, 3]))
            rates = np.stack([np.interp(t, ta, A[:, 4 + k]) for k in range(3)], axis=1)
            out["euler"] = np.stack([roll, -pitch, math.pi / 2 - yaw], axis=1)  # (roll, pitch, yaw) of the ENU/FLU convention
            out["quat"] = rot.from_euler(B, out["euler"][:, 0], out["euler"][:, 1], out["euler"][:, 2])
            out["omega"] = rates * np.array([1.0, -1.0, -1.0])
        return out

    def to_archive(self, name: str = "telemetry") -> RunArchive:
        """A single-vehicle :class:`~aevion.datasets.archive.RunArchive` (channels ``t, pos, vel, quat, omega``)."""
        a = self.to_arrays()
        ch: Dict[str, Any] = dict(t=a["t"], pos=a["pos"][:, None, :], vel=a["vel"][:, None, :])
        if "quat" in a:
            ch["quat"] = a["quat"][:, None, :]
            ch["omega"] = a["omega"][:, None, :]
        meta = dict(name=name, source="telemetry log", complete=True, n_heartbeats=self.heartbeats, other_messages=dict(self.other))
        return RunArchive(channels=ch, meta=meta)


def archive_to_tlog(
    archive: RunArchive, path: Any, vehicle: int = 0, rate_hz: float = 10.0, sysid: int = 1, origin: Optional[Tuple[float, float, float]] = None
) -> Path:
    """Write an Aevion run as a ``.tlog``: HEARTBEAT, ATTITUDE, LOCAL_POSITION_NED and GLOBAL_POSITION_INT at ``rate_hz``.

    ``origin`` is ``(lat, lon, alt)`` of the world's (0, 0, 0); defaults to the world's origin in the archive's metadata, else (0, 0, 0)."""
    t = np.asarray(archive["t"], float)
    pos = np.asarray(archive["pos"])[:, vehicle]
    vel = np.asarray(archive["vel"])[:, vehicle]
    q = np.asarray(archive["quat"])[:, vehicle]
    om = np.asarray(archive["omega"])[:, vehicle] if "omega" in archive else np.zeros_like(pos)
    if origin is None:
        o = (archive.meta.get("config") or {}).get("environment", {}).get("origin") if isinstance(archive.meta.get("config"), dict) else None
        origin = (o["lat"], o["lon"], o.get("alt", 0.0)) if o else (0.0, 0.0, 0.0)
    lat0, lon0, alt0 = origin
    eul = rot.to_euler(B, q)  # ENU/FLU Euler
    idx = np.unique(np.searchsorted(t, np.arange(t[0], t[-1] + 1e-9, 1.0 / rate_hz)).clip(0, len(t) - 1))

    def stream() -> Iterator[Tuple[float, Message]]:
        seq = 0
        for _k, i in enumerate(idx):
            ms = round((t[i] - t[0]) * 1e3) & 0xFFFFFFFF
            ned_p, ned_v = enu_to_ned(pos[i]), enu_to_ned(vel[i])
            lat = lat0 + math.degrees(ned_p[0] / EARTH_R)
            lon = lon0 + math.degrees(ned_p[1] / (EARTH_R * math.cos(math.radians(lat0))))
            frames: List[Tuple[str, Dict[str, Any]]] = [
                ("HEARTBEAT", dict(type=2, autopilot=8, base_mode=0x81, system_status=4, mavlink_version=3)),
                (
                    "ATTITUDE",
                    dict(
                        time_boot_ms=ms,
                        roll=float(eul[i, 0]),
                        pitch=float(-eul[i, 1]),
                        yaw=float(math.pi / 2 - eul[i, 2]),
                        rollspeed=float(om[i, 0]),
                        pitchspeed=float(-om[i, 1]),
                        yawspeed=float(-om[i, 2]),
                    ),
                ),
                (
                    "LOCAL_POSITION_NED",
                    dict(
                        time_boot_ms=ms,
                        x=float(ned_p[0]),
                        y=float(ned_p[1]),
                        z=float(ned_p[2]),
                        vx=float(ned_v[0]),
                        vy=float(ned_v[1]),
                        vz=float(ned_v[2]),
                    ),
                ),
                (
                    "GLOBAL_POSITION_INT",
                    dict(
                        time_boot_ms=ms,
                        lat=round(lat * 1e7),
                        lon=round(lon * 1e7),
                        alt=round((alt0 + pos[i, 2]) * 1e3),
                        relative_alt=round(pos[i, 2] * 1e3),
                        vx=round(ned_v[0] * 100),
                        vy=round(ned_v[1] * 100),
                        vz=round(ned_v[2] * 100),
                        hdg=round(math.degrees(math.pi / 2 - eul[i, 2]) % 360 * 100),
                    ),
                ),
            ]
            for name, f in frames:
                yield float(t[i] - t[0]), Message(MESSAGES[name], f, sysid, 1, seq & 0xFF)
                seq += 1

    return write_tlog(path, stream())
