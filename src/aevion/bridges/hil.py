"""Hardware- and software-in-the-loop over MAVLink: the simulator provides sensors and the aircraft, an autopilot flies it.

:class:`HilHook` closes the loop each physics step. It turns the vehicle's sensor readings into ``HIL_SENSOR`` (IMU, magnetometer,
barometer) and ``HIL_GPS`` messages, sends them over a :class:`~aevion.bridges.transport.Transport`, reads the autopilot's
``HIL_ACTUATOR_CONTROLS`` (motor outputs in ``[0, 1]``) and uses them as the rotor commands.

The interface follows the MAVLink definitions and is tested against an in-process autopilot (:class:`ReferenceAutopilot`) over a
loopback transport. It has **not been run against a real PX4 or ArduPilot build**: check the motor mapping and message rates of your
firmware's HIL mode first.

Frames: the autopilot uses NED/FRD, Aevion ENU/FLU; conversions are in :func:`sensor_messages` and :func:`decode_sensors`. Units follow
MAVLink (m/s², rad/s, gauss, hPa, degrees × 10⁷, millimetres, cm/s).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import NumpyBackend
from ..engine.hooks import Hook
from .mavlink import Message, Parser, pack
from .transport import Transport

__all__ = ["HilHook", "ReferenceAutopilot", "sensor_messages", "decode_sensors", "isa_pressure_hpa"]

B = NumpyBackend()
EARTH_R = 6378137.0
FRD = np.array([1.0, -1.0, -1.0])
FIELDS_UPDATED = 0x1BFF  # accel, gyro, mag, absolute pressure, pressure altitude, temperature


def isa_pressure_hpa(alt_m: float) -> float:
    """Static pressure of the International Standard Atmosphere at ``alt_m`` (troposphere) [hPa]."""
    return 1013.25 * (1.0 - 2.25577e-5 * alt_m) ** 5.25588


def _geodetic(origin: Tuple[float, float, float], enu: np.ndarray) -> Tuple[float, float, float]:
    lat0, lon0, alt0 = origin
    lat = lat0 + math.degrees(enu[1] / EARTH_R)
    lon = lon0 + math.degrees(enu[0] / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon, alt0 + enu[2]


def sensor_messages(
    t_s: float,
    accel_flu: np.ndarray,
    gyro_flu: np.ndarray,
    mag_flu_ut: Optional[np.ndarray],
    altitude_m: float,
    pressure_pa: Optional[float] = None,
    gps: Optional[Dict[str, Any]] = None,
    origin: Tuple[float, float, float] = (47.3769, 8.5417, 408.0),
    sysid: int = 1,
    seq: int = 0,
) -> List[bytes]:
    """The frames for one sensor epoch: a ``HIL_SENSOR`` and, when ``gps`` (``pos`` ENU m, ``vel`` ENU m/s, ``fix``, ``hdop``,
    ``n_sat``) is given, a ``HIL_GPS``."""
    a, g = np.asarray(accel_flu, float) * FRD, np.asarray(gyro_flu, float) * FRD
    m = np.zeros(3) if mag_flu_ut is None else np.asarray(mag_flu_ut, float) * FRD * 0.01  # microtesla -> gauss
    p_hpa = (pressure_pa / 100.0) if pressure_pa is not None else isa_pressure_hpa(origin[2] + altitude_m)
    us = round(t_s * 1e6)
    frames = [
        pack(
            "HIL_SENSOR",
            sysid=sysid,
            seq=seq,
            time_usec=us,
            xacc=float(a[0]),
            yacc=float(a[1]),
            zacc=float(a[2]),
            xgyro=float(g[0]),
            ygyro=float(g[1]),
            zgyro=float(g[2]),
            xmag=float(m[0]),
            ymag=float(m[1]),
            zmag=float(m[2]),
            abs_pressure=float(p_hpa),
            diff_pressure=0.0,
            pressure_alt=float(origin[2] + altitude_m),
            temperature=15.0,
            fields_updated=FIELDS_UPDATED,
        )
    ]
    if gps is not None:
        pos, vel = np.asarray(gps["pos"], float), np.asarray(gps["vel"], float)
        lat, lon, alt = _geodetic(origin, pos)
        vn, ve, vd = vel[1], vel[0], -vel[2]
        frames.append(
            pack(
                "HIL_GPS",
                sysid=sysid,
                seq=seq + 1,
                time_usec=us,
                fix_type=3 if gps.get("fix", 1.0) > 0.5 else 1,
                lat=round(lat * 1e7),
                lon=round(lon * 1e7),
                alt=round(alt * 1e3),
                eph=int(min(65535, max(0, round(100.0 * gps.get("hdop", 1.0))))),
                epv=int(min(65535, max(0, round(150.0 * gps.get("hdop", 1.0))))),
                vel=round(100.0 * math.hypot(vn, ve)),
                vn=round(vn * 100),
                ve=round(ve * 100),
                vd=round(vd * 100),
                cog=round(math.degrees(math.atan2(ve, vn)) % 360 * 100),
                satellites_visible=int(gps.get("n_sat", 10)),
            )
        )
    return frames


def decode_sensors(msg: Message, origin: Tuple[float, float, float] = (47.3769, 8.5417, 408.0)) -> Dict[str, Any]:
    """Inverse of :func:`sensor_messages` for one message: Aevion-frame (ENU/FLU) values from a ``HIL_SENSOR`` or ``HIL_GPS``."""
    f = msg.fields
    if msg.name == "HIL_SENSOR":
        return dict(
            kind="imu",
            t=f["time_usec"] * 1e-6,
            accel=np.array([f["xacc"], f["yacc"], f["zacc"]]) * FRD,
            gyro=np.array([f["xgyro"], f["ygyro"], f["zgyro"]]) * FRD,
            field=np.array([f["xmag"], f["ymag"], f["zmag"]]) * FRD * 100.0,
            altitude=f["pressure_alt"] - origin[2],
        )
    if msg.name == "HIL_GPS":
        lat0, lon0, alt0 = origin
        north = math.radians(f["lat"] * 1e-7 - lat0) * EARTH_R
        east = math.radians(f["lon"] * 1e-7 - lon0) * EARTH_R * math.cos(math.radians(lat0))
        return dict(
            kind="gps",
            t=f["time_usec"] * 1e-6,
            pos=np.array([east, north, f["alt"] * 1e-3 - alt0]),
            vel=np.array([f["ve"] * 0.01, f["vn"] * 0.01, -f["vd"] * 0.01]),
            fix=1.0 if f["fix_type"] >= 3 else 0.0,
        )
    raise ValueError(f"not a sensor message: {msg.name}")


class HilHook(Hook):
    """Feed sensors to an autopilot and fly the simulated vehicle with its motor outputs (see the module documentation).

    ``mapping`` converts a normalised motor output ``c`` to a rotor-speed command: ``"thrust"`` (default, ``ω = ω_max sqrt(c)``,
    outputs are proportional to thrust) or ``"speed"`` (``ω = ω_max c``). ``motor_map`` reorders outputs to rotors. With
    ``require_armed`` the commands are zero until the autopilot reports the armed flag. ``wait_s`` is how long (wall time) to
    wait for each actuator message when the peer is a real autopilot (lock-step); ``0`` uses whatever has arrived.
    """

    name = "hil"
    priority = 90

    def __init__(
        self,
        transport: Transport,
        origin: Tuple[float, float, float] = (47.3769, 8.5417, 408.0),
        sysid: int = 1,
        mapping: str = "thrust",
        motor_map: Optional[Sequence[int]] = None,
        require_armed: bool = True,
        wait_s: float = 0.0,
        send_every: int = 1,
    ) -> None:
        if mapping not in ("thrust", "speed"):
            raise ValueError("mapping must be 'thrust' or 'speed'")
        self.tp, self.origin, self.sysid, self.mapping = transport, origin, sysid, mapping
        self.motor_map, self.require_armed, self.wait_s, self.every = motor_map, require_armed, float(wait_s), max(int(send_every), 1)
        self.parser = Parser()
        self.armed = False
        self.controls: Optional[np.ndarray] = None
        self.n_sent = self.n_received = 0
        self._seq = 0

    def on_reset(self, sim: Any) -> None:
        self.controls, self.armed, self._seq = None, False, 0

    def _readings(self, sim: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r, f = sim.ctx.readings or {}, sim.ctx.fresh or {}
        byk: Dict[str, Any] = {}
        for name, v in r.items():
            if v is None:
                continue
            kind = "imu" if "gyro" in v else "gps" if ("pos" in v and "vel" in v) else "baro" if "altitude" in v else "mag" if "field" in v else None
            if kind and kind not in byk:
                byk[kind] = (v, bool(f.get(name, False)))
        return byk, f

    def pre_step(self, sim: Any) -> None:
        B_ = sim.B
        if sim.k % self.every == 0:
            byk, _ = self._readings(sim)
            st = sim.state
            imu, imu_fresh = byk.get("imu", (None, True))
            if imu is not None and not imu_fresh:
                imu = False  # an IMU is configured but has no new sample this step: send nothing (the GPS may still be due)
            if imu is False:
                frames = []
                gps = byk.get("gps", (None, False))
                if gps[0] is not None and gps[1]:
                    g = gps[0]
                    msg = dict(
                        pos=B_.to_numpy(g["pos"])[0], vel=B_.to_numpy(g["vel"])[0], fix=float(B_.to_numpy(g["fix"]).min()) if "fix" in g else 1.0
                    )
                    frames = sensor_messages(sim.t, np.zeros(3), np.zeros(3), None, 0.0, None, msg, self.origin, self.sysid, self._seq)[1:]
                    self._seq += len(frames)
                for fr in frames:
                    self.tp.send(fr)
                    self.n_sent += 1
                self._poll()
                cmd = self._command(sim)
                if cmd is not None:
                    sim.ctx.cmd = B_.asarray(cmd)
                return
            if imu is not None:
                accel, gyro = B_.to_numpy(imu["accel"])[0], B_.to_numpy(imu["gyro"])[0]
            else:  # no IMU configured: use the ideal specific force and rate
                accel, gyro = B_.to_numpy(st.f_body)[0], B_.to_numpy(st.omega)[0]
            mag = byk.get("mag", (None, False))[0]
            baro = byk.get("baro", (None, False))[0]
            gps = byk.get("gps", (None, False))
            gps_msg = None
            if gps[0] is not None and gps[1]:
                g = gps[0]
                gps_msg = dict(
                    pos=B_.to_numpy(g["pos"])[0],
                    vel=B_.to_numpy(g["vel"])[0],
                    fix=float(B_.to_numpy(g["fix"]).min()) if "fix" in g else 1.0,
                    hdop=float(B_.to_numpy(g["hdop"]).mean()) if "hdop" in g else 1.0,
                    n_sat=int(B_.to_numpy(g["n_sat"]).mean()) if "n_sat" in g else 10,
                )
            alt = float(B_.to_numpy(baro["altitude"])[0]) if baro is not None else float(B_.to_numpy(st.pos)[0, 2])
            pa = float(B_.to_numpy(baro["pressure"])[0]) if (baro is not None and "pressure" in baro) else None
            frames = sensor_messages(
                sim.t, accel, gyro, None if mag is None else B_.to_numpy(mag["field"])[0], alt, pa, gps_msg, self.origin, self.sysid, self._seq
            )
            self._seq += len(frames)
            for fr in frames:
                self.tp.send(fr)
                self.n_sent += 1
        self._poll()
        if self.wait_s > 0.0 and self.controls is None:
            import time

            end = time.perf_counter() + self.wait_s
            while self.controls is None and time.perf_counter() < end:
                self._poll()
        cmd = self._command(sim)
        if cmd is not None:
            sim.ctx.cmd = B_.asarray(cmd)

    def _poll(self) -> None:
        for m in self.parser.feed(self.tp.recv()):
            if m.name == "HIL_ACTUATOR_CONTROLS":
                self.controls = np.asarray(m.fields["controls"], float)
                self.armed = bool(m.fields["mode"] & 0x80)
                self.n_received += 1

    def _command(self, sim: Any) -> Optional[np.ndarray]:
        if self.controls is None:
            return None
        M = sim.spec.n_rotors
        idx = list(self.motor_map) if self.motor_map is not None else list(range(M))
        c = np.nan_to_num(self.controls[idx], nan=0.0).clip(0.0, 1.0)
        w_max = sim.spec.w_max
        w = w_max * (np.sqrt(c) if self.mapping == "thrust" else c)
        if self.require_armed and not self.armed:
            w = np.zeros(M)
        return np.ones((sim.n, 1)) * w[None, :]


class ReferenceAutopilot:
    """An in-process stand-in for an autopilot, to exercise the bridge without PX4.

    It decodes the sensor messages, runs Aevion's estimator and geometric controller on them, and answers with
    ``HIL_ACTUATOR_CONTROLS``. Connect it to a :class:`~aevion.bridges.transport.LoopbackTransport` end (``attach``): each sensor
    message that arrives is answered synchronously. ``target`` is the hover point in the world frame.
    """

    def __init__(
        self,
        spec: Any,
        transport: Transport,
        origin: Tuple[float, float, float] = (47.3769, 8.5417, 408.0),
        target: Sequence[float] = (0.0, 0.0, 5.0),
        dt: float = 0.005,
        sysid: int = 1,
    ) -> None:
        from ..control.controllers import GeometricController
        from ..control.reference import Reference
        from ..core.constants import GravityModel
        from ..estimation import Estimator

        self.spec, self.tp, self.origin, self.dt, self.sysid = spec, transport, origin, dt, sysid
        ns = SimpleNamespace(B=B, spec=spec, n=1, world=SimpleNamespace(gravity=GravityModel("constant", g=9.80665)))
        self.est = Estimator(ns, use_mag=True)
        self.est.reset(None)
        self.ctrl = GeometricController(ns)
        self.ctrl.reset(1)
        self.ref = Reference.hover(B, 1, pos=tuple(target))
        self.parser = Parser()
        self.pending: Dict[str, Any] = {}
        self.out_seq = 0
        self.last_t = 0.0
        self.n_answered = 0
        if hasattr(transport, "on_receive"):
            transport.on_receive = self._on_bytes  # type: ignore[attr-defined]

    def _on_bytes(self, data: bytes) -> None:
        for m in self.parser.feed(data):
            if m.name not in ("HIL_SENSOR", "HIL_GPS"):
                continue
            d = decode_sensors(m, self.origin)
            if d["kind"] == "imu":
                self._step(d)
            else:
                self.pending["gps"] = dict(pos=d["pos"][None, :], vel=d["vel"][None, :], fix=np.array([d["fix"]]), t=d["t"])

    def _step(self, d: Dict[str, Any]) -> None:
        readings = {
            "imu": dict(accel=d["accel"][None, :], gyro=d["gyro"][None, :], t=d["t"]),
            "mag": dict(field=d["field"][None, :], t=d["t"]),
            "baro": dict(altitude=np.array([d["altitude"]]), t=d["t"]),
        }
        fresh = {"imu": True, "mag": True, "baro": True, "gps": False}
        if "gps" in self.pending:
            readings["gps"] = self.pending.pop("gps")
            fresh["gps"] = True
        state = self.est.update(readings, fresh)
        if not self.est.started:
            self._answer(d["t"], np.zeros(4), armed=False)
            return
        dt = d["t"] - self.last_t if self.last_t else self.dt
        self.last_t = d["t"]
        w = np.asarray(self.ctrl.act(state, self.ref, dt))[0]
        self._answer(d["t"], np.clip((w / self.spec.w_max) ** 2, 0.0, 1.0), armed=True)  # normalised thrust

    def _answer(self, t: float, c: np.ndarray, armed: bool) -> None:
        controls = [0.0] * 16
        for i, v in enumerate(c[: self.spec.n_rotors]):
            controls[i] = float(v)
        self.tp.send(
            pack(
                "HIL_ACTUATOR_CONTROLS",
                sysid=self.sysid,
                seq=self.out_seq,
                time_usec=round(t * 1e6),
                controls=controls,
                mode=0x80 if armed else 0,
                flags=0,
            )
        )
        self.out_seq += 1
        self.n_answered += 1
