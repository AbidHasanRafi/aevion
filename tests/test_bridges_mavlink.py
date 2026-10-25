"""The MAVLink codec, telemetry logs and PX4 ULog files - against published constants and round trips."""

import math
import struct

import numpy as np
import pytest

from aevion.bridges import mavlink as ml
from aevion.bridges.mavlink import BY_ID, MESSAGES, MavlinkError, Message, Parser, pack, x25
from aevion.bridges.telemetry import TelemetryLog, archive_to_tlog, enu_to_ned, ned_to_enu, read_tlog, write_tlog
from aevion.bridges.ulog import ULog, matrix_to_quat, ned_frd_to_enu_flu, write_ulog
from aevion.core import rotation as rot
from aevion.core.backend import NumpyBackend
from aevion.errors import ArchiveError

B = NumpyBackend()


# ------------------------------------------------------------------------------------------------ the checksum and the definitions
def test_the_checksum_matches_the_published_check_value_and_the_derived_crc_extras_match_the_mavlink_constants():
    assert x25(b"123456789") == 0x6F91  # the CRC-16/MCRF4XX (X.25 without the final inversion) check value
    published = {  # CRC_EXTRA of common.xml: derived here from the field definitions, so this checks both the algorithm and the layouts
        "HEARTBEAT": 50,
        "SYS_STATUS": 124,
        "SYSTEM_TIME": 137,
        "ATTITUDE": 39,
        "ATTITUDE_QUATERNION": 246,
        "LOCAL_POSITION_NED": 185,
        "GLOBAL_POSITION_INT": 104,
        "COMMAND_LONG": 152,
        "SET_POSITION_TARGET_LOCAL_NED": 143,
        "HIL_ACTUATOR_CONTROLS": 47,
        "HIGHRES_IMU": 93,
        "HIL_SENSOR": 108,
        "HIL_GPS": 124,
        "HIL_STATE_QUATERNION": 4,
        "STATUSTEXT": 83,
    }
    for name, want in published.items():
        assert MESSAGES[name].crc_extra == want, name
    assert len(BY_ID) == len(MESSAGES)


def test_the_wire_order_sorts_fields_by_size_and_keeps_extensions_last():
    hb = MESSAGES["HEARTBEAT"]
    assert [f.name for f in hb.wire] == ["custom_mode", "type", "autopilot", "base_mode", "system_status", "mavlink_version"] and hb.base_len == 9
    att = MESSAGES["ATTITUDE"]
    assert [f.name for f in att.wire][0] == "time_boot_ms" and att.base_len == 28
    assert [f.name for f in MESSAGES["HIL_GPS"].wire][-2:] == ["id", "yaw"] and MESSAGES["HIL_GPS"].base_len == 36  # extensions follow
    hil = MESSAGES["HIL_ACTUATOR_CONTROLS"]
    assert [f.name for f in hil.wire] == [
        "time_usec",
        "flags",
        "controls",
        "mode",
    ] and hil.max_len == 81  # 8-byte fields first, then the 4-byte array, then bytes


# ------------------------------------------------------------------------------------------------ encode / decode
def sample_value(f, rng):
    if f.type == "char":
        return "hello" if f.count else b"A"
    if f.type in ("float", "double"):
        v = (rng.normal(size=f.count) if f.count else rng.normal()) if True else 0
        if f.type == "float":
            v = np.float32(v).astype(float)
        return [float(x) for x in v] if f.count else float(v)
    lo, hi = {
        "int8_t": (-128, 127),
        "uint8_t": (1, 255),
        "int16_t": (-30000, 30000),
        "uint16_t": (1, 60000),
        "int32_t": (-2_000_000_000, 2_000_000_000),
        "uint32_t": (1, 4_000_000_000),
        "int64_t": (-(2**60), 2**60),
        "uint64_t": (1, 2**62),
    }[f.type]
    return [int(x) for x in rng.integers(lo, hi, f.count)] if f.count else int(rng.integers(lo, hi))


@pytest.mark.parametrize("v2", [True, False])
def test_every_message_survives_a_round_trip(v2):
    rng = np.random.default_rng(0)
    for name, spec in MESSAGES.items():
        if not v2 and spec.id > 255:
            continue
        values = {f.name: sample_value(f, rng) for f in spec.fields if v2 or not f.extension}
        frame = pack(name, sysid=7, compid=3, seq=42, v2=v2, **values)
        (m,) = Parser().feed(frame)
        assert m.name == name and (m.sysid, m.compid, m.seq) == (7, 3, 42)
        for k, v in values.items():
            got = m.fields[k]
            assert got == pytest.approx(v) if not isinstance(v, str) else got == v, (name, k)


def test_mavlink_2_trims_trailing_zeros_and_mavlink_1_does_not():
    frame = pack("HEARTBEAT")  # everything zero: a single payload byte is kept
    assert frame[1] == 1 and len(frame) == 1 + 9 + 1 + 2
    (m,) = Parser().feed(frame)
    assert m.fields["custom_mode"] == 0 and m.fields["mavlink_version"] == 0  # the trimmed zeros are restored on decoding
    v1 = pack("HEARTBEAT", v2=False)
    assert v1[0] == ml.STX_V1 and v1[1] == 9 and len(v1) == 6 + 9 + 2
    a = pack("ATTITUDE", time_boot_ms=5, roll=1.0)  # the zero floats after `roll` are not sent
    assert a[1] == 8 and Parser().feed(a)[0].fields["pitch"] == 0.0
    ext = pack("HIL_GPS", lat=10, fix_type=3, yaw=4500)  # an extension that is set is sent, in the right place
    assert Parser().feed(ext)[0].fields["yaw"] == 4500 and ext[1] == 36 + 1 + 2  # the extensions (id, yaw) follow the 36 base bytes


def test_the_parser_resynchronises_counts_errors_and_handles_split_input():
    frames = [pack("ATTITUDE", time_boot_ms=i, roll=0.1 * i) for i in range(5)]
    stream = b"\x00\xaa\x13garbage" + frames[0] + b"\x01\x02" + frames[1] + frames[2][:7]
    p = Parser()
    out = p.feed(stream)
    assert [m.fields["time_boot_ms"] for m in out] == [0, 1] and p.skipped_bytes >= 8 and p.good == 2
    out += p.feed(frames[2][7:] + frames[3] + frames[4])  # the rest arrives later: nothing is lost
    assert [m.fields["time_boot_ms"] for m in out] == [0, 1, 2, 3, 4]
    byte_by_byte = Parser()
    got = []
    for b in b"".join(frames):
        got += byte_by_byte.feed(bytes([b]))
    assert [m.fields["time_boot_ms"] for m in got] == [0, 1, 2, 3, 4]
    bad = bytearray(frames[0])
    bad[12] ^= 0xFF  # corrupt the payload
    p2 = Parser()
    survivors = p2.feed(bytes(bad) + frames[1])
    assert p2.bad_crc >= 1 and [m.fields["time_boot_ms"] for m in survivors] == [1]  # the corrupt frame is dropped, the next one is read


def test_a_frame_from_a_different_definition_is_rejected_by_the_crc_extra():
    spec = MESSAGES["ATTITUDE"]
    frame = bytearray(pack("ATTITUDE", time_boot_ms=1, roll=0.5, pitch=0.25))
    head, payload = bytes(frame[1:10]), bytes(frame[10:-2])
    forged = (
        bytes([ml.STX_V2]) + head + payload + struct.pack("<H", x25(head + payload + bytes([(spec.crc_extra + 1) & 0xFF])))
    )  # a peer with another layout
    p = Parser()
    assert p.feed(forged) == [] and p.bad_crc == 1


def test_signed_frames_are_skipped_over_and_unknown_messages_are_counted():
    base = bytearray(pack("ATTITUDE", time_boot_ms=3, roll=1.0))
    base[2] |= 1  # incompat flag: a 13-byte signature follows the checksum
    head = bytes(base[1:10])
    payload = bytes(base[10:-2])
    crc = x25(head + payload + bytes([MESSAGES["ATTITUDE"].crc_extra]))
    signed = bytes([ml.STX_V2]) + head + payload + struct.pack("<H", crc) + bytes(13)
    p = Parser()
    out = p.feed(signed + pack("HEARTBEAT", type=2))
    assert [m.name for m in out] == ["ATTITUDE", "HEARTBEAT"]
    unknown = bytes([ml.STX_V2, 1, 0, 0, 0, 1, 1, 0xE7, 0x03, 0x00, 0xAB, 0, 0])  # message id 999: no definition, no way to check it
    q = Parser()
    assert q.feed(unknown + pack("HEARTBEAT")) and q.unknown == 1


def test_encoding_errors_are_reported():
    with pytest.raises(MavlinkError, match="unknown message"):
        pack("NOPE")
    with pytest.raises(MavlinkError, match="no field"):
        pack("ATTITUDE", bananas=1)
    with pytest.raises(MavlinkError, match="cannot encode"):
        pack("HEARTBEAT", type=1000)  # does not fit a byte
    ml.define("TEST_BIG_ID", 300, ("uint8_t", "a"))
    with pytest.raises(MavlinkError, match="below 256"):
        pack("TEST_BIG_ID", v2=False)
    assert Parser().feed(pack("TEST_BIG_ID", a=9))[0].fields["a"] == 9  # three-byte message ids exist only in MAVLink 2
    with pytest.raises(MavlinkError, match="unknown field type"):
        ml.define("BAD", 4000, ("complex", "z"))
    assert Parser().feed(pack("STATUSTEXT", severity=6, text="landing gear down"))[0].fields["text"] == "landing gear down"


# ------------------------------------------------------------------------------------------------ telemetry logs
def test_tlog_files_store_big_endian_microsecond_timestamps_and_survive_damage(tmp_path):
    msgs = [
        (
            0.5 * i,
            Message(
                MESSAGES["ATTITUDE"],
                dict(time_boot_ms=500 * i, roll=0.1 * i, pitch=0.0, yaw=1.0, rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0),
                1,
                1,
                i,
            ),
        )
        for i in range(6)
    ]
    path = write_tlog(tmp_path / "a.tlog", msgs)
    raw = path.read_bytes()
    assert struct.unpack(">Q", raw[:8])[0] == 0 and raw[8] == ml.STX_V2
    back = list(read_tlog(path))
    assert [round(t, 3) for t, _ in back] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5] and back[3][1].fields["roll"] == pytest.approx(0.3, abs=1e-6)
    damaged = bytearray(raw)
    damaged[len(raw) // 2] ^= 0xFF
    assert 4 <= len(list(read_tlog(_write(tmp_path / "b.tlog", bytes(damaged))))) <= 6  # a flipped byte costs at most a frame
    assert (
        list(read_tlog(_write(tmp_path / "c.tlog", raw[: len(raw) - 5]))) and len(list(read_tlog(_write(tmp_path / "c.tlog", raw[:-5])))) == 5
    )  # truncated last frame


def _write(path, data):
    path.write_bytes(data)
    return path


def test_ned_enu_conversions_are_involutions_and_match_the_compass():
    v = np.array([[3.0, -4.0, 5.0]])  # north 3, east -4, down 5
    assert np.allclose(ned_to_enu(v), [[-4.0, 3.0, -5.0]]) and np.allclose(enu_to_ned(ned_to_enu(v)), v)
    # a vehicle facing north (yaw 0 in MAVLink) faces +y (yaw 90 deg) in ENU; one facing east (MAVLink yaw 90 deg) has ENU yaw 0
    log = TelemetryLog()
    log.local = [(0.0, 0, 0, 0, 0, 0, 0), (1.0, 0, 0, 0, 0, 0, 0)]
    log.attitude = [(0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0), (1.0, 0.1, 0.2, math.pi / 2, 0.0, 0.0, 0.0)]
    a = log.to_arrays()
    assert a["euler"][0, 2] == pytest.approx(math.pi / 2) and a["euler"][1, 2] == pytest.approx(0.0, abs=1e-12)
    assert a["euler"][0, 0] == pytest.approx(0.1) and a["euler"][0, 1] == pytest.approx(
        -0.2
    )  # roll unchanged, nose-up pitch becomes negative FLU pitch


def test_a_simulated_flight_round_trips_through_a_tlog(tmp_path):
    from aevion.engine import Simulation

    sim = Simulation(
        {
            "duration": 8.0,
            "trajectory": {"type": "circle", "params": {"center": [0, 0, 6.0], "radius": 4.0, "omega": 0.5}},
            "logging": {"record_every": 4},
        }
    )
    a = sim.run()
    path = archive_to_tlog(a, tmp_path / "flight.tlog", rate_hz=20.0, origin=(47.3769, 8.5417, 408.0))
    log = TelemetryLog.from_tlog(path)
    assert log.heartbeats > 10 and len(log.local) == len(log.attitude) and len(log.global_) == len(log.local)
    b = log.to_archive("replayed")
    # compare on the replayed time base
    tt = np.asarray(b["t"])
    pos = np.stack([np.interp(tt, a.t - a.t[0], np.asarray(a["pos"])[:, 0, k]) for k in range(3)], axis=1)
    assert np.abs(np.asarray(b["pos"])[:, 0] - pos).max() < 1e-3  # positions survive float32 and millisecond stamps
    q_ref = np.stack([np.interp(tt, a.t - a.t[0], np.asarray(a["quat"])[:, 0, k]) for k in range(4)], axis=1)
    q_ref /= np.linalg.norm(q_ref, axis=1, keepdims=True)
    assert rot.angle_between(B, np.asarray(b["quat"])[:, 0], q_ref).max() < 5e-3  # attitude within 0.3 degrees
    lat0 = log.global_[0][1]
    assert lat0 == pytest.approx(47.3769 + math.degrees(a["pos"][0, 0, 1] / 6378137.0), abs=1e-5)  # the geodetic position follows the origin
    assert b.meta["source"] == "telemetry log" and b.n_vehicles == 1


# ------------------------------------------------------------------------------------------------ ULog
FORMAT = "uint64_t timestamp;float x;float y;float z;float vx;float vy;float vz"
ATT = "uint64_t timestamp;float[4] q"


def make_log(tmp_path, n=50):
    t = (np.arange(n) * 20000 + 1_000_000).astype(np.uint64)
    cols = {
        "timestamp": t,
        "x": np.linspace(0, 10, n),
        "y": np.linspace(0, 5, n),
        "z": -np.linspace(2, 4, n),
        "vx": np.ones(n),
        "vy": np.full(n, 0.5),
        "vz": np.full(n, -0.1),
    }
    q_ned = rot.from_euler(B, np.zeros(n), np.zeros(n), np.linspace(0, 1.0, n))  # FRD->NED: yaw about down; a clockwise heading
    att = {"timestamp": t, "q": np.asarray(q_ned, np.float32)}
    path = write_ulog(
        tmp_path / "f.ulg",
        {"vehicle_local_position": (FORMAT, cols), "vehicle_attitude": (ATT, att)},
        parameters={"MPC_XY_VEL_MAX": 12.0},
        info={"sys_name": "PX4"},
        start_us=1_000_000,
    )
    return path, cols, att


def test_ulog_columns_parameters_and_info_survive_a_round_trip(tmp_path):
    path, cols, _ = make_log(tmp_path)
    log = ULog(path)
    assert (
        log.version == 1
        and ("vehicle_local_position", 0) in log.topics()
        and log.parameters["MPC_XY_VEL_MAX"] == pytest.approx(12.0)
        and log.info["sys_name"] == "PX4"
    )
    d = log.data("vehicle_local_position")
    assert (
        np.array_equal(d["timestamp"], cols["timestamp"])
        and np.allclose(d["x"], cols["x"], atol=1e-5)
        and np.allclose(d["vz"], cols["vz"], atol=1e-6)
    )
    assert log.data("vehicle_attitude")["q"].shape == (50, 4)
    assert log.duration_s == pytest.approx(0.98, abs=0.01) and not log.truncated
    with pytest.raises(KeyError):
        log.data("nope")


def test_ulog_nested_formats_padding_arrays_and_strings(tmp_path):
    def msg(t, body):
        return struct.pack("<HB", len(body), ord(t)) + body

    raw = bytearray(ml_ulog_header())
    raw += msg("F", b"vec3:float x;float y;float z;")
    raw += msg("F", b"sample:uint64_t timestamp;vec3 pos;uint8_t[3] _padding0;int16_t[2] counts;bool ok;")
    raw += msg("A", struct.pack("<BH", 0, 5) + b"sample")
    raw += msg("A", struct.pack("<BH", 1, 6) + b"sample")  # a second instance of the same topic
    for ts, base, mid in ((10, 1.0, 5), (20, 2.0, 5), (30, 3.0, 6)):
        raw += msg("D", struct.pack("<H", mid) + struct.pack("<Qfff3xhh?", ts, base, base + 1, base + 2, 7, -8, True))
    raw += msg("L", struct.pack("<BQ", 6, 15) + b"armed")
    raw += msg("O", struct.pack("<H", 40))
    log = ULog(bytes(raw))
    d = log.data("sample")
    assert (
        list(d["timestamp"]) == [10, 20] and np.allclose(d["pos.y"], [2.0, 3.0]) and d["counts"].tolist() == [[7, -8], [7, -8]] and bool(d["ok"][0])
    )
    assert log.data("sample", 1)["pos.z"].tolist() == [5.0]  # instance 1
    assert log.messages == [(6, 15, "armed")] and log.dropouts == [(0, 40)] and ("sample", 1) in log.topics()


def ml_ulog_header():
    return b"\x55\x4c\x6f\x67\x01\x12\x35\x01" + struct.pack("<Q", 0)


def test_truncated_and_foreign_files_are_handled(tmp_path):
    path, cols, _ = make_log(tmp_path)
    raw = path.read_bytes()
    log = ULog(raw[: len(raw) - 40])  # a log cut short (a crash): everything complete is kept
    assert log.truncated and 40 <= len(log.data("vehicle_local_position")["x"]) < 50
    with pytest.raises(ArchiveError, match="bad magic"):
        ULog(b"not a log at all, just text of some length")

    def msg(t, body):
        return struct.pack("<HB", len(body), ord(t)) + body

    bad = bytearray(ml_ulog_header())
    bad += msg("F", b"t:wombat_t foo;")
    bad += msg("A", struct.pack("<BH", 0, 1) + b"t")
    bad += msg("D", struct.pack("<H", 1) + b"\x00\x00")
    with pytest.raises(ArchiveError, match="unknown ULog type"):
        ULog(bytes(bad)).data("t")


def test_px4_attitudes_convert_to_the_aevion_convention_and_the_archive_is_in_enu(tmp_path):
    # PX4: FRD -> NED quaternion. Level and facing north is the identity; in ENU/FLU that is a yaw of +90 degrees.
    q = ned_frd_to_enu_flu(np.array([[1.0, 0, 0, 0]]))
    assert rot.angle_between(B, q, rot.from_euler(B, np.array([0.0]), np.array([0.0]), np.array([math.pi / 2]))) < 1e-9
    # facing east (NED yaw +90 deg) is ENU yaw 0; a nose-up pitch of 20 degrees (FRD) is a negative FLU Euler pitch
    q_east_up = rot.from_euler(B, np.array([0.0]), np.array([math.radians(20.0)]), np.array([math.pi / 2]))
    conv = ned_frd_to_enu_flu(q_east_up)
    eul = rot.to_euler(B, conv)[0]
    assert (
        eul[2] == pytest.approx(0.0, abs=1e-9) and eul[1] == pytest.approx(-math.radians(20.0), abs=1e-9) and eul[0] == pytest.approx(0.0, abs=1e-9)
    )
    qs = rot.qnormalize(B, np.random.default_rng(0).normal(size=(5, 4)))
    assert np.allclose(
        rot.to_matrix(B, matrix_to_quat(rot.to_matrix(B, qs))), rot.to_matrix(B, qs), atol=1e-9
    )  # rotation matrix -> quaternion -> matrix
    path, cols, att = make_log(tmp_path)
    a = ULog(path).to_archive()
    assert a.n_steps == 50 and a.n_vehicles == 1
    assert (
        np.allclose(a["pos"][:, 0, 0], cols["y"], atol=1e-5)
        and np.allclose(a["pos"][:, 0, 1], cols["x"], atol=1e-5)
        and np.allclose(a["pos"][:, 0, 2], -cols["z"], atol=1e-5)
    )  # NED -> ENU
    yaw_enu = rot.to_euler(B, a["quat"][:, 0])[:, 2]
    assert (
        yaw_enu[0] == pytest.approx(math.pi / 2, abs=1e-6) and yaw_enu[-1] < yaw_enu[0]
    )  # a clockwise (increasing NED yaw) turn is a decreasing ENU yaw
