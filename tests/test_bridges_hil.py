"""Hardware/software-in-the-loop over MAVLink - frame conversions, a closed loop through an in-process autopilot, transports."""

import math
import time

import numpy as np
import pytest

from aevion.bridges.hil import HilHook, ReferenceAutopilot, decode_sensors, isa_pressure_hpa, sensor_messages
from aevion.bridges.mavlink import MESSAGES, Message, Parser
from aevion.bridges.transport import LoopbackTransport, UdpTransport
from aevion.engine import Simulation
from aevion.vehicles.spec import VehicleSpec

ORIGIN = (47.3769, 8.5417, 408.0)
G = 9.80665


# ------------------------------------------------------------------------------------------------ frames and units
def test_a_level_vehicle_at_rest_reports_the_reaction_to_gravity_along_minus_z_in_frd():
    frames = sensor_messages(1.5, [0.0, 0.0, G], [0.0, 0.0, 0.0], [20.0, 0.0, -43.0], 12.0, None, None, ORIGIN)
    (m,) = Parser().feed(frames[0])
    assert m.name == "HIL_SENSOR" and m.fields["time_usec"] == 1_500_000
    assert m.fields["zacc"] == pytest.approx(-G, rel=1e-6)  # FRD z points down: the accelerometer of a level vehicle reads -g
    assert m.fields["xmag"] == pytest.approx(0.2, abs=1e-6) and m.fields["zmag"] == pytest.approx(0.43, abs=1e-6)  # microtesla -> gauss, FLU -> FRD
    assert m.fields["abs_pressure"] == pytest.approx(isa_pressure_hpa(ORIGIN[2] + 12.0), rel=1e-6) and m.fields["fields_updated"] == 0x1BFF
    assert isa_pressure_hpa(0.0) == pytest.approx(1013.25) and isa_pressure_hpa(1000.0) == pytest.approx(898.7, abs=0.5)


def test_rates_and_gps_use_ned_and_survive_the_wire_quantisation():
    gps = dict(pos=[100.0, 250.0, 30.0], vel=[2.0, -3.0, 0.5], fix=1.0, hdop=0.9, n_sat=11)
    frames = sensor_messages(2.0, [1.0, 2.0, 9.0], [0.1, 0.2, 0.3], None, 30.0, None, gps, ORIGIN)
    imu, gp = (Parser().feed(f)[0] for f in frames)
    assert (imu.fields["xgyro"], imu.fields["ygyro"], imu.fields["zgyro"]) == pytest.approx((0.1, -0.2, -0.3), abs=1e-6)  # rates (p, -q, -r)
    assert gp.name == "HIL_GPS" and gp.fields["fix_type"] == 3 and gp.fields["satellites_visible"] == 11
    assert gp.fields["vn"] == -300 and gp.fields["ve"] == 200 and gp.fields["vd"] == -50  # ENU (2, -3, 0.5): north -3, east 2, down -0.5 m/s, in cm/s
    assert gp.fields["lat"] * 1e-7 == pytest.approx(ORIGIN[0] + math.degrees(250.0 / 6378137.0), abs=2e-7)
    d = decode_sensors(gp, ORIGIN)
    assert np.allclose(d["pos"], gps["pos"], atol=0.03) and np.allclose(d["vel"], gps["vel"], atol=0.011)  # centimetre resolution
    d = decode_sensors(imu, ORIGIN)
    assert np.allclose(d["accel"], [1.0, 2.0, 9.0], atol=1e-6) and np.allclose(d["gyro"], [0.1, 0.2, 0.3], atol=1e-6)
    with pytest.raises(ValueError, match="not a sensor message"):
        decode_sensors(Message(MESSAGES["HEARTBEAT"], {}, 1, 1, 0))


# ------------------------------------------------------------------------------------------------ closed loop
CFG = {
    "duration": 10.0,
    "trajectory": {"type": "hover", "params": {"pos": [0, 0, 5.0]}},
    "environment": {"origin": {"lat": ORIGIN[0], "lon": ORIGIN[1], "alt": ORIGIN[2]}},
    "sensors": ["imu", "mag", "baro", {"type": "gps", "params": {"rate": 10.0, "latency": 0.0}}],
}


def run_hil(cfg=None, arm=True, **hook_kw):
    a, b = LoopbackTransport.pair()
    hook = HilHook(a, origin=ORIGIN, **hook_kw)
    ap = ReferenceAutopilot(VehicleSpec.generic_1kg(), b, origin=ORIGIN, target=(0.0, 0.0, 5.0))
    if not arm:
        orig = ap._answer
        ap._answer = lambda t, c, armed=False: orig(t, c, False)  # an autopilot that never arms
    sim = Simulation(cfg or CFG, hooks=[hook])
    return sim.run(), hook, ap


def test_an_autopilot_on_the_other_end_of_the_bridge_hovers_the_vehicle():
    arch, hook, ap = run_hil()
    pos = arch["pos"][:, 0]
    late = arch.t > 5.0
    assert hook.armed and hook.n_received == ap.n_answered > 1500
    assert np.abs(pos[late] - [0, 0, 5.0]).max() < 0.8 and abs(pos[late, 2].mean() - 5.0) < 0.15  # GPS-limited, but it holds the point
    assert not any(e["type"] == "collision" for e in arch.meta["events"])
    assert hook.parser.bad_crc == 0 and ap.parser.bad_crc == 0 and ap.parser.good > 1500  # both directions carry valid frames only
    cmd = arch["cmd"][late, 0]
    assert 0.5 < cmd.mean() / VehicleSpec.generic_1kg().w_max < 0.9  # hover is a bit above half of the rotor speed range


def test_without_the_armed_flag_the_motors_stay_off_and_the_vehicle_falls():
    arch, hook, _ = run_hil(arm=False)
    assert not hook.armed and arch["pos"][-1, 0, 2] < 0.2
    free, _, _ = run_hil(arm=False, require_armed=False)
    assert free["pos"][-1, 0, 2] > 1.0  # the same stream, accepted without the check, flies


def test_the_output_mapping_and_motor_order_matter():
    from types import SimpleNamespace

    spec = VehicleSpec.generic_1kg()
    sim = SimpleNamespace(spec=spec, n=2)
    ctrl = np.zeros(16)
    ctrl[:4] = [0.25, 0.04, 1.0, 0.0]
    thrust, speed = HilHook(LoopbackTransport()), HilHook(LoopbackTransport(), mapping="speed")
    for h in (thrust, speed):
        h.controls, h.armed = ctrl, True
    assert np.allclose(thrust._command(sim)[0] / spec.w_max, [0.5, 0.2, 1.0, 0.0]) and np.allclose(
        speed._command(sim)[0] / spec.w_max, [0.25, 0.04, 1.0, 0.0]
    )  # thrust ~ output, or speed ~ output
    assert thrust._command(sim).shape == (2, 4)  # one row per simulated vehicle
    reordered = HilHook(LoopbackTransport(), motor_map=[3, 2, 1, 0])
    reordered.controls, reordered.armed = ctrl, True
    assert np.allclose(reordered._command(sim)[0] / spec.w_max, [0.0, 1.0, 0.2, 0.5])
    thrust.controls = np.full(16, np.nan)  # a corrupt message must not make NaN rotor commands
    assert np.all(np.isfinite(thrust._command(sim)))
    thrust.armed = False
    assert np.all(thrust._command(sim) == 0.0)  # disarmed: motors off
    with pytest.raises(ValueError, match="mapping"):
        HilHook(LoopbackTransport(), mapping="pwm")
    swapped, _, _ = run_hil(cfg={**CFG, "duration": 4.0}, motor_map=[1, 0, 2, 3])  # motors 0 and 1 exchanged: the attitude loop pushes the wrong way
    assert np.abs(swapped["pos"][-1, 0] - [0, 0, 5.0]).max() > 1.0


def test_a_peer_that_never_answers_costs_only_the_wait_and_the_vehicle_keeps_flying_on_its_own_controller():
    a, _ = LoopbackTransport.pair()  # nobody listens on the other end
    hook = HilHook(a, origin=ORIGIN, wait_s=0.001)
    arch = Simulation({**CFG, "duration": 0.2}, hooks=[hook]).run()
    assert hook.n_sent > 20 and hook.n_received == 0 and hook.controls is None
    assert arch["pos"][-1, 0, 2] > 4.5  # no command overrode the simulator's own controller


# ------------------------------------------------------------------------------------------------ transports
def test_udp_transport_exchanges_datagrams_on_the_loopback_interface_only():
    a = UdpTransport(0)
    b = UdpTransport(0, remote_port=a.port)
    try:
        b.send(b"hello")
        got = b""
        for _ in range(400):
            got = a.recv()
            if got:
                break
            time.sleep(0.005)
        assert got == b"hello" and a.last_sender is not None
        a.send(b"reply")  # answers to whoever spoke last
        back = b""
        for _ in range(400):
            back = b.recv()
            if back:
                break
            time.sleep(0.005)
        assert back == b"reply" and b.recv() == b""
    finally:
        a.close()
        b.close()
        a.close()  # closing twice is harmless
    for bad in ("0.0.0.0", "192.168.1.20", "example.com"):
        with pytest.raises(ValueError, match="non-loopback"):
            UdpTransport(0, remote_port=1, remote_host=bad)
        with pytest.raises(ValueError, match="non-loopback"):
            UdpTransport(0, bind_host=bad)
    remote_ok = UdpTransport(0, remote_port=14560, remote_host="192.0.2.1", allow_remote=True)  # an explicit decision to leave the machine
    remote_ok.close()


def test_loopback_pairs_deliver_in_order_and_report_missing_peers():
    a, b = LoopbackTransport.pair()
    a.send(b"abc")
    a.send(b"def")
    assert b.recv() == b"abcdef" and b.recv() == b"" and a.sent == 6
    seen = []
    b.on_receive = seen.append
    a.send(b"x")
    assert seen == [b"x"]
    with pytest.raises(RuntimeError, match="no peer"):
        LoopbackTransport().send(b"x")
