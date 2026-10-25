from pathlib import Path

import numpy as np
import pytest

from aevion.config import SimulationConfig
from aevion.engine import CallbackHook, EventBus, Hook, RngStreams, Scheduler, Simulation, StopOnEventHook
from aevion.engine.faults import FaultError, RotorFailure
from aevion.errors import SchedulerError


def make(duration=3.0, **kw):
    d = {"duration": duration, "trajectory": {"type": "hover", "pos": [0, 0, 2.0]}}
    d.update(kw)
    return Simulation(SimulationConfig.from_dict(d))


# ---------------------------------------------------------------------------------- scheduler
def test_scheduler_rates_phases_and_priorities():
    S = Scheduler(dt=0.01)
    log = []
    S.add("fast", lambda: log.append(("fast", S.tick)))
    S.add("slow", lambda: log.append(("slow", S.tick)), rate_hz=20.0)  # every 5 ticks
    S.add("late", lambda: log.append(("late", S.tick)), every_ticks=5, phase=2)
    S.add("first", lambda: log.append(("first", S.tick)), every_ticks=10, priority=1)
    for _ in range(12):
        S.run_tick()
        S.advance()
    ticks = lambda n: [t for k, t in log if k == n]
    assert ticks("fast") == list(range(12)) and ticks("slow") == [0, 5, 10]
    assert ticks("late") == [2, 7] and ticks("first") == [0, 10]
    at0 = [k for k, t in log if t == 0]
    assert at0 == ["first", "fast", "slow"]  # priority order, then insertion order
    assert S.effective_rate("slow") == pytest.approx(20.0)


def test_scheduler_validation():
    S = Scheduler(dt=0.01)
    with pytest.raises(SchedulerError, match="exceeds the physics rate"):
        S.add("x", lambda: 0, rate_hz=500.0)
    with pytest.raises(SchedulerError, match="not both"):
        S.add("x", lambda: 0, rate_hz=10.0, every_ticks=3)
    S.add("ok", lambda: 0)
    with pytest.raises(SchedulerError, match="duplicate"):
        S.add("ok", lambda: 0)
    with pytest.raises(SchedulerError):
        Scheduler(dt=0.0)
    with pytest.warns(UserWarning, match="not a whole multiple"):
        S.add("odd", lambda: 0, rate_hz=70.0)  # 100/70 = 1.43 ticks -> rounds to 1 tick (100 Hz)


def test_scheduler_profile_report():
    S = Scheduler(dt=0.01, profile=True)
    S.add("work", lambda: sum(range(1000)))
    for _ in range(5):
        S.run_tick()
        S.advance()
    assert S.get("work").calls == 5 and S.get("work").wall_ns > 0
    assert "work" in S.report()


# ------------------------------------------------------------------------------------- events
def test_event_bus_patterns_priority_history():
    bus = EventBus(history=3)
    got = []
    unsub = bus.subscribe("fault.*", lambda e: got.append(("f", e.name)), priority=5)
    bus.subscribe("*", lambda e: got.append(("all", e.name)), priority=1)
    bus.emit("fault.start", 0.1, 1, 0, fault="x")
    bus.emit("collision", 0.2, 2)
    assert got == [("all", "fault.start"), ("f", "fault.start"), ("all", "collision")]
    unsub()
    bus.emit("fault.end", 0.3, 3)
    assert got[-1] == ("all", "fault.end")
    for i in range(5):
        bus.emit("x", i, i)
    assert len(bus.history) == 3 and bus.count("x") == 5 and bus.count("fault.start") == 1
    assert [e.name for e in bus.events("x")] == ["x"] * 3
    rec = bus.events("x")[0].as_record()
    assert rec["type"] == "x" and "t" in rec


def test_event_record_uses_fault_type_for_fault_events():
    bus = EventBus()
    e = bus.emit("fault.start", 1.0, 200, 2, fault="rotor_failure", rotor=1)
    assert e.as_record() == {"type": "rotor_failure", "t": 1.0, "name": "fault.start", "vehicle": 2, "fault": "rotor_failure", "rotor": 1}


# ---------------------------------------------------------------------------------------- rng
def test_rng_streams_are_independent_reproducible_and_order_insensitive():
    a, b = RngStreams(7), RngStreams(7)
    x1 = a.stream("imu").normal(size=5)
    y1 = a.stream("gps").normal(size=5)
    y2 = b.stream("gps").normal(size=5)  # b never touched 'imu'
    x2 = b.stream("imu").normal(size=5)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(x1, x2)
    assert not np.allclose(x1, y1)
    assert not np.allclose(RngStreams(8).stream("imu").normal(size=5), x1)
    np.testing.assert_array_equal(a.fresh("imu").normal(size=5), x1)  # fresh restarts the stream
    assert a.int_seed("imu") == b.int_seed("imu") != a.int_seed("gps")


# -------------------------------------------------------------------------------------- hooks
def test_hook_lifecycle_and_ctx_command_override():
    calls = []

    class H(Hook):
        name = "h"

        def on_start(self, sim):
            calls.append("start")

        def pre_step(self, sim):
            if sim.k == 3:
                calls.append("pre3")

        def post_step(self, sim):
            if sim.k == 3:
                calls.append("post3")

        def on_end(self, sim, archive=None):
            calls.append("end")

    sim = make(0.1)
    sim.hooks.add(H())
    sim.run()
    assert calls[0] == "start" and "pre3" in calls and "post3" in calls and calls[-1] == "end"

    # a hook can replace the controller command (here: cut all rotors -> the drone must fall)
    sim = make(1.0)
    sim.hooks.add(CallbackHook(pre_step=lambda s: setattr(s.ctx, "cmd", s.ctx.cmd * 0.0) if s.t > 0.2 else None))
    arch = sim.run()
    assert arch["pos"][-1, 0, 2] < 1.5


def test_hook_errors_name_the_hook_or_are_isolated():
    from aevion.engine.hooks import HookError

    sim = make(0.1)
    sim.hooks.add(CallbackHook("broken", pre_step=lambda s: 1 / 0))
    with pytest.raises(HookError, match="broken"):
        sim.run()
    sim = make(0.1)
    sim.hooks.isolate = True
    sim.hooks.add(CallbackHook("broken", pre_step=lambda s: 1 / 0))
    with pytest.warns(UserWarning, match="disabled"):
        sim.run()
    assert sim.hooks.failures and sim.hooks.failures[0][0] == "broken"


def test_stop_on_event_hook_and_builtin_events():
    cfg = {
        "duration": 5.0,
        "trajectory": {"type": "hover", "pos": [0, 0, 6.0]},
        "controller": "geometric",
        "faults": [{"type": "rotor_failure", "t_start": 0.5, "rotor": 0, "severity": 1.0}],
        "health": {"enabled": False},
    }
    sim = Simulation(SimulationConfig.from_dict(cfg), hooks=[StopOnEventHook("crash")])
    arch = sim.run()
    assert sim.stop_reason and "crash" in sim.stop_reason
    types = [e["type"] for e in arch.meta["events"]]
    assert "rotor_failure" in types and "ground.impact" in types and "crash" in types
    assert arch.t[-1] < 5.0


# ------------------------------------------------------------------------------------- faults
def test_rotor_failure_is_applied_and_reverted():
    cfg = {
        "duration": 2.0,
        "vehicle": {"model": "hex_3kg"},
        "trajectory": {"type": "hover", "pos": [0, 0, 20.0]},
        "faults": [{"type": "rotor_failure", "t_start": 0.5, "t_end": 1.0, "rotor": 2, "vehicle": 0}],
    }
    sim = Simulation(SimulationConfig.from_dict(cfg))
    healths = []
    sim.hooks.add(CallbackHook(post_step=lambda s: healths.append(float(s.dyn.params.rotor_health[0, 2]))))
    arch = sim.run()
    h = np.array(healths)
    assert h[:100].min() == 1.0 and h[101:200].max() == 0.0 and h[205:].min() == 1.0
    ev = [(e["name"], round(e["t"], 3)) for e in arch.meta["events"] if e["name"].startswith("fault.")]
    assert ev == [("fault.start", 0.5), ("fault.end", 1.0)]


def test_sensor_faults_change_what_the_estimator_sees():
    base = {"duration": 4.0, "sensors": ["imu", "gps", "baro", "mag"], "trajectory": {"type": "hover", "pos": [0, 0, 2.0]}}
    sim = Simulation(
        SimulationConfig.from_dict(
            {
                **base,
                "faults": [
                    {"type": "gps_outage", "t_start": 1.0, "t_end": 2.0},
                    {"type": "sensor_bias_step", "t_start": 2.5, "sensor": "imu", "field": "gyro", "bias": [0.0, 0.0, 0.5]},
                ],
            }
        )
    )
    gps_fresh, gyro_z = [], []
    sim.hooks.add(
        CallbackHook(
            post_step=lambda s: (
                gps_fresh.append((s.t, s.suite["gps"].fresh)),
                gyro_z.append((s.t, float(s.suite["imu"].latest["gyro"][0, 2]) if s.suite["imu"].latest else 0.0)),
            )
        )
    )
    sim.run()
    during = [f for t, f in gps_fresh if 1.2 < t < 1.9]
    outside = [f for t, f in gps_fresh if 2.2 < t < 2.9]
    assert not any(during) and any(outside)
    assert np.mean([g for t, g in gyro_z if t > 2.6]) > 0.3 and abs(np.mean([g for t, g in gyro_z if t < 2.4])) < 0.1
    assert sim.suite["imu"].filters  # persists (no t_end)


def test_other_faults_payload_thrust_stuck_wind_battery():
    base = {"duration": 3.0, "trajectory": {"type": "hover", "pos": [0, 0, 3.0]}, "health": {"enabled": False}}
    s = Simulation(SimulationConfig.from_dict({**base, "faults": [{"type": "payload_change", "t_start": 0.5, "mass": 0.3}]}))
    s.run()
    assert s.dyn.params.payload_mass[0, 0] == pytest.approx(0.3)
    s = Simulation(SimulationConfig.from_dict({**base, "faults": [{"type": "thrust_loss", "t_start": 0.5, "t_end": 1.0, "factor": 0.5}]}))
    k0 = s.dyn.params.kT.copy()
    s.run()
    np.testing.assert_allclose(s.dyn.params.kT, k0)  # restored
    s = Simulation(SimulationConfig.from_dict({**base, "faults": [{"type": "actuator_stuck", "t_start": 0.5, "rotor": 1, "speed": 700.0}]}))
    arch = s.run()
    assert arch["cmd"][-1, 0, 1] == pytest.approx(700.0)
    s = Simulation(SimulationConfig.from_dict({**base, "faults": [{"type": "battery_drop", "t_start": 1.0, "soc": 0.1}]}))
    arch = s.run()
    assert arch["soc"][-1, 0] <= 0.1 + 1e-9 and any(e["type"] == "battery.low" for e in arch.meta["events"])
    s = Simulation(
        SimulationConfig.from_dict({**base, "faults": [{"type": "wind_disturbance", "t_start": 1.0, "t_end": 2.0, "velocity": [6.0, 0, 0]}]})
    )
    arch = s.run()
    wx = arch["wind"][:, 0, 0]
    assert wx[arch.t < 0.9].max() == 0.0 and wx[(arch.t > 1.1) & (arch.t < 1.9)].min() == pytest.approx(6.0) and wx[arch.t > 2.1].max() == 0.0
    s = Simulation(SimulationConfig.from_dict({**base, "faults": [{"type": "wind_gust", "t_start": 1.0, "amplitude": [8.0, 0, 0], "duration": 1.0}]}))
    arch = s.run()
    assert arch["wind"][:, 0, 0].max() == pytest.approx(8.0, rel=0.02)


def test_fault_errors_are_actionable():
    with pytest.raises(FaultError, match="no sensor named 'gps'"):
        make(1.0, faults=[{"type": "gps_outage", "t_start": 0.1}]).run()
    with pytest.raises(FaultError, match="rotor 9 out of range"):
        make(1.0, faults=[{"type": "rotor_failure", "t_start": 0.1, "rotor": 9}]).run()
    with pytest.raises(FaultError, match="no swarm/network"):
        make(1.0, faults=[{"type": "comm_blackout", "t_start": 0.1}]).run()
    with pytest.raises(FaultError):
        RotorFailure(severity=2.0)


def test_comm_blackout_stops_swarm_deliveries():
    cfg = {
        "duration": 3.0,
        "vehicle": {"count": 3, "formation": {"type": "line", "spacing": 1.5}},
        "swarm": {"avoid_radius": 1.0, "comm": {"rate_hz": 20}},
        "faults": [{"type": "comm_blackout", "t_start": 1.0, "t_end": 2.0, "target": "swarm"}],
        "trajectory": {"type": "hover", "pos": [0, 0, 2.0]},
    }
    sim = Simulation(SimulationConfig.from_dict(cfg))
    sim.run()
    assert 0.55 < sim.swarm.delivery_ratio() < 0.75  # one of three seconds blacked out


# ---------------------------------------------------------------------------------- simulation
def test_multirate_control_zero_order_hold():
    sim = make(1.0, engine={"controller_rate_hz": 50.0})
    cmds = []
    sim.hooks.add(CallbackHook(post_step=lambda s: cmds.append(s.ctx.cmd.copy())))
    sim.run()
    changes = sum(not np.array_equal(cmds[i], cmds[i - 1]) for i in range(1, len(cmds)))
    assert 0.8 * 49 <= changes <= 50  # new command only every 4th physics tick
    assert sim.scheduler.effective_rate("control") == pytest.approx(50.0)


def test_logging_rate_and_channel_filter():
    sim = make(1.0, logging={"rate_hz": 25.0, "channels": ["pos", "ref_pos"], "vehicles": None})
    arch = sim.run()
    assert set(arch.channels) == {"t", "pos", "ref_pos"} and arch.n_steps == 26
    assert arch.dt == pytest.approx(0.04)


def test_reset_replays_identically_and_run_twice_is_equal():
    cfg = {
        "duration": 2.0,
        "wind": {"type": "dryden", "w20": 4.0},
        "sensors": ["imu", "gps"],
        "seed": 3,
        "trajectory": {"type": "circle", "center": [0, 0, 2], "radius": 1.0, "omega": 1.0},
    }
    sim = Simulation(SimulationConfig.from_dict(cfg))
    a = sim.run()
    sim.reset()
    b = sim.run()
    np.testing.assert_array_equal(a["pos"], b["pos"])
    c = Simulation(SimulationConfig.from_dict(cfg)).run()
    np.testing.assert_array_equal(a["pos"], c["pos"])
    d = Simulation(SimulationConfig.from_dict({**cfg, "seed": 4})).run()
    assert np.abs(a["pos"] - d["pos"]).max() > 1e-8


def test_named_sensor_instances_and_extrinsics():
    cfg = {
        "duration": 0.5,
        "world": {"type": "forest", "n_trees": 5, "half_extent": 6},
        "sensors": [
            {
                "type": "lidar",
                "name": "front",
                "channels": 1,
                "samples": 8,
                "v_fov": [0, 0],
                "extrinsics": {"rpy_deg": [0, 0, 0], "translation": [0.1, 0, 0]},
            },
            {"type": "lidar", "name": "rear", "channels": 1, "samples": 8, "v_fov": [0, 0], "extrinsics": {"rpy_deg": [0, 0, 180]}},
        ],
        "logging": {"sensors": ["front", "rear"]},
    }
    sim = Simulation(SimulationConfig.from_dict(cfg))
    arch = sim.run()
    assert set(sim.suite.sensors) == {"front", "rear"}
    assert "sensors/front" in arch and "sensors/rear" in arch
    np.testing.assert_allclose(sim.suite["rear"].dirs_body[:, :2], -sim.suite["front"].dirs_body[:, :2], atol=1e-12)  # rotated 180 deg
    np.testing.assert_allclose(sim.suite["front"].mount, [0.1, 0, 0])


def test_provenance_and_config_are_embedded():
    arch = make(0.5).run()
    p = arch.provenance
    assert p["aevion"] and p["config_hash"] and p["kernel"] in ("numba", "reference") and p["package_set_hash"]
    assert "hostname" not in p and "user" not in p  # no personal data
    assert arch.config["duration"] == 0.5
    cfg2 = SimulationConfig.from_dict(arch.config)
    assert p["config_hash"] == __import__("aevion.config", fromlist=["config_hash"]).config_hash(cfg2)


def test_verify_uses_embedded_config():
    from aevion.scenario import verify

    arch = make(1.0, wind={"type": "dryden", "w20": 3.0}, seed=2).run()
    r = verify(arch)
    assert r["ok"] and r["max_abs_diff"] == 0.0


# ------------------------------------------------------------------- legacy golden regression
GOLDEN = Path(__file__).parent / "golden" / "v01_scenarios.npz"


def _golden_cases():
    from aevion.scenario import Scenario
    from aevion.wireless import AirToGroundChannel, BaseStation, CellularNetwork

    net = CellularNetwork([BaseStation((-30, -30, 20))], AirToGroundChannel("suburban", los_mode="mean"))
    return {
        "plain_circle": Scenario(trajectory=dict(kind="circle", center=[0, 0, 2.5], radius=1.2, omega=1.0), duration=4.0, record_every=4),
        "wind_sensors": Scenario(
            world={"kind": "forest", "n_trees": 6, "half_extent": 10},
            wind={
                "type": "CompositeWind",
                "models": [{"type": "ConstantWind", "velocity": [2, 1, 0]}, {"type": "DrydenTurbulence", "w20": 4.0, "severity": 1.0}],
            },
            controller="cascaded",
            trajectory=dict(kind="figure8", center=[0, 0, 3.0], size=2.0, period=6.0),
            sensors=["imu", "gps", "baro", "mag", "lidar"],
            record_sensors=["imu", "lidar"],
            duration=3.0,
            seed=5,
            record_every=4,
        ),
        "estimator": Scenario(
            sensors=["imu", "gps", "baro", "mag"],
            use_estimator=True,
            estimator_handover_s=1.5,
            duration=4.0,
            trajectory=dict(kind="hover", pos=[0, 0, 2.0]),
            seed=2,
            record_every=4,
        ),
        "wireless": Scenario(
            wireless=net.to_dict(), duration=3.0, trajectory=dict(kind="circle", center=[0, 0, 2.5], radius=1.2, omega=1.0), seed=3, record_every=4
        ),
        "swarm": Scenario(
            n=4,
            formation=dict(kind="line", spacing=1.0),
            swarm=dict(avoid_radius=1.2, comm=dict(loss=0.2, delay_s=0.05)),
            trajectory=dict(kind="hover", pos=[0, 0, 2.0]),
            duration=3.0,
            seed=4,
            record_every=4,
        ),
        "rotor_failure": Scenario(
            vehicle="hex_3kg",
            duration=3.0,
            perturbation=dict(kind="rotor_failure", t=1.0, vehicle=0, rotor=1, severity=1.0),
            trajectory=dict(kind="hover", pos=[0, 0, 5.0]),
            seed=1,
            record_every=4,
        ),
        "waypoints_lqr": Scenario(
            controller="lqr", trajectory=dict(kind="waypoints", points=[[0, 0, 1], [3, 0, 2], [3, 3, 2]], speed=1.5), duration=5.0, record_every=2
        ),
    }


@pytest.mark.parametrize("name", ["plain_circle", "wind_sensors", "estimator", "wireless", "swarm", "rotor_failure", "waypoints_lqr"])
def test_engine_reproduces_frozen_v01_outputs_bit_exactly(name):
    G = np.load(GOLDEN)
    arch = _golden_cases()[name].run()
    checked = 0
    for key in G.files:
        if not key.startswith(name + "/"):
            continue
        k = key.split("/", 1)[1]
        if k == "events":
            ev = np.array([e["t"] for e in arch.meta["events"] if e["type"] in ("rotor_failure", "collision")])
            np.testing.assert_allclose(ev, G[key])
            continue
        got = arch["sensors/imu"] if k == "sensors_imu" else arch["sensors/lidar"] if k == "sensors_lidar" else arch[k]
        np.testing.assert_array_equal(got, G[key], err_msg=f"{name}/{k}")
        checked += 1
    assert checked >= 8
