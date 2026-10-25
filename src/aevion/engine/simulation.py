"""The simulation engine.

A :class:`Simulation` is built from a :class:`~aevion.config.SimulationConfig` and owns every component:
world, dynamics, controller, mission (trajectory), sensors, estimator, network, swarm coordinator, fault
timeline, hooks, health monitor, event bus, scheduler and recorder.

One *tick* is one physics step. Within a tick the scheduler runs the periodic tasks in priority order::

    05 faults      start/stop scheduled faults
    10 sense       update every sensor (each has its own rate/latency/dropout)
    20 estimate    sensor fusion; choose true or estimated state for control
    30 mission     reference from the trajectory (+ swarm repulsion)
    40 control     controller → rotor command (zero-order hold between control ticks)
    80 network     radio-link evaluation
    90 record      archive row (+ sensor streams)

then ``pre_step`` hooks, the physics step, event detection, health checks and ``post_step`` hooks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..config import SimulationConfig, config_hash, load_config
from ..datasets.archive import Recorder, RunArchive, StreamRecorder
from ..datasets.provenance import collect as collect_provenance
from ..dynamics.multirotor import State
from ..errors import SimulationDiverged
from ..health import HealthContext
from .builder import build_parts
from .events import Event, EventBus
from .hooks import HookManager
from .rng import RngStreams
from .scheduler import Scheduler

P_FAULTS, P_SENSE, P_ESTIMATE, P_MISSION, P_CONTROL, P_NETWORK, P_RECORD = 5, 10, 20, 30, 40, 80, 90


@dataclass
class StepContext:
    """The per-tick blackboard shared by tasks and hooks."""

    readings: Optional[dict] = None
    fresh: Optional[dict] = None
    est_state: Any = None
    use_estimate: bool = False
    control_state: Any = None
    ref: Any = None
    cmd: Any = None
    link: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)  # free-form values hooks may add (recorded if numeric arrays)


class Simulation:
    state: State  # created by reset() at the end of __init__
    recorder: Recorder  # (re)created by reset()

    def __init__(self, config: Union[SimulationConfig, dict, str], hooks: Any = (), isolate_hooks: bool = False) -> None:
        self.cfg: SimulationConfig = config if isinstance(config, SimulationConfig) else load_config(config)
        cfg = self.cfg
        self.config_hash = config_hash(cfg)
        parts = build_parts(cfg)
        self.spec, self.world, self.dyn, self.ctrl, self.traj = parts.spec, parts.world, parts.dyn, parts.ctrl, parts.traj
        self.offsets, self.suite, self.est, self.net, self.swarm = parts.offsets, parts.suite, parts.est, parts.net, parts.swarm
        self.faults, self.health = parts.faults, parts.health
        self.recorded_sensors = parts.recorded_sensors
        self.B, self.dt = self.dyn.B, cfg.engine.dt
        self.n_steps_total = cfg.n_steps
        self.rng = RngStreams(cfg.seed)
        self.bus = EventBus()
        self.hooks = HookManager(list(parts.hooks) + list(hooks), isolate=isolate_hooks)
        self.scheduler = Scheduler(self.dt)
        self.ctx = StepContext()
        self.k = 0
        self.archive: Optional[RunArchive] = None
        self.stop_reason: Optional[str] = None
        self._events_out: List[dict] = []
        self._soc_low_warned: set = set()
        self._wall = 0.0
        self._setup_schedule()
        self.bus.subscribe("*", self._on_any_event, priority=-100)
        self.reset()

    # ------------------------------------------------------------------ properties
    @property
    def t(self) -> float:
        return float(self.state.t) if self.state is not None else 0.0

    @property
    def n(self) -> int:
        return self.dyn.n

    def emit(self, name: str, vehicle: Optional[int] = None, **data: Any) -> Event:
        return self.bus.emit(name, self.t, self.k, vehicle, **data)

    def _on_any_event(self, ev: Event) -> None:
        self._events_out.append(ev.as_record())
        if self.recorder is not None:
            pass  # events are merged into the archive manifest at the end
        self.hooks.call("on_event", self, ev)

    def request_stop(self, reason: str = "requested") -> None:
        self.stop_reason = reason

    # --------------------------------------------------------------------- setup
    def _setup_schedule(self) -> None:
        cfg, S = self.cfg, self.scheduler
        S.add("faults", self._task_faults, priority=P_FAULTS)
        if self.suite is not None:
            S.add("sense", self._task_sense, priority=P_SENSE)
        if self.est is not None:
            S.add("estimate", self._task_estimate, priority=P_ESTIMATE)
        S.add("mission", self._task_mission, priority=P_MISSION)
        c_rate = cfg.engine.controller_rate_hz
        S.add("control", self._task_control, priority=P_CONTROL, rate_hz=c_rate or None)
        lg = cfg.logging
        self.record_every = S.ticks_for_rate(lg.rate_hz, "record") if lg.rate_hz else lg.record_every
        self._net_dt = 0.0
        if self.net is not None:
            nc = cfg.network
            assert nc is not None  # self.net exists only when a network is configured
            net_every = S.ticks_for_rate(nc.rate_hz, "network") if nc.rate_hz else self.record_every
            self._net_dt = net_every * self.dt
            S.add("network", self._task_network, every_ticks=net_every, priority=P_NETWORK)
        S.add("record", self._task_record, every_ticks=self.record_every, priority=P_RECORD)
        if self.suite is not None and self.recorded_sensors:
            S.add("record_sensors", self._task_record_sensors, priority=P_RECORD + 1)  # every fresh sample, not only on record ticks

    def _make_recorder(self) -> Recorder:
        lg = self.cfg.logging
        # the scheduler decides *when* to record, so the recorder itself records every call
        if lg.mode == "stream":
            if not lg.path:
                raise ValueError("logging.path is required for stream mode")
            prov = collect_provenance(self.config_hash, self.cfg.seed, self.dyn.kernel.name, self.B.name) if lg.provenance else {}
            return StreamRecorder(
                lg.path,
                every=1,
                vehicles=lg.vehicles,
                channels=lg.channels,
                chunk_steps=lg.chunk_steps,
                provenance=prov,
                overwrite=True,
                collision_events=False,
            )
        return Recorder(every=1, vehicles=lg.vehicles, channels=lg.channels, collision_events=False)

    # ----------------------------------------------------------------------- reset
    def reset(self) -> Any:
        """(Re)initialise the vehicle(s), sensors, estimator, faults and recorder to the configured start."""
        cfg, _B = self.cfg, self.B
        ini = cfg.vehicle.initial
        if hasattr(self.traj, "reset"):
            self.traj.reset()  # stateful trajectories (behaviour-tree missions) start over
        p0, v0 = self.traj.sample(0.0)[:2]
        pos0 = np.asarray(ini.pos if ini.pos is not None else p0, float)
        if self.offsets is not None:
            pos0 = pos0 + self.offsets
        self.state = self.dyn.reset(pos=pos0, vel=ini.vel if ini.vel is not None else v0, euler=ini.euler, omega=ini.omega, soc=ini.soc)
        if self.suite is not None:
            self.suite.reset(self.state)
        if self.est is not None:
            self.est.reset(None)
        if hasattr(self.ctrl, "reset"):
            self.ctrl.reset(self.n)
        self.faults.reset()
        if self.health is not None:
            self.health.reset()
        self.scheduler.reset()
        self.bus.clear()
        self._events_out.clear()
        self._soc_low_warned.clear()
        self.k = 0
        self.stop_reason = None
        self.ctx = StepContext()
        self.recorder = self._make_recorder()
        self.archive = None
        self._started = False
        self._prev_collided = np.zeros(self.n, bool)
        self.hooks.call("on_reset", self)
        return self.state

    # ----------------------------------------------------------------------- tasks
    def _task_faults(self) -> None:
        self.faults.update(self, self.t, self.k)

    def _task_sense(self) -> None:
        self.ctx.readings = self.suite.update(self.state)
        self.ctx.fresh = self.suite.fresh()

    def _task_estimate(self) -> None:
        c = self.ctx
        c.est_state = self.est.update(c.readings, c.fresh)
        ec = self.cfg.estimator
        assert ec is not None  # the estimate task exists only when an estimator is configured
        if self.t >= ec.handover_s:
            c.use_estimate = True

    def _task_mission(self) -> None:
        c, B = self.ctx, self.B
        observe = getattr(self.traj, "observe", None)
        if observe is not None:  # missions react to the vehicle (battery, position)
            observe(pos=B.to_numpy(self.state.pos)[0], soc=float(np.mean(B.to_numpy(self.state.soc))))
        ref = self.traj.reference(B, self.t, self.n, self.offsets)
        if self.swarm is not None:
            pos_np = B.to_numpy(self.state.pos)
            self.swarm.update(self.t, pos_np)
            ref.pos = ref.pos + B.asarray(self.swarm.repulsion(pos_np))
            if self.swarm.laws:
                rp, rv = B.to_numpy(ref.pos), B.to_numpy(ref.vel)
                for law in self.swarm.laws:
                    rp, rv = law.apply(self.t, pos_np, self.swarm.known, rp, rv)
                ref.pos, ref.vel = B.asarray(rp), B.asarray(rv)
        c.ref = ref

    def _task_control(self) -> None:
        c = self.ctx
        c.control_state = c.est_state if (self.est is not None and c.use_estimate) else self.state
        c.cmd = self.ctrl.act(c.control_state, c.ref, self.dt)

    def _task_network(self) -> None:
        c = self.ctx
        pos = self.B.to_numpy(self.state.pos)
        if getattr(self.net, "wants_state", False):  # time-aware networks (handover timers, fading) also get kinematics and attitude
            c.link = self.net.evaluate(pos, t=self.t, vel=self.B.to_numpy(self.state.vel), quat=self.B.to_numpy(self.state.quat), dt=self._net_dt)
        else:
            c.link = self.net.evaluate(pos)

    def _task_record(self) -> None:
        self._record_row()

    def _extras(self) -> dict:
        c = self.ctx
        ex: Dict[str, Any] = {}
        ex.update(ref_pos=c.ref.pos, ref_vel=c.ref.vel, cmd=c.cmd)
        if self.est is not None and c.est_state is not None:
            ex.update(est_pos=c.est_state.pos, est_quat=c.est_state.quat)
            std = getattr(self.est, "position_std", None)
            if std is not None:
                ex["est_pos_std"] = std()
        if self.net is not None and c.link is not None:
            r = c.link
            ex.update(
                link_sinr_db=r.sinr_db,
                link_rssi_dbm=r.rssi_dbm,
                link_rate_mbps=r.rate_bps / 1e6,
                link_serving=r.serving.astype(float),
                link_los=r.los.astype(float),
                link_outage=r.outage.astype(float),
            )
            for k, v in r.extras.items():
                ex[f"link_{k}"] = v
        for k, v in c.extras.items():
            ex[k] = v
        if self.world.actors and self.recorder.vehicles is None:
            ex["actor_pos"] = np.array([a.position(self.t) for a, _, _ in self.world.actors])  # (A, 3): where every moving obstacle is
        return ex

    def _record_row(self) -> None:
        self.recorder.record(self.B, self.state, **self._extras())

    def _task_record_sensors(self) -> None:
        rec, B = self.recorder, self.B
        for name in self.recorded_sensors:
            s = self.suite[name]
            if s.fresh and s.latest is not None:
                data = _sensor_payload(name, s, B)
                if data is not None:
                    rec.record_side(name, s.stamp, data)

    # ------------------------------------------------------------------- stepping
    def _detect_events(self, prev_collided: Any) -> Any:
        s, B = self.state, self.B
        imp = B.to_numpy(s.impact_speed)
        for i in np.nonzero(imp > 2.0)[0]:
            self.emit("ground.impact", int(i), speed=float(imp[i]))
            if imp[i] > 4.0:
                self.emit("crash", int(i), speed=float(imp[i]))
        col = B.to_numpy(s.collided).astype(bool)
        for i in np.nonzero(col & ~prev_collided)[0]:
            self.emit("collision", int(i))
        soc = B.to_numpy(s.soc)
        for i in np.nonzero(soc < 0.2)[0]:
            if int(i) not in self._soc_low_warned:
                self._soc_low_warned.add(int(i))
                self.emit("battery.low", int(i), soc=float(soc[i]))
        return col

    def step(self) -> Any:
        """Advance one physics tick (all due tasks, hooks, physics, events, health)."""
        if not self._started:
            self._started = True
            self.hooks.call("on_start", self)
            self.emit("sim.start")
        S = self.scheduler
        S.run_tick()
        self.hooks.call("pre_step", self)
        cmd = self.ctx.cmd
        if len(self.faults):
            cmd = self.faults.modify_command(self, cmd)
            self.ctx.cmd = cmd
        prev_col = self._prev_collided
        self.state = self.dyn.step(self.state, cmd)
        S.advance()
        self.k += 1
        self._prev_collided = self._detect_events(prev_col)
        if self.health is not None:
            self.health.check(self.state, HealthContext(self.B, self.spec, self.dt * self.health.every, self.emit), self.k, self.t)
            if self.health.should_stop:
                v = self.health.violations[-1]
                self.emit("health.violation", v.vehicle, check=v.check, message=v.message)
                self.request_stop(f"health: {v.message}")
        self.hooks.call("post_step", self)
        return self.state

    # ------------------------------------------------------------------------ run
    def run(self, duration: Optional[float] = None, progress: bool = False) -> RunArchive:
        """Run to the end of the configured duration (or ``duration`` seconds) and return the archive."""
        steps = self.n_steps_total if duration is None else round(duration / self.dt)
        if progress:
            from .hooks import ProgressHook

            self.hooks.add(ProgressHook())
        t0 = time.perf_counter()
        try:
            while self.k < steps and self.stop_reason is None:
                self.step()
        except BaseException as ex:
            self._abort(ex)
            raise
        self._wall += time.perf_counter() - t0
        return self.finish()

    def _abort(self, ex: Any) -> None:
        rec = self.recorder
        if isinstance(rec, StreamRecorder):
            status = "diverged" if isinstance(ex, SimulationDiverged) else "failed"
            rec.abort(meta=dict(self._meta(), status=status, error=f"{type(ex).__name__}: {ex}"))
        self.emit("sim.abort", error=f"{type(ex).__name__}: {ex}")
        self.hooks.call("on_end", self, None)

    def _meta(self) -> dict:
        cfg, d = self.cfg, self.dyn
        return dict(
            name=cfg.name,
            config=cfg.to_dict(),
            scenario=legacy_scenario_view(cfg),
            vehicle=self.spec.to_dict(),
            world=self.world.to_dict(),
            controller=cfg.controller.type,
            backend=d.B.name,
            device=d.B.device,
            fast_kernel=d.fast,
            kernel=d.kernel.name,
            n_sub=d.n_sub,
            sim=d.info(),
            swarm_delivery_ratio=self.swarm.delivery_ratio() if self.swarm is not None else None,
            wireless=self.net.to_dict() if self.net is not None else None,
            faults=self.faults.describe(),
            health=self.health.summary() if self.health is not None else None,
            events=list(self._events_out),
            stop_reason=self.stop_reason,
            wall_s=self._wall,
            schema_config=cfg.schema_version,
        )

    def finish(self) -> RunArchive:
        """Record the final state, finalise the archive and run ``on_end`` hooks."""
        if self.ctx.ref is not None:  # final state (with the last reference/command, as in the legacy scenario runner)
            if self.net is not None:
                self._task_network()
            self._record_row()
        meta = self._meta()
        prov = collect_provenance(self.config_hash, self.cfg.seed, self.dyn.kernel.name, self.B.name) if self.cfg.logging.provenance else {}
        meta["provenance"] = prov
        self.recorder.events = list(self._events_out)  # collisions come from the engine's onset events
        if isinstance(self.recorder, StreamRecorder):
            self.archive = self.recorder.build(meta, prov)
        else:
            self.archive = self.recorder.build(meta)
        self.hooks.call("on_end", self, self.archive)
        return self.archive


def _sensor_payload(name: str, s: Any, B: Any) -> Any:
    """The array stored for a recorded sensor stream (same conventions as the legacy scenario runner)."""
    r = s.latest
    typ = type(s).__name__
    if hasattr(s, "record_payload"):  # sensors that define their own stored layout
        return s.record_payload(r)
    if typ == "IMU":
        return B.to_numpy(B.cat([r["accel"], r["gyro"]], axis=-1))
    if typ == "GPS":
        return B.to_numpy(r["pos"])
    if typ == "Barometer":
        return B.to_numpy(r["altitude"])
    if typ == "Magnetometer":
        return B.to_numpy(r["field"])
    if typ == "RangeFinder":
        return B.to_numpy(r["range"])
    if typ == "Lidar":
        return B.to_numpy(r["ranges"]).astype(np.float32)
    if typ == "Camera":
        key = "depth" if "depth" in r else next(iter(k for k in r if k != "t"), None)
        return None if key is None else B.to_numpy(r[key]).astype(np.float32)
    # unknown sensor type: store the first array-valued field
    for k, v in r.items():
        if hasattr(v, "shape") and k != "t":
            return B.to_numpy(v)
    return None


def legacy_scenario_view(cfg: SimulationConfig) -> dict:
    """The subset of fields older tools (viewer, report, ``verify``) read from ``meta['scenario']``."""
    return dict(
        name=cfg.name,
        controller=cfg.controller.type,
        trajectory=cfg.trajectory.to_dict() | {"kind": cfg.trajectory.type},
        seed=cfg.seed,
        duration=cfg.duration,
        dt=cfg.engine.dt,
        n=cfg.vehicle.count,
    )
