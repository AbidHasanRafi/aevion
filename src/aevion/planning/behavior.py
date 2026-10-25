"""Behaviour-tree missions.

A behaviour tree (Colledanchise & Ögren 2018) is ticked every control step; each node returns ``SUCCESS``, ``FAILURE`` or ``RUNNING``.
Composites (``Sequence``, ``Fallback``, ``Parallel``) combine children, decorators (``Inverter``, ``Repeat``, ``Retry``, ``Timeout``,
...) change a child's outcome, and the leaves are conditions and actions, for example *take off, visit the waypoints, and if the
battery is low fly home and land*.

:class:`BehaviorTreeMission` is a :class:`~aevion.control.reference.Trajectory`: the engine asks it for the reference at time ``t``,
it ticks the tree, and the action nodes (:class:`Takeoff`, :class:`GoTo`, :class:`Hover`, :class:`Land`) move a reference state along
minimum-snap trajectories (planned around obstacles when a planning space is available). Nodes can read the vehicle state through
``observe``, for instance its battery. Trees are written in code or declaratively (:func:`tree_from_dict`) for configuration files.
Ticking is deterministic: the same tree and observations always produce the same reference.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Callable, Dict, List, Optional
from typing import Sequence as Seq

import numpy as np

from ..control.reference import Trajectory

__all__ = [
    "Status",
    "Node",
    "Sequence",
    "Fallback",
    "Parallel",
    "Inverter",
    "ForceSuccess",
    "ForceFailure",
    "Repeat",
    "Retry",
    "Timeout",
    "Condition",
    "Action",
    "Wait",
    "MissionContext",
    "Takeoff",
    "GoTo",
    "Hover",
    "Land",
    "BatteryAbove",
    "AltitudeAbove",
    "BehaviorTreeMission",
    "tree_from_dict",
]


class Status(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING


class MissionContext:
    """What nodes read and write: the clock, the reference state, an observation of the vehicle and a blackboard."""

    def __init__(
        self, space: Any = None, home: Seq[float] = (0.0, 0.0, 0.0), v_max: float = 3.0, a_max: float = 2.5, planner: str = "theta_star"
    ) -> None:
        self.t = 0.0
        self.dt = 0.0
        self.pos = np.array(home, float)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.jerk = np.zeros(3)
        self.yaw = 0.0
        self.home = np.array(home, float)
        self.space = space
        self.v_max, self.a_max, self.planner = float(v_max), float(a_max), planner
        self.battery: Optional[float] = None  # state of charge in [0, 1] when observed
        self.vehicle_pos: Optional[np.ndarray] = None
        self.ground_z = 0.0
        self.blackboard: Dict[Any, Any] = {}
        self.log: List[str] = []

    def set_ref(self, p: Any, v: Any = None, a: Any = None, j: Any = None, yaw: Optional[float] = None) -> None:
        self.pos = np.asarray(p, float)
        self.vel = np.zeros(3) if v is None else np.asarray(v, float)
        self.acc = np.zeros(3) if a is None else np.asarray(a, float)
        self.jerk = np.zeros(3) if j is None else np.asarray(j, float)
        if yaw is not None:
            self.yaw = float(yaw)


# ======================================================================================================== nodes
class Node:
    """Base node. Subclasses implement :meth:`tick`; :meth:`reset` clears any state (called when a subtree is interrupted)."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    def tick(self, ctx: MissionContext) -> Status:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def children_nodes(self) -> List["Node"]:
        return []

    def describe(self, depth: int = 0) -> str:
        return "  " * depth + self.name + "".join("\n" + c.describe(depth + 1) for c in self.children_nodes())


class _Composite(Node):
    def __init__(self, children: Seq[Node], name: Optional[str] = None) -> None:
        super().__init__(name)
        self.children = list(children)
        if not self.children:
            raise ValueError(f"{type(self).__name__} needs at least one child")
        self._i = 0

    def children_nodes(self) -> List[Node]:
        return self.children

    def reset(self) -> None:
        self._i = 0
        for c in self.children:
            c.reset()


class Sequence(_Composite):
    """Run children in order; fail at the first failure, succeed when all succeeded. ``memory`` resumes at the running child
    (default); with ``memory=False`` every tick restarts from the first child, so earlier conditions are re-checked (reactive)."""

    def __init__(self, children: Seq[Node], memory: bool = True, name: Optional[str] = None) -> None:
        super().__init__(children, name)
        self.memory = memory

    def tick(self, ctx: MissionContext) -> Status:
        start = self._i if self.memory else 0
        for i in range(start, len(self.children)):
            st = self.children[i].tick(ctx)
            if st is R:
                if not self.memory and i < self._i:
                    for later in self.children[i + 1 :]:
                        later.reset()
                self._i = i
                return R
            if st is F:
                self.reset()
                return F
        self.reset()
        return S


class Fallback(_Composite):
    """Try children in order; succeed at the first success, fail when all failed (a priority list: plan A, else plan B)."""

    def __init__(self, children: Seq[Node], memory: bool = False, name: Optional[str] = None) -> None:
        super().__init__(children, name)
        self.memory = memory

    def tick(self, ctx: MissionContext) -> Status:
        start = self._i if self.memory else 0
        for i in range(start, len(self.children)):
            st = self.children[i].tick(ctx)
            if st is R:
                if i != self._i:  # a higher-priority child took over: abandon the one that was running
                    for c in self.children[i + 1 :]:
                        c.reset()
                self._i = i
                return R
            if st is S:
                self.reset()
                return S
        self.reset()
        return F


class Parallel(_Composite):
    """Tick all children every step. ``policy``: ``"all"`` succeeds when every child has (fails if any failed), ``"any"`` when
    the first one has, or an integer ``k``: succeeds when ``k`` have succeeded."""

    def __init__(self, children: Seq[Node], policy: Any = "all", name: Optional[str] = None) -> None:
        super().__init__(children, name)
        self.policy = policy
        self._done: Dict[int, Status] = {}

    def reset(self) -> None:
        super().reset()
        self._done = {}

    def tick(self, ctx: MissionContext) -> Status:
        for i, c in enumerate(self.children):
            if i in self._done:
                continue
            st = c.tick(ctx)
            if st is not R:
                self._done[i] = st
        ok = sum(1 for v in self._done.values() if v is S)
        bad = sum(1 for v in self._done.values() if v is F)
        need = len(self.children) if self.policy == "all" else 1 if self.policy == "any" else int(self.policy)
        if ok >= need:
            self.reset()
            return S
        if bad > len(self.children) - need:
            self.reset()
            return F
        return R


class _Decorator(Node):
    def __init__(self, child: Node, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.child = child

    def children_nodes(self) -> List[Node]:
        return [self.child]

    def reset(self) -> None:
        self.child.reset()


class Inverter(_Decorator):
    def tick(self, ctx: MissionContext) -> Status:
        st = self.child.tick(ctx)
        return F if st is S else S if st is F else R


class ForceSuccess(_Decorator):
    def tick(self, ctx: MissionContext) -> Status:
        st = self.child.tick(ctx)
        return R if st is R else S


class ForceFailure(_Decorator):
    def tick(self, ctx: MissionContext) -> Status:
        st = self.child.tick(ctx)
        return R if st is R else F


class Repeat(_Decorator):
    """Run the child ``times`` times in a row (``times=None``: forever); fails as soon as one run fails."""

    def __init__(self, child: Node, times: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(child, name)
        self.times, self._n = times, 0

    def reset(self) -> None:
        super().reset()
        self._n = 0

    def tick(self, ctx: MissionContext) -> Status:
        st = self.child.tick(ctx)
        if st is F:
            self.reset()
            return F
        if st is S:
            self._n += 1
            if self.times is not None and self._n >= self.times:
                self.reset()
                return S
        return R


class Retry(_Decorator):
    """Run the child until it succeeds, at most ``attempts`` times."""

    def __init__(self, child: Node, attempts: int = 3, name: Optional[str] = None) -> None:
        super().__init__(child, name)
        self.attempts, self._n = int(attempts), 0

    def reset(self) -> None:
        super().reset()
        self._n = 0

    def tick(self, ctx: MissionContext) -> Status:
        st = self.child.tick(ctx)
        if st is S:
            self.reset()
            return S
        if st is F:
            self._n += 1
            if self._n >= self.attempts:
                self.reset()
                return F
            self.child.reset()
        return R


class Timeout(_Decorator):
    """Fail if the child is still running after ``seconds`` of mission time."""

    def __init__(self, child: Node, seconds: float, name: Optional[str] = None) -> None:
        super().__init__(child, name)
        self.seconds = float(seconds)
        self._t0: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._t0 = None

    def tick(self, ctx: MissionContext) -> Status:
        if self._t0 is None:
            self._t0 = ctx.t
        st = self.child.tick(ctx)
        if st is not R:
            self.reset()
            return st
        if ctx.t - self._t0 >= self.seconds:
            self.child.reset()
            self.reset()
            return F
        return R


class Condition(Node):
    """A leaf that returns ``SUCCESS`` when ``fn(ctx)`` is true and ``FAILURE`` otherwise."""

    def __init__(self, fn: Callable[[MissionContext], bool], name: Optional[str] = None) -> None:
        super().__init__(name or getattr(fn, "__name__", "Condition"))
        self.fn = fn

    def tick(self, ctx: MissionContext) -> Status:
        return S if self.fn(ctx) else F


class Action(Node):
    """A leaf calling ``fn(ctx)``, which returns a :class:`Status` (or ``True``/``False``/``None`` for success/failure/running)."""

    def __init__(self, fn: Callable[[MissionContext], Any], name: Optional[str] = None) -> None:
        super().__init__(name or getattr(fn, "__name__", "Action"))
        self.fn = fn

    def tick(self, ctx: MissionContext) -> Status:
        r = self.fn(ctx)
        return r if isinstance(r, Status) else R if r is None else S if r else F


class Wait(Node):
    """Running for ``seconds`` of mission time, then success (the reference keeps its last value)."""

    def __init__(self, seconds: float, name: Optional[str] = None) -> None:
        super().__init__(name or f"Wait({seconds:g}s)")
        self.seconds = float(seconds)
        self._t0: Optional[float] = None

    def reset(self) -> None:
        self._t0 = None

    def tick(self, ctx: MissionContext) -> Status:
        if self._t0 is None:
            self._t0 = ctx.t
        if ctx.t - self._t0 >= self.seconds - 1e-9:
            self._t0 = None
            return S
        return R


class BatteryAbove(Condition):
    """Success while the observed state of charge is at least ``level`` (success when no battery has been observed)."""

    def __init__(self, level: float, name: Optional[str] = None) -> None:
        lv = float(level)
        super().__init__(lambda c: c.battery is None or c.battery >= lv, name or f"BatteryAbove({level:g})")


class AltitudeAbove(Condition):
    def __init__(self, z: float, name: Optional[str] = None) -> None:
        zz = float(z)
        super().__init__(lambda c: float(c.pos[2]) >= zz, name or f"AltitudeAbove({z:g})")


# ======================================================================================================== mission actions
class _Move(Node):
    """Base for actions that fly a smooth trajectory from the current reference state to a target."""

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self._traj: Any = None
        self._t0 = 0.0

    def reset(self) -> None:
        self._traj = None

    def _start(self, ctx: MissionContext, target: np.ndarray, v_max: float, a_max: float, plan: bool = True) -> bool:
        from .trajectory import plan_path, plan_trajectory

        start = ctx.pos.copy()
        if float(np.linalg.norm(target - start)) < 1e-6 and float(np.linalg.norm(ctx.vel)) < 1e-6:
            self._traj = None
            return True
        path = np.vstack([start, target])
        space = ctx.space if plan else None
        if space is not None and not space.segment_free(start, target):
            res = plan_path(space, start, target, ctx.planner)
            if not res.success:
                ctx.log.append(f"{self.name}: no path ({res.reason})")
                return False
            path = res.path
        sd = np.vstack([ctx.vel, ctx.acc, ctx.jerk])
        try:
            self._traj = plan_trajectory(space, path, v_max=v_max, a_max=a_max, yaw=ctx.yaw, start_derivs=sd)
        except RuntimeError as ex:
            ctx.log.append(f"{self.name}: {ex}")
            return False
        self._t0 = ctx.t
        return True

    def _follow(self, ctx: MissionContext) -> Status:
        if self._traj is None:
            return S
        tt = ctx.t - self._t0
        p, v, a, j, _, _ = self._traj.sample(tt)
        ctx.set_ref(p, v, a, j)
        if tt >= self._traj.duration - 1e-9:
            self._traj = None
            return S
        return R


class Takeoff(_Move):
    """Climb vertically to ``altitude`` (metres above the ground)."""

    def __init__(self, altitude: float = 3.0, speed: Optional[float] = None, name: Optional[str] = None) -> None:
        super().__init__(name or f"Takeoff({altitude:g} m)")
        self.altitude, self.speed = float(altitude), speed
        self._started = False

    def reset(self) -> None:
        super().reset()
        self._started = False

    def tick(self, ctx: MissionContext) -> Status:
        if not self._started:
            target = ctx.pos.copy()
            target[2] = ctx.ground_z + self.altitude
            if not self._start(ctx, target, self.speed or ctx.v_max * 0.5, ctx.a_max, plan=False):
                return F
            self._started = True
        st = self._follow(ctx)
        if st is not R:
            self._started = False
        return st


class GoTo(_Move):
    """Fly to a waypoint, planning around obstacles when the context has a planning space. ``tolerance`` (optional) also requires
    the *vehicle* to be that close (needs observations) before the action succeeds."""

    def __init__(
        self,
        waypoint: Seq[float],
        speed: Optional[float] = None,
        tolerance: Optional[float] = None,
        retry_after: float = 5.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f"GoTo({', '.join(f'{x:g}' for x in waypoint)})")
        self.waypoint = np.asarray(waypoint, float)
        self.speed, self.tol, self.retry_after = speed, tolerance, float(retry_after)
        self._started = False

    def reset(self) -> None:
        super().reset()
        self._started = False

    def tick(self, ctx: MissionContext) -> Status:
        if not self._started:
            key = ("goto_failed", id(self))  # a priority list re-ticks its first child every step: do not re-plan a hopeless route each time
            failed = ctx.blackboard.get(key)
            if failed is not None and ctx.t - failed < self.retry_after:
                return F
            if not self._start(ctx, self.waypoint, self.speed or ctx.v_max, ctx.a_max):
                ctx.blackboard[key] = ctx.t
                return F
            ctx.blackboard.pop(key, None)
            self._started = True
        st = self._follow(ctx)
        if st is S and self.tol is not None and ctx.vehicle_pos is not None:
            if float(np.linalg.norm(ctx.vehicle_pos - self.waypoint)) > self.tol:
                return R  # arrived on paper, not yet in reality
        if st is not R:
            self._started = False
        return st


class Hover(Node):
    """Hold the current reference for ``seconds``."""

    def __init__(self, seconds: float = 2.0, name: Optional[str] = None) -> None:
        super().__init__(name or f"Hover({seconds:g}s)")
        self.seconds = float(seconds)
        self._t0: Optional[float] = None

    def reset(self) -> None:
        self._t0 = None

    def tick(self, ctx: MissionContext) -> Status:
        if self._t0 is None:
            self._t0 = ctx.t
            ctx.set_ref(ctx.pos)
        if ctx.t - self._t0 >= self.seconds - 1e-9:
            self._t0 = None
            return S
        return R


class Land(_Move):
    """Descend to the ground at ``speed`` m/s above the ground, then succeed and mark the context ``landed``."""

    def __init__(self, speed: float = 0.8, name: Optional[str] = None) -> None:
        super().__init__(name or "Land")
        self.speed, self._started = float(speed), False

    def reset(self) -> None:
        super().reset()
        self._started = False

    def tick(self, ctx: MissionContext) -> Status:
        if not self._started:
            target = ctx.pos.copy()
            target[2] = ctx.ground_z
            if not self._start(ctx, target, self.speed, ctx.a_max * 0.5, plan=False):
                return F
            self._started = True
        st = self._follow(ctx)
        if st is S:
            ctx.blackboard["landed"] = True
            self._started = False
        return st


# ======================================================================================================== mission trajectory
class BehaviorTreeMission(Trajectory):
    """A mission written as a behaviour tree, usable wherever a trajectory is (``trajectory: {type: mission, ...}``).

    ``sample(t)`` ticks the tree once per distinct time and returns the reference state the action nodes maintain. Call
    :meth:`observe` with the vehicle state (the engine does) to let conditions see the battery and position. When the tree
    finishes, the reference holds its last pose; ``on_finish`` is ``"hold"`` or ``"stop"`` (ends the run, ``duration`` becomes
    the finishing time).
    """

    def __init__(
        self,
        tree: Node,
        space: Any = None,
        home: Seq[float] = (0.0, 0.0, 0.0),
        v_max: float = 3.0,
        a_max: float = 2.5,
        planner: str = "theta_star",
        yaw: float = 0.0,
        on_finish: str = "hold",
    ) -> None:
        self.tree = tree
        self.ctx = MissionContext(space, home, v_max, a_max, planner)
        self.ctx.yaw = float(yaw)
        self.on_finish = on_finish
        self.finished: Optional[Status] = None
        self.finished_at: Optional[float] = None
        self._last_t: Optional[float] = None
        self.duration = math.inf
        self._home0 = np.array(home, float)

    def reset(self) -> None:
        self.tree.reset()
        c = self.ctx
        c.t, c.dt, c.blackboard, c.log = 0.0, 0.0, {}, []
        c.set_ref(self._home0)
        self.finished = self.finished_at = self._last_t = None
        self.duration = math.inf

    def observe(self, pos: Any = None, soc: Optional[float] = None) -> None:
        if pos is not None:
            self.ctx.vehicle_pos = np.asarray(pos, float)
        if soc is not None:
            self.ctx.battery = float(soc)

    def sample(self, t: float) -> Any:
        c = self.ctx
        if self._last_t is None or t > self._last_t + 1e-12:
            c.dt = 0.0 if self._last_t is None else t - self._last_t
            c.t = float(t)
            self._last_t = float(t)
            if self.finished is None:
                st = self.tree.tick(c)
                if st is not R:
                    self.finished, self.finished_at = st, float(t)
                    if self.on_finish == "stop":
                        self.duration = float(t)
        return c.pos.copy(), c.vel.copy(), c.acc.copy(), c.jerk.copy(), float(c.yaw), 0.0


# ======================================================================================================== declarative trees
_LEAVES: Dict[str, Callable[..., Node]] = {
    "takeoff": lambda altitude=3.0, speed=None, **k: Takeoff(altitude, speed, **k),
    "goto": lambda waypoint, speed=None, tolerance=None, retry_after=5.0, **k: GoTo(waypoint, speed, tolerance, retry_after, **k),
    "hover": lambda seconds=2.0, **k: Hover(seconds, **k),
    "land": lambda speed=0.8, **k: Land(speed, **k),
    "wait": lambda seconds, **k: Wait(seconds, **k),
    "battery_above": lambda level, **k: BatteryAbove(level, **k),
    "altitude_above": lambda z, **k: AltitudeAbove(z, **k),
}


def tree_from_dict(spec: Any) -> Node:
    """Build a tree from nested dictionaries, e.g.::

        {"type": "sequence", "children": [
            {"type": "takeoff", "altitude": 5},
            {"type": "goto", "waypoint": [10, 0, 5]},
            {"type": "fallback", "children": [{"type": "battery_above", "level": 0.3}, {"type": "land"}]}]}

    Composites: ``sequence``, ``reactive_sequence``, ``fallback``, ``parallel`` (``policy``). Decorators (``child``): ``inverter``,
    ``force_success``, ``force_failure``, ``repeat`` (``times``), ``retry`` (``attempts``), ``timeout`` (``seconds``).
    """
    if isinstance(spec, Node):
        return spec
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"a tree node needs a dictionary with a 'type', got {spec!r}")
    d = dict(spec)
    kind = str(d.pop("type")).lower()
    name = d.pop("name", None)
    if kind in ("sequence", "reactive_sequence", "fallback", "parallel"):
        kids = [tree_from_dict(c) for c in d.pop("children", [])]
        if kind == "sequence":
            return Sequence(kids, memory=d.pop("memory", True), name=name)
        if kind == "reactive_sequence":
            return Sequence(kids, memory=False, name=name)
        if kind == "fallback":
            return Fallback(kids, memory=d.pop("memory", False), name=name)
        return Parallel(kids, policy=d.pop("policy", "all"), name=name)
    deco = {"inverter": Inverter, "force_success": ForceSuccess, "force_failure": ForceFailure, "repeat": Repeat, "retry": Retry, "timeout": Timeout}
    if kind in deco:
        child = tree_from_dict(d.pop("child"))
        return deco[kind](child, name=name, **d)
    if kind in _LEAVES:
        return _LEAVES[kind](**d, **({"name": name} if name else {}))
    raise ValueError(
        f"unknown behaviour-tree node {kind!r}; known: sequence, reactive_sequence, fallback, parallel, {', '.join(sorted(list(deco) + list(_LEAVES)))}"
    )
