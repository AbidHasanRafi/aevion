"""Behaviour trees, missions, and the planned / mission trajectories inside the engine."""

import numpy as np
import pytest

from aevion.engine import Simulation
from aevion.environments.world import World
from aevion.planning import BehaviorTreeMission, PlanningSpace, tree_from_dict
from aevion.planning.behavior import (
    Action,
    BatteryAbove,
    Condition,
    Fallback,
    ForceFailure,
    ForceSuccess,
    GoTo,
    Inverter,
    Land,
    MissionContext,
    Parallel,
    Repeat,
    Retry,
    Sequence,
    Status,
    Takeoff,
    Timeout,
    Wait,
)

S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING


def leaf(*results):
    """An action returning the given statuses in turn (then repeating the last)."""
    it = iter(results)
    last = [results[-1]]

    def fn(ctx):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return Action(fn)


def counting(status):
    calls = []

    def fn(ctx):
        calls.append(ctx.t)
        return status

    n = Action(fn)
    n.calls = calls
    return n


# ------------------------------------------------------------------------------------------------ tree semantics
def test_sequence_and_fallback_truth_tables():
    ctx = MissionContext()
    assert Sequence([leaf(S), leaf(S)]).tick(ctx) is S
    assert Sequence([leaf(S), leaf(F), leaf(S)]).tick(ctx) is F
    assert Sequence([leaf(S), leaf(R), leaf(F)]).tick(ctx) is R
    assert Fallback([leaf(F), leaf(F)]).tick(ctx) is F
    assert Fallback([leaf(F), leaf(S), leaf(F)]).tick(ctx) is S
    assert Fallback([leaf(F), leaf(R)]).tick(ctx) is R
    with pytest.raises(ValueError):
        Sequence([])


def test_sequence_with_memory_resumes_and_reactive_sequence_rechecks_its_conditions():
    flag = {"ok": True}
    cond = Condition(lambda c: flag["ok"], "guard")
    work = leaf(R, R, R, S)
    mem = Sequence([cond, work], memory=True)
    flag["ok"] = True
    assert mem.tick(MissionContext()) is R
    flag["ok"] = False  # a memory sequence does not look at the guard again while the second child runs
    assert mem.tick(MissionContext()) is R
    flag["ok"] = True
    work2 = leaf(R, R, R, S)
    reactive = Sequence([cond, work2], memory=False)
    assert reactive.tick(MissionContext()) is R
    flag["ok"] = False
    assert reactive.tick(MissionContext()) is F  # the guard is re-evaluated every tick and now fails


def test_fallback_gives_priority_to_the_first_child_even_while_a_later_one_runs():
    cheap_ok = {"v": False}
    first = Condition(lambda c: cheap_ok["v"], "preferred")
    long = counting(R)
    fb = Fallback([first, long])
    assert fb.tick(MissionContext()) is R and len(long.calls) == 1
    cheap_ok["v"] = True
    assert fb.tick(MissionContext()) is S and len(long.calls) == 1  # the preferred plan took over; the other was not ticked again


def test_parallel_policies():
    ctx = MissionContext()
    assert Parallel([leaf(S), leaf(S)], "all").tick(ctx) is S
    p = Parallel([leaf(S), leaf(R, S)], "all")
    assert p.tick(ctx) is R and p.tick(ctx) is S
    assert Parallel([leaf(R), leaf(S)], "any").tick(ctx) is S
    assert Parallel([leaf(F), leaf(S), leaf(S)], 2).tick(ctx) is S
    assert Parallel([leaf(F), leaf(F), leaf(S)], 2).tick(ctx) is F
    assert Parallel([leaf(F), leaf(R)], "all").tick(ctx) is F


def test_decorators():
    ctx = MissionContext()
    assert Inverter(leaf(S)).tick(ctx) is F and Inverter(leaf(F)).tick(ctx) is S and Inverter(leaf(R)).tick(ctx) is R
    assert ForceSuccess(leaf(F)).tick(ctx) is S and ForceFailure(leaf(S)).tick(ctx) is F and ForceSuccess(leaf(R)).tick(ctx) is R
    rep = Repeat(counting(S), times=3)
    assert [rep.tick(ctx) for _ in range(3)] == [R, R, S]
    assert Repeat(leaf(S, F), times=5).tick(ctx) is R
    r = Repeat(leaf(S, F), times=5)
    r.tick(ctx)
    assert r.tick(ctx) is F
    retry = Retry(leaf(F, F, S), attempts=3)
    assert [retry.tick(ctx) for _ in range(3)] == [R, R, S]
    assert Retry(leaf(F), attempts=2).tick(ctx) is R
    r2 = Retry(leaf(F), attempts=2)
    r2.tick(ctx)
    assert r2.tick(ctx) is F


def test_timeout_and_wait_follow_mission_time():
    ctx = MissionContext()
    to = Timeout(leaf(R), seconds=2.0)
    ctx.t = 10.0
    assert to.tick(ctx) is R
    ctx.t = 11.5
    assert to.tick(ctx) is R
    ctx.t = 12.0
    assert to.tick(ctx) is F
    w = Wait(1.0)
    ctx.t = 20.0
    assert w.tick(ctx) is R
    ctx.t = 21.0
    assert w.tick(ctx) is S
    assert Timeout(leaf(S), 1.0).tick(ctx) is S


def test_conditions_actions_and_the_description():
    ctx = MissionContext()
    ctx.battery = 0.2
    assert BatteryAbove(0.3).tick(ctx) is F
    ctx.battery = None
    assert BatteryAbove(0.3).tick(ctx) is S  # unobserved: do not abort
    assert Action(lambda c: True).tick(ctx) is S and Action(lambda c: False).tick(ctx) is F and Action(lambda c: None).tick(ctx) is R
    tree = Sequence([Takeoff(5), Fallback([BatteryAbove(0.3), Land()])], name="mission")
    d = tree.describe()
    assert "mission" in d and "Takeoff(5 m)" in d and "Land" in d and d.count("\n") == 4


def test_trees_from_dictionaries_and_their_validation():
    spec = {
        "type": "sequence",
        "children": [
            {"type": "takeoff", "altitude": 4},
            {"type": "retry", "attempts": 2, "child": {"type": "goto", "waypoint": [5, 0, 4]}},
            {"type": "fallback", "children": [{"type": "battery_above", "level": 0.3}, {"type": "land"}]},
            {"type": "parallel", "policy": "any", "children": [{"type": "hover", "seconds": 1}, {"type": "wait", "seconds": 5}]},
        ],
    }
    tree = tree_from_dict(spec)
    assert isinstance(tree, Sequence) and len(tree.children) == 4 and isinstance(tree.children[1].child, GoTo)
    for bad in ({"type": "dance"}, {"children": []}, "sequence", {"type": "sequence", "children": [{"type": "goto"}]}):
        with pytest.raises((ValueError, TypeError)):
            tree_from_dict(bad)


# ------------------------------------------------------------------------------------------------ kinematic missions
def run_mission(tree, space=None, seconds=60.0, dt=0.02, home=(0.0, 0.0, 0.0), observe=None, **kw):
    m = BehaviorTreeMission(tree, space, home, **kw)
    m.reset()
    out = []
    for k in range(int(seconds / dt)):
        t = k * dt
        if observe:
            observe(m, t)
        p, v, a, j, yaw, _ = m.sample(t)
        out.append((t, p, v, a))
        if m.finished is not None and t - m.finished_at > 0.5:
            break
    return m, out


def test_takeoff_goto_hover_land_flies_a_smooth_reference_and_ends_on_the_ground():
    tree = tree_from_dict(
        {
            "type": "sequence",
            "children": [
                {"type": "takeoff", "altitude": 5},
                {"type": "goto", "waypoint": [12, 4, 5]},
                {"type": "hover", "seconds": 1.0},
                {"type": "land"},
            ],
        }
    )
    m, out = run_mission(tree, v_max=3.0, a_max=2.0)
    assert m.finished is S and m.ctx.blackboard["landed"]
    P = np.array([o[1] for o in out])
    V = np.array([o[2] for o in out])
    A = np.array([o[3] for o in out])
    assert P[:, 2].max() == pytest.approx(5.0, abs=0.05) and np.allclose(P[-1], [12, 4, 0.0], atol=0.02)
    assert np.linalg.norm(V, axis=1).max() <= 3.0 * 1.02 and np.linalg.norm(A, axis=1).max() <= 2.0 * 1.05
    assert np.abs(np.diff(V, axis=0)).max() < 0.2  # no velocity jumps at the action boundaries (start derivatives are carried over)
    assert np.allclose(V[-1], 0.0, atol=1e-3)


def test_goto_plans_around_obstacles_when_the_context_has_a_space():
    w = World()
    w.add_box((10.0, 0.0, 5.0), (2.0, 20.0, 10.0))  # a wall across the straight line, open above 10 m and beyond |y| = 10
    space = PlanningSpace.from_world(w, bounds=(-5, -15, 0, 25, 15, 14), resolution=0.5, inflation=0.5, ground=False)
    tree = tree_from_dict({"type": "goto", "waypoint": [20, 0, 5]})
    m, out = run_mission(tree, space, home=(0.0, 0.0, 5.0), v_max=4.0, a_max=3.0, seconds=60.0)
    assert m.finished is S
    P = np.array([o[1] for o in out])
    assert (space.is_free(P).all() and P[:, 2].max() > 10.5) or np.abs(P[:, 1]).max() > 10.0  # went over or round the wall
    assert np.allclose(P[-1], [20, 0, 5], atol=0.05)


def test_an_unreachable_waypoint_fails_the_action_and_a_fallback_recovers():
    w = World()
    w.add_box((10.0, 0.0, 5.0), (2.0, 40.0, 10.0))
    space = PlanningSpace.from_world(w, bounds=(-5, -15, 0, 25, 15, 10.4), resolution=0.5, inflation=0.5, ground=False)
    space.occ[:, :, -1:] = True  # the ceiling is closed: the wall spans the whole space
    tree = Fallback([GoTo([20, 0, 5], name="far side"), GoTo([3, 0, 5], name="stay near")])
    m, out = run_mission(tree, space, home=(0.0, 0.0, 5.0), seconds=30.0)
    assert m.finished is S and any("no path" in line for line in m.ctx.log)
    assert np.allclose(out[-1][1], [3, 0, 5], atol=0.05)


def test_the_battery_condition_diverts_the_mission_home():
    tree = tree_from_dict(
        {
            "type": "fallback",
            "children": [
                {
                    "type": "sequence",
                    "memory": False,
                    "children": [{"type": "battery_above", "level": 0.3}, {"type": "goto", "waypoint": [40, 0, 5]}],
                },
                {"type": "sequence", "children": [{"type": "goto", "waypoint": [0, 0, 5]}, {"type": "land"}]},
            ],
        }
    )

    def drain(m, t):
        m.observe(soc=1.0 - 0.045 * t)  # crosses 30 % at t = 15.6 s

    m, out = run_mission(tree, None, home=(0.0, 0.0, 5.0), seconds=80.0, observe=drain, v_max=3.0)
    P = np.array([o[1] for o in out])
    assert m.finished is S and m.ctx.blackboard["landed"]
    assert P[:, 0].max() > 15.0 and np.allclose(P[-1], [0, 0, 0], atol=0.05)  # it set off, turned back, and landed at home


def test_missions_are_deterministic_and_reset_cleanly():
    spec = {"type": "sequence", "children": [{"type": "takeoff", "altitude": 3}, {"type": "goto", "waypoint": [6, 0, 3]}]}
    m1, o1 = run_mission(tree_from_dict(spec), seconds=20.0)
    m1.reset()
    o1b = []
    for k in range(len(o1)):
        o1b.append(m1.sample(k * 0.02)[0])
    assert np.array_equal(np.array([o[1] for o in o1]), np.array(o1b))
    m2, o2 = run_mission(tree_from_dict(spec), seconds=20.0)
    assert np.array_equal(np.array([o[1] for o in o1]), np.array([o[1] for o in o2]))


def test_a_mission_can_end_the_run_when_it_finishes():
    m = BehaviorTreeMission(tree_from_dict({"type": "hover", "seconds": 2.0}), on_finish="stop", home=(0, 0, 1))
    m.reset()
    for k in range(200):
        m.sample(k * 0.02)
    assert m.finished is S and m.duration == pytest.approx(m.finished_at)


# ------------------------------------------------------------------------------------------------ in the engine
BOXES = [{"center": [15, 0, 6], "size": [4, 14, 12]}, {"center": [8, -8, 4], "size": [4, 4, 8]}]


def _world(boxes=BOXES):
    return {"type": "scene", "params": {"boxes": boxes, "bounds": [-5, -15, 0, 35, 15, 20]}}


def test_the_planned_trajectory_flies_around_buildings_without_a_collision():
    cfg = {
        "duration": 22.0,
        "world": _world(),
        "trajectory": {
            "type": "planned",
            "params": {"start": [0, 0, 3], "goal": [30, 0, 3], "v_max": 4.0, "a_max": 3.0, "inflation": 0.8, "resolution": 1.0},
        },
    }
    sim = Simulation(cfg)
    arch = sim.run()
    pos = arch["pos"][:, 0]
    tr = sim.traj
    assert tr.info["peak_speed"] <= 4.0 * 1.02 and tr.info["planner"] == "theta_star"
    assert np.linalg.norm(pos[-1] - [30, 0, 3]) < 0.2 and np.linalg.norm(pos - arch["ref_pos"][:, 0], axis=1).max() < 0.3
    assert not any(e["type"] == "collision" for e in arch.meta["events"])
    dist_to_box = lambda p, c, s: np.linalg.norm(np.maximum(np.abs(p - np.asarray(c)) - np.asarray(s) / 2.0, 0.0))
    assert min(dist_to_box(p, b["center"], b["size"]) for p in pos[::10] for b in BOXES) > 0.4  # clear of the walls (vehicle radius + slack)


def test_an_impossible_goal_is_reported_when_the_simulation_is_built():
    cfg = {
        "duration": 5.0,
        "world": _world(),
        "trajectory": {"type": "planned", "params": {"start": [0, 0, 3], "goal": [15, 0, 6], "resolution": 1.0}},
    }
    with pytest.raises(RuntimeError, match="no theta_star path"):  # the goal is inside the building
        Simulation(cfg)


def test_a_mission_in_the_engine_lands_and_reacts_to_the_battery():
    tree = {
        "type": "sequence",
        "children": [
            {"type": "takeoff", "altitude": 4},
            {"type": "goto", "waypoint": [10, 0, 4]},
            {"type": "land"},
        ],
    }
    cfg = {
        "duration": 30.0,
        "world": _world([]),
        "trajectory": {"type": "mission", "params": {"tree": tree, "home": [0, 0, 0.05], "v_max": 3.0, "resolution": 1.0}},
    }
    sim = Simulation(cfg)
    arch = sim.run()
    pos = arch["pos"][:, 0]
    assert sim.traj.finished is S and pos[:, 2].max() > 3.8 and np.linalg.norm(pos[-1, :2] - [10, 0]) < 0.3 and pos[-1, 2] < 0.3
    # the same mission, started with a nearly empty battery, is cut short by the guard
    guarded = {
        "type": "fallback",
        "children": [
            {"type": "sequence", "memory": False, "children": [{"type": "battery_above", "level": 0.3}, tree]},
            {"type": "land"},
        ],
    }
    cfg2 = {
        **cfg,
        "trajectory": {"type": "mission", "params": {"tree": guarded, "home": [0, 0, 0.05], "resolution": 1.0}},
        "vehicle": {"initial": {"soc": 0.2}},
    }
    sim2 = Simulation(cfg2)
    arch2 = sim2.run()
    assert arch2["pos"][:, 0, 2].max() < 0.5  # never took off: the first branch failed its battery check, the fallback landed on the spot
    assert sim2.traj.ctx.blackboard.get("landed")


def test_planned_and_mission_runs_are_bit_reproducible_from_their_archives():
    cfg = {
        "duration": 8.0,
        "world": _world([]),
        "trajectory": {"type": "planned", "params": {"start": [0, 0, 2], "goal": [10, 5, 3], "resolution": 1.0}},
    }
    a = Simulation(cfg).run()
    b = Simulation(cfg).run()
    assert np.array_equal(a["pos"], b["pos"])
    from aevion.scenario import verify

    assert verify(a)["ok"]


def test_planning_example_runs_end_to_end(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [sys.executable, str(root / "examples" / "14_planning_estimation_tour.py"), "--out", str(tmp_path), "--duration", "6"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-2500:]
    assert "theta_star" in res.stdout and "estimator" in res.stdout and "collision False" in res.stdout
    assert (tmp_path / "replay.html").stat().st_size > 500_000
