"""Dynamic actors and geofences."""

import json
import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments.actors import BouncingActor, CircularActor, LinearActor, RandomWalkActor, WaypointActor, actor_from_dict
from aevion.environments.geofence import (
    AltitudeBand,
    BoxZone,
    CylinderZone,
    Geofence,
    GeofenceCheck,
    GeofenceSet,
    PolygonZone,
    SphereZone,
    zone_from_dict,
)
from aevion.environments.world import World
from aevion.sensors.raycast import cast_rays

B = NumpyBackend()


# ----------------------------------------------------------------------------------------------- actors
def test_linear_and_circular_motion_are_analytic():
    a = LinearActor(start=(1, 2, 3), velocity=(2, 0, -1))
    np.testing.assert_allclose(a.position(4.0), [9, 2, -1])
    np.testing.assert_allclose(a.velocity(4.0), [2, 0, -1])
    c = CircularActor(center=(5, 5, 8), path_radius=10.0, omega=0.5, phase=0.3)
    for t in (0.0, 1.7, 9.0):
        p = c.position(t) - [5, 5, 8]
        assert np.linalg.norm(p[:2]) == pytest.approx(10.0) and p[2] == 0.0
        assert np.linalg.norm(c.velocity(t)) == pytest.approx(5.0)  # r * omega
        assert np.dot(c.velocity(t), p) == pytest.approx(0.0, abs=1e-9)  # tangent
    h = 1e-5
    np.testing.assert_allclose(c.velocity(2.0), (c.position(2.0 + h) - c.position(2.0 - h)) / (2 * h), atol=1e-6)


def test_waypoint_modes():
    pts = [(0, 0, 0), (10, 0, 0), (10, 10, 0)]
    loop = WaypointActor(pts, speed=2.0, mode="loop")
    assert loop._len == pytest.approx(10 + 10 + math.hypot(10, 10))
    np.testing.assert_allclose(loop.position(5.0), [10, 0, 0])
    np.testing.assert_allclose(loop.position(10.0), [10, 10, 0])
    np.testing.assert_allclose(loop.position(loop._len / 2.0), [0, 0, 0], atol=1e-9)  # one full lap
    np.testing.assert_allclose(loop.velocity(1.0), [2, 0, 0])
    pp = WaypointActor(pts, speed=5.0, mode="pingpong")
    np.testing.assert_allclose(pp.position(4.0), [10, 10, 0])  # 20 m: the far end
    np.testing.assert_allclose(pp.position(5.0), [10, 5, 0])  # on the way back
    np.testing.assert_allclose(pp.velocity(5.0), [0, -5, 0])
    stop = WaypointActor(pts, speed=5.0, mode="stop")
    np.testing.assert_allclose(stop.position(100.0), [10, 10, 0])
    np.testing.assert_allclose(stop.velocity(100.0), 0.0)
    assert np.linalg.norm(loop.velocity(3.3)) == pytest.approx(2.0)


def test_bouncing_actor_stays_inside_and_reflects():
    a = BouncingActor(start=(0, 0, 5), velocity=(3.0, 2.0, 1.0), bounds=(-10, -10, 2, 10, 10, 12))
    ts = np.linspace(0, 200, 4001)
    P = np.array([a.position(t) for t in ts])
    assert np.all(np.array([-10, -10, 2]) - 1e-9 <= P) and np.all(np.array([10, 10, 12]) + 1e-9 >= P)
    assert np.any(np.abs(P[:, 0]) > 9.9)  # it does reach the walls
    assert a.velocity(0.0)[0] == 3.0 and a.velocity(4.0)[0] == -3.0  # x = 12 unfolds to 8 on the way back
    np.testing.assert_allclose(a.position(10 / 3.0), [10, 6.6667, 8.3333], atol=1e-3)
    with pytest.raises(ValueError):
        BouncingActor(start=(50, 0, 5), bounds=(-10, -10, 2, 10, 10, 12))


def test_random_walk_is_deterministic_bounded_and_query_order_independent():
    a, b = RandomWalkActor(seed=3), RandomWalkActor(seed=3)
    ts = [0.0, 7.0, 3.3, 20.0, 1.1]
    forward = [a.position(t) for t in ts]
    backward = [b.position(t) for t in reversed(ts)][::-1]
    np.testing.assert_allclose(forward, backward)
    assert not np.allclose(RandomWalkActor(seed=4).position(7.0), forward[1])
    P = np.array([a.position(t) for t in np.linspace(0, 120, 500)])
    assert np.all(a.bounds[:3] - 1e-9 <= P) and np.all(a.bounds[3:] + 1e-9 >= P)
    assert np.std(P[:, 0]) > 1.0  # it actually wanders
    assert np.linalg.norm(a.velocity(5.0)) < 20.0 and a.position(0.0) == pytest.approx(a.start)


def test_actor_validation_and_serialisation():
    for bad in (dict(shape="torus"), dict(radius=0.0), dict(size=(1, 0, 1))):
        with pytest.raises(ValueError):
            LinearActor(**bad)
    with pytest.raises(ValueError):
        WaypointActor([(0, 0, 0)])
    with pytest.raises(ValueError):
        WaypointActor([(0, 0, 0), (0, 0, 0)])
    with pytest.raises(ValueError):
        CircularActor(path_radius=-1)
    with pytest.raises(ValueError):
        LinearActor(start=(0, 0, float("nan")))
    for a in (
        LinearActor(shape="box", size=(2, 2, 2), color=(1, 0, 0)),
        CircularActor(),
        WaypointActor(mode="pingpong"),
        BouncingActor(),
        RandomWalkActor(seed=9),
    ):
        again = actor_from_dict(json.loads(json.dumps(a.to_dict())))
        assert type(again) is type(a) and again.to_dict() == a.to_dict()
        np.testing.assert_allclose(again.position(3.0), a.position(3.0))
    with pytest.raises(ValueError, match="unknown actor"):
        actor_from_dict({"type": "ufo"})


# ------------------------------------------------------------------------------------- world integration
def test_world_moves_actors_and_every_query_sees_them():
    w = World()
    w.add_actor(LinearActor(start=(20, 0, 3), velocity=(-2.0, 0, 0), radius=1.0))  # a sphere flying towards the origin
    w.add_actor(CircularActor(center=(0, 30, 3), path_radius=5, omega=0.4, shape="box", size=(2, 2, 2)))
    w.add_actor(LinearActor(start=(0, -20, 0), velocity=(0, 0, 0), shape="cylinder", radius=1.0, height=6.0))
    ray = (np.array([0.0, 0.0, 3.0]), np.array([[1.0, 0, 0]]))
    w.advance(0.0)
    assert cast_rays(w, ray[0], ray[1], 100.0).t[0] == pytest.approx(19.0)
    w.advance(5.0)  # sphere now at x = 10
    assert cast_rays(w, ray[0], ray[1], 100.0).t[0] == pytest.approx(9.0)
    assert w.distance_to_nearest(np.array([[0.0, 0.0, 3.0]]))[0] == pytest.approx(9.0)
    hit, depth, _ = w.collide(B, np.array([[8.6, 0.0, 3.0]]), 0.5)  # actor sphere (r = 1) at x = 10: contact within 1.5 m
    assert hit[0] and depth[0] == pytest.approx(0.1)
    ids = cast_rays(w, ray[0], ray[1], 100.0).prim
    np.testing.assert_allclose(w.primitive_velocity(ids, 5.0), [[-2.0, 0, 0]])
    assert np.allclose(
        w.primitive_velocity(np.array([-1, -2, 0]), 5.0), np.array([[0, 0, 0], [0, 0, 0], w.actors[1][0].velocity(5.0)])
    )  # misses/ground: zero
    w.advance(5.0)
    assert w.cylinders[0, 3] - w.cylinders[0, 2] == pytest.approx(6.0)


def test_actor_world_round_trip_does_not_duplicate_primitives():
    w = World()
    w.add_box((50, 0, 1), (2, 2, 2))
    w.add_actor(LinearActor(start=(0, 0, 3), velocity=(1, 0, 0)))
    w.advance(2.0)
    w2 = World.from_dict(json.loads(json.dumps(w.to_dict())))
    assert w2.boxes.shape == w.boxes.shape and w2.spheres.shape == w.spheres.shape and len(w2.actors) == 1
    w2.advance(10.0)
    assert w2.spheres[0, 0] == pytest.approx(10.0)


def test_a_vehicle_crashes_into_a_moving_obstacle_in_the_engine():
    from aevion.engine import Simulation

    cfg = {
        "duration": 6.0,
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 3]}},
        "world": {"type": "empty"},
    }
    sim = Simulation(cfg)
    sim.world.add_actor(LinearActor(start=(12.0, 0.0, 3.0), velocity=(-3.0, 0.0, 0.0), radius=1.0))  # reaches x ~ 1.5 around t = 3.5
    arch = sim.run()
    assert arch.world is not None and "actor_pos" in arch
    ap = arch["actor_pos"]
    assert ap.shape == (arch.n_steps, 1, 3) and ap[0, 0, 0] == pytest.approx(12.0, abs=0.05) and ap[-1, 0, 0] < 0.0
    events = [e["type"] for e in arch.meta["events"]]
    assert "collision" in events
    t_hit = min(e["t"] for e in arch.meta["events"] if e["type"] == "collision")
    assert 2.8 < t_hit < 4.2


def test_actor_registry_and_documented_types():
    from aevion.registry import registries

    assert set(registries.actors.names()) == {"linear", "circular", "waypoints", "bouncing", "random_walk"}
    a = registries.actors.create("circular", path_radius=4.0)
    assert isinstance(a, CircularActor)


# ----------------------------------------------------------------------------------------------- zones
def test_zone_signed_distances_are_exact():
    p = np.array([[3.0, 0, 0], [0, 0, 0], [1.0, 1.0, 0]])
    np.testing.assert_allclose(SphereZone((0, 0, 0), 1.0).signed_distance(p), [2.0, -1.0, math.sqrt(2) - 1])
    np.testing.assert_allclose(
        BoxZone((-1, -1, -1), (1, 1, 1)).signed_distance(np.array([[3.0, 0, 0], [0, 0, 0], [2.0, 2.0, 0]])), [2.0, -1.0, math.sqrt(2)]
    )
    cyl = CylinderZone((0, 0), 2.0, 0.0, 10.0)
    np.testing.assert_allclose(cyl.signed_distance(np.array([[5.0, 0, 5], [0, 0, 5], [0, 0, 13], [5.0, 0, 14]])), [3.0, -2.0, 3.0, 5.0])
    np.testing.assert_allclose(CylinderZone((0, 0), 2.0).signed_distance(np.array([[5.0, 0, 1e6], [0, 0, 1e6]])), [3.0, -2.0])  # unbounded column
    np.testing.assert_allclose(
        AltitudeBand(z_max=100.0).signed_distance(np.array([[0, 0, 120.0], [0, 0, 50.0], [0, 0, -5.0]])), [20.0, -50.0, -105.0]
    )
    floor = AltitudeBand(z_min=10.0, z_max=np.inf)
    np.testing.assert_allclose(floor.signed_distance(np.array([[0, 0, 4.0], [0, 0, 30.0]])), [6.0, -20.0])


def test_polygon_zone_matches_a_brute_force_distance():
    verts = [(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)]  # concave "M"-like shape
    z = PolygonZone(verts, z_min=0.0, z_max=20.0)
    rng = np.random.default_rng(0)
    q = rng.uniform([-5, -5, -5], [15, 15, 25], (400, 3))

    def seg_dist(pt, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = np.clip(np.dot(pt - a, b - a) / np.dot(b - a, b - a), 0, 1)
        return np.linalg.norm(pt - (a + t * (b - a)))

    def inside(pt):
        x, y, c = pt[0], pt[1], False
        for i in range(len(verts)):
            (x1, y1), (x2, y2) = verts[i], verts[(i + 1) % len(verts)]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                c = not c
        return c

    for pt, d in zip(q, z.signed_distance(q)):
        d2 = min(seg_dist(pt[:2], verts[i], verts[(i + 1) % 5]) for i in range(5))
        s2 = -d2 if inside(pt) else d2
        dz = abs(pt[2] - 10.0) - 10.0
        want = math.hypot(max(s2, 0), max(dz, 0)) + min(max(s2, dz), 0)
        assert d == pytest.approx(want, abs=1e-9)
    cw = PolygonZone(verts[::-1], 0, 20)
    np.testing.assert_allclose(cw.signed_distance(q), z.signed_distance(q), atol=1e-12)  # winding does not matter


def test_zone_validation_and_serialisation():
    for bad in (
        lambda: SphereZone((0, 0, 0), 0),
        lambda: BoxZone((0, 0, 0), (0, 1, 1)),
        lambda: CylinderZone((0, 0), 1, 5, 5),
        lambda: PolygonZone([(0, 0), (1, 1)]),
        lambda: AltitudeBand(5, 5),
    ):
        with pytest.raises(ValueError):
            bad()
    for z in (
        SphereZone((1, 2, 3), 4),
        BoxZone((0, 0, 0), (1, 2, 3)),
        CylinderZone((0, 0), 3),
        PolygonZone([(0, 0), (1, 0), (0, 1)], 0, 5),
        AltitudeBand(z_max=120.0),
    ):
        again = zone_from_dict(json.loads(json.dumps(z.to_dict())))  # infinities travel as null
        pts = np.array([[0.5, 0.5, 2.0], [9.0, 9.0, 200.0]])
        np.testing.assert_allclose(again.signed_distance(pts), z.signed_distance(pts))
    with pytest.raises(ValueError, match="unknown zone"):
        zone_from_dict({"type": "blob"})


def test_fence_rules_margins_and_sets():
    zone = CylinderZone((0, 0), 100.0)
    inc, exc = Geofence(zone, "inclusion", margin=10.0, name="area"), Geofence(SphereZone((50, 0, 20), 10.0), "exclusion", margin=5.0, name="tower")
    pts = np.array([[0, 0, 20.0], [95, 0, 20.0], [50, 0, 20.0], [50, 0, 33.0], [50, 0, 40.0]])
    np.testing.assert_allclose(inc.clearance(pts)[:2], [90.0, -5.0])  # the inclusion margin eats into the area
    np.testing.assert_allclose(exc.clearance(pts)[2:], [-15.0, -2.0, 5.0])
    fs = GeofenceSet([inc, exc])
    assert fs.clearances(pts).shape == (5, 2) and fs.breaches(pts).tolist() == [
        [False, False],
        [True, False],
        [False, True],
        [False, True],
        [False, False],
    ]
    np.testing.assert_allclose(fs.nearest_boundary(pts), [35.0, -5.0, -15.0, -2.0, 5.0])
    assert np.all(np.isinf(GeofenceSet([]).nearest_boundary(pts)))
    assert GeofenceSet.from_list(fs.to_list()).breaches(pts).tolist() == fs.breaches(pts).tolist()
    with pytest.raises(ValueError, match="unique"):
        GeofenceSet([inc, Geofence(zone, "inclusion", name="area")])
    with pytest.raises(ValueError):
        Geofence(zone, "sideways")
    with pytest.raises(ValueError):
        Geofence(zone, "inclusion", margin=-1)


# ---------------------------------------------------------------------------------------- engine events
LINE = {"type": "waypoints", "params": {"points": [[-40, 0, 5], [40, 0, 5]], "speed": 6.0}}


def run_with_fences(fences, stop=False, policy="warn", duration=14.0):
    from aevion.engine import Simulation

    cfg = {
        "duration": duration,
        "trajectory": LINE,
        "vehicle": {"initial": {"pos": [-40, 0, 5]}},
        "health": {"on_diverge": policy, "checks": [{"type": "geofence", "params": {"fences": fences, "stop": stop}}]},
    }
    sim = Simulation(cfg)
    return sim, sim.run()


def test_exclusion_zone_emits_breach_and_clear_events_once_per_crossing():
    fences = [{"kind": "exclusion", "name": "airport", "zone": {"type": "cylinder", "center": [0, 0], "radius": 10.0}}]
    with pytest.warns(UserWarning, match="geofence"):
        sim, arch = run_with_fences(fences)
    ev = [(e["type"], e.get("fence")) for e in arch.meta["events"] if e["type"].startswith("geofence")]
    assert ev == [("geofence.breach", "airport"), ("geofence.clear", "airport")]
    t = {e["type"]: e["t"] for e in arch.meta["events"] if e["type"].startswith("geofence")}
    x = arch["pos"][:, 0, 0]
    tt = arch.t
    assert abs(float(np.interp(t["geofence.breach"], tt, x))) == pytest.approx(10.0, abs=1.0)  # entered at the boundary (x = -10)
    assert float(np.interp(t["geofence.clear"], tt, x)) == pytest.approx(10.0, abs=1.0)
    assert sim.health.summary()["violations"]["geofence"] == 1


def test_inclusion_zone_and_ceiling():
    fences = [
        {"kind": "inclusion", "name": "box", "zone": {"type": "box", "lo": [-30, -10, 0], "hi": [30, 10, 50]}},
        {"kind": "exclusion", "name": "ceiling", "zone": {"type": "altitude", "z_min": 100.0}},
    ]
    with pytest.warns(UserWarning):
        sim, arch = run_with_fences(fences)
    ev = [(e["type"], e["fence"]) for e in arch.meta["events"] if e["type"].startswith("geofence")]
    assert ("geofence.breach", "box") in ev and all(f != "ceiling" for _, f in ev)  # leaves the box, never reaches the ceiling
    assert ev[0] == ("geofence.breach", "box")


def test_hard_geofence_stops_the_run_cleanly():
    fences = [{"kind": "exclusion", "name": "wall", "zone": {"type": "box", "lo": [-5, -20, 0], "hi": [5, 20, 30]}}]
    sim, arch = run_with_fences(fences, stop=True, policy="stop")
    assert sim.stop_reason and "geofence" in sim.stop_reason and arch.t[-1] < 12.0
    assert any(e["type"] == "geofence.breach" for e in arch.meta["events"])


def test_geofence_check_without_fences_is_inert_and_reset_forgets_state():
    c = GeofenceCheck([])
    assert c.check(type("S", (), {"pos": np.zeros((2, 3))})(), type("C", (), {"B": None})()) == []
    c = GeofenceCheck([{"kind": "exclusion", "name": "a", "zone": {"type": "sphere", "center": [0, 0, 0], "radius": 5}}])
    ctx = type("C", (), {"B": None, "emit": None})()
    st = type("S", (), {"pos": np.zeros((1, 3))})()
    assert len(c.check(st, ctx)) == 1 and c.check(st, ctx) == []  # reported once while inside
    c.reset()
    assert len(c.check(st, ctx)) == 1
