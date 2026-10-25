"""Glue between the planners and the simulation engine: the ``planned`` and ``mission`` trajectory types.

Both are registered in the ``trajectories`` registry. The engine hands them the built world (they declare a ``world`` parameter),
and they voxelise it into a :class:`~aevion.planning.space.PlanningSpace`, so a configuration can say::

    trajectory: {type: planned, params: {start: [0, 0, 2], goal: [30, 20, 6], planner: theta_star, inflation: 0.6, v_max: 4}}
    trajectory: {type: mission, params: {tree: {type: sequence, children: [{type: takeoff, altitude: 6}, {type: goto, waypoint: [20, 0, 6]}, {type: land}]}}}
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .behavior import BehaviorTreeMission, tree_from_dict
from .space import PlanningSpace
from .trajectory import PlannedTrajectory, plan_path, plan_trajectory

__all__ = ["space_from_world", "planned_trajectory", "mission_trajectory"]


def space_from_world(
    world: Any, resolution: float = 0.5, inflation: float = 0.5, bounds: Optional[Sequence[float]] = None, ground: bool = True
) -> PlanningSpace:
    """Voxelise a world for planning; ``bounds`` default to the world's own, padded by a metre so paths can hug the edge."""
    b = [float(x) for x in (world.bounds if bounds is None else bounds)]
    return PlanningSpace.from_world(world, bounds=b, resolution=resolution, inflation=inflation, ground=ground)


def planned_trajectory(
    start: Sequence[float],
    goal: Sequence[float],
    via: Sequence[Sequence[float]] = (),
    world: Any = None,
    planner: str = "theta_star",
    resolution: float = 0.5,
    inflation: float = 0.5,
    bounds: Optional[Sequence[float]] = None,
    v_max: float = 3.0,
    a_max: float = 2.5,
    margin: float = 0.0,
    yaw: float = 0.0,
    yaw_follow: bool = True,
    ground: bool = True,
    seed: int = 0,
    planner_options: Optional[Dict[str, Any]] = None,
) -> PlannedTrajectory:
    """Plan around the world's obstacles from ``start`` through ``via`` to ``goal`` and smooth the result (minimum snap).

    Without a world the path is the straight line through the waypoints. A ``RuntimeError`` explains when no path exists.
    """
    pts = [np.asarray(start, float)] + [np.asarray(p, float) for p in via] + [np.asarray(goal, float)]
    if world is None:
        return plan_trajectory(None, np.array(pts), v_max=v_max, a_max=a_max, yaw=yaw, yaw_follow=yaw_follow)
    space = space_from_world(world, resolution, inflation, bounds, ground)
    legs: List[np.ndarray] = []
    for a, b in itertools.pairwise(pts):
        r = plan_path(space, a, b, planner, seed=seed, **(planner_options or {}))
        if not r.success:
            raise RuntimeError(f"no {planner} path from {a.tolist()} to {b.tolist()}: {r.reason}")
        legs.append(r.path if not legs else r.path[1:])
    path = np.vstack(legs)
    tr = plan_trajectory(space, path, v_max=v_max, a_max=a_max, margin=margin, yaw=yaw, yaw_follow=yaw_follow)
    tr.info["planner"] = planner
    return tr


def mission_trajectory(
    tree: Any,
    world: Any = None,
    home: Optional[Sequence[float]] = None,
    planner: str = "theta_star",
    resolution: float = 0.5,
    inflation: float = 0.5,
    bounds: Optional[Sequence[float]] = None,
    v_max: float = 3.0,
    a_max: float = 2.5,
    yaw: float = 0.0,
    on_finish: str = "hold",
    ground: bool = True,
) -> BehaviorTreeMission:
    """A behaviour-tree mission (``tree`` as nested dictionaries, see :func:`~aevion.planning.behavior.tree_from_dict`)."""
    space = None if world is None else space_from_world(world, resolution, inflation, bounds, ground)
    return BehaviorTreeMission(tree_from_dict(tree), space, home if home is not None else (0.0, 0.0, 0.0), v_max, a_max, planner, yaw, on_finish)
