"""A tour of the planning and estimation stack: plan through a small city, fly it with MPC, estimate with the ESKF, and check the result.

Plans a collision-free, dynamically feasible route between two points with Theta* and minimum-snap, flies it with the MPC controller
in a steady wind, feeds the controller's estimate from an error-state Kalman filter (IMU, magnetometer, barometer, delayed GNSS),
then reports the clearance to the buildings, the tracking error, and the estimator's error against its own claimed uncertainty.

    python examples/14_planning_estimation_tour.py --out planning_tour --duration 30
"""

import argparse
from pathlib import Path

import numpy as np

from aevion.engine import Simulation
from aevion.viewer import export_html

BUILDINGS = [
    {"center": [15, 0, 6], "size": [4, 14, 12]},
    {"center": [8, -8, 4], "size": [4, 4, 8]},
    {"center": [24, 6, 5], "size": [5, 5, 10]},
]


def box_distance(p: np.ndarray, box: dict) -> float:
    return float(np.linalg.norm(np.maximum(np.abs(p - np.asarray(box["center"])) - np.asarray(box["size"]) / 2.0, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="planning_tour", help="output directory")
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "duration": args.duration,
        "seed": 4,
        "world": {"type": "scene", "params": {"boxes": BUILDINGS, "bounds": [-5, -15, 0, 40, 15, 20]}},
        "wind": {"type": "constant", "velocity": [2.0, 1.0, 0.0]},
        "trajectory": {
            "type": "planned",
            "params": {"start": [0, 0, 3], "goal": [34, 0, 3], "v_max": 4.0, "a_max": 3.0, "inflation": 0.8, "resolution": 1.0},
        },
        "controller": {"type": "mpc", "params": {"horizon": 20}},
        "sensors": ["imu", "mag", "baro", {"type": "gps", "params": {"latency": 0.1}}],
        "estimator": {"type": "eskf", "handover_s": 3.0},
    }
    sim = Simulation(cfg)
    traj = sim.traj
    info = traj.info
    print(f"plan     : {info['planner']}, {len(traj.waypoints)} waypoints, peak speed {info['peak_speed']:.2f} m/s")

    arch = sim.run()
    pos = arch["pos"][:, 0]
    clearance = min(box_distance(p, b) for p in pos[::5] for b in BUILDINGS)
    err = np.linalg.norm(pos - arch["ref_pos"][:, 0], axis=1)
    late = arch.t > 4.0
    est = np.linalg.norm(arch["est_pos"][:, 0] - pos, axis=1)
    collided = any(e["type"] == "collision" for e in arch.meta["events"])
    print(f"flight   : final distance to goal {np.linalg.norm(pos[-1] - [34, 0, 3]):.2f} m, collision {collided}")
    print(f"clearance: closest approach to a building {clearance:.2f} m")
    print(f"tracking : mean error {err[late].mean():.3f} m (true position vs reference; includes the GNSS wander the controller sees)")
    print(f"estimator: mean position error {est[late].mean():.2f} m, reported 1-sigma {arch['est_pos_std'][late, 0].mean():.2f} m per axis")

    page = export_html(arch, out / "replay.html", title="Planning tour", max_frames=300)
    print(f"replay   : {page}")


if __name__ == "__main__":
    main()
