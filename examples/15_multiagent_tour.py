"""A tour of the multi-agent toolkit: consensus, task allocation, coverage, deconfliction and a swarm law in the engine.

python examples/15_multiagent_tour.py --out multiagent_tour
"""

import argparse
from pathlib import Path

import numpy as np

from aevion.engine import Simulation
from aevion.multiagent import (
    CoverageDomain,
    SpaceTimePlanner,
    Trajectory4D,
    TravelScore,
    adjacency,
    algebraic_connectivity,
    average_consensus,
    brute_force_allocation,
    cbba,
    consensus_rate,
    coverage_cost,
    detect_conflicts,
    formation_error,
    lloyd,
    make_formation,
    schedule_departures,
)
from aevion.planning import PlanningSpace
from aevion.viewer import export_html


def consensus_section() -> None:
    ring = adjacency("ring", 10)
    x0 = np.random.default_rng(0).normal(size=10)
    traj = average_consensus(x0, ring, steps=120)
    d = np.linalg.norm(traj - x0.mean(), axis=1)
    print(
        f"consensus : ring of 10, lambda_2 = {algebraic_connectivity(ring):.3f}, predicted rate {consensus_rate(ring):.3f}, measured {(d[100] / d[60]) ** (1 / 40):.3f}"
    )


def allocation_section() -> None:
    rng = np.random.default_rng(1)
    score = TravelScore(rng.uniform(0, 10, (3, 2)), rng.uniform(0, 10, (6, 2)), value=rng.uniform(1, 5, 6), speed=2.0, discount=0.9)
    res = cbba(score, 3, 6, adjacency=adjacency("ring", 3))
    best, _ = brute_force_allocation(score, 3, 6)
    print(
        f"CBBA      : conflict-free {res.conflict_free}, {res.rounds} rounds, score {res.score:.3f} = {100 * res.score / best:.1f} % of the optimum {best:.3f}"
    )


def coverage_section() -> None:
    dom = CoverageDomain([0, 0, 10, 10], 0.25)
    start = np.random.default_rng(2).uniform(0, 10, (6, 2))
    res = lloyd(start, dom, iterations=80)
    print(f"coverage  : Lloyd lowered the locational cost from {coverage_cost(start, dom):.0f} to {res.cost[-1]:.0f} in {res.iterations} iterations")


def utm_section() -> None:
    a = Trajectory4D([0.0, 10.0], [[0, 0, 5], [100, 0, 5]])
    b = Trajectory4D([0.0, 10.0], [[100, 0, 5], [0, 0, 5]])
    c = detect_conflicts([a, b], 10.0)[0]
    delays, ok = schedule_departures([a, b], 10.0, max_delay=60.0)
    print(
        f"UTM       : head-on flights are within 10 m from t = {c.t_start:.1f} s to {c.t_end:.1f} s; ground delays {delays.tolist()} resolve it ({ok})"
    )
    space = PlanningSpace(np.zeros((12, 5, 3), bool), (0, 0, 0), 1.0)
    plan = SpaceTimePlanner(space, dt=1.0, separation=1.5).plan([[0.5, 2.5, 1.5], [10.5, 2.5, 1.5]], [[10.5, 2.5, 1.5], [0.5, 2.5, 1.5]])
    print(
        f"            swapping places in a corridor: success {plan.success}, makespan {plan.makespan():.0f} s, conflicts {len(plan.conflicts(1.5))}"
    )


def swarm_section(out: Path, duration: float):
    offsets = make_formation("ring", 6, 3.0)

    def run(laws):
        cfg = {
            "duration": duration,
            "seed": 1,
            "trajectory": {"type": "circle", "params": {"center": [0, 0, 8.0], "radius": 6.0, "omega": 0.3}},
            "vehicle": {"count": 6, "formation": {"type": "ring", "params": {"spacing": 3.0}}},
            "swarm": {"avoid_radius": 1.0, "comm": {"delay_s": 0.05}, "laws": laws},
            "faults": [{"type": "thrust_loss", "t_start": 4.0, "t_end": 9.0, "vehicle": 2, "params": {"factor": 0.55}}],
        }
        arch = Simulation(cfg).run()
        err = np.array([formation_error(p, offsets) for p in np.asarray(arch["pos"])])
        during = (arch.t > 4.5) & (arch.t < 9.0)
        return arch, float(err[during].mean())

    _, plain = run([])
    arch, kept = run([{"type": "formation_keeping", "params": {"gain": 0.4}}])
    print(
        f"swarm law : vehicle 2 loses 45 % thrust for 5 s: formation error {plain * 100:.1f} cm without the law, {kept * 100:.1f} cm with formation_keeping"
    )
    return arch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="multiagent_tour")
    ap.add_argument("--duration", type=float, default=12.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    consensus_section()
    allocation_section()
    coverage_section()
    utm_section()
    arch = swarm_section(out, args.duration)
    page = export_html(arch, out / "swarm_replay.html", title="Formation keeping under a thrust loss", max_frames=250)
    print(f"replay    : {page}")


if __name__ == "__main__":
    main()
