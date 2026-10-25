"""Cooperative swarm laws inside the engine, through the lossy inter-agent link."""

import numpy as np
import pytest

from aevion.config import SimulationConfig
from aevion.engine import Simulation
from aevion.errors import ConfigError
from aevion.multiagent import CoverageDomain, coverage_cost, formation_error, make_formation
from aevion.multiagent.laws import CoverageLaw, FormationKeeping, Rendezvous, static_graph
from aevion.registry import registries

OFFSETS = make_formation("ring", 6, 3.0)


def formation_run(laws, loss=0.0, delay=0.05, fault=True):
    cfg = {
        "duration": 14.0,
        "seed": 1,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 8.0], "radius": 6.0, "omega": 0.3}},
        "vehicle": {"count": 6, "formation": {"type": "ring", "params": {"spacing": 3.0}}},
        "swarm": {"avoid_radius": 1.0, "comm": {"loss": loss, "delay_s": delay}, "laws": laws},
        "faults": [{"type": "thrust_loss", "t_start": 4.0, "t_end": 9.0, "vehicle": 2, "params": {"factor": 0.55}}] if fault else [],
    }
    a = Simulation(cfg).run()
    err = np.array([formation_error(p, OFFSETS) for p in np.asarray(a["pos"])])
    return a, err


def test_formation_keeping_holds_the_shape_when_one_vehicle_is_disturbed():
    a, plain = formation_run([])
    b, kept = formation_run([{"type": "formation_keeping", "params": {"gain": 0.4}}])
    during = (a.t > 4.5) & (a.t < 9.0)
    assert plain[during].mean() > 0.04  # a vehicle that lost 45 % of its thrust drags the shape out of line
    assert kept[during].mean() < 0.6 * plain[during].mean()  # its neighbours wait for it
    assert kept[a.t < 4.0].max() < 0.02  # and the law costs (almost) nothing when everything is nominal
    assert not any(e["type"] == "collision" for e in b.meta["events"])


def test_the_law_degrades_with_the_link_it_depends_on():
    during = lambda a: (a.t > 4.5) & (a.t < 9.0)
    a, _ = formation_run([])
    _, plain = formation_run([])
    _, good = formation_run([{"type": "formation_keeping", "params": {"gain": 0.4}}])
    _, dead = formation_run([{"type": "formation_keeping", "params": {"gain": 0.4}}], loss=1.0)
    # with every packet lost the beliefs freeze at the initial positions, so the law can no longer see the disturbance
    assert good[during(a)].mean() < dead[during(a)].mean()
    assert dead[during(a)].mean() > 0.8 * plain[during(a)].mean()


def test_rendezvous_brings_scattered_vehicles_together_over_a_lossy_link():
    cfg = {
        "duration": 20.0,
        "seed": 2,
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 6.0]}},
        "vehicle": {"count": 5, "formation": {"type": "line", "params": {"spacing": 6.0}}},
        "swarm": {
            "avoid_radius": 1.5,
            "comm": {"loss": 0.3, "delay_s": 0.1},
            "laws": [{"type": "rendezvous", "params": {"start": 1.0, "gain": 0.5}}],
        },
    }
    a = Simulation(cfg).run()
    pos = np.asarray(a["pos"])
    spread = lambda p: np.linalg.norm(p[:, :2] - p[:, :2].mean(axis=0), axis=1).max()
    assert spread(pos[0]) > 10 and spread(pos[-1]) < 3.0  # avoidance keeps them apart by ~1.5 m, so "together" means a tight cluster
    assert np.linalg.norm(pos[-1].mean(axis=0)[:2] - pos[0].mean(axis=0)[:2]) < 1.5  # the cluster forms near the initial centroid
    assert not any(e["type"] == "collision" for e in a.meta["events"])


def test_coverage_spreads_vehicles_over_the_region_and_lowers_the_locational_cost():
    cfg = {
        "duration": 40.0,
        "seed": 3,
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 6.0]}},
        "vehicle": {"count": 4, "formation": {"type": "ring", "params": {"spacing": 2.0}}},
        "swarm": {
            "avoid_radius": 1.0,
            "laws": [{"type": "coverage", "params": {"bounds": [-10, -10, 10, 10], "resolution": 0.5, "gain": 0.5, "max_step": 1.5}}],
        },
    }
    a = Simulation(cfg).run()
    pos = np.asarray(a["pos"])[:, :, :2]
    dom = CoverageDomain([-10, -10, 10, 10], 0.5)
    c0, c1 = coverage_cost(pos[0], dom), coverage_cost(pos[-1], dom)
    from aevion.multiagent import lloyd

    best = lloyd(pos[0], dom, 300).cost[-1]
    assert c1 < 0.5 * c0 and c1 < 1.5 * best  # close to what Lloyd's algorithm reaches without a vehicle in the loop
    assert np.asarray(a["pos"])[-1, :, 2].min() > 5.0


def test_coverage_density_attracts_vehicles_to_hotspots():
    law = CoverageLaw(3, bounds=(-10, -10, 10, 10), resolution=0.5, hotspots=[dict(center=[6.0, 6.0], sigma=1.5, weight=20.0)])
    pos = np.array([[0.0, 0, 5], [1.0, 0, 5], [-1.0, 0, 5]])
    known = np.repeat(pos[None], 3, axis=0)
    p = pos.copy()
    for _ in range(80):
        known = np.repeat(p[None], 3, axis=0)
        p, _ = law.apply(0.0, p, known, p, np.zeros_like(p))
    assert np.linalg.norm(p[:, :2] - [6.0, 6.0], axis=1).min() < 3.0


def test_laws_are_registered_validated_and_reject_bad_input():
    assert {"formation_keeping", "rendezvous", "coverage"} <= set(registries.swarm_laws.names())
    ok = SimulationConfig.from_dict(
        {"duration": 1.0, "vehicle": {"count": 3, "formation": {"type": "line"}}, "swarm": {"laws": [{"type": "rendezvous", "gain": 0.5}]}}
    )
    assert ok.swarm.laws[0].type == "rendezvous"
    with pytest.raises(ConfigError):
        SimulationConfig.from_dict({"duration": 1.0, "vehicle": {"count": 3, "formation": {"type": "line"}}, "swarm": {"laws": [{"type": "flock"}]}})
    with pytest.raises(ConfigError):
        SimulationConfig.from_dict(
            {
                "duration": 1.0,
                "vehicle": {"count": 3, "formation": {"type": "line"}},
                "swarm": {"laws": [{"type": "rendezvous", "params": {"speed": 3}}]},
            }
        )
    with pytest.raises(ValueError):
        FormationKeeping(3, gain=-1)
    with pytest.raises(ValueError):
        CoverageLaw(3, bounds=(0, 0, 1))
    with pytest.raises(ValueError, match="formation"):
        static_graph("knn", None, 4)
    with pytest.raises(ValueError, match="unknown graph"):
        static_graph("hypercube", OFFSETS, 6)
    g = static_graph("knn", OFFSETS, 6, k=2)
    assert np.all(g == g.T) and g.sum(axis=1).min() >= 2
    r = Rendezvous(2)
    r.bind(None, 0.01)
    pos = np.array([[0.0, 0, 1], [4.0, 0, 1]])
    out, _ = r.apply(0.0, pos, np.repeat(pos[None], 2, axis=0), pos, np.zeros((2, 3)))
    assert out[0, 0] > 0 and out[1, 0] < 4.0  # each steps towards the other
