"""Performance benchmarks for Aevion's hot paths.

Each benchmark is a ``setup() -> run`` pair: ``setup`` builds everything (untimed) and returns a zero-argument callable
that does the measured work. ``units`` is the amount of work one call performs (vehicle-steps, rays, rows, ...), used
to report a throughput next to the time.

Run with ``python tools/bench.py``; compare against the stored baseline with ``python tools/bench_check.py``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np


@dataclass
class Benchmark:
    name: str
    setup: Callable[[], Callable[[], object]]
    units: float
    unit: str
    description: str = ""
    requires: str = ""  # optional dependency name; the benchmark is skipped when it is unavailable


SUITE: Dict[str, Benchmark] = {}


def bench(name: str, units: float, unit: str, description: str = "", requires: str = ""):
    def deco(setup):
        SUITE[name] = Benchmark(name, setup, units, unit, description, requires)
        return setup

    return deco


def available(b: Benchmark) -> bool:
    if not b.requires:
        return True
    from importlib import metadata

    try:
        metadata.version(b.requires)
        return True
    except metadata.PackageNotFoundError:
        return False


def names() -> List[str]:
    return list(SUITE)


# ------------------------------------------------------------------------------------------- physics
def _stepper(n: int, steps: int, **kw):
    from aevion.dynamics import Multirotor

    sim = Multirotor("generic_1kg", n=n, dt=0.005, **kw)
    s0 = sim.reset()
    cmd = sim.hover_command()

    def run():
        s = s0
        for _ in range(steps):
            s = sim.step(s, cmd)
        return s

    run()  # warm-up (JIT compilation, caches)
    return run


@bench("physics_reference_n1", 300, "vehicle-steps", "reference dynamics, one vehicle (latency-bound)")
def _p1():
    return _stepper(1, 300, kernel="reference")


@bench("physics_reference_n256", 256 * 100, "vehicle-steps", "reference dynamics, 256 vehicles (NumPy-vectorised)")
def _p256():
    return _stepper(256, 100, kernel="reference")


@bench("physics_numba_n256", 256 * 100, "vehicle-steps", "fused Numba kernel, 256 vehicles", requires="numba")
def _pn256():
    return _stepper(256, 100, kernel="numba")


@bench("physics_numba_n4096", 4096 * 40, "vehicle-steps", "fused Numba kernel, 4096 vehicles", requires="numba")
def _pn4096():
    return _stepper(4096, 40, kernel="numba")


# -------------------------------------------------------------------------------------------- engine
def _engine(cfg: dict, duration: float):
    from aevion.engine import Simulation

    def run():
        return Simulation(cfg).run()

    return run


@bench("engine_single_imu", 400, "physics steps", "full engine loop: 1 vehicle, geometric controller, IMU, 2 s")
def _e1():
    return _engine({"duration": 2.0, "sensors": ["imu"], "trajectory": {"type": "figure8"}}, 2.0)


@bench("engine_swarm_16", 400 * 16, "vehicle-steps", "full engine loop: 16-vehicle swarm with avoidance, 2 s")
def _e16():
    cfg = {
        "duration": 2.0,
        "vehicle": {"count": 16, "formation": {"type": "ring"}},
        "swarm": {"avoid_radius": 1.0},
        "trajectory": {"type": "hover", "params": {"pos": [0, 0, 3.0]}},
    }
    return _engine(cfg, 2.0)


@bench("engine_forest_lidar", 400, "physics steps", "1 vehicle with an 8x180 LiDAR in a 60-tree forest, 2 s")
def _elidar():
    cfg = {
        "duration": 2.0,
        "world": {"type": "forest", "params": {"n_trees": 60}},
        "sensors": [{"type": "lidar", "params": {"rate": 10}}],
    }
    return _engine(cfg, 2.0)


# ------------------------------------------------------------------------------------------- sensing
@bench("raycast_forest_65k", 65536, "rays", "65,536 rays through a 100-tree forest")
def _rays():
    from aevion.environments.world import World
    from aevion.sensors.raycast import cast_rays

    w = World.random_forest(100, seed=1)
    d = np.random.default_rng(0).normal(size=(65536, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    cast_rays(w, [0, 0, 2.0], d[:1000], 60.0)
    return lambda: cast_rays(w, [0, 0, 2.0], d, 60.0)


# --------------------------------------------------------------------------------------- config / IO
@bench("config_load_validate", 200, "configs", "validate + build 200 typical configurations")
def _cfg():
    from aevion.config import load_config

    doc = {
        "schema_version": 1,
        "duration": 10.0,
        "vehicle": {"count": 4, "formation": {"type": "ring"}},
        "sensors": ["imu", {"type": "gps", "params": {"rate": 5}}, "baro"],
        "wind": {"type": "dryden", "params": {"w20": 3.0}},
        "faults": [{"type": "rotor_failure", "t_start": 5.0}],
    }

    def run():
        for _ in range(200):
            load_config(doc)

    run()
    return run


@bench("archive_write_read", 5000, "rows", "write and re-read 5,000 rows x 3 channels through the chunked store")
def _store():
    from aevion.datasets.store import ArchiveReader, ArchiveWriter

    rows = np.random.default_rng(0).normal(size=(5000, 13))

    def run():
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.aevd"
            with ArchiveWriter(p, chunk_steps=1024) as w:
                for r in rows:
                    w.append("state", r)
                    w.append("t", r[0])
                    w.append("pos", r[:3])
            with ArchiveReader(p) as rd:
                for ch in rd.channels.values():
                    ch.to_numpy()

    run()
    return run


# --------------------------------------------------------------------------------------- figures
@bench("figure_render", 1, "figures", "render and save the research-grade 'tracking' figure", requires="matplotlib")
def _fig():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aevion.analysis import figures as F
    from aevion.engine import Simulation

    arch = Simulation({"duration": 4.0, "trajectory": {"type": "figure8"}}).run()

    def run():
        fig = F.fig_tracking(arch)
        with tempfile.TemporaryDirectory() as d:
            fig.savefig(Path(d) / "f.png", dpi=150)
        plt.close(fig)

    run()
    return run


# ----------------------------------------------------------------------------------------------- wireless
@bench(
    "wireless_nr_network_n64",
    64 * 20,
    "link evaluations",
    "integrated NR network: 64 UAVs, 3 sites, fading + handover + link adaptation, 20 intervals",
)
def _nr():
    from aevion.wireless.mac_network import NrCellularNetwork

    net = NrCellularNetwork(sites=[{"pos": [0, 0, 25]}, {"pos": [500, 0, 25]}, {"pos": [250, 430, 25]}], seed=1, background_users=2)
    rng = np.random.default_rng(0)
    pos0 = rng.uniform([0, 0, 40], [500, 400, 90], size=(64, 3))
    vel = np.tile([15.0, 0.0, 0.0], (64, 1))

    def run():
        net.reset(1)
        for k in range(20):
            net.evaluate(pos0 + vel * 0.1 * k, t=0.1 * (k + 1), vel=vel, dt=0.1)

    run()
    return run


@bench("wireless_rem_80x80", 6400, "grid points", "radio environment map, 3 sites, 80 x 80 grid")
def _rem():
    from aevion.wireless.mac_network import NrCellularNetwork
    from aevion.wireless.maps import radio_environment_map

    net = NrCellularNetwork(sites=[{"pos": [0, 0, 25]}, {"pos": [500, 0, 25]}, {"pos": [250, 430, 25]}], seed=1)
    return lambda: radio_environment_map(net, 60.0, (-100, 700, -100, 600), 10.0)


@bench("wireless_ray_trace_city", 200, "links", "image-method ray tracing of 200 links through a 3 x 3 city")
def _rt():
    from aevion.environments.world import World
    from aevion.wireless.propagation import trace_paths

    w = World.city(blocks=(3, 3), seed=1)
    rng = np.random.default_rng(0)
    tx = np.tile([0.0, 0.0, 30.0], (200, 1))
    rx = rng.uniform([-30, -30, 2], [60, 60, 40], size=(200, 3))
    return lambda: trace_paths(w, tx, rx, 3.5e9)


@bench("wireless_jakes_fading", 2000 * 100, "link-steps", "Jakes fading process: 2,000 links x 100 steps")
def _jakes():
    from aevion.wireless.fading import JakesProcess

    p = JakesProcess(2000, rng=np.random.default_rng(0))
    return lambda: [p.step(100.0, 1e-3) for _ in range(100)]


def _viewer_archive():
    from aevion.scenario import Scenario

    return Scenario(
        trajectory=dict(kind="circle", center=[0, 0, 3.0], radius=4.0, omega=0.5),
        n=4,
        formation=dict(kind="ring", spacing=1.2),
        duration=10.0,
        record_every=4,
    ).run()


@bench("viewer_payload_build", 1, "payloads", "aevion.view/2 payload of a 4-vehicle, 10 s run (2,400 frame cap)")
def _vpayload():
    from aevion.viewer import build_payload

    a = _viewer_archive()
    return lambda: build_payload(a, max_frames=2400)


@bench("viewer_frame_bundle", 50, "windows", "50 binary frame windows (500 frames, all channels) as served by `aevion replay --serve`")
def _vbundle():
    from aevion.viewer.payload import FrameSet

    fs = FrameSet(_viewer_archive())
    return lambda: [fs.bundle(fs.names, 1.0, 8.0, 500) for _ in range(50)]


# ------------------------------------------------------------------------------------------- environment and sensors
def _city_mesh_index(n: int = 12):
    from aevion.environments.mesh import TriangleMesh
    from aevion.environments.mesh_index import MeshIndex

    rng = np.random.default_rng(3)
    meshes = []
    for i in range(n * n):
        gx, gy = divmod(i, n)
        meshes.append(TriangleMesh.box(center=(gx * 30.0, gy * 30.0, 0.0), size=tuple(rng.uniform(8.0, 18.0, 2)) + (float(rng.uniform(10.0, 40.0)),)))
    meshes[0] = TriangleMesh.sphere(center=(0.0, 0.0, 50.0), radius=6.0, n_lat=24, n_lon=48)  # a finer mesh among the boxes
    return MeshIndex(meshes)


@bench("mesh_raycast_65k", 65536, "rays", "65,536 rays through a 144-building mesh city (BVH, numba)", requires="numba")
def _mesh_cast():
    idx = _city_mesh_index()
    rng = np.random.default_rng(0)
    o = np.tile([180.0, 180.0, 60.0], (65536, 1))
    d = rng.normal(size=(65536, 3))
    d[:, 2] = -np.abs(d[:, 2]) * 0.3
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    idx.cast(o[:8], d[:8], 500.0)  # compile outside the timed region
    return lambda: idx.cast(o, d, 500.0)


@bench("mesh_nearest_16k", 16384, "queries", "16,384 nearest-surface queries in a 144-building mesh city (BVH, numba)", requires="numba")
def _mesh_nearest():
    idx = _city_mesh_index()
    p = np.random.default_rng(1).uniform([0.0, 0.0, 1.0], [330.0, 330.0, 60.0], (16384, 3))
    idx.nearest(p[:8])
    return lambda: idx.nearest(p)


@bench("gnss_receiver_epochs", 64 * 20, "receiver-epochs", "GNSS pseudorange receiver: 64 vehicles x 20 epochs (GPS + Galileo, WLS fix)")
def _gnss():
    from aevion.core.backend import NumpyBackend
    from aevion.environments.world import World
    from aevion.sensors import GnssReceiver

    B = NumpyBackend()
    n = 64
    sim = type("S", (), {})()
    sim.B, sim.n, sim.world, sim.dt = B, n, World(), 0.01
    sim.hover_speed = lambda: 400.0
    rx = GnssReceiver(rate=5.0)
    rx.bind(sim, 1)
    st = type("St", (), {})()
    st.pos = np.column_stack([np.linspace(-200, 200, n), np.zeros(n), np.full(n, 30.0)])
    st.vel = np.zeros((n, 3))
    st.t = 0.0

    def run():
        for k in range(20):
            st.t = k * 0.2
            rx.measure(st)

    return run


@bench("radar_frames", 32 * 15, "beam-frames", "FMCW radar: 32 frames of a 15 x 3 beam fan against a mesh wall", requires="numba")
def _radar():
    from aevion.core.backend import NumpyBackend
    from aevion.environments.mesh import TriangleMesh
    from aevion.environments.world import World
    from aevion.sensors import Radar

    w = World()
    w.add_mesh(TriangleMesh.box(center=(30, 0, 5), size=(0.5, 60, 30)))
    sim = type("S", (), {})()
    sim.B, sim.n, sim.world, sim.dt = NumpyBackend(), 1, w, 0.01
    sim.hover_speed = lambda: 400.0
    r = Radar(ground_clutter=False)
    r.bind(sim, 1)
    st = type("St", (), {})()
    st.pos, st.vel = np.array([[0.0, 0.0, 5.0]]), np.zeros((1, 3))
    st.quat, st.t = np.array([[1.0, 0, 0, 0]]), 0.0
    r.measure(st)
    return lambda: [r.measure(st) for _ in range(32)]


# ------------------------------------------------------------------------------------- estimation, planning, vehicles
@bench("eskf_step_n64", 64 * 200, "vehicle-steps", "error-state Kalman filter: 64 vehicles, 200 predicts with a GNSS and a baro update every 10th")
def _eskf():
    from aevion.core.backend import NumpyBackend
    from aevion.estimation.eskf import ESKF

    n = 64
    f = ESKF(NumpyBackend(), n)
    f.reset()
    gyro, acc = np.zeros((n, 3)), np.tile([0.0, 0.0, 9.80665], (n, 1))
    pos = np.zeros((n, 3))

    def run():
        for k in range(200):
            f.predict(gyro, acc, 0.005)
            if k % 10 == 0:
                f.update_gnss(pos, None, 1.0)
                f.update_baro(pos[:, 2], 0.5)

    return run


@bench("planning_astar_64k", 1, "plans", "A* and Theta* across a 40 x 40 x 40 grid with a wall and a gap (26-connected)")
def _astar():
    from aevion.planning import PlanningSpace, astar, theta_star

    occ = np.zeros((40, 40, 40), bool)
    occ[20, :30, :] = True
    s = PlanningSpace(occ, (0.0, 0.0, 0.0), 1.0)

    def run():
        a = astar(s, (2.0, 2.0, 5.0), (38.0, 2.0, 5.0))
        t = theta_star(s, (2.0, 2.0, 5.0), (38.0, 2.0, 5.0))
        assert a.success and t.success

    return run


@bench("planning_minsnap_20wp", 1, "fits", "minimum-snap trajectory through 20 waypoints, sampled at 100 Hz")
def _minsnap():
    from aevion.planning import MinSnap

    rng = np.random.default_rng(0)
    pts = np.cumsum(rng.normal(0, 2.0, (20, 3)), axis=0)
    times = np.full(19, 1.5)

    def run():
        m = MinSnap(pts, times)
        m.sample_many(np.arange(0.0, m.duration, 0.01))

    return run


@bench("qp_admm_60", 100, "solves", "ADMM QP solver: 100 solves of a 60-variable problem with box and 30 inequality rows")
def _qp():
    from aevion.planning import solve_qp

    rng = np.random.default_rng(1)
    M = rng.normal(size=(60, 60))
    P = M @ M.T + 0.1 * np.eye(60)
    q = rng.normal(size=60)
    A = rng.normal(size=(30, 60))
    lo, hi = np.full(30, -1.0), np.full(30, 1.0)

    def run():
        for _ in range(100):
            solve_qp(P, q, A, lo, hi)

    return run


@bench("engine_mpc_hover", 400, "physics steps", "full engine loop: MPC controller with its disturbance observer, 2 s")
def _empc():
    return _engine({"duration": 2.0, "controller": {"type": "mpc", "params": {"horizon": 20}}, "trajectory": {"type": "figure8"}}, 2.0)


@bench("engine_eskf_flight", 400, "physics steps", "full engine loop: IMU, magnetometer, barometer and GNSS feeding the ESKF, 2 s")
def _eeskf():
    cfg = {
        "duration": 2.0,
        "sensors": ["imu", "mag", "baro", {"type": "gps", "params": {"latency": 0.1}}],
        "estimator": {"type": "eskf", "handover_s": 0.5},
        "trajectory": {"type": "figure8"},
    }
    return _engine(cfg, 2.0)


@bench("bemt_propeller_sweep", 40 * 10, "operating points", "blade-element-momentum propeller: 40 rpm values x 10 airspeeds")
def _bemt():
    from aevion.vehicles.propeller import get_propeller

    p = get_propeller("10x4.5")
    rpms = np.linspace(2000.0, 8000.0, 40)
    speeds = np.linspace(0.0, 15.0, 10)

    def run():
        for r in rpms:
            for v in speeds:
                p.operating_point(float(r), float(v))

    return run


# ------------------------------------------------------------------------------------- multi-agent, ML, experiments
@bench("multiagent_hungarian_200", 20, "assignments", "Hungarian method: 20 random 200 x 200 assignment problems")
def _hung():
    from aevion.multiagent.allocation import hungarian

    rng = np.random.default_rng(0)
    mats = [rng.uniform(0, 100, (200, 200)) for _ in range(20)]

    def run():
        for c in mats:
            hungarian(c)

    return run


@bench("multiagent_cbba_6x12", 1, "allocations", "CBBA: 6 agents, 12 tasks, ring communication graph")
def _cbba():
    from aevion.multiagent import TravelScore, adjacency, cbba

    rng = np.random.default_rng(1)
    score = TravelScore(rng.uniform(0, 50, (6, 2)), rng.uniform(0, 50, (12, 2)), value=rng.uniform(1, 5, 12), speed=3.0, discount=0.95)
    A = adjacency("ring", 6)
    return lambda: cbba(score, 6, 12, adjacency=A)


@bench("multiagent_lloyd_20x80", 20 * 80, "site-iterations", "Lloyd coverage: 20 sites on a 80 x 80 grid, 80 iterations")
def _lloyd():
    from aevion.multiagent import CoverageDomain, lloyd

    dom = CoverageDomain([0, 0, 20, 20], 0.25)
    start = np.random.default_rng(2).uniform(0, 20, (20, 2))
    return lambda: lloyd(start, dom, iterations=80, tol=0.0)


@bench("multiagent_spacetime_4", 1, "plans", "prioritised space-time planning: 4 vehicles crossing a 16 x 16 x 3 volume")
def _stp():
    from aevion.multiagent import SpaceTimePlanner
    from aevion.planning import PlanningSpace

    space = PlanningSpace(np.zeros((16, 16, 3), bool), (0.0, 0.0, 0.0), 1.0)
    starts = [[0.5, 0.5, 1.5], [15.5, 0.5, 1.5], [0.5, 15.5, 1.5], [15.5, 15.5, 1.5]]
    goals = [[15.5, 15.5, 1.5], [0.5, 15.5, 1.5], [15.5, 0.5, 1.5], [0.5, 0.5, 1.5]]
    planner = SpaceTimePlanner(space, dt=1.0, separation=1.5, connectivity=18)

    def run():
        assert planner.plan(starts, goals).success

    return run


@bench(
    "swarm_env_8x8", 8 * 8 * 50, "vehicle-steps", "cooperative swarm environment: 8 teams of 8 vehicles, 50 control steps with the scripted policy"
)
def _swarm():
    from aevion.envs import VecSwarmEnv
    from aevion.envs.swarm_policies import SwarmScriptedPolicy

    env = VecSwarmEnv(8, task="formation", n_agents=8, k_neighbors=4, seed=0)
    pol = SwarmScriptedPolicy(env)
    obs = env.reset(seed=1)

    def run():
        o = obs
        for _ in range(50):
            o, *_ = env.step(pol(o))

    return run


@bench("gp_fit_predict_120", 1, "fits", "Gaussian process: fit 4 length scales + variance on 120 points, predict 2,000")
def _gp():
    from aevion.experiments.bayesopt import GaussianProcess

    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (120, 4))
    y = np.sin(5 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.normal(size=120)
    Xs = rng.uniform(0, 1, (2000, 4))

    def run():
        GaussianProcess().fit(X, y, restarts=2).predict(Xs)

    return run


@bench("sobol_indices_ishigami", 4096 * 5, "evaluations", "Sobol indices of the Ishigami function with 4,096 base samples (20,480 model evaluations)")
def _sobol():
    import math

    from aevion.experiments import sobol_indices

    def f(X):
        return np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

    box = [(-math.pi, math.pi)] * 3
    return lambda: sobol_indices(f, box, n=4096, vectorized=True, n_boot=50)


@bench("optical_flow_192x144", 192 * 144, "pixels", "optical-flow ground truth from depth and two poses, 192 x 144 camera")
def _flow():
    from aevion.datasets.vision import optical_flow
    from aevion.sensors.camera_models import PinholeLens

    lens = PinholeLens(192, 144, 90.0)
    depth = np.random.default_rng(4).uniform(2.0, 30.0, (144, 192))
    p0 = (np.zeros(3), np.eye(3))
    c, s = np.cos(0.05), np.sin(0.05)
    p1 = (np.array([0.5, 0.2, 0.0]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))
    return lambda: optical_flow(depth, lens, p0, p1, depth)
