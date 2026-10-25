import numpy as np
import pytest

from aevion import Multirotor, World
from aevion.control import CascadedPID, Circle, Gains, GeometricController, LQRHover, Reference, Waypoints
from aevion.environments.wind import ConstantWind
from aevion.metrics import step_response_metrics


def fly(ctrl_cls, ref_fn, T=6.0, vehicle="generic_1kg", n=2, world=None, dt=0.005, **kw):
    sim = Multirotor(vehicle, n=n, dt=dt, world=world or World())
    ctrl = ctrl_cls(sim, **kw)
    if hasattr(ctrl, "reset"):
        ctrl.reset(n)
    s = sim.reset(pos=ref_fn(0.0).pos[0])
    errs = []
    for _ in range(int(T / dt)):
        ref = ref_fn(s.t)
        s = sim.step(s, ctrl.act(s, ref, dt))
        errs.append(np.linalg.norm(s.pos - ref.pos, axis=1).mean())
    return np.array(errs), s


@pytest.mark.parametrize("cls", [GeometricController, CascadedPID, LQRHover])
@pytest.mark.parametrize("vehicle", ["crazyflie", "racing_5inch", "generic_1kg", "camera_2kg", "hex_3kg"])
def test_all_controllers_reach_setpoint_on_all_vehicles(cls, vehicle):
    B = Multirotor(vehicle, n=1).B
    hold = lambda t: Reference.hover(B, 2, (1.0, -1.0, 2.0))
    Reference.hover(B, 2, (0.0, 0.0, 1.0))

    sim = Multirotor(vehicle, n=2, dt=0.005)
    ctrl = cls(sim)
    if hasattr(ctrl, "reset"):
        ctrl.reset(2)
    s = sim.reset(pos=[0, 0, 1.0])
    ref = hold(0.0)
    for _ in range(int(7.0 / 0.005)):
        s = sim.step(s, ctrl.act(s, ref, 0.005))
    assert np.linalg.norm(s.pos - ref.pos, axis=1).max() < 0.05, (cls.__name__, vehicle)


def test_geometric_tracks_circle_accurately():
    B = Multirotor("generic_1kg", n=1).B
    tr = Circle(center=(0, 0, 2), radius=1.5, omega=1.2)
    errs, _ = fly(GeometricController, lambda t: tr.reference(B, t, 2), T=10.0)
    assert np.sqrt((errs[len(errs) // 2 :] ** 2).mean()) < 0.02


def test_integral_action_rejects_constant_wind():
    B = Multirotor("generic_1kg", n=1).B
    hold = lambda t: Reference.hover(B, 2, (0, 0, 3.0))
    wf = lambda: World(wind=ConstantWind([6.0, 3.0, 0.0]))

    def run(**kw):
        sim = Multirotor("generic_1kg", n=2, world=wf())
        c = GeometricController(sim, **kw)
        c.reset(2)
        s = sim.reset(pos=[0, 0, 3.0])
        for _ in range(int(20 / 0.005)):
            s = sim.step(s, c.act(s, hold(0), 0.005))
        return np.linalg.norm(s.pos - hold(0).pos, axis=1).mean()

    e_p, e_i = run(), run(gains=Gains(ki_pos=4.0))
    assert e_i < 0.25 * e_p and e_i < 0.03


def test_step_response_has_no_large_overshoot():
    sim = Multirotor("generic_1kg", n=1)
    ctrl = GeometricController(sim)
    ctrl.reset(1)
    s = sim.reset(pos=[0, 0, 1.0])
    ref = Reference.hover(sim.B, 1, (0, 0, 2.0))
    t, z = [], []
    for _ in range(1600):
        s = sim.step(s, ctrl.act(s, ref, 0.005))
        t.append(s.t)
        z.append(s.pos[0, 2])
    m = step_response_metrics(np.array(t), np.array(z), 2.0, 1.0)
    assert m["overshoot_pct"] < 10 and m["steady_state_error"] < 1e-3 and m["settling_time"] < 5


def test_hexacopter_survives_single_motor_failure_with_fault_tolerant_allocation():
    """Naive pseudo-inverse allocation fails after a rotor loss; bounded LS with de-prioritised yaw holds hover."""
    from aevion.control import FaultTolerantAllocator

    def hold(alloc_fn, T=8.0):
        sim = Multirotor("hex_3kg", n=1, dt=0.004)
        sim.params.set_rotor_failure(0, 2, 1.0)
        ctrl = GeometricController(sim)
        ctrl.reset(1)
        s = sim.reset(pos=[0, 0, 5.0])
        ref = Reference.hover(sim.B, 1, (0, 0, 5.0))
        for _ in range(int(T / 0.004)):
            thrust, tau = ctrl.wrench(s, ref, 0.004)
            s = sim.step(s, alloc_fn(sim, thrust, tau))
        return np.linalg.norm(s.pos[0] - [0, 0, 5.0])

    ft = FaultTolerantAllocator(Multirotor("hex_3kg", n=1).spec, Multirotor("hex_3kg", n=1).B)
    tolerant = hold(lambda sim, T, tau: ft.to_rotor_speeds(T, tau, sim.params.rotor_health))
    naive = hold(lambda sim, T, tau: GeometricController(sim).alloc.to_rotor_speeds(T, tau))
    assert tolerant < 0.2
    assert naive > 5 * tolerant


def test_waypoint_trajectory_is_smooth_and_hits_points():
    pts = [[0, 0, 1], [3, 0, 2], [3, 3, 2], [0, 3, 1]]
    wp = Waypoints(pts, speed=1.5)
    for k, p in enumerate(pts):
        np.testing.assert_allclose(wp.sample(wp.knots[k])[0], p, atol=1e-9)
    np.testing.assert_allclose(wp.sample(0.0)[1], 0, atol=1e-9)
    np.testing.assert_allclose(wp.sample(wp.duration)[1], 0, atol=1e-9)
    v = np.array([wp.sample(t)[1] for t in np.linspace(0, wp.duration, 400)])
    assert np.abs(np.diff(v, axis=0)).max() < 0.2  # no velocity jumps


# ------------------------------------------------------------------ environments
gym = pytest.importorskip("gymnasium")


def test_gymnasium_env_passes_checker():
    from gymnasium.utils.env_checker import check_env

    from aevion.envs import QuadrotorEnv

    for mode in ("thrust_rates", "rotor", "velocity", "position"):
        check_env(QuadrotorEnv(task="hover", action_mode=mode, max_episode_s=2.0), skip_render_check=True)


def test_registered_ids():
    env = gym.make("Aevion-Hover-v0")
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    for _ in range(5):
        env.step(env.action_space.sample())


def test_vec_env_determinism_and_autoreset():
    from aevion.envs import VecQuadEnv

    def roll(seed):
        env = VecQuadEnv(8, task="hover", action_mode="rotor", max_episode_s=1.0, seed=seed)
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(0)
        rets, dones = 0, 0
        for _ in range(80):
            obs, r, te, tr, info = env.step(rng.uniform(-1, 1, (8, env.act_dim)))
            rets = rets + r.sum()
            dones += int(info["done"].sum())
        return obs.copy(), rets, dones

    a, b = roll(5), roll(5)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2] and a[2] > 0
    assert not np.allclose(roll(6)[0], a[0])


def test_domain_randomization_varies_parameters():
    from aevion.envs import DomainRandomization, VecQuadEnv

    env = VecQuadEnv(64, task="hover", domain_randomization=DomainRandomization.strong())
    m = env.sim.params.mass[:, 0]
    assert m.std() / m.mean() > 0.05 and m.min() >= 0.74 and m.max() <= 1.26
    assert env.sim.params.kT.std() > 0
    env.reset(0)
    assert np.abs(env.world.wind.w).max() > 0


def test_motor_failure_randomization():
    from aevion.envs import DomainRandomization, VecQuadEnv

    env = VecQuadEnv(200, task="hover", domain_randomization=DomainRandomization(rotor_failure_prob=0.5))
    dead = env.sim.params.rotor_health.min(1) == 0
    assert 0.35 < dead.mean() < 0.65


def test_navigation_task_with_lidar():
    from aevion.envs import VecQuadEnv

    w = World.random_forest(20, half_extent=10, seed=1)
    env = VecQuadEnv(4, task="navigation", world=w, lidar=True, lidar_samples=12, action_mode="velocity", bounds=(-14, -12, 0, 14, 12, 8))
    obs = env.reset(0)
    assert obs.shape[1] == env.obs_dim == 3 + 3 + 9 + 3 + 4 + 12
    for _ in range(30):
        obs, *_ = env.step(env.hover_action())
    assert np.isfinite(obs).all()


def test_proportional_policy_solves_hover_and_beats_random():
    from aevion.envs import VecQuadEnv
    from aevion.experiments import ProportionalPolicy, evaluate_policy

    mk = lambda seed: VecQuadEnv(16, task="hover", action_mode="position", max_episode_s=4.0, seed=seed)
    pol = ProportionalPolicy(mk(0))
    good = evaluate_policy(pol, mk, seeds=(0, 1), episodes_per_seed=16)
    rng = np.random.default_rng(0)
    bad = evaluate_policy(lambda o: rng.uniform(-1, 1, (16, 4)), mk, seeds=(0, 1), episodes_per_seed=16)
    assert good.per_seed["success_rate"].mean() > 0.9 and bad.per_seed["success_rate"].mean() < 0.2
    assert good.per_seed["return"].mean() > bad.per_seed["return"].mean()


def test_sb3_adapter_trains():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from aevion.envs import make_sb3_vec_env

    env = make_sb3_vec_env(4, task="hover", action_mode="velocity", max_episode_s=1.0)
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=64, verbose=0, device="cpu")
    model.learn(256)
    a, _ = model.predict(env.reset())
    assert a.shape == (4, 4)
