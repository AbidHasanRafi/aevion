"""PyTorch backend: parity with NumPy and differentiability."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aevion import Multirotor, World  # noqa: E402
from aevion.control.controllers import CascadedPID, GeometricController, LQRHover  # noqa: E402
from aevion.control.reference import Reference  # noqa: E402
from aevion.environments.wind import ConstantWind  # noqa: E402


def _fly(backend, steps=300, **kw):
    w = World(wind=ConstantWind([1.0, 0.5, 0.0]))
    w.add_box((1.5, 0, 1.5), (1, 1, 3))
    sim = Multirotor("generic_1kg", n=4, dt=0.004, backend=backend, world=w, kernel="reference", **kw)
    rng = np.random.default_rng(1)
    s = sim.reset(pos=rng.uniform([-1, -1, 1], [1, 1, 3], (4, 3)), euler=rng.normal(size=(4, 3)) * 0.2, vel=rng.normal(size=(4, 3)))
    hov = sim.hover_speed()
    for k in range(steps):
        s = sim.step(s, hov * (1 + 0.1 * np.sin(0.07 * k + np.arange(4)[None, :] + np.arange(4)[:, None])))
    return sim, s


def test_torch_float64_matches_numpy():
    _, a = _fly("numpy")
    sim, b = _fly("torch", dtype="float64")
    for name in ("pos", "vel", "quat", "omega", "rotor", "soc"):
        np.testing.assert_allclose(sim.B.to_numpy(getattr(b, name)), getattr(a, name), rtol=1e-9, atol=1e-9, err_msg=name)


def test_torch_float32_is_close():
    _, a = _fly("numpy", steps=150)
    sim, b = _fly("torch", steps=150)
    np.testing.assert_allclose(sim.B.to_numpy(b.pos), a.pos, atol=5e-3)


def test_autograd_through_dynamics_matches_finite_difference():
    sim = Multirotor("generic_1kg", n=1, dt=0.005, backend="torch", dtype="float64", kernel="reference")
    hov = sim.hover_speed()

    def final_z(scale):
        s = sim.reset(pos=[0, 0, 2.0])
        cmd = hov * scale * torch.ones(1, 4, dtype=torch.float64)
        for _ in range(100):
            s = sim.step(s, cmd)
        return s.pos[0, 2]

    scale = torch.tensor(1.02, dtype=torch.float64, requires_grad=True)
    z = final_z(scale)
    z.backward()
    g_auto = float(scale.grad)
    eps = 1e-6
    g_fd = (float(final_z(torch.tensor(1.02 + eps, dtype=torch.float64))) - float(final_z(torch.tensor(1.02 - eps, dtype=torch.float64)))) / (2 * eps)
    assert g_auto == pytest.approx(g_fd, rel=1e-5)
    assert abs(g_auto) > 1.0


def test_controllers_run_on_torch():
    for cls in (GeometricController, CascadedPID, LQRHover):
        sim = Multirotor("generic_1kg", n=2, dt=0.005, backend="torch", dtype="float64", kernel="reference")
        ctrl = cls(sim)
        if hasattr(ctrl, "reset"):
            ctrl.reset(2)
        s = sim.reset(pos=[0, 0, 1.0])
        ref = Reference.hover(sim.B, 2, (1.0, 0.5, 2.0))
        for _ in range(1200):
            s = sim.step(s, ctrl.act(s, ref, 0.005))
        assert float(torch.linalg.norm(s.pos - ref.pos, dim=-1).max()) < 0.05, cls.__name__


def test_torch_raycast_matches_numpy():
    from aevion.sensors.raycast import cast_rays

    w = World.random_forest(30, half_extent=15, seed=2)
    w.add_box((3, 3, 1.5), (2, 2, 3))
    w.add_sphere((-4, 2, 2), 1.0)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.array([0.0, 0.0, 2.0])
    a = cast_rays(w, o, d, 60.0)
    b = cast_rays(w, torch.tensor(o), torch.tensor(d), 60.0)
    np.testing.assert_allclose(b.t.numpy(), a.t, atol=1e-6)
    np.testing.assert_array_equal(b.prim.numpy(), a.prim)


def test_torch_env_runs():
    from aevion.envs import VecQuadEnv

    env = VecQuadEnv(8, task="hover", backend="torch", action_mode="thrust_rates")
    obs = env.reset(seed=0)
    assert isinstance(obs, torch.Tensor)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(torch.zeros(8, 4))
    assert obs.shape == (8, env.obs_dim) and torch.isfinite(obs).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    a = Multirotor("generic_1kg", n=64, backend="torch:cpu", dtype="float64", kernel="reference")
    b = Multirotor("generic_1kg", n=64, backend="cuda", dtype="float64", kernel="reference")
    sa, sb = a.reset(), b.reset()
    for _ in range(100):
        sa, sb = a.step(sa, a.hover_command()), b.step(sb, b.hover_command())
    np.testing.assert_allclose(sb.pos.cpu().numpy(), sa.pos.numpy(), atol=1e-9)
