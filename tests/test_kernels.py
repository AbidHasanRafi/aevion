import numpy as np
import pytest

from aevion import Multirotor, VehicleSpec, World
from aevion.dynamics import fast, kernels
from aevion.dynamics.kernels import KernelUnavailable, parity_check

pytestmark = pytest.mark.skipif(not fast.HAVE_NUMBA, reason="numba not installed")


def test_auto_selects_numba_on_cpu_and_reference_otherwise():
    assert Multirotor("generic_1kg", n=2).kernel.name == "numba"
    assert Multirotor("generic_1kg", n=2, kernel="reference").kernel.name == "reference"
    assert Multirotor("generic_1kg", n=2, kernel="reference").kernel.name == "reference"  # legacy flag
    assert Multirotor("generic_1kg", n=2).fast is True
    pytest.importorskip("torch")
    assert Multirotor("generic_1kg", n=2, backend="torch").kernel.name == "reference"


def test_explicit_numba_on_wrong_backend_raises_clearly():
    pytest.importorskip("torch")
    with pytest.raises(KernelUnavailable, match="needs the NumPy backend"):
        Multirotor("generic_1kg", n=2, backend="torch", kernel="numba")
    with pytest.raises(KernelUnavailable, match="unknown kernel"):
        Multirotor("generic_1kg", n=2, kernel="cuda")


def test_parity_check_passes_for_all_presets_and_is_cached():
    for name in ("crazyflie", "racing_5inch", "generic_1kg", "camera_2kg", "hex_3kg"):
        sim = Multirotor(name, n=2, verify_kernel=False)
        ok, err = parity_check(sim, use_cache=False)
        assert ok and err < 1e-6, (name, err)
    sim = Multirotor("generic_1kg", n=2)
    assert parity_check(sim) is parity_check(sim)  # same cached tuple object


def test_parity_check_catches_a_broken_kernel(monkeypatch):
    """Simulate a kernel bug: the guard must refuse it (explicit) or fall back with a warning (auto)."""
    real = fast.step_kernel

    def broken(*args, **kw):
        real(*args, **kw)
        args[-14][...] += 1e-3  # corrupt o_pos (the first output array)

    kernels._parity_cache.clear()
    monkeypatch.setattr(fast, "step_kernel", broken)
    with pytest.raises(KernelUnavailable, match="disagrees with the reference"):
        Multirotor("generic_1kg", n=2, kernel="numba")
    kernels._parity_cache.clear()
    with pytest.warns(UserWarning, match="falling back to the reference"):
        sim = Multirotor("generic_1kg", n=2, kernel="auto")
    assert sim.kernel.name == "reference"
    kernels._parity_cache.clear()


def test_kernel_choice_is_part_of_the_cache_key_for_distinct_configs():
    k1 = kernels._config_key(Multirotor("generic_1kg", n=1, verify_kernel=False))
    k2 = kernels._config_key(Multirotor("generic_1kg", n=1, dt=0.01, verify_kernel=False))
    k3 = kernels._config_key(Multirotor(VehicleSpec.generic_1kg().copy(mass=1.2), n=1, verify_kernel=False))
    w = World().set_heightmap(np.zeros((4, 4)))
    k4 = kernels._config_key(Multirotor("generic_1kg", n=1, world=w, verify_kernel=False))
    assert len({k1, k2, k3, k4}) == 4


def test_engine_kernel_setting_is_honoured():
    from aevion.config import SimulationConfig
    from aevion.engine import Simulation

    for choice, expected in (("reference", "reference"), ("numba", "numba"), ("auto", "numba")):
        sim = Simulation(SimulationConfig.from_dict({"engine": {"kernel": choice}, "duration": 0.2}))
        assert sim.dyn.kernel.name == expected
        assert sim.run().provenance["kernel"] == expected


def test_parallel_kernel_allocates_nothing_inside_the_prange_loop():
    """Regression: scratch arrays allocated inside the prange body produced rare NaNs under CPU load (a race inside numba).

    The kernel must take its scratch memory as per-vehicle rows of arrays allocated *before* the parallel loop.
    """
    import inspect

    from aevion.dynamics import fast

    src = inspect.getsource(fast).split("for i in prange(N)")[1]
    body = src.split("METHOD_IDS")[0]
    assert "np.empty" not in body and "np.zeros(" not in body and "np.array(" not in body, "allocation inside the parallel loop"
