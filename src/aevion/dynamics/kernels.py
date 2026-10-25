"""Dynamics kernels: interchangeable implementations of one physics step.

* ``reference``: the backend-generic implementation (NumPy, PyTorch CPU/CUDA). It is the *definition* of the
  model: differentiable, readable, and the source of truth.
* ``numba``: a fused, multi-threaded CPU kernel (NumPy backend only) that re-implements the same equations.

The two are kept honest by :func:`parity_check`, which runs both on randomised states and compares the
results. It runs automatically (once per distinct configuration per process) when the fast kernel is selected;
if the kernels disagree, ``auto`` falls back to the reference and warns, and an explicit ``kernel="numba"`` raises.
"""

from __future__ import annotations

import json
import warnings
from typing import Dict, Tuple

import numpy as np

from ..errors import AevionError


class KernelUnavailable(AevionError):
    """The requested kernel cannot be used here (missing dependency, wrong backend, parity failure)."""


class Kernel:
    name = "kernel"
    description = ""

    def available(self, sim) -> Tuple[bool, str]:
        return True, ""

    def step(self, sim, state, cmd, dt):  # pragma: no cover - interface
        raise NotImplementedError


class ReferenceKernel(Kernel):
    name = "reference"
    description = "backend-generic reference implementation (NumPy / torch, differentiable)"

    def step(self, sim, state, cmd, dt):
        return sim._step_reference(state, cmd, dt)


class NumbaKernel(Kernel):
    name = "numba"
    description = "fused multi-threaded Numba kernel (NumPy backend)"

    def available(self, sim):
        from . import fast

        if not fast.HAVE_NUMBA:
            return False, "numba is not installed (pip install aevion[fast])"
        if sim.B.name != "numpy":
            return False, f"the numba kernel needs the NumPy backend (got '{sim.B.name}')"
        return True, ""

    def step(self, sim, state, cmd, dt):
        return sim._step_fast(state, cmd, dt)


KERNELS: Dict[str, Kernel] = {k.name: k for k in (ReferenceKernel(), NumbaKernel())}


# ------------------------------------------------------------------------------- parity check
_parity_cache: Dict[str, Tuple[bool, float]] = {}


def _config_key(sim) -> str:
    w = sim.world
    return json.dumps(
        dict(
            spec=sim.spec.to_dict(),
            dt=sim.dt,
            integrator=sim.integrator_name,
            gravity=w.gravity.to_dict(),
            atmosphere=w.atmosphere.to_dict(),
            heightmap=w.heightmap is not None,
            hm=[w.hm_res, list(w.hm_origin)] if w.heightmap is not None else None,
            obstacles=w.has_obstacles,
            ground_offset=sim.ground_offset,
        ),
        sort_keys=True,
        default=str,
    )


def parity_check(sim, steps: int = 12, tol: float = 1e-6, seed: int = 12345, use_cache: bool = True) -> Tuple[bool, float]:
    """Compare the numba and reference kernels for ``sim``'s configuration on randomised inputs.

    Uses a throw-away twin simulator (4 vehicles, same spec / gravity / atmosphere / terrain), so ``sim`` itself is
    untouched. Returns ``(ok, max_abs_difference)``.
    """
    key = _config_key(sim)
    if use_cache and key in _parity_cache:
        return _parity_cache[key]
    from ..environments.world import World
    from .multirotor import Multirotor

    def twin(kernel):
        w = World(gravity=sim.world.gravity, atmosphere=sim.world.atmosphere)
        if sim.world.heightmap is not None:
            w.set_heightmap(sim.world.heightmap, sim.world.hm_origin, sim.world.hm_res)
        if sim.world.has_obstacles:
            w.boxes, w.spheres, w.cylinders = sim.world.boxes.copy(), sim.world.spheres.copy(), sim.world.cylinders.copy()
        return Multirotor(sim.spec, n=4, backend="numpy", dt=sim.dt, integrator=sim.integrator_name, world=w, kernel=kernel, verify_kernel=False)

    ref, fast = twin("reference"), twin("numba")
    rng = np.random.default_rng(seed)
    p0 = rng.uniform([-2, -2, 0.5], [2, 2, 4], (4, 3))
    kw = dict(pos=p0, vel=rng.normal(size=(4, 3)), euler=rng.normal(size=(4, 3)) * 0.3, omega=rng.normal(size=(4, 3)) * 0.5)
    sr, sf = ref.reset(**kw), fast.reset(**kw)
    hover = ref.hover_speed()
    worst = 0.0
    for k in range(steps):
        cmd = hover * (1 + 0.2 * np.sin(0.3 * k + np.arange(ref.n_rotors)[None, :] + np.arange(4)[:, None]))
        sr, sf = ref.step(sr, cmd), fast.step(sf, cmd)
    for f in ("pos", "vel", "quat", "omega", "rotor", "soc"):
        worst = max(worst, float(np.max(np.abs(getattr(sr, f) - getattr(sf, f)))))
    ok = bool(np.isfinite(worst) and worst <= tol)
    _parity_cache[key] = (ok, worst)
    return ok, worst


def select(sim, requested: str = "auto", verify: bool = True) -> Kernel:
    """Choose the kernel for ``sim`` (see module docstring for the fallback rules)."""
    if requested not in ("auto", "reference", "numba"):
        raise KernelUnavailable(f"unknown kernel '{requested}' (choose 'auto', 'reference' or 'numba')")
    if requested == "reference":
        return KERNELS["reference"]
    nb = KERNELS["numba"]
    ok, why = nb.available(sim)
    if not ok:
        if requested == "numba":
            raise KernelUnavailable(f"kernel 'numba' unavailable: {why}")
        return KERNELS["reference"]
    if verify:
        good, err = parity_check(sim)
        if not good:
            msg = f"numba kernel disagrees with the reference implementation (max |difference| {err:.3g}) for this configuration"
            if requested == "numba":
                raise KernelUnavailable(msg)
            warnings.warn(msg + "; falling back to the reference kernel", stacklevel=3)
            return KERNELS["reference"]
    return nb
