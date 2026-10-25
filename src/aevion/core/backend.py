"""Array backends.

Every numerical routine in Aevion is written against the small :class:`Backend`
interface below, so the same code runs on NumPy (CPU, no extra dependencies),
on PyTorch CPU tensors, or on PyTorch CUDA tensors. Torch tensors keep autograd
history, which makes the dynamics differentiable.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

# torch is optional *and slow to import* (seconds): look for it without importing, and import on first real use.
try:
    _HAVE_TORCH = importlib.util.find_spec("torch") is not None
except (ImportError, ValueError):  # pragma: no cover - broken installation
    _HAVE_TORCH = False


class _LazyTorch:
    """Stand-in for the ``torch`` module that performs the import on first attribute access."""

    def __getattr__(self, name):
        import torch

        return getattr(torch, name)


_torch: Any = _LazyTorch() if _HAVE_TORCH else None


def _is_torch_tensor(x) -> bool:
    """True for torch tensors without importing torch: an object can only be a tensor if torch has already been imported."""
    mod = sys.modules.get("torch")
    return mod is not None and isinstance(x, mod.Tensor)


def _dims(shape: tuple) -> tuple:
    """``zeros(3, 4)`` and ``zeros((3, 4))`` both mean shape (3, 4)."""
    return tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)


class Backend:
    """Minimal array-API shim. Subclasses implement the handful of ops we need."""

    name = "abstract"
    device = "cpu"

    # -- creation ---------------------------------------------------------
    def asarray(self, x, dtype=None):
        raise NotImplementedError

    def zeros(self, *shape):
        raise NotImplementedError

    def ones(self, *shape):
        raise NotImplementedError

    def zeros_like(self, x):
        raise NotImplementedError

    def full(self, shape, value):
        raise NotImplementedError

    def eye(self, n):
        raise NotImplementedError

    # -- structure --------------------------------------------------------
    def stack(self, xs, axis=0):
        raise NotImplementedError

    def cat(self, xs, axis=0):
        raise NotImplementedError

    def to_numpy(self, x):
        raise NotImplementedError

    def clone(self, x):
        raise NotImplementedError

    # -- math -------------------------------------------------------------
    def sqrt(self, x):
        raise NotImplementedError

    def sin(self, x):
        raise NotImplementedError

    def cos(self, x):
        raise NotImplementedError

    def exp(self, x):
        raise NotImplementedError

    def atan2(self, y, x):
        raise NotImplementedError

    def arcsin(self, x):
        raise NotImplementedError

    def arccos(self, x):
        raise NotImplementedError

    def abs(self, x):
        raise NotImplementedError

    def clip(self, x, lo, hi):
        raise NotImplementedError

    def where(self, c, a, b):
        raise NotImplementedError

    def maximum(self, a, b):
        raise NotImplementedError

    def minimum(self, a, b):
        raise NotImplementedError

    def cross(self, a, b):
        raise NotImplementedError

    def einsum(self, eq, *ops):
        raise NotImplementedError

    def norm(self, x, axis=-1, keepdims=False):
        return self.sqrt(self.sum(x * x, axis=axis, keepdims=keepdims))

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def mean(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def randn(self, shape, gen=None):
        raise NotImplementedError

    def rand(self, shape, gen=None):
        raise NotImplementedError

    def make_rng(self, seed):
        raise NotImplementedError

    def inv(self, x):
        raise NotImplementedError

    def sign(self, x):
        raise NotImplementedError

    def floor(self, x):
        raise NotImplementedError

    def tanh(self, x):
        raise NotImplementedError

    def isfinite(self, x):
        raise NotImplementedError


class NumpyBackend(Backend):
    name = "numpy"

    def __init__(self, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        self.device = "cpu"

    def asarray(self, x, dtype=None):
        if _is_torch_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype or self.dtype)

    def zeros(self, *shape):
        return np.zeros(_dims(shape), dtype=self.dtype)

    def ones(self, *shape):
        return np.ones(_dims(shape), dtype=self.dtype)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def full(self, shape, value):
        return np.full(shape, value, dtype=self.dtype)

    def eye(self, n):
        return np.eye(n, dtype=self.dtype)

    def stack(self, xs, axis=0):
        return np.stack(xs, axis=axis)

    def cat(self, xs, axis=0):
        return np.concatenate(xs, axis=axis)

    def to_numpy(self, x):
        return np.asarray(x)

    def clone(self, x):
        return np.array(x, copy=True)

    def sqrt(self, x):
        return np.sqrt(x)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def exp(self, x):
        return np.exp(x)

    def atan2(self, y, x):
        return np.arctan2(y, x)

    def arcsin(self, x):
        return np.arcsin(np.clip(x, -1.0, 1.0))

    def arccos(self, x):
        return np.arccos(np.clip(x, -1.0, 1.0))

    def abs(self, x):
        return np.abs(x)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def where(self, c, a, b):
        return np.where(c, a, b)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def cross(self, a, b):
        a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
        b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
        return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)

    def einsum(self, eq, *ops):
        return np.einsum(eq, *ops)

    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def make_rng(self, seed):
        return np.random.default_rng(seed)

    def randn(self, shape, gen=None):
        gen = gen if gen is not None else np.random.default_rng()
        return gen.standard_normal(shape).astype(self.dtype)

    def rand(self, shape, gen=None):
        gen = gen if gen is not None else np.random.default_rng()
        return gen.random(shape).astype(self.dtype)

    def inv(self, x):
        return np.linalg.inv(x)

    def sign(self, x):
        return np.sign(x)

    def floor(self, x):
        return np.floor(x)

    def tanh(self, x):
        return np.tanh(x)

    def isfinite(self, x):
        return np.isfinite(x)


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, device="cpu", dtype=None):
        if _torch is None:
            raise ImportError("PyTorch is required for the torch backend: pip install aevion[torch]")
        self.device = str(device)
        self.dtype = dtype if dtype is not None else _torch.float32
        if self.device.startswith("cuda") and not _torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    @property
    def _kw(self):
        return dict(dtype=self.dtype, device=self.device)

    def asarray(self, x, dtype=None):
        t = _torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=dtype or self.dtype)
        return t.as_tensor(np.asarray(x), dtype=dtype or self.dtype, device=self.device)

    def zeros(self, *shape):
        return _torch.zeros(*_dims(shape), **self._kw)

    def ones(self, *shape):
        return _torch.ones(*_dims(shape), **self._kw)

    def zeros_like(self, x):
        return _torch.zeros_like(x)

    def full(self, shape, value):
        return _torch.full(tuple(shape), float(value), **self._kw)

    def eye(self, n):
        return _torch.eye(n, **self._kw)

    def stack(self, xs, axis=0):
        return _torch.stack(list(xs), dim=axis)

    def cat(self, xs, axis=0):
        return _torch.cat(list(xs), dim=axis)

    def to_numpy(self, x):
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def clone(self, x):
        return x.clone()

    def sqrt(self, x):
        return _torch.sqrt(x)

    def sin(self, x):
        return _torch.sin(x)

    def cos(self, x):
        return _torch.cos(x)

    def exp(self, x):
        return _torch.exp(x)

    def atan2(self, y, x):
        return _torch.atan2(y, x)

    def arcsin(self, x):
        return _torch.asin(_torch.clamp(x, -1.0, 1.0))

    def arccos(self, x):
        return _torch.acos(_torch.clamp(x, -1.0, 1.0))

    def abs(self, x):
        return _torch.abs(x)

    def clip(self, x, lo, hi):
        if isinstance(lo, _torch.Tensor) or isinstance(hi, _torch.Tensor):
            out = x
            if lo is not None:
                out = _torch.maximum(out, lo if isinstance(lo, _torch.Tensor) else _torch.as_tensor(lo, **self._kw))
            if hi is not None:
                out = _torch.minimum(out, hi if isinstance(hi, _torch.Tensor) else _torch.as_tensor(hi, **self._kw))
            return out
        return _torch.clamp(x, lo, hi)

    def where(self, c, a, b):
        a = a if isinstance(a, _torch.Tensor) else _torch.as_tensor(a, **self._kw)
        b = b if isinstance(b, _torch.Tensor) else _torch.as_tensor(b, **self._kw)
        return _torch.where(c, a, b)

    def maximum(self, a, b):
        a = a if isinstance(a, _torch.Tensor) else _torch.as_tensor(a, **self._kw)
        b = b if isinstance(b, _torch.Tensor) else _torch.as_tensor(b, **self._kw)
        return _torch.maximum(a, b)

    def minimum(self, a, b):
        a = a if isinstance(a, _torch.Tensor) else _torch.as_tensor(a, **self._kw)
        b = b if isinstance(b, _torch.Tensor) else _torch.as_tensor(b, **self._kw)
        return _torch.minimum(a, b)

    def cross(self, a, b):
        a, b = _torch.broadcast_tensors(a, b)
        return _torch.linalg.cross(a, b, dim=-1)

    def einsum(self, eq, *ops):
        return _torch.einsum(eq, *ops)

    def sum(self, x, axis=None, keepdims=False):
        if axis is None:
            return _torch.sum(x)
        return _torch.sum(x, dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        if axis is None:
            return _torch.mean(x)
        return _torch.mean(x, dim=axis, keepdim=keepdims)

    def make_rng(self, seed):
        g = _torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return g

    def randn(self, shape, gen=None):
        return _torch.randn(*shape, generator=gen, **self._kw)

    def rand(self, shape, gen=None):
        return _torch.rand(*shape, generator=gen, **self._kw)

    def inv(self, x):
        return _torch.linalg.inv(x)

    def sign(self, x):
        return _torch.sign(x)

    def floor(self, x):
        return _torch.floor(x)

    def tanh(self, x):
        return _torch.tanh(x)

    def isfinite(self, x):
        return _torch.isfinite(x)


_DTYPES = {"float32": "float32", "float64": "float64", "f32": "float32", "f64": "float64"}


def get_backend(spec="numpy", dtype=None) -> Backend:
    """Resolve a backend.

    ``spec`` may be ``"numpy"``, ``"torch"``, ``"torch:cpu"``, ``"cuda"``,
    ``"torch:cuda:0"`` or an existing :class:`Backend`. ``dtype`` is
    ``"float32"`` or ``"float64"`` (default: float64 for NumPy, float32 for torch).
    """
    if isinstance(spec, Backend):
        return spec
    s = str(spec).lower()
    dt = _DTYPES.get(str(dtype).lower()) if dtype is not None else None
    if s == "numpy":
        return NumpyBackend(np.float32 if dt == "float32" else np.float64)
    if s.startswith(("torch", "cuda")) or s == "cpu":
        if s.startswith("torch:"):
            device = s.split(":", 1)[1]
        elif s.startswith("cuda"):
            device = s
        else:
            device = "cpu"
        if _torch is None:
            raise ImportError("PyTorch is required for the torch backend: pip install aevion[torch]")
        tdt = _torch.float64 if dt == "float64" else _torch.float32
        return TorchBackend(device=device, dtype=tdt)
    raise ValueError(f"unknown backend spec {spec!r}")


def torch_available() -> bool:
    return _torch is not None


def cuda_available() -> bool:
    return _torch is not None and _torch.cuda.is_available()


def is_tensor(x) -> bool:
    return _is_torch_tensor(x)
