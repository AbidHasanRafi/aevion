"""A small deployable policy network and its export to ONNX and TorchScript.

:class:`MLPPolicy` normalises its input with statistics stored *inside* the module, so the exported graph takes raw observations and returns actions in
``[-1, 1]``; nothing else is needed to run it on a flight computer. :func:`export_onnx` and :func:`export_torchscript` write the files and
:func:`verify_onnx` checks the exported graph against the PyTorch module on random inputs (ONNX Runtime is optional).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["MLPPolicy", "export_onnx", "export_torchscript", "verify_onnx", "load_policy"]


class MLPPolicy(nn.Module):
    """``tanh(MLP((obs − mean) / std))``; accepts NumPy arrays or tensors and returns the same kind."""

    obs_mean: torch.Tensor
    obs_std: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (128, 128),
        obs_mean: Optional[Any] = None,
        obs_std: Optional[Any] = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden = int(obs_dim), int(act_dim), tuple(int(h) for h in hidden)
        acts = dict(relu=nn.ReLU, tanh=nn.Tanh, elu=nn.ELU)
        if activation not in acts:
            raise ValueError(f"activation must be one of {sorted(acts)}")
        self.activation = activation
        layers: list = []
        d = self.obs_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), acts[activation]()]
            d = h
        layers += [nn.Linear(d, self.act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        mean = torch.zeros(self.obs_dim) if obs_mean is None else torch.as_tensor(np.asarray(obs_mean), dtype=torch.float32)
        std = torch.ones(self.obs_dim) if obs_std is None else torch.as_tensor(np.asarray(obs_std), dtype=torch.float32)
        self.register_buffer("obs_mean", mean)
        self.register_buffer("obs_std", std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def __call__(self, obs: Any) -> Any:  # type: ignore[override]
        if isinstance(obs, torch.Tensor):
            out = self.forward(obs.to(self.obs_mean.dtype))
            return out.to(obs.dtype) if obs.dtype.is_floating_point else out
        x = torch.as_tensor(np.asarray(obs, np.float32).reshape(-1, self.obs_dim))
        return self.forward(x).numpy()

    def config(self) -> dict:
        return dict(obs_dim=self.obs_dim, act_dim=self.act_dim, hidden=list(self.hidden), activation=self.activation)

    def save(self, path: Any) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(config=self.config(), state=self.state_dict()), p)
        return p


def load_policy(path: Any) -> MLPPolicy:
    """Load a policy written by :meth:`MLPPolicy.save` (weights only; no pickled code is executed)."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    pol = MLPPolicy(**blob["config"])
    pol.load_state_dict(blob["state"])
    pol.eval()
    return pol


def export_onnx(policy: MLPPolicy, path: Any, opset: int = 17) -> Path:
    """Write an ONNX graph with input ``obs`` ``(batch, obs_dim)`` and output ``action`` ``(batch, act_dim)``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    policy = policy.eval().cpu()
    x = torch.zeros(1, policy.obs_dim)
    kw: Dict[str, Any] = dict(
        input_names=["obs"], output_names=["action"], dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=opset
    )
    try:
        torch.onnx.export(policy, (x,), str(p), dynamo=False, **kw)  # the TorchScript-based exporter needs no extra packages
    except TypeError:  # older torch has no `dynamo` argument and always uses it
        torch.onnx.export(policy, (x,), str(p), **kw)
    return p


def export_torchscript(policy: MLPPolicy, path: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(policy.eval().cpu(), torch.zeros(1, policy.obs_dim))
    traced.save(str(p))
    return p


def verify_onnx(policy: MLPPolicy, path: Any, n: int = 64, seed: int = 0, atol: float = 1e-5) -> float:
    """Largest absolute difference between the ONNX graph (run with ONNX Runtime) and the PyTorch policy on random observations; raises if above ``atol``."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, policy.obs_dim)).astype(np.float32) * 2.0
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["action"], {"obs": x})[0]
    want = policy(x)
    err = float(np.abs(got - want).max())
    if err > atol:
        raise AssertionError(f"ONNX output differs from the PyTorch policy by {err:.3g} (> {atol})")
    return err
