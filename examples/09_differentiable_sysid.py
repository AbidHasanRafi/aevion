"""System identification by gradient descent through the simulator.

A "real" vehicle with an unknown thrust coefficient and motor time constant is flown with a known command
sequence. We recover both parameters by minimising the error of the measured specific force (what an IMU sees)
with autograd (torch backend). Fitting altitude alone would identify kT but not the lag: position
double-integrates thrust and barely reflects it.
"""

import numpy as np
import torch

from aevion import Multirotor, VehicleSpec

true_spec = VehicleSpec.generic_1kg().copy(kT=1.12e-5, motor_tau=0.040)
guess = VehicleSpec.generic_1kg()  # nominal parameters (wrong)


def rollout(kT, tau, steps=120):
    spec = guess.copy()
    sim = Multirotor(spec, n=1, backend="torch", dtype="float64", kernel="reference", dt=0.01)
    sim.params.kT = sim.params.kT * 0 + kT  # differentiable parameters
    sim.params.tau_up = sim.params.tau_up * 0 + tau
    sim.params.tau_down = sim.params.tau_down * 0 + tau
    s = sim.reset(pos=[0, 0, 2.0])
    out = []
    t = torch.arange(steps, dtype=torch.float64)
    for k in range(steps):
        # rich excitation (a fast and a slow component) so both the gain and the lag are observable
        cmd = sim.hover_speed() * (1.0 + 0.10 * torch.sin(0.9 * t[k]) + 0.08 * torch.sin(0.17 * t[k])) * torch.ones(1, 4, dtype=torch.float64)
        s = sim.step(s, cmd)
        out.append(s.f_body[0, 2])  # accelerometer-like signal: observes thrust (hence kT and the motor lag) directly
    return torch.stack(out)


with torch.no_grad():
    target = rollout(torch.tensor(1.12e-5, dtype=torch.float64), torch.tensor(0.040, dtype=torch.float64))

log_kT = torch.tensor(np.log(1.0e-5), dtype=torch.float64, requires_grad=True)  # optimise in log space
log_tau = torch.tensor(np.log(0.025), dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([log_kT, log_tau], lr=0.08)
for it in range(80):
    opt.zero_grad()
    loss = ((rollout(log_kT.exp(), log_tau.exp()) - target) ** 2).mean()
    loss.backward()
    opt.step()
    if it % 10 == 0:
        print(f"iter {it:3d}  loss {loss.item():.3e}  kT {log_kT.exp().item():.3e}  tau {log_tau.exp().item():.4f}")
print(f"identified kT = {log_kT.exp().item():.4e} (true 1.12e-5), tau = {log_tau.exp().item():.4f} s (true 0.040)")
