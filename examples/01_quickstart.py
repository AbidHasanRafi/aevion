"""Fly 1000 drones to a setpoint on the CPU and print the throughput."""

import time

from aevion import Multirotor
from aevion.control import GeometricController, Reference

sim = Multirotor("generic_1kg", n=1000, dt=0.005)  # backend="cuda" for a GPU
ctrl = GeometricController(sim)
ctrl.reset(sim.n)
state = sim.reset(pos=[0, 0, 1.0])
ref = Reference.hover(sim.B, sim.n, pos=(1.0, 2.0, 3.0))

t0 = time.perf_counter()
for _ in range(1000):
    state = sim.step(state, ctrl.act(state, ref, sim.dt))
dt = time.perf_counter() - t0
print(f"final position of drone 0: {state.pos[0].round(3)}")
print(f"{sim.n * 1000 / dt / 1e6:.2f} M vehicle-steps/s (physics + controller), {sim.info()}")
