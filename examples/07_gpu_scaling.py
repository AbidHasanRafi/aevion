"""Same code, three backends: NumPy, NumPy+Numba kernel, and (if available) CUDA."""

import time

from aevion import Multirotor, cuda_available


def bench(label, n, **kw):
    sim = Multirotor("generic_1kg", n=n, dt=0.005, **kw)
    s = sim.reset()
    cmd = sim.hover_command()
    for _ in range(5):
        s = sim.step(s, cmd)
    t0 = time.perf_counter()
    for _ in range(50):
        s = sim.step(s, cmd)
    if sim.B.device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    print(f"{label:24s} N={n:7d}: {n * 50 / (time.perf_counter() - t0) / 1e6:7.2f} M vehicle-steps/s")


bench("numpy reference", 4096, kernel="reference")
bench("numpy + numba", 4096)
if cuda_available():
    for n in (4096, 65536, 1_000_000):
        bench("cuda (float32)", n, backend="cuda")
else:
    print("no CUDA device found; everything above ran on the CPU")
