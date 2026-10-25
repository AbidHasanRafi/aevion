from aevion.engine.hooks import Hook
from aevion.sensors.base import Sensor


class Sonar(Sensor):
    """Downward sonar: altitude with Gaussian noise, clipped at ``max_range``."""

    name = "sonar"

    def __init__(self, rate=20.0, max_range=5.0, noise_std=0.02, **kw):  # **kw forwards latency/dropout/name/clock/extrinsics
        super().__init__(rate=rate, **kw)
        self.max_range, self.noise_std = max_range, noise_std

    def measure(self, state):
        B = self.B
        h = state.pos[:, 2] + self.noise(state.pos[:, 2].shape, self.noise_std)
        return dict(range=B.minimum(h, self.max_range * B.ones(1)), t=state.t)


class FlightLogger(Hook):
    """Print the altitude of vehicle 0 once per simulated second."""

    name = "flight_logger"

    def __init__(self, every_s=1.0):
        self.every_s, self._next = every_s, 0.0

    def post_step(self, sim):
        if sim.t >= self._next:
            self._next += self.every_s
            print(f"[flight_logger] t={sim.t:5.1f}s  z={float(sim.state.pos[0, 2]):6.2f} m")
