"""Example Aevion plugin: a sonar sensor, a thrust-loss fault with ramp, and a custom hook."""

from .components import FlightLogger, Sonar


def register(registries):
    """Entry point declared in pyproject.toml (group 'aevion.plugins')."""
    registries.sensors.register("sonar", Sonar, source="my_aevion_ext", doc="downward sonar with a hard range limit")
    registries.hooks.register("flight_logger", FlightLogger, source="my_aevion_ext", doc="print altitude once per second")
