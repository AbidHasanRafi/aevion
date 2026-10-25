"""The simulation engine: declarative config, fault injection, hooks, streaming archive, provenance and diff."""

import tempfile
from pathlib import Path

import numpy as np

from aevion.config import SimulationConfig, config_hash
from aevion.datasets import diff_archives
from aevion.engine import CallbackHook, EventLogHook, Simulation

cfg = SimulationConfig.from_dict(
    {
        "name": "engine_demo",
        "duration": 12.0,
        "seed": 3,
        "vehicle": {"model": "hex_3kg"},
        "trajectory": {"type": "hover", "pos": [0, 0, 8.0]},
        "wind": {"type": "dryden", "w20": 4.0},
        "sensors": ["imu", "gps", "baro", "mag"],
        "faults": [
            {"type": "gps_outage", "t_start": 3.0, "t_end": 5.0},  # transient sensor fault
            {"type": "payload_change", "t_start": 6.0, "mass": 0.4},  # drop-in payload
            {"type": "wind_gust", "t_start": 8.0, "amplitude": [8.0, 0, 0], "duration": 1.5},
        ],
    }
)
print("config hash:", config_hash(cfg))

# a hook that watches the run and a built-in event logger
peak = {"tilt": 0.0}
watch = CallbackHook("tilt_watch", post_step=lambda s: peak.__setitem__("tilt", max(peak["tilt"], float(np.abs(s.state.euler(s.B)[0, :2]).max()))))
sim = Simulation(cfg, hooks=[watch, EventLogHook("fault.*")])
archive = sim.run()
print(f"peak tilt {np.degrees(peak['tilt']):.1f} deg; events: {[e['type'] for e in archive.meta['events']]}")
print(archive.summary())

# the same run, streamed to disk with bounded memory, is bit-identical
out = Path(tempfile.mkdtemp()) / "stream.aevd"
cfg_stream = cfg.replace(logging=cfg.logging.replace(mode="stream", path=str(out), chunk_steps=64))
streamed = Simulation(cfg_stream).run()
print(
    "streamed identical to in-memory:",
    diff_archives(archive, streamed, with_metrics=False).identical,
    "| integrity:",
    streamed.verify_integrity() or "OK",
)

# change one thing and see exactly what differs
other = Simulation(cfg.replace(seed=4)).run()
print(diff_archives(archive, other).summary(max_items=5))
