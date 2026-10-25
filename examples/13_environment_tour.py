"""A tour of the environment and sensors, from one scenario file.

Loads ``scenarios/urban_environment.yaml`` (buildings from GeoJSON at a real place and time, weather, a moving obstacle, geofences,
urban-wake wind, GNSS and radar), flies it, reports what the sensors saw and writes a one-file replay.

    python examples/13_environment_tour.py --out tour --duration 20
"""

import argparse
from pathlib import Path

import numpy as np

from aevion.config import load_config
from aevion.engine import Simulation
from aevion.environments.solar import sun_position
from aevion.sensors.allan import allan_deviation, fit_noise
from aevion.sensors.inertial import IMU_GRADES
from aevion.viewer import export_html

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="environment_tour", help="output directory")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(HERE / "scenarios" / "urban_environment.yaml")
    cfg.duration = args.duration
    sim = Simulation(cfg)
    arch = sim.run()
    world = arch.world

    o = world.origin
    sun = sun_position(o.lat, o.lon, world.utc)
    print(f"place    : {o.lat:.4f} N, {o.lon:.4f} E   time {world.utc}   sun elevation {float(sun['elevation_deg']):.1f} deg")
    print(f"weather  : visibility {world.weather.visibility_m:.0f} m, air density {world.weather.air_density():.3f} kg/m3")
    print(f"scene    : {len(world.meshes)} mesh(es), {world.n_mesh_triangles} triangles, {len(world.actors)} moving obstacle(s)")

    gps = arch["sensors/gps"][:, 0]
    print(
        f"GNSS     : {int(gps[:, 7].min())}-{int(gps[:, 7].max())} satellites, HDOP {gps[:, 8].mean():.2f}, fix {100 * gps[:, 6].mean():.0f} % of epochs"
    )
    det = arch["sensors/radar"][:, 0]
    print(f"radar    : {int((det[..., 0] > 0).sum())} beam detections over {len(det)} frames")
    events = [e["type"] for e in arch.meta["events"] if e["type"].startswith("geofence.")]
    print(f"geofences: {len(events)} event(s) {sorted(set(events))}")

    # What an Allan-variance fit recovers from white noise of a datasheet-grade gyro
    grade = IMU_GRADES["tactical"]
    density = grade["gyro_arw"] * np.pi / 180.0 / 60.0  # deg/sqrt(h) -> rad/s/sqrt(Hz)
    dt = 0.01
    x = np.random.default_rng(0).normal(0.0, density / np.sqrt(dt), (60_000, 2))
    taus, adev = allan_deviation(x, dt)
    print(
        f"IMU      : tactical-grade ARW {grade['gyro_arw']} deg/sqrt(h) -> fitted {fit_noise(taus, adev, ('N',))['N'] / density:.3f} x the specification"
    )

    page = export_html(arch, out / "replay.html", title="Urban environment tour", max_frames=300)
    print(f"replay   : {page}")


if __name__ == "__main__":
    main()
