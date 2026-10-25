"""Generate the documentation pages that are derived from code and repository files.

Written into ``docs/`` (git-ignored): the home page (from README.md), the guide (from DOCUMENTATION.md), the API
reference, the CLI reference and the figure gallery. ``mkdocs build`` runs this automatically through ``tools/mkdocs_hooks.py``.

python tools/gen_site.py            # everything
python tools/gen_site.py --no-gallery
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "src"))

REPO_BLOB = "https://github.com/aevion/aevion/blob/main/"
GENERATED_DIRS = ["api", "gallery", "assets"]
GENERATED_FILES = ["cli.md", "guide.md", "index.md"]

API_BLURB = {
    "aevion": "Top-level convenience imports.",
    "aevion.errors": "Exception hierarchy.",
    "aevion.deprecation": "Deprecation machinery used by the library (and by plugins).",
    "aevion.config": "Typed configuration schema, validation, migration and file I/O.",
    "aevion.registry": "Component registries and plugin discovery.",
    "aevion.engine": "The `Simulation` engine, scheduler, hooks, faults and event bus.",
    "aevion.health": "Run-time health monitors (divergence, saturation, ...).",
    "aevion.frames": "Rigid transforms, frame trees and clock models.",
    "aevion.datasets": "Run archives, chunked store, provenance, run registry, diff and export.",
    "aevion.dynamics": "Multirotor dynamics and kernels.",
    "aevion.core.backend": "NumPy / torch array backend abstraction.",
    "aevion.environments.world": "Obstacle worlds and terrain.",
    "aevion.environments.wind": "Wind and turbulence models.",
    "aevion.environments.geodesy": "WGS84 geodesy: ECEF/geodetic/local ENU, Vincenty and haversine.",
    "aevion.environments.geomagnetic": "Geomagnetic field models (tilted dipole, fixed field).",
    "aevion.environments.solar": "NOAA solar ephemeris, sunrise/sunset, clear-sky irradiance.",
    "aevion.environments.weather": "Weather presets, moist-air density, rain and fog extinction.",
    "aevion.environments.wind_fields": "Log-law, gridded and urban-wake wind fields.",
    "aevion.environments.mesh": "Triangle meshes and OBJ / STL / glTF loaders.",
    "aevion.environments.mesh_index": "BVH acceleration structure for ray, distance and collision queries.",
    "aevion.environments.terrain": "GeoTIFF / ASCII-grid / npz terrain loaders.",
    "aevion.environments.osm": "OpenStreetMap and GeoJSON building footprints.",
    "aevion.environments.sdf": "Signed and Euclidean distance fields.",
    "aevion.environments.scene": "The `scene` world: terrain, buildings, meshes and primitives from a configuration.",
    "aevion.environments.actors": "Moving obstacles (linear, circular, waypoint, bouncing, random walk).",
    "aevion.environments.geofence": "Geofences, no-fly zones and the geofence health check.",
    "aevion.sensors.allan": "Allan variance and noise-parameter fitting.",
    "aevion.sensors.gnss": "GNSS pseudorange receiver, constellations and DOP.",
    "aevion.sensors.airdata": "Pitot, air-data and optical-flow sensors.",
    "aevion.sensors.proximity": "Ultrasonic and UWB ranging sensors.",
    "aevion.sensors.radar": "FMCW radar with CFAR detection.",
    "aevion.sensors.scan_patterns": "LiDAR scan patterns.",
    "aevion.sensors.camera_models": "Camera lens models.",
    "aevion.sensors.event_camera": "Event camera.",
    "aevion.vehicles.spec": "Vehicle specifications and presets.",
    "aevion.control": "Controllers, allocation and trajectories.",
    "aevion.sensors": "Sensor models (IMU, GPS, barometer, LiDAR, camera, ...).",
    "aevion.estimation": "State estimators: complementary filter, position-velocity KF and the error-state Kalman filter, with consistency (NEES/NIS) and observability tools.",
    "aevion.estimation.mapping": "Log-odds occupancy grid.",
    "aevion.planning": "Planning space, A*/Theta*/D* Lite/RRT*/PRM, minimum-snap and time-optimal trajectories, ORCA-3D and control-barrier avoidance, behaviour-tree missions.",
    "aevion.control.advanced": "MPC, MPPI, INDI, L1 adaptive and sliding-mode controllers, and the batched QP and disturbance observer they use.",
    "aevion.vehicles.propeller": "Blade-element-momentum propeller model and the built-in propeller database.",
    "aevion.vehicles.motor": "BLDC motor, ESC and motor-propeller operating point.",
    "aevion.vehicles.battery_pack": "Battery pack with Peukert, temperature, thermal and ageing effects.",
    "aevion.engine.effects": "Slung load, proximity (ceiling and wall) effects and battery-model hooks.",
    "aevion.aircraft": "Fixed-wing and VTOL aircraft, trim, linearisation and autopilots.",
    "aevion.identification": "System identification by multiple shooting.",
    "aevion.bridges.mavlink": "MAVLink v1/v2 codec and streaming parser.",
    "aevion.bridges.telemetry": "MAVLink .tlog reading/writing and NED/ENU conversions.",
    "aevion.bridges.ulog": "PX4 ULog reader and writer.",
    "aevion.bridges.transport": "Loopback and (loopback-only) UDP byte transports.",
    "aevion.bridges.hil": "Hardware/software-in-the-loop over MAVLink.",
    "aevion.bridges.ros2": "ROS 2 message construction and an optional rclpy publisher.",
    "aevion.wireless": "Wireless package overview and the original channel/network classes.",
    "aevion.wireless.propagation": "Path-loss models (free space, two-ray, log-distance, Al-Hourani, 3GPP TR 38.901 / 36.777), knife-edge diffraction, image-method ray tracing.",
    "aevion.wireless.fading": "Rayleigh/Rician/Nakagami fading, time-correlated Jakes fading, correlated shadowing, Doppler statistics.",
    "aevion.wireless.antennas": "Antenna patterns, phased arrays, beam codebooks, beamforming and UAV-attitude-dependent gain.",
    "aevion.wireless.channel": "Air-to-ground channel, MIMO, blockage, delay spread and mobility kinematics.",
    "aevion.wireless.phy": "NR numerology, MCS/CQI tables, BLER, link adaptation with HARQ, OFDM effective SINR.",
    "aevion.wireless.mac_network": "Schedulers, handover, radio-link failure, relay, mesh routing, CSMA and the integrated NR cellular network.",
    "aevion.wireless.ris": "Reconfigurable intelligent surfaces: geometry, cascaded channel, phase configuration.",
    "aevion.wireless.ntn": "Satellite links: geometry, link budget, Doppler, atmospheric loss.",
    "aevion.wireless.maps": "Radio environment maps, coverage statistics and map reconstruction.",
    "aevion.wireless.security": "Interference, GNSS jamming/spoofing threat models and matching engine faults.",
    "aevion.wireless.metrics": "Outage, capacity, age of information, latency and run-level link metrics.",
    "aevion.wireless.interop": "ns-3 mobility/link-trace export and OBJ geometry export.",
    "aevion.multiagent": "Formations, swarm coordination, consensus, task allocation, coverage and UTM-style deconfliction.",
    "aevion.multiagent.graphs": "Communication graphs, Laplacian spectra and consensus weight matrices.",
    "aevion.multiagent.consensus": "Average, delayed and lossy consensus, push-sum and displacement-based formation control.",
    "aevion.multiagent.allocation": "Hungarian, auction and CBBA task allocation.",
    "aevion.multiagent.coverage": "Voronoi coverage control (Lloyd's algorithm).",
    "aevion.multiagent.utm": "4-D conflict detection, strategic ground delays and prioritised space-time planning.",
    "aevion.multiagent.laws": "Swarm laws acting through the lossy inter-agent link: formation keeping, rendezvous, coverage.",
    "aevion.ml": "Datasets, behaviour cloning and DAgger, policy export, curricula, tracking, tuning and parallel rollouts.",
    "aevion.ml.dataset": "D4RL-style transition datasets and the collection loop.",
    "aevion.ml.policies": "A deployable MLP policy with ONNX and TorchScript export.",
    "aevion.ml.imitation": "Behaviour cloning and DAgger.",
    "aevion.ml.curriculum": "Success-driven curriculum scheduler.",
    "aevion.ml.tracking": "Training-run loggers: JSONL, CSV, TensorBoard, MLflow, Weights & Biases.",
    "aevion.ml.tuning": "Optuna hyper-parameter search.",
    "aevion.ml.rollouts": "Parallel data collection across processes.",
    "aevion.envs.swarm_env": "Cooperative multi-agent environment and PettingZoo-style API.",
    "aevion.envs.tasks_ext": "Gate racing, landing, evasive-target tracking, inspection and area coverage tasks.",
    "aevion.envs.experts": "Privileged scripted experts for the extended tasks.",
    "aevion.datasets.vision": "COCO / YOLO labels, PNG, depth and optical-flow ground truth from the camera.",
    "aevion.experiments.design": "Latin hypercube, Sobol and Halton designs and named parameter spaces.",
    "aevion.experiments.sensitivity": "Morris elementary effects and Sobol variance-based indices.",
    "aevion.experiments.bayesopt": "Gaussian-process regression and Bayesian optimisation.",
    "aevion.experiments.cache": "Content-addressed result cache.",
    "aevion.experiments.runners": "Serial and process runners, Slurm job-array script generator.",
    "aevion.experiments.database": "SQLite result database and leaderboards.",
    "aevion.experiments.config_experiment": "Experiments over full simulation configurations.",
    "aevion.experiments.suites": "Benchmark suites with leaderboards.",
    "aevion.analysis.diagnostics": "Frequency response, stability margins, ROC and kernel density numerics.",
    "aevion.analysis.figures_diag": "Bode, pole-zero, Allan, NEES, ROC, ridgeline and step-response figures.",
    "aevion.analysis.templates": "Journal templates: figure sizes, typefaces and LaTeX wrappers.",
    "aevion.analysis.appendix": "Auto-generated methods appendix.",
    "aevion.envs": "Gymnasium environments.",
    "aevion.experiments": "Experiment grids, statistics and training logs.",
    "aevion.scenario": "The original `Scenario` API (kept for compatibility).",
    "aevion.metrics": "Flight-performance metrics.",
    "aevion.analysis.style": "Research-grade figure style: journal column sizes, serif text, boxed axes.",
    "aevion.analysis.figures": "Single-run figures.",
    "aevion.analysis.figures_stats": "Statistical comparison figures.",
    "aevion.analysis.stats": "Statistics helpers.",
    "aevion.analysis.report": "HTML report builder.",
    "aevion.viewer": "The viewer: single-file replay, local server, live streaming, run comparison, video/screenshot export, notebook widget and the scene editor.",
    "aevion.safety": "Resource limits for untrusted configurations.",
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def clean() -> None:
    for d in GENERATED_DIRS:
        shutil.rmtree(DOCS / d, ignore_errors=True)
    for f in GENERATED_FILES:
        (DOCS / f).unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------- API
def gen_api() -> None:
    from aevion._api import PUBLIC_MODULES

    mods = list(PUBLIC_MODULES)
    lines = [
        "# API reference",
        "",
        "Everything listed here is public API and covered by the [deprecation policy](../guide.md#versioning-and-deprecation-policy). Names that start with an underscore are private.",
        "",
        "| Module | Contents |",
        "| --- | --- |",
    ]
    for m in mods:
        lines.append(f"| [`{m}`]({m}.md) | {API_BLURB.get(m, '')} |")
        _write(DOCS / "api" / f"{m}.md", f"# `{m}`\n\n::: {m}\n    options:\n      heading_level: 2\n")
    _write(DOCS / "api" / "index.md", "\n".join(lines) + "\n")


# -------------------------------------------------------------------------------------- home and guide
def _relink(text: str, *, guide: str) -> str:
    """Rewrite links written for the repository root so they resolve inside the site.

    Repository files that are not site pages (source, examples, LICENSE, ...) point at the repository on GitHub.
    """

    def sub(m: re.Match[str]) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"^([a-z][a-z0-9+.-]*:|#)", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        if path == "README.md":
            return f"[{label}](index.md{'#' + frag if frag else ''})"
        if path == "DOCUMENTATION.md":
            return f"[{label}]({guide}#{frag})" if frag else f"[{label}]({guide})"
        if path == "docs/simulation.schema.json":
            return f"[{label}](simulation.schema.json)"
        return f"[{label}]({REPO_BLOB}{path}{'#' + frag if frag else ''})"

    return re.sub(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)", sub, text)


def gen_home_and_guide() -> None:
    guide = (ROOT / "DOCUMENTATION.md").read_text(encoding="utf-8")
    _write(DOCS / "guide.md", _relink(guide, guide=""))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    cover = ROOT / "aevion.png"
    if cover.exists():
        (DOCS / "assets").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cover, DOCS / "assets" / cover.name)
        readme = readme.replace(f'src="{cover.name}"', f'src="assets/{cover.name}"')
    _write(DOCS / "index.md", _relink(readme, guide="guide.md"))


# ------------------------------------------------------------------------------------------------- CLI
def gen_cli() -> None:
    import argparse as ap

    from aevion.cli import build_parser

    parser = build_parser()
    parser.prog = "aevion"
    out = ["# Command line", "", "Generated from the argument parser (`aevion --help`).", "", "```text", parser.format_help().rstrip(), "```", ""]
    sub = next(a for a in parser._actions if isinstance(a, ap._SubParsersAction))
    for name, p in sub.choices.items():
        p.prog = f"aevion {name}"
        out += [f"## `aevion {name}`", "", "```text", p.format_help().rstrip(), "```", ""]
    _write(DOCS / "cli.md", "\n".join(out))


# ---------------------------------------------------------------------------------------------- gallery
GALLERY = [
    ("trajectory_3d", "3-D trajectory coloured by speed", "fig_trajectory_3d"),
    ("topdown", "Top-down view with the reference path", "fig_topdown"),
    ("tracking", "Position tracking and error", "fig_tracking"),
    ("attitude", "Attitude angles", "fig_attitude"),
    ("rotors", "Rotor speeds", "fig_rotors"),
    ("energy", "Power and battery state of charge", "fig_energy"),
    ("error_distribution", "Error distribution (histogram and ECDF)", "fig_error_distribution"),
    ("psd", "Power spectral density of the tracking error", "fig_psd"),
    ("overview", "Six-panel overview, double-column", "fig_overview"),
]


def gen_gallery() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aevion.analysis import figures as F
    from aevion.engine import Simulation

    arch = Simulation(
        {
            "name": "gallery",
            "seed": 7,
            "duration": 12.0,
            "world": {"type": "forest", "params": {"n_trees": 30}},
            "wind": {"type": "composite", "params": {"models": [{"type": "constant", "velocity": [1.5, 0.5, 0.0]}, {"type": "dryden", "w20": 3.0}]}},
            "trajectory": {"type": "figure8", "params": {"center": [0, 0, 2.5], "size": 3.0, "period": 8.0}},
            "sensors": ["imu"],
        }
    ).run()
    page = [
        "# Gallery",
        "",
        "Every figure on this page is produced by Aevion's analysis module from one 12-second figure-eight flight in a forest with wind and turbulence "
        "(`seed: 7`). They are research-grade figures with journal column sizes, serif text and boxed axes with inward ticks, so they can go into a paper unchanged: "
        "`aevion.analysis.figures.fig_*(archive).savefig('name.pdf')`.",
        "",
    ]
    for stem, caption, fn in GALLERY:
        fig = getattr(F, fn)(arch)
        out = DOCS / "gallery" / f"{stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
        page += [f"## {caption}", "", f"![{caption}]({stem}.png)", "", f"`{fn}(archive)`", ""]
    _write(DOCS / "gallery" / "index.md", "\n".join(page))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gallery", action="store_true", help="skip figure rendering (fast)")
    args = ap.parse_args(argv)
    clean()
    gen_api()
    gen_home_and_guide()
    gen_cli()
    if not args.no_gallery:
        gen_gallery()
    else:
        _write(DOCS / "gallery" / "index.md", "# Gallery\n\n*(skipped in this build)*\n")
    print("generated docs/index.md, docs/guide.md, docs/api, docs/cli.md, docs/gallery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
