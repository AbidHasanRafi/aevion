"""The ``environment:`` configuration block, weather in the dynamics, the magnetic model, and input-file safety."""

import math

import numpy as np
import pytest

from aevion.config import SimulationConfig, config_hash, dump_config, json_schema, load_config
from aevion.core.backend import NumpyBackend
from aevion.engine import Simulation
from aevion.environments.geodesy import LocalFrame
from aevion.environments.geomagnetic import TiltedDipole
from aevion.environments.weather import Weather
from aevion.environments.world import World
from aevion.errors import ConfigError
from aevion.safety import ResourceLimitError, ResourceLimits, check_config
from aevion.sensors.inertial import Magnetometer

B = NumpyBackend()
ENV = {
    "origin": {"lat": 47.3769, "lon": 8.5417, "alt": 408.0},
    "utc": "2026-06-21T10:00:00Z",
    "weather": {"preset": "rain", "rain_mm_h": 6.0},
    "actors": [{"type": "circular", "params": {"center": [0, 0, 6.0], "path_radius": 8.0, "omega": 0.4}}],
    "geofences": [
        {"name": "area", "kind": "inclusion", "margin": 2.0, "zone": {"type": "cylinder", "center": [0, 0], "radius": 30.0, "z_max": 60.0}}
    ],
}


def hover(env=None, **extra):
    cfg = {"duration": 2.0, "trajectory": {"type": "hover", "params": {"pos": [0, 0, 5.0]}}, **extra}
    if env is not None:
        cfg["environment"] = env
    return cfg


# ------------------------------------------------------------------------------------------------ config
def test_environment_block_parses_dumps_and_is_in_the_schema():
    cfg = SimulationConfig.from_dict(hover(ENV))
    e = cfg.environment
    assert (
        e.origin.lat == 47.3769
        and e.utc.startswith("2026")
        and e.weather["rain_mm_h"] == 6.0
        and e.actors[0].type == "circular"
        and e.geofences[0].name == "area"
    )
    again = SimulationConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict() and config_hash(again) == config_hash(cfg)
    props = json_schema(SimulationConfig)["properties"]
    assert "environment" in props


def test_the_hash_of_configurations_without_an_environment_is_unchanged():
    plain = SimulationConfig.from_dict({"duration": 3.0, "seed": 4})
    explicit = SimulationConfig.from_dict({"duration": 3.0, "seed": 4, "environment": {}})
    assert config_hash(plain) == config_hash(explicit)
    assert "environment" not in __import__("aevion.config", fromlist=["canonical_json"]).canonical_json(plain)
    assert config_hash(SimulationConfig.from_dict(hover(ENV))) != config_hash(SimulationConfig.from_dict(hover()))


@pytest.mark.parametrize(
    "env,needle",
    [
        ({"utc": "tomorrow"}, "ISO-8601"),
        ({"weather": {"preset": "hail"}}, "hail"),
        ({"weather": {"humidity": 3.0}}, "humidity"),
        ({"origin": {"lat": 120.0, "lon": 0.0}}, "lat"),
        ({"actors": [{"type": "ufo"}]}, "ufo"),
        ({"geofences": [{"zone": {"type": "blob"}}]}, "zone type"),
        ({"geofences": [{"name": "a", "zone": {"type": "box"}}, {"name": "a", "zone": {"type": "box"}}]}, "duplicate"),
        ({"geofence_stop": True}, "on_diverge"),
    ],
)
def test_bad_environments_are_reported_with_the_path(env, needle):
    with pytest.raises(ConfigError) as e:
        SimulationConfig.from_dict(hover(env))
    assert "environment" in str(e.value) and needle in str(e.value)


def test_yaml_round_trip(tmp_path):
    p = tmp_path / "c.json"
    dump_config(SimulationConfig.from_dict(hover(ENV)), p)
    assert load_config(p).environment.origin.alt == 408.0


# --------------------------------------------------------------------------------------------- applying
def test_environment_is_applied_to_the_world_and_recorded():
    sim = Simulation(hover(ENV))
    w = sim.world
    assert w.origin.lat == 47.3769 and w.utc == "2026-06-21T10:00:00Z" and w.weather.rain_mm_h == 6.0 and len(w.actors) == 1
    assert w.atmosphere.humidity == pytest.approx(0.95) and w.atmosphere.pressure_sl < 101325.0  # the rain preset
    arch = sim.run()
    assert arch.world.origin.to_dict() == w.origin.to_dict() and arch["actor_pos"].shape == (arch.n_steps, 1, 3)
    assert np.allclose(np.linalg.norm(arch["actor_pos"][:, 0, :2], axis=1), 8.0, atol=1e-6)  # the actor flies its circle
    assert arch.meta["world"]["weather"]["rain_mm_h"] == 6.0


def test_wgs84_gravity_follows_the_origin_latitude():
    def g_at(lat):
        return Simulation(hover({"origin": {"lat": lat, "lon": 0.0, "alt": 0.0}, "gravity": "wgs84"})).world.gravity.latitude_g()

    assert (
        g_at(0.0) == pytest.approx(9.7803, abs=1e-3)
        and g_at(90.0) == pytest.approx(9.8322, abs=1e-3)
        and g_at(45.0) == pytest.approx(9.8062, abs=1e-3)
    )


def test_geofence_events_come_from_the_config():
    cfg = hover({"geofences": [{"name": "wall", "kind": "exclusion", "zone": {"type": "box", "lo": [-1, -1, 0], "hi": [1, 1, 20]}}]})
    with pytest.warns(UserWarning, match="geofence"):
        arch = Simulation(cfg).run()
    assert any(e["type"] == "geofence.breach" and e["fence"] == "wall" for e in arch.meta["events"])  # the hovering vehicle sits inside the box
    stop = hover(
        {"geofence_stop": True, "geofences": [{"name": "wall", "zone": {"type": "box", "lo": [-1, -1, 0], "hi": [1, 1, 20]}}]},
        health={"on_diverge": "stop"},
    )
    sim = Simulation(stop)
    sim.run()
    assert sim.stop_reason and "geofence" in sim.stop_reason


# ----------------------------------------------------------------------------- weather in the dynamics
@pytest.mark.parametrize("kernel", ["reference", "numba"])
def test_hover_rotor_speed_scales_with_inverse_square_root_of_air_density(kernel):
    def mean_rotor(weather):
        cfg = hover({"weather": weather} if weather else None)
        cfg["engine"] = {"kernel": kernel}
        arch = Simulation(cfg).run()
        return float(arch["rotor"][len(arch["rotor"]) // 2 :].mean())

    rho_isa = float(World().atmosphere.density(B, np.array([5.0]))[0])
    w = Weather.preset("hot")
    w_world = World()
    w.apply(w_world)
    rho_hot = float(w_world.atmosphere.density(B, np.array([5.0]))[0])
    ratio = mean_rotor({"preset": "hot"}) / mean_rotor(None)
    assert ratio == pytest.approx(math.sqrt(rho_isa / rho_hot), rel=0.01)  # thrust ~ rho w^2 must carry the same weight
    assert ratio > 1.04  # hot air needs visibly faster rotors


def test_fused_kernel_and_reference_agree_in_foul_weather():
    runs = {}
    for k in ("reference", "numba"):
        cfg = hover({"weather": {"preset": "storm"}}, controller={"type": "geometric"})
        cfg["engine"] = {"kernel": k}
        cfg["trajectory"] = {"type": "circle", "params": {"center": [0, 0, 4.0], "radius": 2.0, "omega": 1.0}}
        runs[k] = Simulation(cfg).run()["pos"]
    assert np.max(np.abs(runs["reference"] - runs["numba"])) < 1e-5


def test_barometer_reads_the_weather_pressure():
    cfg = hover(
        {"weather": {"temperature_c": 15.0, "pressure_sl_pa": 99_000.0, "humidity": 0.0}},
        sensors=[{"type": "baro", "params": {"noise_pa": 0.0, "drift_pa": 0.0}}],
    )
    sim = Simulation(cfg)
    sim.run()
    p = float(sim.suite["baro"].latest["pressure"][0])
    assert p == pytest.approx(99_000.0 * (1 - 0.0065 * 5.0 / 288.15) ** 5.2559, rel=1e-4)  # QNH scales the whole pressure profile


# -------------------------------------------------------------------------------------------- magnetic
def test_magnetometer_with_the_dipole_model_follows_position_and_attitude():
    w = World()
    w.origin, w.utc = LocalFrame(47.3769, 8.5417, 408.0), "2026-06-21T10:00:00Z"
    sim = type("S", (), {})()
    sim.B, sim.n, sim.world = B, 2, w
    m = Magnetometer(model="dipole", ideal=True)
    m.bind(sim, 1)
    st = type("S", (), {})()
    st.pos, st.quat, st.t = np.array([[0.0, 0.0, 0.0], [0.0, 300_000.0, 0.0]]), np.tile([1.0, 0, 0, 0], (2, 1)), 0.0
    f = m.measure(st)["field"]
    ref = TiltedDipole().field_enu(*w.origin.to_lla(st.pos).T, 2026.47)
    np.testing.assert_allclose(f, ref, rtol=2e-3)
    assert np.linalg.norm(f[0] - f[1]) > 1.0  # 300 km north changes the field by microtesla
    yawed = np.tile(
        [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)], (2, 1)
    )  # body yawed 90 degrees: east field component moves to the body axes
    st.quat = yawed
    f2 = m.measure(st)["field"]
    assert np.linalg.norm(f2[0]) == pytest.approx(np.linalg.norm(f[0]), rel=1e-9) and f2[0][1] == pytest.approx(-f[0][0], abs=1e-6)
    default = Magnetometer(ideal=True)
    default.bind(sim, 1)
    assert default._model is None  # unchanged default: the uniform field
    with pytest.raises(ValueError, match="unknown magnetic model"):
        bad = Magnetometer(model="igrf14")
        bad.bind(sim, 1)


# ---------------------------------------------------------------------------------------------- safety
def test_untrusted_configurations_cannot_name_input_files():
    cfg = SimulationConfig.from_dict(
        {
            "duration": 1.0,
            "world": {"type": "scene", "params": {"meshes": [{"file": "/etc/passwd"}], "terrain": {"file": "dem.hgt"}}},
            "wind": {"type": "gridded", "params": {"file": "w.npz"}},
        }
    )
    with pytest.raises(ResourceLimitError) as e:
        check_config(cfg, ResourceLimits.untrusted())
    msg = str(e.value)
    assert msg.count("names a file") == 3 and "world" in msg and "wind" in msg
    check_config(cfg, ResourceLimits())  # trusted runs may read files
    inline = SimulationConfig.from_dict(
        {"duration": 1.0, "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [5, 0, 1], "size": [1, 1, 1]}]}}}
    )
    check_config(inline, ResourceLimits.untrusted())  # geometry given inline is fine


# ------------------------------------------------------------------------------------------------ examples
def test_urban_scenario_file_loads_its_map_next_to_itself_and_does_not_change_the_hash():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "urban_environment.yaml"
    cfg = load_config(path)
    h = config_hash(cfg)
    cfg.duration = 2.0
    arch = Simulation(cfg).run()  # relative `old_town.geojson` is found beside the YAML, whatever the working directory
    assert arch.world.n_mesh_triangles == 48 and arch.world.origin.lat == pytest.approx(47.3769)
    assert "_source_dir" not in cfg.to_dict() and config_hash(load_config(path)) == h


def test_environment_tour_example_runs_end_to_end(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [sys.executable, str(root / "examples" / "13_environment_tour.py"), "--out", str(tmp_path), "--duration", "4"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-2500:]
    assert "GNSS" in res.stdout and "fitted" in res.stdout
    assert (tmp_path / "replay.html").stat().st_size > 500_000
