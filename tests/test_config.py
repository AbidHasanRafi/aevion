import json
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import pytest

from aevion.config import ConfigError, SimulationConfig, config_hash, dump_config, json_schema, load_config
from aevion.config.schema import Block, spec


def test_minimal_config_has_sensible_defaults():
    cfg = SimulationConfig.from_dict({})
    assert cfg.controller.type == "geometric" and cfg.trajectory.type == "hover" and cfg.wind.type == "none"
    assert cfg.engine.dt == 0.005 and cfg.n_steps == 2000 and cfg.vehicle.count == 1
    assert cfg.engine.integrator == "rk4" and cfg.health.on_diverge == "raise"


def test_shorthand_component_forms():
    cfg = SimulationConfig.from_dict(
        {
            "controller": "cascaded",
            "world": "forest",
            "sensors": ["imu", {"type": "lidar", "name": "front", "channels": 4, "record": True}],
            "vehicle": "crazyflie",
        }
    )
    assert cfg.controller.type == "cascaded" and cfg.world.type == "forest"
    assert [s.key for s in cfg.sensors] == ["imu", "front"]
    assert cfg.sensors[1].params == {"channels": 4} and cfg.sensors[1].record
    assert cfg.vehicle.model == "crazyflie"
    assert cfg.to_dict()["sensors"][1] == {"type": "lidar", "name": "front", "record": True, "channels": 4}


def test_every_problem_is_reported_with_its_path():
    bad = {
        "duration": -1,
        "seed": "x",
        "engine": {"dt": 0.5, "integrator": "rk5"},
        "controller": {"type": "geometrik"},
        "sensors": ["imu", {"type": "lidar", "chanels": 4}, {"type": "nope"}],
        "vehicle": {"count": 0},
        "unknown_key": 1,
        "faults": [{"type": "rotor_failure", "t_start": 5, "t_end": 2}],
        "wind": {"type": "dryden", "w21": 3},
    }
    with pytest.raises(ConfigError) as e:
        SimulationConfig.from_dict(bad)
    paths = {i.path for i in e.value.issues}
    assert {
        "duration",
        "seed",
        "engine.dt",
        "engine.integrator",
        "controller.type",
        "sensors[1].chanels",
        "sensors[2].type",
        "vehicle.count",
        "unknown_key",
        "faults[0].t_end",
        "wind.w21",
    } <= paths, paths
    msg = str(e.value)
    assert "did you mean 'geometric'" in msg and "did you mean 'channels'" in msg and "did you mean 'w20'" in msg
    assert len(e.value.issues) >= 11


def test_cross_field_validation():
    with pytest.raises(ConfigError, match="needs an 'imu'"):
        SimulationConfig.from_dict({"estimator": "mahony_kf", "sensors": ["gps"]})
    with pytest.raises(ConfigError, match="duplicate sensor name"):
        SimulationConfig.from_dict({"sensors": ["imu", "imu"]})
    with pytest.raises(ConfigError, match="stream mode needs"):
        SimulationConfig.from_dict({"logging": {"mode": "stream"}})
    with pytest.raises(ConfigError, match="cannot exceed the physics rate"):
        SimulationConfig.from_dict({"engine": {"controller_rate_hz": 1000.0}})
    with pytest.raises(ConfigError, match="out of range"):
        SimulationConfig.from_dict({"faults": [{"type": "rotor_failure", "vehicle": 3}]})
    with pytest.raises(ConfigError, match="not a configured sensor"):
        SimulationConfig.from_dict({"logging": {"sensors": ["imu"]}})
    with pytest.raises(ConfigError, match="coincident"):
        SimulationConfig.from_dict({"vehicle": {"count": 3}, "swarm": {}})
    with pytest.raises(ConfigError, match="unknown vehicle preset"):
        SimulationConfig.from_dict({"vehicle": "crazyfly"})


def test_roundtrip_dict_yaml_json_and_hash_stability(tmp_path):
    cfg = SimulationConfig.from_dict(
        {
            "name": "rt",
            "duration": 3.5,
            "world": {"type": "forest", "n_trees": 7},
            "sensors": ["imu", {"type": "gps", "latency": 0.05}],
            "faults": [{"type": "gps_outage", "t_start": 1.0, "t_end": 2.0}],
            "swarm": {"avoid_radius": 1.1},
            "vehicle": {"count": 2, "formation": "line"},
        }
    )
    for suf in (".yaml", ".json"):
        p = dump_config(cfg, tmp_path / f"c{suf}")
        again = load_config(p)
        assert again.to_dict() == cfg.to_dict(), suf
        assert config_hash(again) == config_hash(cfg)
    assert config_hash(cfg) == config_hash(cfg.to_dict())
    assert config_hash(cfg) != config_hash(cfg.replace(seed=1))
    slim = cfg.to_dict(skip_defaults=True)
    assert "engine" not in slim or slim["engine"] == {}


def test_file_errors_name_the_file(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("duration: [1, 2\n  oops", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot parse"):
        load_config(p)
    p = tmp_path / "x.txt"
    p.write_text("a: 1", encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported extension"):
        load_config(p)
    p = tmp_path / "list.json"
    p.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(p)
    p = tmp_path / "typo.yaml"
    p.write_text("duraton: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert str(p) in str(e.value) and "did you mean 'duration'" in str(e.value)


def test_v1_scenario_migrates_losslessly():
    from aevion.scenario import Scenario
    from aevion.wireless import AirToGroundChannel, BaseStation, CellularNetwork

    net = CellularNetwork([BaseStation((0, 0, 20))], AirToGroundChannel("urban", los_mode="mean"))
    sc = Scenario(
        name="old",
        vehicle="hex_3kg",
        world={"kind": "city", "blocks": [2, 2]},
        wind={"type": "ConstantWind", "velocity": [1, 0, 0]},
        controller="cascaded",
        controller_kwargs={"kp_pos": 1.5},
        trajectory=dict(kind="circle", radius=2.0),
        duration=7.0,
        dt=0.004,
        n=3,
        formation=dict(kind="ring", spacing=1.5),
        seed=9,
        sensors=["imu", "gps"],
        record_sensors=["imu"],
        use_estimator=True,
        record_every=5,
        initial={"pos": [1, 2, 3]},
        perturbation=dict(kind="rotor_failure", t=2.0, vehicle=1, rotor=2, severity=0.5),
        wireless=net.to_dict(),
        swarm=dict(avoid_radius=1.3, comm=dict(loss=0.1)),
        notes="hello",
    )
    cfg = sc.to_config()
    assert cfg.name == "old" and cfg.description == "hello" and cfg.vehicle.count == 3 and cfg.vehicle.model == "hex_3kg"
    assert cfg.engine.dt == 0.004 and cfg.controller.params == {"kp_pos": 1.5} and cfg.world.type == "city"
    assert cfg.vehicle.formation.type == "ring" and cfg.vehicle.formation.params == {"spacing": 1.5}
    assert cfg.sensors[0].record and not cfg.sensors[1].record and cfg.estimator is not None
    assert cfg.faults[0].type == "rotor_failure" and cfg.faults[0].vehicle == 1 and cfg.faults[0].params == {"rotor": 2, "severity": 0.5}
    assert cfg.network.params["bs"][0]["pos"] == [0.0, 0.0, 20.0] and cfg.swarm.comm.loss == 0.1 and cfg.logging.record_every == 5
    assert cfg.vehicle.initial.pos == (1.0, 2.0, 3.0)


def test_future_and_unknown_schema_versions_are_rejected():
    with pytest.raises(ConfigError, match="newer than this Aevion"):
        load_config({"schema_version": 99})
    with pytest.raises(ConfigError, match="no migration"):
        load_config({"schema_version": 0})


def test_json_schema_is_valid_and_describes_constraints():
    import jsonschema

    sch = json_schema(SimulationConfig)
    jsonschema.Draft202012Validator.check_schema(sch)
    eng = sch["$defs"]["EngineConfig"]["properties"]["dt"]
    assert eng["exclusiveMinimum"] == 0.0 and eng["maximum"] == 0.05 and "physics step" in eng["description"]
    assert sch["$defs"]["EngineConfig"]["properties"]["integrator"]["enum"] == ["euler", "semi_implicit", "midpoint", "rk4"]
    ok = SimulationConfig.from_dict({"duration": 3}).to_dict()
    ok["engine"].pop("dtype", None)
    v = jsonschema.Draft202012Validator(sch)
    errs = [e for e in v.iter_errors({"duration": 3.0, "engine": {"dt": 0.01}}) if "$defs" not in str(e.schema_path)]
    assert not errs, errs


def test_generic_schema_engine_on_custom_blocks():
    @dataclass
    class Leaf(Block):
        x: float = spec(1.0, gt=0, unit="m")
        mode: Literal["a", "b"] = "a"

    @dataclass
    class Root(Block):
        leaves: List[Leaf] = field(default_factory=list)
        table: Dict[str, int] = field(default_factory=dict)
        pair: Optional[Tuple[float, float]] = None
        name: str = spec(min_len=2, default="ab")

    r = Root.from_dict({"leaves": [{"x": 2}, {"mode": "b"}], "table": {"k": 3}, "pair": [1, 2]})
    assert r.leaves[0].x == 2.0 and r.pair == (1.0, 2.0)
    issues = Root.validate_dict({"leaves": [{"x": 0}, {"mode": "c"}, 5], "table": {"k": "v"}, "pair": [1], "name": "a"})
    paths = {i.path for i in issues}
    assert {"leaves[0].x", "leaves[1].mode", "leaves[2]", "table.k", "pair", "name"} <= paths
    assert not Root.validate_dict({})
    with pytest.raises(ConfigError):
        Root.from_dict({"leaves": "nope"})
    assert Leaf.validate_dict({"x": True})  # bool is not a number
    assert Leaf.from_dict({"x": 3}).x == 3.0 and isinstance(Leaf.from_dict({"x": 3}).x, float)


def test_generated_docs_are_in_sync_with_the_code():
    """docs/simulation.schema.json and the configuration reference in DOCUMENTATION.md are generated; they must not drift from the dataclasses."""
    import dataclasses
    from pathlib import Path

    from aevion.config import SimulationConfig, blocks

    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "docs" / "simulation.schema.json").read_text(encoding="utf-8"))
    assert schema == json.loads(json.dumps(json_schema(SimulationConfig, "Aevion simulation configuration"))), "run: python tools/gen_config_docs.py"
    guide = (root / "DOCUMENTATION.md").read_text(encoding="utf-8")
    ref = guide.split("<!-- config-reference:start -->")[1].split("<!-- config-reference:end -->")[0]
    for cls in (blocks.SimulationConfig, blocks.EngineConfig, blocks.LoggingConfig, blocks.HealthConfig, blocks.FaultConfig, blocks.SwarmConfig):
        assert f"#### `{cls.__name__}`" in ref
        for f in dataclasses.fields(cls):
            assert f"`{f.name}`" in ref, (cls.__name__, f.name)
