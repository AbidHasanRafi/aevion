"""Regressions found by the property tests: malformed inputs must give ConfigError, never a stray exception."""

import pytest

from aevion.config import load_config
from aevion.errors import ConfigError


@pytest.mark.parametrize(
    "doc",
    [
        {"engine": None},
        {"duration": None},
        {"schema_version": []},
        {"schema_version": "two"},
        {"schema_version": True},
        {"vehicle": None},
        {"vehicle": {"count": "many"}},
        {"engine": {"dt": None}, "duration": 1.0},
        {"controller": 5},
        {"sensors": "imu"},
        {"sensors": [None]},
        {"logging": {"record_every": []}},
        {"controller": {"type": "geometric", "params": 3}},
        {"duration": [1, 2]},
    ],
)
def test_malformed_documents_raise_config_error(doc):
    with pytest.raises(ConfigError):
        load_config(doc)


def test_type_errors_are_reported_before_cross_field_checks():
    with pytest.raises(ConfigError) as ei:
        load_config({"duration": None, "engine": {"dt": "fast"}})
    paths = {i.path for i in ei.value.issues}
    assert {"duration", "engine.dt"} <= paths


def test_unversioned_v2_documents_are_not_mistaken_for_v0_1_scenarios():
    """Regression: {'vehicle': {'count': ...}} was once migrated as a legacy scenario and its contents silently dropped."""
    cfg = load_config({"vehicle": {"count": 3, "formation": {"type": "line"}}})
    assert cfg.vehicle.count == 3
    assert load_config({"sensors": [{"type": "gps"}]}).sensors[0].type == "gps"
    assert load_config({"trajectory": {"type": "hover", "params": {"pos": [0, 0, 3]}}}).trajectory.params["pos"] == [0, 0, 3]


def test_params_must_be_a_mapping_but_none_means_empty():
    assert load_config({"trajectory": {"type": "hover", "params": None}}).trajectory.params == {}
    with pytest.raises(ConfigError, match="params"):
        load_config({"controller": {"type": "geometric", "params": 3}})


def test_yaml_numbers_with_unsigned_exponents_are_floats_not_strings(tmp_path):
    """PyYAML (YAML 1.1) reads ``3.5e9`` as a *string*; engineering configs are full of such numbers."""
    from aevion.config.io import yaml_loads

    d = yaml_loads("a: 3.5e9\nb: 1e-3\nc: 2E6\nd: 1.0e+5\ne: 10\nf: abc\ng: 1e\nh: [1e3, 2.5]\ni: j5\n")
    assert d == {"a": 3.5e9, "b": 1e-3, "c": 2e6, "d": 1e5, "e": 10, "f": "abc", "g": "1e", "h": [1e3, 2.5], "i": "j5"}
    p = tmp_path / "c.yaml"
    p.write_text("schema_version: 2\nduration: 2\nnetwork:\n  type: nr_cellular\n  params: {freq_hz: 3.5e9, sites: [{pos: [0, 0, 25]}]}\n")
    assert load_config(p).network.params["freq_hz"] == 3.5e9


def test_yaml_loader_still_refuses_python_tags():
    import yaml

    from aevion.config.io import yaml_loads

    with pytest.raises(yaml.YAMLError):
        yaml_loads("x: !!python/object/apply:os.getcwd []")
