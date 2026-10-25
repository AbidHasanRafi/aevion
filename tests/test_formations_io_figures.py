"""Formations, configuration file round-trips and the remaining research-grade figures."""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aevion.analysis import figures as F
from aevion.config import SimulationConfig, config_hash, dump_config, load_config
from aevion.config.io import canonical_json
from aevion.engine import Simulation
from aevion.multiagent.formations import FORMATIONS, make_formation


# ------------------------------------------------------------------------------------------ formations
@pytest.mark.parametrize("kind", sorted(FORMATIONS))
@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_formations_have_the_requested_size_and_keep_vehicles_apart(kind, n):
    spacing = 1.5
    p = make_formation(kind, n, spacing)
    assert p.shape == (n, 3) and np.all(np.isfinite(p))
    if n > 1:
        d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(n) * 1e9
        assert d.min() > 0.5 * spacing, f"{kind} with n={n} places vehicles too close ({d.min():.2f} m)"


def test_ring_and_line_spacing_are_exact():
    r = make_formation("ring", 6, 2.0)
    chords = np.linalg.norm(r - np.roll(r, -1, axis=0), axis=1)
    assert np.allclose(chords, 2.0)
    ln = make_formation("line", 4, 3.0)
    assert np.allclose(np.diff(ln[:, 1]), 3.0) and np.isclose(ln[:, 1].mean(), 0.0)


def test_unknown_formation_lists_the_choices():
    with pytest.raises(ValueError, match="ring"):
        make_formation("zigzag", 3)


# ------------------------------------------------------------------------------------------ config files
DOC = {"schema_version": 2, "name": "io", "duration": 3.0, "seed": 5, "vehicle": {"count": 2, "formation": {"type": "ring"}}, "sensors": ["imu"]}


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
@pytest.mark.parametrize("skip_defaults", [False, True])
def test_dump_and_load_roundtrip_preserves_the_configuration(tmp_path, suffix, skip_defaults):
    cfg = load_config(DOC)
    path = dump_config(cfg, tmp_path / f"c{suffix}", skip_defaults=skip_defaults)
    again = load_config(path)
    assert config_hash(again) == config_hash(cfg) and again.to_dict() == cfg.to_dict()
    if skip_defaults:
        assert len(path.read_text(encoding="utf-8")) < len(dump_config(cfg, tmp_path / f"full{suffix}").read_text(encoding="utf-8"))


def test_canonical_json_is_order_independent_and_hash_is_stable():
    a = {"b": 1, "a": {"y": 2, "x": [1, 2]}}
    b = {"a": {"x": [1, 2], "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b) == json.dumps(a, sort_keys=True, separators=(",", ":"))
    assert config_hash(load_config(DOC)) == config_hash(load_config(dict(reversed(list(DOC.items())))))


def test_numpy_values_serialise_in_configs(tmp_path):
    cfg = SimulationConfig.from_dict({"vehicle": {"initial": {"pos": np.array([1.0, 2.0, 3.0])}}})
    path = dump_config(cfg, tmp_path / "np.json")
    assert load_config(path).vehicle.initial.pos == (1.0, 2.0, 3.0)
    assert canonical_json({"v": np.float32(0.5), "n": np.int64(3), "a": np.arange(2)}) == '{"a":[0,1],"n":3,"v":0.5}'
    with pytest.raises(TypeError):
        canonical_json({"bad": object()})


def test_unsupported_extension_and_bad_files_are_config_errors(tmp_path):
    from aevion.errors import ConfigError

    p = tmp_path / "c.toml"
    p.write_text("x = 1")
    with pytest.raises(ConfigError, match="extension"):
        load_config(p)
    p = tmp_path / "bad.yaml"
    p.write_text("a: [1, 2")
    with pytest.raises(ConfigError, match="parse"):
        load_config(p)
    p = tmp_path / "list.json"
    p.write_text("[1, 2]")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(p)


# ------------------------------------------------------------------------------------------ figures
@pytest.fixture(scope="module")
def rich_archive():
    cfg = {
        "duration": 5.0,
        "seed": 4,
        "world": {"type": "forest", "params": {"n_trees": 12}},
        "wind": {"type": "dryden", "params": {"w20": 3.0}},
        "trajectory": {"type": "circle"},
        "sensors": [{"type": "imu", "record": True}, {"type": "lidar", "record": True, "params": {"rate": 5}}],
        "estimator": {"type": "mahony_kf", "handover_s": 1.0},
        "logging": {"sensors": ["imu", "lidar"]},
    }
    return Simulation(cfg).run()


@pytest.mark.parametrize(
    "fn",
    ["fig_wind", "fig_phase_portrait", "fig_lidar_scan", "fig_imu", "fig_estimation", "fig_psd", "fig_error_distribution"],
)
def test_remaining_figures_render(rich_archive, fn):
    fig = getattr(F, fn)(rich_archive)
    assert fig.get_axes()
    plt.close(fig)


def test_figures_report_missing_data_with_a_helpful_error(rich_archive):
    bare = Simulation({"duration": 1.0}).run()
    with pytest.raises(KeyError, match="lidar"):
        F.fig_lidar_scan(bare)
    with pytest.raises(KeyError, match="imu"):
        F.fig_imu(bare)
    with pytest.raises(KeyError, match="wireless"):
        F.fig_link(bare)


def test_multi_vehicle_topdown_and_options():
    arch = Simulation(
        {"duration": 2.0, "vehicle": {"count": 3, "formation": {"type": "line"}}, "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2.0]}}}
    ).run()
    for kw in (dict(all_vehicles=True), dict(vehicle=1), dict(columns="double", venue="acm")):
        fig = F.fig_topdown(arch, **kw)
        assert fig.get_axes()
        plt.close(fig)
