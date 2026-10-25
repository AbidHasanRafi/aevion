import json
import zipfile

import numpy as np
import pytest

from aevion.cli import main
from aevion.config import SimulationConfig
from aevion.datasets import RunArchive
from aevion.datasets.store import ArchiveReader, pack, unpack, validate_manifest
from aevion.errors import ArchiveError, ConfigError
from aevion.safety import ResourceLimitError, ResourceLimits, check_config, estimate_cost, load_config_checked, measure_tree, run_limited

SMALL = {"duration": 0.5, "sensors": ["imu"]}


def cfg_of(**kw):
    return SimulationConfig.from_dict({**SMALL, **kw})


# ------------------------------------------------------------------------------------- resource limits
def test_a_modest_config_passes_the_untrusted_preset():
    check_config(cfg_of(), ResourceLimits.untrusted())


def test_every_violation_is_reported_at_once():
    cfg = cfg_of(duration=100000.0, vehicle={"count": 500, "formation": {"type": "line"}})
    with pytest.raises(ResourceLimitError) as ei:
        check_config(cfg, ResourceLimits.untrusted())
    text = str(ei.value)
    assert len(ei.value.violations) >= 3 and "duration" in text and "vehicles" in text and "vehicle-steps" in text


def test_sensor_ray_throughput_is_estimated_and_limited():
    cfg = cfg_of(
        sensors=[{"type": "lidar", "params": {"channels": 128, "samples": 2048, "rate": 20}}], vehicle={"count": 4, "formation": {"type": "line"}}
    )
    est = estimate_cost(cfg)
    assert est["rays_per_second"] == 128 * 2048 * 20 * 4
    with pytest.raises(ResourceLimitError, match="ray casts"):
        check_config(cfg, ResourceLimits.untrusted())
    check_config(cfg, ResourceLimits())  # fine for a trusted user


def test_camera_resolution_is_limited():
    cfg = cfg_of(sensors=[{"type": "camera", "params": {"width": 1920, "height": 1080}}])
    with pytest.raises(ResourceLimitError, match="pixels"):
        check_config(cfg, ResourceLimits.untrusted())


def test_world_object_count_is_estimated_without_building():
    cfg = cfg_of(world={"type": "forest", "params": {"n_trees": 10**7}})
    with pytest.raises(ResourceLimitError, match="world objects"):
        check_config(cfg, ResourceLimits.untrusted())


def test_output_path_is_confined(tmp_path):
    lim = ResourceLimits.untrusted(output_root=tmp_path)
    check_config(cfg_of(logging={"mode": "stream", "path": str(tmp_path / "ok")}), lim)
    with pytest.raises(ResourceLimitError, match="outside"):
        check_config(cfg_of(logging={"mode": "stream", "path": str(tmp_path.parent / "elsewhere")}), lim)
    with pytest.raises(ResourceLimitError, match="outside"):
        check_config(cfg_of(logging={"mode": "stream", "path": str(tmp_path / ".." / "x")}), lim)


def test_plugin_components_are_rejected_when_builtin_only():
    from aevion.registry import registries

    registries.sensors.register("sneaky_sensor", lambda **kw: None, source="evil_plugin", replace=True)
    try:
        cfg = cfg_of(sensors=["sneaky_sensor"])
        with pytest.raises(ResourceLimitError, match="evil_plugin"):
            check_config(cfg, ResourceLimits.untrusted())
        check_config(cfg, ResourceLimits())
    finally:
        registries.sensors._entries.pop("sneaky_sensor", None)


# ------------------------------------------------------------------------------------- parser bombs
def test_oversized_config_files_are_refused(tmp_path):
    p = tmp_path / "big.json"
    p.write_text(json.dumps({"description": "x" * 400_000}))
    with pytest.raises(ResourceLimitError, match="bytes"):
        load_config_checked(p, ResourceLimits.untrusted())


def test_yaml_alias_bombs_are_refused(tmp_path):
    bomb = "a: &a [x, x, x, x, x, x, x, x, x]\n" + "".join(
        f"{chr(98 + i)}: &{chr(98 + i)} [*{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}]\n"
        for i in range(8)
    )
    p = tmp_path / "bomb.yaml"
    p.write_text(bomb)
    with pytest.raises(ResourceLimitError, match="alias"):
        load_config_checked(p, ResourceLimits.untrusted())


def test_expanded_node_count_is_bounded_even_with_aliases_allowed():
    shared = [0] * 100
    data: dict = {"l0": shared}
    for i in range(1, 8):
        data[f"l{i}"] = [data[f"l{i - 1}"]] * 100  # 100**8 expanded nodes from a few objects
    with pytest.raises(ResourceLimitError, match="nodes"):
        measure_tree(data, 10_000, 64)


def test_deep_nesting_is_refused(tmp_path):
    p = tmp_path / "deep.json"
    p.write_text("[" * 5000 + "]" * 5000)
    with pytest.raises(ResourceLimitError):
        load_config_checked(p, ResourceLimits.untrusted())
    with pytest.raises(ResourceLimitError, match="deeper"):
        measure_tree(json.loads("[" * 40 + "]" * 40), 1000, 24)


def test_malformed_files_raise_config_errors_not_crashes(tmp_path):
    for name, text in [("a.yaml", "a: [unclosed"), ("b.json", "{nope"), ("c.yaml", "- just\n- a list\n")]:
        p = tmp_path / name
        p.write_text(text)
        with pytest.raises(ConfigError):
            load_config_checked(p, ResourceLimits.untrusted())


# ------------------------------------------------------------------------------------- running
def test_run_limited_runs_a_safe_config(tmp_path):
    a = run_limited(dict(SMALL, duration=0.3), ResourceLimits.untrusted(output_root=tmp_path))
    assert a.n_steps > 10


def test_run_limited_refuses_before_running():
    with pytest.raises(ResourceLimitError):
        run_limited(dict(SMALL, duration=1e5), ResourceLimits.untrusted())


def test_wall_clock_limit_stops_a_long_run():
    lim = ResourceLimits(max_wall_s=0.05)
    with pytest.raises(ResourceLimitError, match="wall-clock"):
        run_limited({"duration": 600.0, "engine": {"dt": 0.002}}, lim)


def test_plugins_can_be_disabled_by_environment(monkeypatch):
    from aevion.registry import plugins

    monkeypatch.setenv("AEVION_DISABLE_PLUGINS", "1")
    assert plugins.plugins_disabled()
    called = []
    monkeypatch.setattr(plugins, "_entry_points", lambda: called.append(1) or [])
    monkeypatch.setattr(plugins, "_loaded", False)
    plugins.discover(object())
    assert not called  # never even enumerated


# ------------------------------------------------------------------------------------- CLI
def test_cli_validate_and_run_untrusted(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text("schema_version: 2\nduration: 0.4\nsensors: [imu]\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 2\nduration: 100000\n")
    with pytest.raises(SystemExit) as e:
        main(["validate", "--untrusted", str(good)])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        main(["validate", "--untrusted", str(bad)])
    assert e.value.code == 1
    main(["run", "--untrusted", str(good), "--out", str(tmp_path / "out"), "--no-report"])
    with pytest.raises(SystemExit) as e:
        main(["run", "--untrusted", str(bad), "--out", str(tmp_path / "out2"), "--no-report"])
    assert e.value.code == 1
    assert "exceeds" in capsys.readouterr().err


# ------------------------------------------------------------------------------------- archives
@pytest.fixture
def store(tmp_path):
    a = run_limited(dict(SMALL, duration=0.3), ResourceLimits())
    path = a.save(tmp_path / "run.aevd")
    return path


def edit_manifest(path, fn):
    mp = path / "manifest.json"
    m = json.loads(mp.read_text())
    fn(m)
    mp.write_text(json.dumps(m))


@pytest.mark.parametrize(
    "mutate,why",
    [
        (lambda m: m["channels"]["t"]["chunks"][0].update(file="../../outside.npy"), "unsafe"),
        (lambda m: m["channels"]["t"]["chunks"][0].update(file="/etc/passwd"), "unsafe"),
        (lambda m: m["channels"]["t"]["chunks"][0].update(file="C:/Windows/x.npy"), "unsafe"),
        (lambda m: m["channels"]["t"].update(dtype="O"), "dtype"),
        (lambda m: m["channels"]["t"].update(dtype="U8"), "dtype"),
        (lambda m: m["channels"]["t"].update(length=-5), "length"),
        (lambda m: m["channels"]["t"].update(length="many"), "length"),
        (lambda m: m["channels"]["t"].update(tail="abc"), "tail"),
        (lambda m: m["channels"]["t"].update(tail=[2**40]), "tail"),
        (lambda m: m["channels"]["t"].update(chunks="nope"), "chunks"),
        (lambda m: m["channels"]["t"]["chunks"][0].update(n=10**12), "chunk"),
        (lambda m: m.update(channels=[1, 2]), "channels"),
    ],
)
def test_hostile_manifests_are_rejected_with_archive_errors(store, mutate, why):
    edit_manifest(store, mutate)
    with pytest.raises(ArchiveError, match=why):
        ArchiveReader(store)


def test_manifest_that_is_not_an_object_is_rejected(store):
    (store / "manifest.json").write_text("[1, 2, 3]")
    with pytest.raises(ArchiveError):
        ArchiveReader(store)
    with pytest.raises(ArchiveError):
        validate_manifest("nope")


def test_chunk_that_disagrees_with_the_manifest_is_rejected(store):
    m = json.loads((store / "manifest.json").read_text())
    f = store / m["channels"]["t"]["chunks"][0]["file"]
    np.save(f, np.zeros((3,), np.float32))
    r = ArchiveReader(store)
    with pytest.raises(ArchiveError, match="manifest says"):
        r.channels["t"].to_numpy()


def test_pickled_chunks_are_never_loaded(store):
    m = json.loads((store / "manifest.json").read_text())
    f = store / m["channels"]["t"]["chunks"][0]["file"]
    with open(f, "wb") as fh:
        np.save(fh, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    r = ArchiveReader(store)
    with pytest.raises(ArchiveError, match="cannot read chunk"):
        r.channels["t"].to_numpy()


def test_corrupt_chunk_files_raise_archive_errors(store):
    m = json.loads((store / "manifest.json").read_text())
    (store / m["channels"]["t"]["chunks"][0]["file"]).write_bytes(b"not an npy file")
    with pytest.raises(ArchiveError):
        ArchiveReader(store).channels["t"].to_numpy()


def test_malicious_packed_archives_cannot_escape(tmp_path, store):
    evil = tmp_path / "evil.aevz"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("manifest.json", (store / "manifest.json").read_text())
        z.writestr("../../escape.txt", "pwned")
    dst = tmp_path / "dst"
    with pytest.raises(ArchiveError):
        unpack(evil, dst)
    assert not (tmp_path / "escape.txt").exists() and not dst.exists()


def test_packed_archive_with_bad_zip_is_an_archive_error(tmp_path):
    p = tmp_path / "x.aevz"
    p.write_bytes(b"this is not a zip")
    with pytest.raises(ArchiveError):
        ArchiveReader(p)


def test_zip_bombs_are_refused(tmp_path):
    bomb = tmp_path / "bomb.aevz"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", "{}")
        z.writestr("channels/x/000000.npy", b"\0" * 50_000_000)
    with pytest.raises(ArchiveError):
        unpack(bomb, tmp_path / "out")


def test_roundtrip_still_works_after_hardening(tmp_path, store):
    packed = pack(store, tmp_path / "p.aevz")
    out = unpack(packed, tmp_path / "again.aevd")
    a, b = RunArchive.load(store), RunArchive.load(out)
    assert np.array_equal(a["pos"], b["pos"])
