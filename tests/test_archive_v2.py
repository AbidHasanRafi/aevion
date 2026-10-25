import json
import shutil

import numpy as np
import pytest

from aevion.config import SimulationConfig
from aevion.datasets import ArchiveReader, ArchiveWriter, RunArchive, RunRegistry, diff_archives, diff_dicts, is_store, pack, unpack
from aevion.engine import Simulation
from aevion.errors import ArchiveError


def sim_archive(duration=2.0, **kw):
    cfg = {
        "duration": duration,
        "trajectory": {"type": "circle", "center": [0, 0, 2], "radius": 1.0, "omega": 1.0},
        "sensors": ["imu"],
        "logging": {"sensors": ["imu"]},
    }
    cfg.update(kw)
    return Simulation(SimulationConfig.from_dict(cfg)).run()


# ----------------------------------------------------------------------------- writer / reader
def test_writer_reader_roundtrip_with_chunk_boundaries(tmp_path):
    p = tmp_path / "a.aevd"
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1037, 2, 3))
    with ArchiveWriter(p, meta={"k": 1}, chunk_steps=100) as w:
        for row in data[:500]:
            w.append("x", row)
        w.append_block("x", data[500:])
        w.append_block("t", np.arange(1037.0))
        w.append_block("flags", (np.arange(1037) % 3 == 0).astype(np.int8))
    r = ArchiveReader(p)
    ch = r.channels["x"]
    assert ch.shape == (1037, 2, 3) and ch.n_chunks == 11 and r.complete and r.meta == {"k": 1}
    np.testing.assert_array_equal(np.asarray(ch), data)
    np.testing.assert_array_equal(ch[95:105], data[95:105])  # across a chunk boundary
    np.testing.assert_array_equal(ch[::37], data[::37])
    np.testing.assert_array_equal(ch[-3:], data[-3:])
    np.testing.assert_array_equal(ch[500, 1], data[500, 1])
    np.testing.assert_array_equal(ch[10:20, 0, 2], data[10:20, 0, 2])
    np.testing.assert_array_equal(ch[[3, 250, 999, 4]], data[[3, 250, 999, 4]])
    np.testing.assert_array_equal(ch[::-1][:5], data[::-1][:5])
    np.testing.assert_array_equal(ch[np.arange(1037) % 100 == 0], data[np.arange(1037) % 100 == 0])
    assert ch[5:5].shape == (0, 2, 3) and r.channels["flags"].dtype == np.int8
    with pytest.raises(IndexError):
        ch[5000]
    assert r.verify() == []
    assert sum(1 for _ in ch.iter_chunks()) == 11


def test_writer_rejects_inconsistent_rows_and_overwrite(tmp_path):
    p = tmp_path / "a.aevd"
    w = ArchiveWriter(p)
    w.append("x", np.zeros(3))
    with pytest.raises(ArchiveError, match="row shape"):
        w.append("x", np.zeros(4))
    w.close()
    with pytest.raises(ArchiveError, match="closed"):
        w.append("x", np.zeros(3))
    with pytest.raises(ArchiveError, match="already exists"):
        ArchiveWriter(p)
    ArchiveWriter(p, overwrite=True).close()


def test_crashed_run_is_readable_up_to_last_flush(tmp_path):
    p = tmp_path / "crash.aevd"
    w = ArchiveWriter(p, chunk_steps=50)
    for i in range(130):
        w.append("x", np.array([float(i)]))
    # simulate a crash: never call close()
    r = ArchiveReader(p)
    assert not r.complete and r.status == "incomplete"
    assert len(r.channels["x"]) == 100  # two full chunks flushed; the 30 buffered rows are lost, nothing is corrupt
    np.testing.assert_array_equal(np.asarray(r.channels["x"]).ravel(), np.arange(100.0))
    assert any("did not finish cleanly" in m for m in r.verify())
    w.abort()
    assert len(ArchiveReader(p).channels["x"]) == 130


def test_corruption_is_detected(tmp_path):
    p = tmp_path / "c.aevd"
    with ArchiveWriter(p, chunk_steps=10) as w:
        w.append_block("x", np.arange(30.0))
    chunk = next((p / "channels").rglob("000001.npy"))
    raw = bytearray(chunk.read_bytes())
    raw[-3] ^= 0xFF
    chunk.write_bytes(bytes(raw))
    probs = ArchiveReader(p).verify()
    assert probs and "hash mismatch" in probs[0]
    assert ArchiveReader(p).verify(deep=False) == []  # shallow check does not read data
    chunk.unlink()
    r = ArchiveReader(p)
    assert any("missing" in m for m in r.verify())
    with pytest.raises(ArchiveError, match="missing chunk"):
        np.asarray(r.channels["x"])


def test_bad_inputs_raise_archive_errors(tmp_path):
    with pytest.raises(ArchiveError, match="not an Aevion run archive"):
        ArchiveReader(tmp_path)
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ArchiveError, match="corrupt manifest"):
        ArchiveReader(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "other/1", "channels": {}}), encoding="utf-8")
    with pytest.raises(ArchiveError, match="unrecognised schema"):
        ArchiveReader(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "aevion.run/9", "channels": {}}), encoding="utf-8")
    with pytest.raises(ArchiveError, match="newer than this Aevion"):
        ArchiveReader(tmp_path)
    with pytest.raises(ArchiveError, match="does not exist"):
        RunArchive.load(tmp_path / "nope.aev")
    junk = tmp_path / "junk.aev"
    junk.write_bytes(b"not an npz")
    with pytest.raises(ArchiveError, match="cannot read"):
        RunArchive.load(junk)


def test_pack_unpack_and_read_in_place(tmp_path):
    a = sim_archive(1.0)
    d = a.save(tmp_path / "r.aevd")
    z = pack(d)
    assert z.suffix == ".aevz" and is_store(z) and is_store(d) and not is_store(tmp_path / "nothing")
    b = RunArchive.load(z)
    np.testing.assert_array_equal(b["pos"], a["pos"])
    assert b.verify_integrity() == []
    out = unpack(z, tmp_path / "back.aevd")
    assert RunArchive.load(out).verify_integrity() == []
    a.save(tmp_path / "direct.aevz")
    assert RunArchive.load(tmp_path / "direct.aevz").n_steps == a.n_steps


# ------------------------------------------------------------------------------------ RunArchive
def test_v1_and_v2_roundtrip_are_identical(tmp_path):
    a = sim_archive(1.5)
    f1 = a.save(tmp_path / "x.aev")
    f2 = a.save(tmp_path / "x.aevd")
    r1, r2 = RunArchive.load(f1), RunArchive.load(f2)
    assert r2.is_lazy and not r1.is_lazy
    for k in a.channels:
        np.testing.assert_array_equal(r1[k], a[k])
        np.testing.assert_array_equal(r2[k], a[k])
    assert r2.complete and r2.meta["config"] == a.meta["config"] and r2.provenance["config_hash"] == a.provenance["config_hash"]
    assert "v2 store" in r2.summary() and "v1 file" in r1.summary()
    assert r2.world is not None and r2.spec.name == "generic_1kg"


def test_lazy_access_reads_only_requested_rows(tmp_path):
    a = sim_archive(4.0)
    d = a.save(tmp_path / "x.aevd", chunk_steps=20)
    r = RunArchive.load(d)
    lazy = r.lazy("pos")
    part = lazy[30:45]
    np.testing.assert_array_equal(part, a["pos"][30:45])
    assert "pos" not in r._cache  # nothing materialised yet
    full = r["pos"]
    np.testing.assert_array_equal(full, a["pos"])
    assert "pos" in r._cache
    r.release()
    assert not r._cache
    sl = r.slice_time(1.0, 2.0)
    assert sl.t[0] >= 1.0 - 1e-9 and sl.t[-1] <= 2.0 + 1e-9 and not sl.is_lazy
    assert (sl["sensors/imu/t"] >= 1.0 - 1e-9).all() and (sl["sensors/imu/t"] <= 2.0 + 1e-9).all()


def test_existing_consumers_work_on_lazy_archives(tmp_path):
    from aevion import metrics
    from aevion.analysis.report import insights

    a = sim_archive(3.0)
    r = RunArchive.load(a.save(tmp_path / "x.aevd"))
    m1, m2 = metrics.flight_metrics(a), metrics.flight_metrics(r)
    for k in m1:
        np.testing.assert_allclose(m1[k], m2[k])
    assert {i["title"] for i in insights(r)} >= {"Tracking accuracy", "Energy"}
    r.to_csv(tmp_path / "x.csv")
    assert (tmp_path / "x.csv").read_text().splitlines()[0].startswith("t,")
    from aevion.viewer import build_payload

    assert build_payload(r, max_frames=50)["meta"]["T"] <= 50


def test_streaming_mode_matches_memory_mode_and_bounds_memory(tmp_path):
    cfg = {
        "duration": 3.0,
        "trajectory": {"type": "figure8", "center": [0, 0, 3], "size": 2.0, "period": 6.0},
        "sensors": ["imu", "gps"],
        "seed": 4,
        "wind": {"type": "dryden", "w20": 3.0},
    }
    mem = Simulation(SimulationConfig.from_dict({**cfg, "logging": {"sensors": ["imu"]}})).run()
    stream = Simulation(
        SimulationConfig.from_dict({**cfg, "logging": {"sensors": ["imu"], "mode": "stream", "path": str(tmp_path / "s.aevd"), "chunk_steps": 16}})
    ).run()
    assert stream.is_lazy and stream.complete and stream.lazy("pos").n_chunks > 3
    for k in mem.channels:
        np.testing.assert_array_equal(stream[k], mem[k], err_msg=k)
    assert stream.verify_integrity() == []


def test_failed_stream_run_leaves_a_readable_incomplete_archive(tmp_path):
    from aevion.errors import SimulationDiverged

    cfg = {
        "duration": 4.0,
        "trajectory": {"type": "circle", "radius": 2.0, "omega": 1.5},
        "health": {"on_diverge": "raise", "max_speed": 0.8},
        "logging": {"mode": "stream", "path": str(tmp_path / "f.aevd"), "chunk_steps": 16, "record_every": 1},
    }
    with pytest.raises(SimulationDiverged):
        Simulation(SimulationConfig.from_dict(cfg)).run()
    r = RunArchive.load(tmp_path / "f.aevd")
    assert not r.complete and r.meta["status"] == "diverged" and "speed" in r.meta["error"]
    assert r.n_steps > 0 and np.isfinite(r["pos"]).all()


# ------------------------------------------------------------------------------------------ diff
def test_diff_identical_and_changed_runs():
    a = sim_archive(2.0)
    b = sim_archive(2.0)
    d = diff_archives(a, b)
    assert d.identical and d.config_identical and "IDENTICAL" in d.summary()
    c = sim_archive(2.0, seed=5, wind={"type": "constant", "velocity": [2.0, 0, 0]}, controller={"type": "geometric", "wn_pos": [3, 3, 4]})
    d = diff_archives(a, c)
    assert not d.identical and not d.config_identical
    paths = {p for p, _, _ in d.meta_changes}
    assert "config.seed" in paths and "config.wind.type" in paths and any(p.startswith("config.controller") for p in paths)
    assert d.channels["pos"].max_abs > 1e-3 and d.channels["pos"].first_time is not None
    assert "rmse_pos" in d.metric_deltas
    text = d.summary()
    assert "config.wind.type" in text and "pos" in text


def test_diff_shape_mismatch_and_tolerance():
    a = sim_archive(2.0)
    b = sim_archive(1.0)
    d = diff_archives(a, b, with_metrics=False)
    assert d.channels["pos"].shape_a != d.channels["pos"].shape_b and not d.identical
    x = sim_archive(1.0)
    y = RunArchive({k: np.asarray(x[k]).copy() for k in x.channels}, dict(x.meta))
    y.channels["pos"][5] += 1e-9
    assert not diff_archives(x, y, tol=0.0, with_metrics=False).identical
    assert diff_archives(x, y, tol=1e-6, with_metrics=False).identical


def test_diff_dicts_ignores_volatile_keys_and_reports_lists():
    a = {"a": 1, "created": "x", "n": [1, 2, 3], "d": {"q": 1.0}}
    b = {"a": 1, "created": "y", "n": [1, 5, 3], "d": {"q": 1.0 + 1e-15}, "new": 1}
    assert diff_dicts(a, b) == [("n[1]", 2, 5), ("new", "<absent>", 1)]
    assert diff_dicts({"x": [1]}, {"x": [1, 2]}) == [("x.<length>", 1, 2)]


# ------------------------------------------------------------------------------------- registry
def test_run_registry_register_query_tag_remove(tmp_path):
    reg = RunRegistry(tmp_path / "runs.db")
    a = sim_archive(1.0, name="alpha")
    b = sim_archive(1.0, name="beta", seed=2)
    pa, pb = a.save(tmp_path / "a.aevd"), b.save(tmp_path / "b.aev")
    ida = reg.register(pa, tags=["baseline"])
    idb = reg.register(RunArchive.load(pb), tags=["variant", "wind"], notes="seed 2")
    assert reg.count() == 2 and ida != idb
    assert [r.name for r in reg.list(name="alp")] == ["alpha"] and [r.name for r in reg.list(tag="wind")] == ["beta"]
    rec = reg.get(idb[:20])
    assert rec.notes == "seed 2" and rec.seed == 2 and rec.exists and "rmse_pos" in rec.metrics and rec.n_vehicles == 1
    assert reg.find_by_config(rec.config_hash)[0].run_id == idb
    assert reg.tag(ida, "good", "final") == ["baseline", "final", "good"] and reg.tag(ida, "final", remove=True) == ["baseline", "good"]
    reg.update_metrics(ida, {"custom": 3.5})
    assert reg.get(ida).metrics["custom"] == 3.5
    assert "alpha" in reg.table() and "beta" in reg.table()
    assert rec.open().n_steps == b.n_steps
    shutil.rmtree(pa)
    assert not reg.get(ida).exists and "[missing]" in reg.table()
    assert reg.prune_missing() == 1 and reg.count() == 1
    reg.remove(idb, delete_files=True)
    assert reg.count() == 0 and not pb.exists()
    with pytest.raises(ArchiveError, match="no registered run"):
        reg.get("zzz")
    with pytest.raises(ArchiveError, match="not been saved"):
        reg.register(sim_archive(0.5))


def test_registry_ambiguous_prefix_and_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AEVION_HOME", str(tmp_path / "home"))
    reg = RunRegistry()
    assert reg.path == tmp_path / "home" / "runs.db"
    a = sim_archive(0.5)
    p = a.save(tmp_path / "a.aevd")
    i1, i2 = reg.register(p), reg.register(p)
    assert i1 != i2
    with pytest.raises(ArchiveError, match="ambiguous"):
        reg.get(i1[:8])
    assert reg.get(i2).run_id == i2


# -------------------------------------------------------------------------------- exports/etc
PARQUET_CHECK = """
import sys
import numpy as np
import pyarrow.parquet as pq
from aevion.datasets import RunArchive, to_parquet
from aevion.datasets.export import from_parquet

a = RunArchive.load(sys.argv[1])
p = to_parquet(a, sys.argv[2] + "/x.parquet")
df = from_parquet(p)
assert len(df) == a.n_steps and {"t", "pos_0", "pos_2", "rotor_3", "soc"} <= set(df.columns)
np.testing.assert_allclose(df["pos_2"].to_numpy(), a["pos"][:, 0, 2])
assert b"aevion" in pq.read_schema(p).metadata
b = RunArchive.load(sys.argv[3])
n = to_parquet(b, sys.argv[2] + "/all.parquet", vehicle=None)
assert set(from_parquet(n)["vehicle"]) == {0, 1}
print("parquet ok")
"""


def test_parquet_export_roundtrip(tmp_path):
    """Runs in a subprocess: pyarrow occasionally dies with a native access violation on some Windows setups, which must not take
    down a test worker. A genuine assertion failure (exit code 1) still fails the test; a native crash is retried once, then skipped."""
    import subprocess
    import sys

    pytest.importorskip("pyarrow")
    one = sim_archive(1.0).save(tmp_path / "one.aevd")
    two = sim_archive(0.5, vehicle={"count": 2, "formation": "line"}).save(tmp_path / "two.aevd")
    cmd = [sys.executable, "-c", PARQUET_CHECK, str(one), str(tmp_path), str(two)]
    for _attempt in range(2):
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            assert "parquet ok" in res.stdout
            return
        if res.returncode == 1:
            raise AssertionError(res.stderr[-2000:])
    pytest.skip(f"pyarrow crashed natively twice (exit code {res.returncode}); not an Aevion failure")


def test_provenance_content():
    from aevion.datasets.provenance import collect

    p = collect("abc123", seed=5, kernel="numba", backend="numpy")
    assert p["config_hash"] == "abc123" and p["seed"] == 5 and p["libraries"]["numpy"]
    assert isinstance(p["plugins"], list) and p["package_set_hash"] and p["n_packages"] > 5
    forbidden = {"hostname", "user", "username", "env", "environ", "cwd", "home"}
    assert not (forbidden & set(p))
