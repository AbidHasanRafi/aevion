"""Property-based tests (hypothesis): invariants that must hold for *all* inputs, not just the hand-picked ones."""

import copy
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from aevion.config import SimulationConfig, config_hash, load_config  # noqa: E402
from aevion.core.constants import Atmosphere  # noqa: E402
from aevion.datasets.store import ArchiveReader, ArchiveWriter  # noqa: E402
from aevion.engine.scheduler import Scheduler  # noqa: E402
from aevion.errors import ConfigError  # noqa: E402
from aevion.frames import ClockModel, Transform  # noqa: E402
from aevion.frames.transform import (  # noqa: E402
    matrix_to_quat,
    quat_conj,
    quat_from_rpy,
    quat_mul,
    quat_normalize,
    quat_slerp,
    quat_to_matrix,
    quat_to_rpy,
)
from aevion.registry.core import _norm  # noqa: E402

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
angle = st.floats(-math.pi, math.pi)
vec3 = st.tuples(finite, finite, finite)


@st.composite
def quats(draw):
    q = np.array(draw(st.tuples(*[st.floats(-1, 1)] * 4)))
    assume(np.linalg.norm(q) > 1e-3)
    return quat_normalize(q)


@st.composite
def transforms(draw):
    return Transform(draw(vec3), draw(quats()))


# ------------------------------------------------------------------------------------ rotations / frames
@given(quats())
def test_normalised_quaternions_have_unit_norm(q):
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


@given(quats(), quats())
def test_quaternion_product_preserves_unit_norm_and_conjugate_inverts(a, b):
    assert abs(np.linalg.norm(quat_mul(a, b)) - 1.0) < 1e-9
    assert np.allclose(quat_mul(a, quat_conj(a)), [1, 0, 0, 0], atol=1e-9)


@given(quats())
def test_rotation_matrix_is_orthonormal_with_unit_determinant_and_roundtrips(q):
    R = quat_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1.0) < 1e-9
    q2 = matrix_to_quat(R)
    assert np.allclose(q, q2, atol=1e-7) or np.allclose(q, -q2, atol=1e-7)  # q and -q are the same rotation


@given(st.floats(-3.0, 3.0), st.floats(-1.45, 1.45), st.floats(-3.0, 3.0))
def test_euler_roundtrip_away_from_gimbal_lock(r, p, y):
    out = quat_to_rpy(quat_from_rpy(r, p, y))
    assert np.allclose(out, [r, p, y], atol=1e-7)


@given(quats(), quats(), st.floats(0, 1))
def test_slerp_stays_on_the_unit_sphere_and_hits_its_endpoints(a, b, t):
    assert abs(np.linalg.norm(quat_slerp(a, b, t)) - 1.0) < 1e-9
    for end, ref in ((0.0, a), (1.0, b)):
        got = quat_slerp(a, b, end)
        assert np.allclose(got, ref, atol=1e-7) or np.allclose(got, -ref, atol=1e-7)


@given(transforms(), transforms(), transforms())
def test_transform_composition_is_associative_and_inverse_is_exact(a, b, c):
    assert a.compose(b).compose(c).isclose(a.compose(b.compose(c)), atol=1e-6)
    assert a.compose(a.inverse()).isclose(Transform.identity(), atol=1e-6)


@given(transforms(), st.lists(vec3, min_size=1, max_size=8))
def test_rigid_transforms_preserve_distances_and_invert_points(t, pts):
    P = np.array(pts)
    Q = t.apply(P)
    if len(P) > 1:
        assert np.allclose(np.linalg.norm(P[1:] - P[:-1], axis=1), np.linalg.norm(Q[1:] - Q[:-1], axis=1), atol=1e-6)
    assert np.allclose(t.inverse().apply(Q), P, atol=1e-6)


@given(transforms())
def test_transform_dict_and_matrix_roundtrips(t):
    assert Transform.from_dict(t.to_dict()).isclose(t, atol=1e-12)
    assert Transform.from_matrix(t.matrix()).isclose(t, atol=1e-7)


# ------------------------------------------------------------------------------------ clock
@given(st.floats(-5, 5), st.floats(-500, 500), st.floats(0, 1e5))
def test_clock_deterministic_part_is_invertible(offset, ppm, t):
    c = ClockModel(offset_s=offset, drift_ppm=ppm)
    assert math.isclose(c.true_time(c.stamp(t)), t, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(c.stamp(t) - t, c.error_at(t), rel_tol=1e-9, abs_tol=1e-9)


@given(st.integers(0, 2**31), st.floats(1e-6, 1e-2))
def test_clock_jitter_is_reproducible_from_the_seed(seed, std):
    a, b = ClockModel(jitter_std_s=std, seed=seed), ClockModel(jitter_std_s=std, seed=seed)
    assert [a.stamp(i * 0.1) for i in range(5)] == [b.stamp(i * 0.1) for i in range(5)]
    a.reset()
    assert a.stamp(0.0) == ClockModel(jitter_std_s=std, seed=seed).stamp(0.0)


# ------------------------------------------------------------------------------------ atmosphere
@given(st.floats(0.0, 10_000.0), st.floats(0.0, 10_000.0))
def test_isa_is_monotonic_in_altitude(h1, h2):
    atm = Atmosphere()
    lo, hi = sorted((h1, h2))
    assume(hi - lo > 1.0)
    assert atm.density(None, hi) < atm.density(None, lo)
    assert atm.pressure(None, hi) < atm.pressure(None, lo)
    assert atm.temperature(None, hi) < atm.temperature(None, lo)


# ------------------------------------------------------------------------------------ scheduler
@given(st.integers(1, 40), st.integers(0, 10), st.integers(1, 400))
def test_scheduler_call_counts_match_the_arithmetic(every, phase, n_ticks):
    s = Scheduler(0.01)
    count = [0]
    s.add("t", lambda: count.__setitem__(0, count[0] + 1), every_ticks=every, phase=phase)
    for _ in range(n_ticks):
        s.run_tick()
        s.advance()
    expected = 0 if n_ticks <= phase else (n_ticks - 1 - phase) // every + 1
    assert count[0] == expected == s.get("t").calls


@given(st.lists(st.integers(0, 100), min_size=2, max_size=8, unique=True))
def test_scheduler_runs_tasks_in_priority_order(priorities):
    s = Scheduler(0.01)
    order = []
    for p in priorities:
        s.add(f"t{p}", lambda p=p: order.append(p), priority=p)
    s.run_tick()
    assert order == sorted(priorities)


# ------------------------------------------------------------------------------------ registry names
@given(st.text(alphabet=st.characters(categories=("L", "N", "Zs"), max_codepoint=0x7F), max_size=20))
def test_name_normalisation_is_idempotent(name):
    assert _norm(_norm(name)) == _norm(name)


# ------------------------------------------------------------------------------------ store
shapes = st.lists(st.integers(1, 4), min_size=0, max_size=2).map(tuple)


@given(
    shape=shapes,
    n=st.integers(1, 60),
    chunk=st.integers(1, 17),
    dtype=st.sampled_from([np.float64, np.float32, np.int32, np.bool_]),
    data=st.data(),
)
def test_store_roundtrip_and_slicing_equal_numpy(shape, n, chunk, dtype, data):
    arr = data.draw(hnp.arrays(dtype, (n,) + shape, elements=hnp.from_dtype(np.dtype(dtype), allow_nan=False, allow_infinity=False)))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.aevd"
        with ArchiveWriter(path, chunk_steps=chunk) as w:
            for row in arr:
                w.append("x", row)
        r = ArchiveReader(path)
        ch = r.channels["x"]
        assert ch.shape == arr.shape and ch.dtype == arr.dtype and r.verify() == []
        assert np.array_equal(ch.to_numpy(), arr)
        start = data.draw(st.integers(-n - 2, n + 2))
        stop = data.draw(st.integers(-n - 2, n + 2))
        step = data.draw(st.sampled_from([1, 2, 3, -1]))
        assert np.array_equal(ch[start:stop:step], arr[start:stop:step])
        i = data.draw(st.integers(-n, n - 1))
        assert np.array_equal(ch[i], arr[i])
        idx = data.draw(st.lists(st.integers(-n, n - 1), min_size=1, max_size=6))
        assert np.array_equal(ch[idx], arr[idx])
        r.close()


# ------------------------------------------------------------------------------------ configuration
json_leaf = st.one_of(st.none(), st.booleans(), st.integers(-(10**6), 10**6), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=12))
json_tree = st.recursive(json_leaf, lambda c: st.one_of(st.lists(c, max_size=4), st.dictionaries(st.text(max_size=8), c, max_size=4)), max_leaves=12)

VALID = {
    "schema_version": 2,
    "duration": 1.0,
    "seed": 3,
    "vehicle": {"count": 2, "formation": {"type": "line"}},
    "sensors": ["imu", {"type": "gps", "params": {"rate": 5}}],
    "controller": {"type": "geometric"},
    "wind": {"type": "constant", "params": {"velocity": [1, 0, 0]}},
    "engine": {"dt": 0.005},
}


def _paths(node, prefix=()):
    yield prefix
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _paths(v, prefix + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _paths(v, prefix + (i,))


PATHS = [p for p in _paths(VALID) if p]


@given(json_tree)
def test_arbitrary_json_never_crashes_the_config_loader(doc):
    """Whatever you feed it, the only possible outcomes are a config or a ConfigError - never a stray TypeError/KeyError."""
    try:
        load_config(doc if isinstance(doc, dict) else {"x": doc})
    except ConfigError:
        pass


@given(st.sampled_from(PATHS), json_tree)
def test_mutated_valid_configs_never_crash_the_config_loader(path, replacement):
    doc = copy.deepcopy(VALID)
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement
    try:
        cfg = load_config(doc)
    except ConfigError:
        return
    # if it was accepted, it must be self-consistent: serialises, reloads and hashes identically
    again = SimulationConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert config_hash(again) == config_hash(cfg)


@given(st.floats(0.01, 600.0), st.integers(1, 50), st.integers(0, 2**31), st.sampled_from([0.001, 0.002, 0.005, 0.01]))
def test_valid_config_roundtrips_through_dict_and_json(duration, count, seed, dt):
    doc = {"duration": duration, "seed": seed, "engine": {"dt": dt}, "vehicle": {"count": count, "formation": {"type": "line"}}}
    cfg = load_config(doc)
    again = SimulationConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again.to_dict() == cfg.to_dict() and config_hash(again) == config_hash(cfg)
    assert cfg.n_steps == round(duration / dt)
