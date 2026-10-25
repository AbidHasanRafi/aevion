"""The public API may only grow deliberately and never shrink without a deprecation cycle."""

import json
from pathlib import Path

import pytest

from aevion import _api

SNAPSHOT = Path(__file__).parent / "golden" / "public_api.json"


def test_no_breaking_changes_to_the_public_api():
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    breaking, _ = _api.compare(old, _api.snapshot())
    assert not breaking, (
        "breaking public-API changes (deprecate first, see DOCUMENTATION.md; if intentional run tools/api_snapshot.py --update):\n  "
        + "\n  ".join(breaking)
    )


def test_additions_to_the_public_api_are_acknowledged():
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    _, additions = _api.compare(old, _api.snapshot())
    assert not additions, (
        "new public API not in the snapshot; review it, add a CHANGELOG entry, then run tools/api_snapshot.py --update:\n  " + "\n  ".join(additions)
    )


def test_backwards_compatible_reexports_exist():
    """Names that moved must still import from their old locations."""
    from aevion.scenario import SENSOR_FACTORIES, Scenario, make_formation, make_trajectory, verify  # noqa: F401

    assert callable(make_formation) and callable(make_trajectory) and "lidar" in SENSOR_FACTORIES


# ---- the comparison logic itself --------------------------------------------------------------------
def sig(*params):
    return [{"name": n, "kind": k, "required": r} for n, k, r in params]


POK = "POSITIONAL_OR_KEYWORD"


def fn(*params):
    return {"kind": "function", "signature": sig(*params)}


def cmp(old_params, new_params):
    old = {"m": {"f": fn(*old_params)}}
    new = {"m": {"f": fn(*new_params)}}
    return _api.compare(old, new)


def test_compare_accepts_compatible_signature_evolution():
    assert cmp([("a", POK, True)], [("a", POK, True), ("b", POK, False)]) == ([], [])  # added optional parameter
    assert cmp([("a", POK, True), ("b", POK, False)], [("a", POK, True), ("b", POK, False), ("c", "KEYWORD_ONLY", False)]) == ([], [])
    assert cmp([("a", POK, True)], [("a", POK, False)]) == ([], [])  # parameter became optional


@pytest.mark.parametrize(
    "old,new,why",
    [
        ([("a", POK, True), ("b", POK, False)], [("a", POK, True)], "removed"),
        ([("a", POK, True)], [("a", POK, True), ("b", POK, True)], "required parameter"),
        ([("a", POK, True), ("b", POK, True)], [("b", POK, True), ("a", POK, True)], "reordered"),
        ([("a", POK, True), ("kw", "VAR_KEYWORD", False)], [("a", POK, True)], "**kwargs"),
    ],
)
def test_compare_flags_breaking_signature_changes(old, new, why):
    breaking, _ = cmp(old, new)
    assert breaking and why in breaking[0]


def test_compare_flags_removed_names_modules_and_methods_and_reports_additions():
    old = {"m": {"f": fn(), "C": {"kind": "class", "signature": [], "methods": ["a", "b"]}}, "gone": {"x": fn()}}
    new = {"m": {"C": {"kind": "class", "signature": [], "methods": ["a", "c"]}, "g": fn()}}
    breaking, additions = _api.compare(old, new)
    assert "module gone removed" in breaking and "m.f removed" in breaking and "m.C.b() removed" in breaking
    assert "m.g" in additions and "m.C.c()" in additions
    # optional dependencies that are absent in this environment are not breakage
    breaking, _ = _api.compare({"m": {"Q": {"kind": "optional-missing"}}}, {"m": {}})
    assert not breaking


def test_snapshot_describes_real_objects():
    snap = _api.snapshot()
    sim = snap["aevion.engine"]["Simulation"]
    assert sim["kind"] == "class" and "run" in sim["methods"] and any(p["name"] == "config" and p["required"] for p in sim["signature"])
    assert snap["aevion.config"]["load_config"]["kind"] == "function"
    assert "aevion" in snap and len(snap) == len(_api.PUBLIC_MODULES)
