import warnings

import pytest

from aevion import Multirotor, __version__
from aevion import deprecation as D
from aevion.deprecation import AevionDeprecationWarning, Deprecation, deprecated, is_expired, list_deprecations, parse_version, renamed_kwarg


def test_parse_version_and_minor_distance():
    assert parse_version("0.4") == (0, 4, 0) and parse_version("1.2.3") == (1, 2, 3) and parse_version("1.0.0rc1") == (1, 0, 0)
    with pytest.raises(ValueError):
        parse_version("banana")
    assert D.minors_between("0.2.0", "0.4.0") == 2 and D.minors_between("0.9", "1.0") > 2


def test_policy_requires_two_minor_releases():
    with pytest.raises(ValueError, match="at least two minor"):
        D.register(Deprecation("x", "0.2.0", "0.3.0"))
    D.register(Deprecation("test.ok", "0.2.0", "0.4.0"))
    D._REGISTRY.pop("test.ok")


def test_decorator_warns_registers_and_preserves_behaviour():
    @deprecated("0.2.0", "0.5.0", alternative="new_fn()", reason="renamed")
    def old_fn(a, b=2):
        """Doc."""
        return a + b

    with pytest.warns(AevionDeprecationWarning) as w:
        assert old_fn(1, b=3) == 4
    msg = str(w[0].message)
    assert "deprecated since Aevion 0.2.0" in msg and "removed in 0.5.0" in msg and "Use new_fn() instead" in msg and "renamed" in msg
    assert old_fn.__doc__ == "Doc." and old_fn.__name__ == "old_fn" and old_fn.__deprecated__.alternative == "new_fn()"
    assert any(d.what.endswith("old_fn") for d in list_deprecations())
    for k in [k for k in D._REGISTRY if k.endswith("old_fn")]:
        D._REGISTRY.pop(k)


def test_deprecation_warning_is_visible_by_default_to_end_users():
    assert issubclass(AevionDeprecationWarning, FutureWarning) and not issubclass(AevionDeprecationWarning, DeprecationWarning)


def test_renamed_kwarg():
    @renamed_kwarg("old", "new", "0.2.0", "0.4.0")
    def f(new=0):
        return new

    with pytest.warns(AevionDeprecationWarning, match="'new='"):
        assert f(old=5) == 5
    assert f(new=6) == 6
    with pytest.raises(TypeError, match="both"):
        f(old=1, new=2)
    for k in [k for k in D._REGISTRY if "(old=...)" in k]:
        D._REGISTRY.pop(k)


def test_the_fast_flag_was_removed_in_0_5_0_in_favour_of_kernel():
    from aevion.scenario import Scenario

    with pytest.raises(TypeError):
        Multirotor("generic_1kg", n=1, fast=False)
    sc = Scenario(duration=0.2, trajectory=dict(kind="hover", pos=[0, 0, 2.0]))
    with pytest.raises(TypeError):
        sc.run(fast=False)
    assert Multirotor("generic_1kg", n=1, kernel="reference").kernel.name == "reference"
    assert sc.run(kernel="reference").meta["kernel"] == "reference"


def test_the_library_does_not_use_its_own_deprecated_api():
    """Running the main workflows must emit no AevionDeprecationWarning (the library dogfoods the policy)."""
    from aevion.config import SimulationConfig
    from aevion.engine import Simulation
    from aevion.scenario import Scenario

    with warnings.catch_warnings():
        warnings.simplefilter("error", AevionDeprecationWarning)
        Simulation(SimulationConfig.from_dict({"duration": 0.3, "sensors": ["imu"]})).run()
        Scenario(duration=0.3, trajectory=dict(kind="hover", pos=[0, 0, 2.0])).run()
        Multirotor("generic_1kg", n=2).reset()


def test_no_deprecation_is_past_its_removal_version():
    """Promises must not lapse: once the package version reaches a deprecation's removal version, it has to be removed."""
    # importing the modules registers their (decorator-based) deprecations; warn_deprecated-based ones register on first use
    expired = [d for d in list_deprecations() if is_expired(d, __version__)]
    assert not expired, "remove these deprecated features now (or postpone with a changelog note):\n  " + "\n  ".join(
        f"{d.what} (due {d.remove_in})" for d in expired
    )
    assert is_expired(Deprecation("x", "0.2.0", "0.4.0"), "0.4.0") and not is_expired(Deprecation("x", "0.2.0", "0.4.0"), "0.3.9")
