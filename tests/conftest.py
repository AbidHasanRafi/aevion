"""Shared test configuration: isolation from the user's machine, deterministic plotting, hypothesis profiles, markers."""

import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from hypothesis import HealthCheck, settings

    _quiet = [HealthCheck.too_slow, HealthCheck.data_too_large]
    settings.register_profile("dev", max_examples=60, deadline=None, suppress_health_check=_quiet)
    settings.register_profile("ci", max_examples=200, deadline=None, suppress_health_check=_quiet, print_blob=True)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
except ImportError:  # property tests are skipped when hypothesis is not installed
    pass


@pytest.fixture(autouse=True, scope="session")
def _isolated_aevion_home(tmp_path_factory):
    """Run registry, viewer cache etc. live in a throw-away directory, never in the developer's ``~/.aevion``."""
    home = tmp_path_factory.mktemp("aevion_home")
    old = os.environ.get("AEVION_HOME")
    os.environ["AEVION_HOME"] = str(home)
    yield home
    if old is None:
        os.environ.pop("AEVION_HOME", None)
    else:
        os.environ["AEVION_HOME"] = old


def pytest_collection_modifyitems(config, items):
    """``@pytest.mark.gpu`` tests are skipped when CUDA is unavailable."""
    try:
        from aevion.core.backend import cuda_available

        has_cuda = cuda_available()
    except Exception:
        has_cuda = False
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if item.get_closest_marker("gpu") and not has_cuda:
            item.add_marker(skip)


def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true", default=False, help="rewrite the golden files instead of comparing against them")
