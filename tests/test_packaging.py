"""Release hygiene: versions, changelog, metadata and shipped files agree."""

import re
from pathlib import Path

import pytest

try:
    import tomllib  # novermin
except ImportError:  # Python 3.10: pytest itself depends on tomli there
    import tomli as tomllib

import aevion

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = CONFIG["project"]


def test_pyproject_and_package_versions_agree():
    assert PROJECT["version"] == aevion.__version__


GUIDE = (ROOT / "DOCUMENTATION.md").read_text(encoding="utf-8")


def _section(name):
    """The text of one ``<!-- page: name -->`` section of DOCUMENTATION.md."""
    return re.search(rf"^<!-- page: {name} -->$(.*?)(?=^<!-- page: |\Z)", GUIDE, re.M | re.S).group(1)


def test_changelog_has_a_section_for_the_current_version():
    assert re.search(rf"^### \[?{re.escape(aevion.__version__)}\]?\b", _section("changelog"), re.M), "the changelog has no entry for this version"


def test_governance_sections_exist_and_are_linked_from_each_other():
    for name in ("contributing", "code_of_conduct", "security", "versioning", "releasing", "changelog"):
        assert _section(name).strip(), name
    assert (ROOT / "README.md").exists() and (ROOT / "LICENSE").exists()
    assert "(#versioning-and-deprecation-policy)" in _section("contributing")
    assert "(#releasing-aevion)" in _section("versioning")
    for anchor in ("versioning-and-deprecation-policy", "releasing-aevion"):
        assert re.search(rf"^#{{2,3}} {anchor.replace('-', '[ -]')}", GUIDE, re.M | re.I), anchor


def test_metadata_is_complete():
    for key in ("name", "version", "description", "readme", "requires-python", "license", "classifiers", "dependencies", "urls"):
        assert key in PROJECT, key
    assert PROJECT["name"] == "aevion" and "Typing :: Typed" in PROJECT["classifiers"]
    assert (ROOT / "src" / "aevion" / "py.typed").exists()


def test_template_files_are_declared_as_package_data():
    assert "templates/**/*" in CONFIG["tool"]["setuptools"]["package-data"]["aevion"]


@pytest.mark.parametrize("extra", ["fast", "torch", "gym", "viz", "hdf5", "yaml", "parquet", "ml", "all", "test", "dev", "docs"])
def test_documented_extras_exist(extra):
    assert extra in PROJECT["optional-dependencies"]


def test_importing_aevion_does_not_import_torch():
    """torch takes seconds to import; NumPy-only users (and ``aevion info``) must never pay for it."""
    import subprocess
    import sys

    code = "import sys, aevion, aevion.registry, aevion.wireless; from aevion import Multirotor; Multirotor('generic_1kg', n=1).reset(); assert 'torch' not in sys.modules, 'torch was imported'"
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT / "tests",
        env={
            "PYTHONPATH": str(ROOT / "src"),
            "PATH": __import__("os").environ.get("PATH", ""),
            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        },
        check=False,
    )
    assert res.returncode == 0, res.stderr[-1500:]
