"""``aevion new-plugin``: generated projects must be valid, pass their own tests and install."""

import os
import re
import subprocess
import sys

import pytest

from aevion.cli import main
from aevion.scaffold import ScaffoldError, create_plugin, normalise_names, render


def test_names_are_normalised_consistently():
    n = normalise_names("My_Aevion.Ext")
    assert n == dict(dist_name="my-aevion-ext", package="my_aevion_ext", component="my_aevion_ext", class_name="MyAevionExt")
    n = normalise_names("aevion-sonar-plugin")
    assert n["component"] == "sonar" and n["class_name"] == "Sonar" and n["package"] == "aevion_sonar_plugin"


@pytest.mark.parametrize("bad", ["", "1abc", "has space", "../evil", "a/b", "class", "aevion", "os", "tests", "-x"])
def test_unsafe_or_shadowing_names_are_rejected(bad):
    with pytest.raises(ScaffoldError):
        normalise_names(bad)


def test_render_substitutes_all_placeholders():
    assert render("{{a}}-{{b}}-{{a}}", {"a": "1", "b": "2"}) == "1-2-1"


def test_generated_project_layout_and_contents(tmp_path):
    files = create_plugin("acme-sonar", tmp_path / "p", description='A "quoted" sonar')
    rel = {f.relative_to(tmp_path / "p").as_posix() for f in files}
    assert rel == {".gitignore", "pyproject.toml", "README.md", "src/acme_sonar/__init__.py", "src/acme_sonar/components.py", "tests/test_plugin.py"}
    text = "".join(f.read_text(encoding="utf-8") for f in files)
    assert not re.search(r"\{\{[a-z_]+\}\}", text), "unrendered placeholder"
    try:
        import tomllib  # novermin
    except ImportError:  # Python 3.10: pytest itself depends on tomli there
        import tomli as tomllib

    meta = tomllib.loads((tmp_path / "p" / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["name"] == "acme-sonar"
    assert meta["project"]["entry-points"]["aevion.plugins"] == {"acme_sonar": "acme_sonar:register"}
    for f in files:
        if f.suffix == ".py":
            compile(f.read_text(encoding="utf-8"), str(f), "exec")


def test_existing_directories_are_not_overwritten_without_force(tmp_path):
    create_plugin("acme-sonar", tmp_path / "p")
    with pytest.raises(ScaffoldError, match="not empty"):
        create_plugin("acme-sonar", tmp_path / "p")
    create_plugin("acme-sonar", tmp_path / "p", force=True)


def test_cli_creates_the_project(tmp_path, capsys):
    main(["new-plugin", "demo-ext", "--dest", str(tmp_path / "x")])
    assert (tmp_path / "x" / "src" / "demo_ext" / "__init__.py").exists()
    assert "created 6 files" in capsys.readouterr().out


def test_generated_plugin_passes_its_own_tests(tmp_path):
    """End to end: scaffold, then run the generated test-suite in a fresh interpreter against the new package."""
    root = tmp_path / "acme-sonar"
    create_plugin("acme-sonar", root)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]), "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-c",
            str(root / "pyproject.toml"),
            "--rootdir",
            str(root),
            str(root / "tests"),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=300,
        check=False,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    assert "3 passed" in res.stdout


def test_template_is_shipped_with_the_package():
    from aevion.scaffold import TEMPLATE_DIR

    assert len(list(TEMPLATE_DIR.rglob("*.tmpl"))) == 6
