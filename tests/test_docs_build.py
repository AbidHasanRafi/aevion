"""The documentation site must build in strict mode (broken links, missing references and bad docstrings fail)."""

import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def installed(dist):
    try:
        metadata.version(dist)
        return True
    except metadata.PackageNotFoundError:
        return False


def test_nav_entries_exist_or_are_generated():
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    generated = ("api/", "gallery/", "cli.md", "guide.md", "index.md")  # all written by tools/gen_site.py
    for rel in re.findall(r":\s+([\w./-]+\.md)\s*$", nav, re.M):
        assert rel.startswith(generated) or (ROOT / "docs" / rel).exists(), f"mkdocs.yml nav points to missing {rel}"


def test_every_public_module_has_an_api_page_template():
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_site

    from aevion._api import PUBLIC_MODULES

    assert set(PUBLIC_MODULES) <= set(gen_site.API_BLURB), "add a one-line description for the new public module in tools/gen_site.py"


@pytest.mark.slow
@pytest.mark.skipif(
    not (installed("mkdocs") and installed("mkdocs-material") and installed("mkdocstrings-python")), reason="docs extras not installed"
)
def test_site_builds_in_strict_mode(tmp_path):
    env = {**os.environ, "AEVION_DOCS_NO_GALLERY": "1", "PYTHONIOENCODING": "utf-8", "NO_MKDOCS_2_WARNING": "true"}
    res = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(tmp_path / "site")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert res.returncode == 0, (res.stdout + res.stderr)[-4000:]
    assert (tmp_path / "site" / "index.html").exists() and (tmp_path / "site" / "api" / "aevion.engine" / "index.html").exists()
