"""Structural checks for the CI/CD definitions: they cannot be run locally, so validate everything that can be validated."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib  # novermin
except ImportError:  # Python 3.10: pytest itself depends on tomli there
    import tomli as tomllib

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def all_steps(wf):
    for job in wf["jobs"].values():
        yield from job.get("steps", [])


def run_lines(wf):
    for s in all_steps(wf):
        for line in str(s.get("run", "")).splitlines():
            yield line.strip()


def test_expected_workflows_exist():
    assert {p.stem for p in WORKFLOWS} >= {"ci", "docs", "bench", "security", "release", "gpu"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_workflow_is_well_formed_and_least_privilege(path):
    wf = load(path)
    assert wf.get("name") and wf.get(True, wf.get("on")), "needs a name and triggers"  # YAML parses a bare `on:` key as True
    assert "permissions" in wf, "set top-level permissions (least privilege)"
    assert wf["permissions"].get("contents") in ("read", None) or path.stem == "release"
    for name, job in wf["jobs"].items():
        assert "runs-on" in job, name
        assert job.get("steps") or job.get("uses"), name


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_actions_are_pinned_to_versions(path):
    for step in all_steps(load(path)):
        uses = step.get("uses")
        if not uses:
            continue
        action, _, ref = uses.partition("@")
        assert ref, f"{uses} is not pinned"
        assert re.fullmatch(r"v\d+(\.\d+){0,2}|[0-9a-f]{40}", ref) or uses.startswith("pypa/gh-action-pypi-publish@release/v"), (
            f"{uses}: pin a version tag or commit SHA"
        )
        assert not action.startswith("./") or (ROOT / action[2:]).exists()


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_referenced_repository_files_exist(path):
    wf = load(path)
    for line in run_lines(wf):
        for tok in re.findall(r"(?:tools|tests|benchmarks|docs)/[\w./*-]+\.(?:py|md|json|yml)", line):
            if "*" in tok:
                continue
            assert (ROOT / tok).exists(), f"{path.name}: '{tok}' does not exist"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_pip_extras_exist(path):
    extras = set(PYPROJECT["project"]["optional-dependencies"])
    for line in run_lines(load(path)):
        for group in re.findall(r"\.\[([\w,-]+)\]", line):
            for name in group.split(","):
                assert name in extras, f"{path.name}: unknown extra '{name}' in: {line}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_cli_commands_used_in_workflows_exist(path):
    import argparse

    from aevion.cli import build_parser

    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    for line in run_lines(load(path)):
        m = re.match(r"aevion\s+([a-z-]+)", line)
        if m:
            assert m.group(1) in sub.choices, f"{path.name}: unknown CLI command 'aevion {m.group(1)}'"


def test_python_versions_in_the_matrix_are_declared_in_the_classifiers():
    declared = set(re.findall(r"Python :: (3\.\d+)", "\n".join(PYPROJECT["project"]["classifiers"])))
    ci = load(ROOT / ".github" / "workflows" / "ci.yml")
    versions = set(ci["jobs"]["tests"]["strategy"]["matrix"]["python"])
    versions |= {str(i["python"]) for i in ci["jobs"]["tests"]["strategy"]["matrix"].get("include", []) if "python" in i}
    assert versions <= declared, versions - declared
    assert declared <= versions, f"classifier(s) {declared - versions} are never tested"
    assert f">={min(declared, key=lambda v: int(v.split('.')[1]))}" == PYPROJECT["project"]["requires-python"]


def test_ci_success_gate_depends_on_every_other_job():
    ci = load(ROOT / ".github" / "workflows" / "ci.yml")
    others = set(ci["jobs"]) - {"ci-success"}
    assert set(ci["jobs"]["ci-success"]["needs"]) == others


def test_the_main_ci_runs_every_quality_gate():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for needle in (
        "ruff check",
        "ruff format --check",
        "mypy src/aevion",
        "api_snapshot.py",
        "gen_config_docs.py --check",
        "--cov-fail-under=85",
        "smoke_wheel.py",
        "twine check",
    ):
        assert needle in text, f"ci.yml does not run: {needle}"


def test_coverage_threshold_matches_the_project_configuration():
    assert PYPROJECT["tool"]["coverage"]["report"]["fail_under"] == 85


def test_release_uses_trusted_publishing_not_stored_tokens():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    wf = load(ROOT / ".github" / "workflows" / "release.yml")
    assert wf["jobs"]["publish"]["permissions"]["id-token"] == "write" and wf["jobs"]["publish"]["environment"]
    assert not re.search(r"PYPI_(API_)?TOKEN|password:", text), "use trusted publishing instead of a stored token"
    assert set(wf["jobs"]["publish"]["needs"] if isinstance(wf["jobs"]["publish"]["needs"], list) else [wf["jobs"]["publish"]["needs"]]) == {"build"}
    triggers = wf.get(True, wf.get("on"))
    assert list(triggers) == ["push"] and triggers["push"]["tags"] == ["v*"]


def test_dependabot_keeps_actions_and_pip_current():
    cfg = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    assert {u["package-ecosystem"] for u in cfg["updates"]} >= {"github-actions", "pip"}


def test_mkdocs_hook_and_config_are_consistent():
    cfg = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "strict: true" in cfg
    for hook in re.findall(r"^\s+-\s+(tools/\S+\.py)\s*$", cfg, re.M):
        assert (ROOT / hook).exists()


def test_local_ci_runner_lists_gates_that_mirror_the_workflows():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "ci_local.py"), "--list"], capture_output=True, text=True, check=True).stdout
    for gate in ("lint", "format", "types", "api", "docs-sync", "security-static", "tests", "bench", "docs", "wheel"):
        assert re.search(rf"^{gate}\s", out, re.M), gate
    for line in out.splitlines():
        assert len(line.split()) >= 3
