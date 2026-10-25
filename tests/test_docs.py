"""Executable documentation: every python block marked ``<!-- docs-test -->`` runs, top to bottom, one namespace per page.

Pages are the ``<!-- page: name -->`` sections of DOCUMENTATION.md.
"""

import re
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[1] / "DOCUMENTATION.md"
MARK = re.compile(r"<!--\s*docs-test\s*-->\s*\n```python\n(.*?)```", re.S)
PAGE = re.compile(r"^<!-- page: ([\w-]+) -->$", re.M)


def _pages():
    parts = PAGE.split(GUIDE.read_text(encoding="utf-8"))  # [preamble, name, text, name, text, ...]
    return dict(zip(parts[1::2], parts[2::2], strict=True))


PAGES = _pages()
WITH_BLOCKS = [name for name, text in PAGES.items() if MARK.search(text)]


def test_executable_pages_exist():
    assert {"quickstart", "untrusted"} <= set(WITH_BLOCKS)


@pytest.mark.parametrize("page", WITH_BLOCKS)
def test_documentation_examples_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = {"__name__": "docs_example"}
    for i, code in enumerate(MARK.findall(PAGES[page]), 1):
        try:
            exec(compile(code, f"{page}[block {i}]", "exec"), ns)
        except Exception as ex:
            raise AssertionError(f"{page}, block {i} failed: {type(ex).__name__}: {ex}\n--- code ---\n{code}") from ex


def test_internal_links_resolve_to_headings():
    """Every ``](#anchor)`` in DOCUMENTATION.md must hit a heading, using GitHub's slug and duplicate-numbering rules."""
    seen, anchors, fenced = {}, set(), False
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        m = None if fenced else re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            base = re.sub(r"[^\w\- ]", "", re.sub(r"`|\*|<[^>]+>", "", m.group(1)).lower()).strip().replace(" ", "-")
            n = seen.get(base, 0)
            seen[base] = n + 1
            anchors.add(base if n == 0 else f"{base}-{n}")
    broken = sorted(set(re.findall(r"\]\(#([^)]+)\)", GUIDE.read_text(encoding="utf-8"))) - anchors)
    assert not broken, f"links to missing headings: {broken}"


def test_every_unmarked_python_block_at_least_compiles():
    """Illustrative snippets need not run, but they must be valid Python."""
    fence = re.compile(r"```python\n(.*?)```", re.S)
    for name, text in PAGES.items():
        for i, code in enumerate(fence.findall(text), 1):
            if ">>>" in code or "..." in code.replace("...)", "").replace("[...]", ""):
                continue
            try:
                compile(code, f"{name}[{i}]", "exec")
            except SyntaxError as ex:
                raise AssertionError(f"{name} python block {i} has a syntax error: {ex}") from ex
