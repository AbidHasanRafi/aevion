"""Static security rules over the whole source tree. A failure here is a policy violation, not a flaky test."""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "aevion"
FILES = sorted(SRC.rglob("*.py"))

BANNED_CALLS = {
    "eval": "eval() executes arbitrary code",
    "exec": "exec() executes arbitrary code",
    "compile": "compile() of dynamic source",
    "__import__": "dynamic imports must go through importlib with a fixed module name",
}
BANNED_ATTRS = {
    ("os", "system"),
    ("os", "popen"),
    ("pickle", "load"),
    ("pickle", "loads"),
    ("pickle", "Unpickler"),
    ("marshal", "load"),
    ("marshal", "loads"),
    ("shelve", "open"),
    ("yaml", "load"),
    ("yaml", "unsafe_load"),
    ("yaml", "full_load"),
    ("tempfile", "mktemp"),
}
BANNED_IMPORTS = {"pickle", "cPickle", "marshal", "shelve", "dill", "cloudpickle", "telnetlib", "ftplib"}


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def rel(path):
    return path.relative_to(SRC.parent.parent).as_posix()


def test_source_tree_is_found():
    assert len(FILES) > 80


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_no_dangerous_constructs(path):
    problems = []
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_CALLS:
                problems.append(f"{node.lineno}: {f.id}(): {BANNED_CALLS[f.id]}")
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in BANNED_ATTRS:
                problems.append(f"{node.lineno}: {f.value.id}.{f.attr}()")
            for kw in node.keywords:
                if kw.arg == "allow_pickle" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    problems.append(f"{node.lineno}: allow_pickle must be literally False")
                if kw.arg == "shell" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    problems.append(f"{node.lineno}: shell=True / dynamic shell argument")
                if kw.arg == "Loader" and isinstance(f, ast.Attribute) and f.attr == "load":
                    problems.append(f"{node.lineno}: yaml.load with an explicit Loader (use safe_load)")
            if isinstance(f, ast.Attribute) and f.attr in ("extractall", "extract") and "unpack" not in path.name and "store" not in path.name:
                problems.append(f"{node.lineno}: archive extraction outside the validated unpack() path")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED_IMPORTS:
                    problems.append(f"{node.lineno}: import {a.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in BANNED_IMPORTS:
            problems.append(f"{node.lineno}: from {node.module} import ...")
    assert not problems, f"{rel(path)}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_network_access_is_https_only_and_explicit(path):
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"""["'](http://[^"'\s]+)["']""", text):
        url = m.group(1)
        assert url.startswith(("http://localhost", "http://127.0.0.1")), f"{rel(path)}: cleartext URL {url}"
    uses_network = any(tok in text for tok in ("urlopen", "urlretrieve", "requests.", "httpx.", "socket."))
    allowed = {
        "viewer/ws.py",
        "bridges/transport.py",
    }  # the WebSocket layer (live viewer, DevTools client) and the UDP transport of the bridges (loopback-only by default); http.server/urllib are checked below
    assert not uses_network or str(path.relative_to(SRC).as_posix()) in allowed, f"{rel(path)} performs network access; only {sorted(allowed)} may"


SERVER_MODULES = {"http.server": "viewer/server.py", "socketserver": "viewer/server.py", "http.client": "viewer/browser.py"}


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_listening_and_devtools_code_lives_in_named_modules_only(path):
    """Opening a listening socket (the viewer server) or driving a browser are confined to one module each, so they are easy to audit."""
    here = path.relative_to(SRC).as_posix()
    for node in ast.walk(parse(path)):
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
            if isinstance(node, ast.ImportFrom) and node.module
            else []
        )
        for n in names:
            if n in SERVER_MODULES:
                assert here == SERVER_MODULES[n], f"{rel(path)} imports {n}; only {SERVER_MODULES[n]} may"


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_array_loaders_refuse_pickle(path):
    """np.load / torch.load are the places pickle can sneak in through a dependency."""
    problems = []
    for node in ast.walk(parse(path)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)):
            continue
        owner, name = node.func.value.id, node.func.attr
        kws = {k.arg: k.value for k in node.keywords}
        if (owner, name) == ("np", "load") and not (isinstance(kws.get("allow_pickle"), ast.Constant) and kws["allow_pickle"].value is False):
            problems.append(f"{node.lineno}: np.load without allow_pickle=False")
        if (owner, name) == ("torch", "load") and not (isinstance(kws.get("weights_only"), ast.Constant) and kws["weights_only"].value is True):
            problems.append(f"{node.lineno}: torch.load without weights_only=True")
    assert not problems, rel(path) + ":\n  " + "\n  ".join(problems)


def test_no_secrets_or_personal_paths_are_embedded():
    pat = re.compile(r"(?i)(api[_-]?key|secret|passwd|password|token)\s*=\s*['\"][^'\"]{8,}['\"]|C:\\\\Users\\\\|/home/[a-z]+/")
    for path in FILES:
        m = pat.search(path.read_text(encoding="utf-8"))
        assert not m, f"{rel(path)}: suspicious literal {m.group(0)!r}"


def test_bandit_reports_no_medium_or_high_findings():
    pytest.importorskip("bandit")
    import json
    import subprocess
    import sys

    root = SRC.parent.parent
    res = subprocess.run(
        [sys.executable, "-m", "bandit", "-r", str(SRC), "-c", str(root / "pyproject.toml"), "-f", "json", "-q"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    data = json.loads(res.stdout)
    bad = [r for r in data["results"] if r["issue_severity"] in ("MEDIUM", "HIGH")]
    assert not bad, "\n".join(f"{b['filename']}:{b['line_number']} {b['test_id']} {b['issue_text']}" for b in bad)
