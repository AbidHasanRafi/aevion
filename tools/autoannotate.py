"""Conservative auto-annotation of untyped function signatures (development helper).

Adds parameter annotations from well-known parameter names and return annotations from the function's
``return`` statements. It never overwrites an existing annotation; mypy then verifies the result. Anything it
cannot infer gets ``Any`` (honest, and visible to reviewers) rather than a guess.

    python tools/autoannotate.py src/aevion/engine/faults.py [more files...]
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PARAM_TYPES = {
    "sim": '"Simulation"',
    "cfg": '"SimulationConfig"',
    "config": '"SimulationConfig"',
    "name": "str",
    "key": "str",
    "kind": "str",
    "sensor": "str",
    "frame": "str",
    "target": "str",
    "source": "str",
    "pattern": "str",
    "metric": "str",
    "field": "str",
    "t": "float",
    "dt": "float",
    "tol": "float",
    "t0": "float",
    "t1": "float",
    "t_start": "float",
    "t_end": "float",
    "duration": "float",
    "rate_hz": "float",
    "step": "int",
    "k": "int",
    "n": "int",
    "count": "int",
    "vehicle": "Optional[int]",
    "seed": "int",
    "verbose": "bool",
    "deep": "bool",
    "quick": "bool",
    "path": "PathLike",
    "dst": "PathLike",
    "src": "PathLike",
    "out": "PathLike",
    "meta": "Dict[str, Any]",
    "data": "Dict[str, Any]",
    "d": "Dict[str, Any]",
    "archive": '"RunArchive"',
}
ALWAYS_NONE = {
    "__init__",
    "reset",
    "clear",
    "close",
    "abort",
    "flush",
    "run_tick",
    "advance",
    "remove",
    "unregister",
    "on_start",
    "on_end",
    "on_reset",
    "pre_step",
    "post_step",
    "on_event",
}
TYPING_NAMES = ("Optional", "Dict", "List", "Any", "Iterator")
FROM_TYPING = re.compile(r"^from typing import ([^\n(]+)$", re.M)


def annotate_param(a: ast.arg, is_first: bool, in_method: bool) -> None:
    if a.annotation is not None:
        return
    if is_first and in_method and a.arg in ("self", "cls"):
        return
    a.annotation = ast.parse(PARAM_TYPES.get(a.arg, "Any"), mode="eval").body


def return_annotation(fn: ast.FunctionDef) -> str | None:
    if fn.returns is not None:
        return None
    if fn.name in ALWAYS_NONE:
        return "None"
    nested = {id(m) for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.Lambda)) and n is not fn for m in ast.walk(n)}
    rets = [r for r in ast.walk(fn) if isinstance(r, ast.Return) and id(r) not in nested]
    if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn) if id(n) not in nested):
        return "Iterator[Any]"
    vals = [r.value for r in rets if r.value is not None]
    if not vals:
        return "None"
    if fn.name == "__len__":
        return "int"
    if fn.name in ("__repr__", "__str__"):
        return "str"
    if fn.name == "__contains__":
        return "bool"
    if all(isinstance(v, ast.Constant) and isinstance(v.value, bool) for v in vals):
        return "bool"
    if all(isinstance(v, ast.JoinedStr) or (isinstance(v, ast.Constant) and isinstance(v.value, str)) for v in vals):
        return "str"
    if all(isinstance(v, (ast.Dict, ast.DictComp)) for v in vals):
        return "Dict[str, Any]"
    if all(isinstance(v, (ast.List, ast.ListComp)) for v in vals):
        return "List[Any]"
    if all(isinstance(v, ast.Name) and v.id == "self" for v in vals) and len(rets) == len(vals):
        return None  # needs Self / the class name: leave for manual annotation
    return "Any"


def add_imports(text: str, need: set, path: Path) -> str:
    """Add typing / TYPE_CHECKING imports needed by the new annotations."""
    parts = path.resolve().parts
    pkg = parts[parts.index("aevion") + 1 : -1]  # sub-packages below aevion/
    dots = "." * (len(pkg) + 1)
    here = pkg[0] if pkg else ""
    typing_needed = sorted(n for n in need if n in TYPING_NAMES)
    new_lines = []
    m = FROM_TYPING.search(text)
    if typing_needed:
        if m:
            have = [x.strip() for x in m.group(1).split(",")]
            text = text.replace(m.group(0), "from typing import " + ", ".join(sorted(set(have) | set(typing_needed))), 1)
        else:
            new_lines.append("from typing import " + ", ".join(typing_needed))
    if "PathLike" in need and not re.search(r"^\s*(PathLike\s*=|from .* import .*\bPathLike\b)", text, re.M):
        new_lines.append(f"from {dots}_types import PathLike")
    tc = []
    if "Simulation" in need and not re.search(r"^\s*from .* import .*\bSimulation\b", text, re.M):
        tc.append("from .simulation import Simulation" if here == "engine" else f"from {dots}engine.simulation import Simulation")
    if "SimulationConfig" in need and not re.search(r"^\s*from .* import .*\bSimulationConfig\b", text, re.M):
        tc.append("from .blocks import SimulationConfig" if here == "config" else f"from {dots}config import SimulationConfig")
    if "RunArchive" in need and not re.search(r"^\s*from .* import .*\bRunArchive\b", text, re.M):
        tc.append("from .archive import RunArchive" if here == "datasets" else f"from {dots}datasets.archive import RunArchive")
    block = "\n".join(new_lines)
    if tc:
        if not re.search(r"\bTYPE_CHECKING\b", text):
            block += ("\n" if block else "") + "from typing import TYPE_CHECKING"
        block += "\n\nif TYPE_CHECKING:\n" + "\n".join("    " + t for t in tc)
    if not block:
        return text
    tree = ast.parse(text)
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            last = node.end_lineno
        elif last:
            break
    lines = text.splitlines()
    lines[last:last] = ["", *block.splitlines()]
    return "\n".join(lines) + "\n"


def process(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    edits = []
    need: set = set()
    parents = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parents[id(ch)] = node
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        in_method = isinstance(parents.get(id(node)), ast.ClassDef)
        args = node.args
        positional = list(args.posonlyargs) + list(args.args)
        all_args = positional + list(args.kwonlyargs)
        missing = any(a.annotation is None and not (i == 0 and in_method and a.arg in ("self", "cls")) for i, a in enumerate(all_args))
        missing = missing or bool(args.vararg and args.vararg.annotation is None) or bool(args.kwarg and args.kwarg.annotation is None)
        new_ret = return_annotation(node)
        if not missing and new_ret is None:
            continue
        for i, a in enumerate(positional):
            annotate_param(a, i == 0, in_method)
        for a in args.kwonlyargs:
            annotate_param(a, False, in_method)
        if args.vararg and args.vararg.annotation is None:
            args.vararg.annotation = ast.parse("Any", mode="eval").body
        if args.kwarg and args.kwarg.annotation is None:
            args.kwarg.annotation = ast.parse("Any", mode="eval").body
        if new_ret is not None:
            node.returns = ast.parse(new_ret, mode="eval").body
        start, end = node.lineno - 1, node.body[0].lineno - 1
        sig_end = None
        for i in range(start, end + 1):
            code = re.sub(r"\s+#.*$", "", lines[i]).rstrip()
            if code.endswith(":"):
                sig_end = i
                break
        if sig_end is None:
            continue
        m = re.search(r"(\s+#.*)$", lines[sig_end])
        comment = m.group(1) if m else ""
        indent = re.match(r"\s*", lines[start]).group(0)
        stub = ast.FunctionDef(node.name, node.args, [ast.Pass()], [], node.returns, None, lineno=1, col_offset=0)
        sig = ast.unparse(stub).splitlines()[0]
        edits.append((start, sig_end, indent + sig + comment))
        for name in ("Simulation", "SimulationConfig", "RunArchive", "PathLike", *TYPING_NAMES):
            if re.search(rf"\b{name}\b", sig):
                need.add(name)
    if not edits:
        return 0
    for s, e, new in sorted(edits, reverse=True):
        lines[s : e + 1] = [new]
    text = add_imports("\n".join(lines) + "\n", need, path)
    path.write_text(text, encoding="utf-8")
    return len(edits)


if __name__ == "__main__":
    total = 0
    for p in sys.argv[1:]:
        n = process(Path(p))
        total += n
        print(f"{p}: annotated {n} signature(s)")
    print("total", total)
