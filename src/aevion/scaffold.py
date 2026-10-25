"""``aevion new-plugin``: generate a ready-to-install plugin package from the bundled template.

The template lives in ``aevion/templates/plugin`` as ``*.tmpl`` files with ``{{placeholder}}`` markers (in file
contents *and* in directory names). Nothing is executed; files are rendered by plain substitution.
"""

from __future__ import annotations

import keyword
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ._types import PathLike
from .errors import AevionError

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "plugin"
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")


class ScaffoldError(AevionError):
    """The plugin could not be generated (bad name, existing directory, ...)."""


def normalise_names(name: str) -> Dict[str, str]:
    """Derive the distribution name, import package, component name and class name from what the user typed."""
    if not _NAME.fullmatch(name or ""):
        raise ScaffoldError(f"invalid plugin name {name!r}: use letters, digits, '-', '_' or '.', starting with a letter")
    dist = re.sub(r"[-_.]+", "-", name).lower()
    package = dist.replace("-", "_")
    if keyword.iskeyword(package) or package in sys.stdlib_module_names or package in ("aevion", "test", "tests"):
        raise ScaffoldError(f"'{package}' would shadow an existing module; choose another name")
    component = package.removeprefix("aevion_").removesuffix("_plugin") or package
    cls = "".join(p.capitalize() for p in component.split("_")) or "Component"
    if not cls.isidentifier():
        cls = "Plugin" + cls
    return dict(dist_name=dist, package=package, component=component, class_name=cls)


def render(text: str, values: Dict[str, str]) -> str:
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def create_plugin(name: str, dest: Optional[PathLike] = None, description: Optional[str] = None, force: bool = False) -> List[Path]:
    """Write the plugin project to ``dest`` (default ``./<dist-name>``) and return the created files."""
    from . import __version__

    values = normalise_names(name)
    values["description"] = (description or f"Aevion plugin: {values['component']} components").replace('"', "'")
    values["aevion_version"] = ".".join(__version__.split(".")[:2]) + ".0"
    root = Path(dest) if dest is not None else Path.cwd() / values["dist_name"]
    if root.exists() and any(root.iterdir()) and not force:
        raise ScaffoldError(f"'{root}' already exists and is not empty (use force=True / --force to write into it)")
    if not TEMPLATE_DIR.is_dir():
        raise ScaffoldError(f"template directory missing from the installation: {TEMPLATE_DIR}")
    created: List[Path] = []
    for src in sorted(p for p in TEMPLATE_DIR.rglob("*.tmpl") if p.is_file()):
        rel = Path(render(src.relative_to(TEMPLATE_DIR).as_posix(), values))
        name_out = rel.name[: -len(".tmpl")]
        if name_out == "gitignore":
            name_out = ".gitignore"
        target = root / rel.parent / name_out
        resolved = target.resolve()
        if root.resolve() not in resolved.parents:
            raise ScaffoldError(f"refusing to write outside the destination: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(src.read_text(encoding="utf-8"), values), encoding="utf-8", newline="\n")
        created.append(target)
    return created
