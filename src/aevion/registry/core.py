"""Typed registries: name → factory, with aliases, metadata and helpful errors.

A registry maps a short string (``"geometric"``) to a *factory* (a class or function). Scenarios and
config files refer to components by name, and third-party packages add their own through the
``aevion.plugins`` entry-point group (see :mod:`aevion.registry.plugins`).
"""

from __future__ import annotations

import difflib
import inspect
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..errors import RegistryError


@dataclass
class Entry:
    name: str
    factory: Callable
    doc: str = ""
    aliases: tuple = ()
    config: Optional[type] = None  # optional dataclass describing the factory's parameters
    meta: Dict[str, Any] = field(default_factory=dict)
    source: str = "builtin"  # "builtin" or the plugin that registered it


def _norm(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


class Registry:
    """A named collection of factories.

    >>> controllers = Registry("controllers")
    >>> @controllers.register("pid", aliases=("PID",))
    ... def make_pid(sim, **kw): ...
    >>> controllers.create("pid", sim)
    """

    def __init__(self, kind: str, base: Optional[type] = None) -> None:
        self.kind = kind
        self.base = base  # optional interface that created objects should satisfy (checked loosely)
        self._entries: Dict[str, Entry] = {}
        self._aliases: Dict[str, str] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ registration
    def register(
        self,
        name: str,
        factory: Optional[Callable] = None,
        *,
        aliases: Iterable[str] = (),
        doc: str = "",
        config: Optional[type] = None,
        replace: bool = False,
        source: str = "builtin",
        **meta: Any,
    ) -> Any:
        """Register ``factory`` under ``name``. Usable as a decorator when ``factory`` is omitted."""
        if factory is None:

            def deco(f: Any) -> Any:
                self.register(name, f, aliases=aliases, doc=doc, config=config, replace=replace, source=source, **meta)
                return f

            return deco
        key = _norm(name)
        with self._lock:
            if key in self._entries or key in self._aliases:
                if not replace:
                    existing = self._entries.get(key)
                    by = existing.source if existing is not None else f"alias of '{self._aliases.get(key)}'"
                    raise RegistryError(f"{self.kind}: '{name}' is already registered (by {by}); use replace=True to override")
                self._drop(key)
            entry = Entry(
                key, factory, doc or (inspect.getdoc(factory) or "").split("\n")[0], tuple(_norm(a) for a in aliases), config, dict(meta), source
            )
            self._entries[key] = entry
            for a in entry.aliases:
                if a in self._entries or (a in self._aliases and self._aliases[a] != key):
                    raise RegistryError(f"{self.kind}: alias '{a}' collides with an existing entry")
                self._aliases[a] = key
        return factory

    def _drop(self, key: str) -> None:
        e = self._entries.pop(key, None)
        if e:
            for a in e.aliases:
                self._aliases.pop(a, None)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._drop(self._resolve(name))

    # ----------------------------------------------------------------- lookup
    def _resolve(self, name: str) -> str:
        key = _norm(name)
        key = self._aliases.get(key, key)
        if key not in self._entries:
            close = difflib.get_close_matches(key, list(self._entries) + list(self._aliases), n=3, cutoff=0.5)
            hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
            raise RegistryError(f"unknown {self.kind} '{name}'.{hint} Available: {', '.join(sorted(self._entries)) or '(none)'}")
        return key

    def get(self, name: str) -> Callable:
        return self._entries[self._resolve(name)].factory

    def entry(self, name: str) -> Entry:
        return self._entries[self._resolve(name)]

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the named component, wrapping constructor errors with context."""
        e = self.entry(name)
        defaults = e.meta.get("defaults")
        if defaults:
            kwargs = {**defaults, **kwargs}  # entry defaults sit below anything the caller sets
        try:
            return e.factory(*args, **kwargs)
        except TypeError as ex:
            sig = ""
            try:
                sig = f" Signature: {e.name}{inspect.signature(e.factory)}"
            except (TypeError, ValueError):
                pass
            raise RegistryError(f"cannot create {self.kind} '{name}': {ex}.{sig}") from ex

    def build(self, spec: Any, *args: Any, **defaults: Any) -> Any:
        """Create from a spec: a name (``"imu"``) or a mapping with a ``type`` (or ``kind``) key plus parameters.

        ``defaults`` are used for parameters the spec does not set (explicit spec values win).
        """
        if isinstance(spec, str):
            return self.create(spec, *args, **defaults)
        if not isinstance(spec, dict):
            raise RegistryError(f"{self.kind}: expected a name or a mapping with a 'type' key, got {type(spec).__name__}")
        params = dict(spec)
        name = params.pop("type", None) or params.pop("kind", None)
        if name is None:
            raise RegistryError(f"{self.kind}: spec {spec!r} has no 'type' key")
        params.pop("kind", None)
        return self.create(name, *args, **{**defaults, **params})

    def __contains__(self, name: str) -> bool:
        try:
            self._resolve(name)
            return True
        except RegistryError:
            return False

    def names(self) -> List[str]:
        return sorted(self._entries)

    def __iter__(self) -> Any:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[Entry]:
        return [self._entries[n] for n in self.names()]

    def describe(self) -> str:
        w = max((len(n) for n in self._entries), default=4)
        lines = [f"{self.kind} ({len(self)}):"]
        for e in self.entries():
            al = f" [{', '.join(e.aliases)}]" if e.aliases else ""
            src = "" if e.source == "builtin" else f"  <{e.source}>"
            lines.append(f"  {e.name:<{w}}{al}  {e.doc}{src}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self)} entries)"
