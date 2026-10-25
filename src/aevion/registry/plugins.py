"""Plugin discovery through the ``aevion.plugins`` entry-point group.

A third-party package declares in its ``pyproject.toml``::

    [project.entry-points."aevion.plugins"]
    my_plugin = "my_package.aevion_plugin:register"

where ``register(registries)`` receives the :class:`Registries` object and calls e.g.
``registries.sensors.register("sonar", Sonar, source="my_package")``. Failures in one plugin are
isolated: they are recorded (and warned about) but never prevent Aevion or other plugins from loading.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

GROUP = "aevion.plugins"


@dataclass
class PluginRecord:
    name: str
    target: str
    loaded: bool
    error: Optional[str] = None
    version: Optional[str] = None


_records: List[PluginRecord] = []
_loaded = False


def plugins_disabled() -> bool:
    """``AEVION_DISABLE_PLUGINS=1`` skips all third-party plugin code (use it for untrusted or reproducibility-critical runs)."""
    return os.environ.get("AEVION_DISABLE_PLUGINS", "").strip().lower() in ("1", "true", "yes", "on")


def _entry_points() -> Any:
    try:
        eps = md.entry_points()
        return list(eps.select(group=GROUP)) if hasattr(eps, "select") else list(eps.get(GROUP, []))
    except Exception:  # pragma: no cover - metadata backend problems must not break imports
        return []


def load_plugin_function(name: str, fn: Callable, registries: Any, source: Optional[str] = None, version: Optional[str] = None) -> PluginRecord:
    """Run one plugin ``register`` function with error isolation."""
    rec = PluginRecord(name, getattr(fn, "__module__", "?") + "." + getattr(fn, "__name__", "register"), False, None, version)
    try:
        fn(registries)
        rec.loaded = True
    except Exception as ex:
        rec.error = f"{type(ex).__name__}: {ex}"
        warnings.warn(f"aevion plugin '{name}' failed to load: {rec.error}", stacklevel=2)
    _records.append(rec)
    return rec


def discover(registries: Any, force: bool = False) -> List[PluginRecord]:
    """Load all installed plugins once (idempotent unless ``force``)."""
    global _loaded
    if _loaded and not force:
        return list(_records)
    _loaded = True
    if plugins_disabled():
        return list(_records)
    for ep in _entry_points():
        version = None
        try:
            version = ep.dist.version if ep.dist is not None else None
        except Exception:
            pass
        try:
            fn = ep.load()
        except Exception as ex:
            rec = PluginRecord(ep.name, ep.value, False, f"{type(ex).__name__}: {ex}", version)
            _records.append(rec)
            warnings.warn(f"aevion plugin '{ep.name}' could not be imported: {rec.error}", stacklevel=2)
            continue
        load_plugin_function(ep.name, fn, registries, version=version)
    return list(_records)


def records() -> List[PluginRecord]:
    return list(_records)
