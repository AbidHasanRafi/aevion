"""Deprecation machinery implementing the versioning policy in ``DOCUMENTATION.md``.

* A deprecated feature keeps working and emits :class:`AevionDeprecationWarning` (a ``FutureWarning``, so end users
  see it by default) for **at least two minor releases** before removal.
* Every deprecation is recorded in a registry; ``tests/test_deprecation.py`` fails the build when a feature is still
  present after the release in which it was due to be removed, so promises cannot silently lapse.
"""

from __future__ import annotations

import functools
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class AevionDeprecationWarning(FutureWarning):
    """Raised (as a warning) when a deprecated Aevion API is used."""


@dataclass(frozen=True)
class Deprecation:
    what: str
    since: str
    remove_in: str
    alternative: Optional[str] = None
    reason: str = ""

    def message(self) -> str:
        msg = f"{self.what} is deprecated since Aevion {self.since} and will be removed in {self.remove_in}."
        if self.alternative:
            msg += f" Use {self.alternative} instead."
        if self.reason:
            msg += f" ({self.reason})"
        return msg


_REGISTRY: Dict[str, Deprecation] = {}


def parse_version(v: str) -> Tuple[int, int, int]:
    """``'0.4'`` / ``'0.4.1'`` / ``'1.0.0rc1'`` → ``(major, minor, patch)`` (pre-release tags ignored)."""
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", v)
    if not m:
        raise ValueError(f"not a version string: {v!r}")
    return int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)


def minors_between(a: str, b: str) -> int:
    """Number of minor releases from ``a`` to ``b`` (same major; across majors counts as large)."""
    (ma, na, _), (mb, nb, _) = parse_version(a), parse_version(b)
    return (mb - ma) * 1000 + (nb - na)


def register(dep: Deprecation) -> Deprecation:
    if minors_between(dep.since, dep.remove_in) < 2:
        raise ValueError(f"deprecation of {dep.what}: policy requires at least two minor releases between {dep.since} and {dep.remove_in}")
    _REGISTRY[dep.what] = dep
    return dep


def list_deprecations() -> List[Deprecation]:
    return sorted(_REGISTRY.values(), key=lambda d: (parse_version(d.remove_in), d.what))


def is_expired(dep: Deprecation, current: Optional[str] = None) -> bool:
    """True when ``current`` has reached the version in which ``dep`` was due to be removed."""
    if current is None:
        from . import __version__ as package_version  # late import: avoids a cycle at package import

        current = package_version
    return parse_version(current) >= parse_version(dep.remove_in)


def warn_deprecated(what: str, since: str, remove_in: str, alternative: Optional[str] = None, reason: str = "", stacklevel: int = 3) -> None:
    """Emit the warning for ``what`` (and register it the first time)."""
    dep = _REGISTRY.get(what) or register(Deprecation(what, since, remove_in, alternative, reason))
    warnings.warn(dep.message(), AevionDeprecationWarning, stacklevel=stacklevel)


def deprecated(since: str, remove_in: str, alternative: Optional[str] = None, reason: str = "") -> Callable[[F], F]:
    """Decorator marking a function, method or class as deprecated."""

    def deco(obj: F) -> F:
        what = f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"
        register(Deprecation(what, since, remove_in, alternative, reason))

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(what, since, remove_in, alternative, reason, stacklevel=3)
            return obj(*args, **kwargs)

        wrapper.__deprecated__ = _REGISTRY[what]  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return deco


def renamed_kwarg(old: str, new: str, since: str, remove_in: str) -> Callable[[F], F]:
    """Accept a renamed keyword argument under its old name (with a warning)."""

    def deco(fn: F) -> F:
        what = f"{fn.__module__}.{fn.__qualname__}({old}=...)"
        register(Deprecation(what, since, remove_in, f"'{new}='"))

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"{fn.__qualname__}() got both '{old}' (deprecated) and '{new}'")
                warn_deprecated(what, since, remove_in, f"'{new}='", stacklevel=3)
                kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco
