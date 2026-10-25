"""Signature introspection used to validate component parameters before construction."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Set


def accepted_params(factory: Callable, skip_positional: int = 0) -> Optional[Set[str]]:
    """Names of keyword parameters ``factory`` accepts, or ``None`` if it accepts arbitrary keywords.

    For classes the ``__init__`` chain along the MRO is followed while each ``__init__`` forwards ``**kwargs``.
    ``skip_positional`` drops that many leading positional parameters (context arguments such as ``sim``).
    """

    def params_of(fn: Any, skip: Any) -> Any:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return set(), True
        names, var_kw, i = set(), False, 0
        for p in sig.parameters.values():
            if p.name == "self":
                continue
            if p.kind is p.VAR_KEYWORD:
                var_kw = True
            elif p.kind is p.VAR_POSITIONAL:
                continue
            else:
                if i < skip:
                    i += 1
                    continue
                names.add(p.name)
        return names, var_kw

    if inspect.isclass(factory):
        names: Set[str] = set()
        first = True
        for cls in factory.__mro__:
            init = cls.__dict__.get("__init__")
            if init is None or cls is object:
                continue
            n, var_kw = params_of(init, skip_positional if first else 0)
            first = False
            names |= n
            if not var_kw:
                return names
        return None
    n, var_kw = params_of(factory, skip_positional)
    return None if var_kw else n
