"""Dataclass-based configuration schema with path-precise validation.

Declare configuration as dataclasses deriving from :class:`Block`. ``Block.from_dict`` validates input
and collects **all** problems (wrong types, out-of-range values, unknown keys with "did you mean"
suggestions, missing required fields) into one :class:`~aevion.errors.ConfigError`, each with the dotted
path of the offending field (``sensors[2].rate``). :func:`json_schema` exports a JSON-Schema document so
editors can validate and auto-complete scenario files.

Field metadata understood: ``gt, ge, lt, le, choices, min_len, max_len, doc, unit``.
"""

from __future__ import annotations

import dataclasses
import difflib
import enum
import typing
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union, get_args, get_origin

from ..errors import ConfigError, ConfigIssue

__all__ = ["Block", "field", "spec", "json_schema", "parse", "ConfigError", "ConfigIssue"]


def spec(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    choices: Any = None,
    min_len: Any = None,
    max_len: Any = None,
    doc: Any = "",
    unit: Any = "",
    **kw: Any,
) -> Any:
    """``dataclasses.field`` with validation metadata."""
    meta = {
        k: v
        for k, v in dict(gt=gt, ge=ge, lt=lt, le=le, choices=choices, min_len=min_len, max_len=max_len, doc=doc, unit=unit).items()
        if v not in (None, "")
    }
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=meta, **kw)
    if default is not MISSING:
        return field(default=default, metadata=meta, **kw)
    return field(metadata=meta, **kw)


_hint_cache: Dict[type, Dict[str, Any]] = {}


def _hints(cls: Any) -> Dict[str, Any]:
    if cls not in _hint_cache:
        _hint_cache[cls] = typing.get_type_hints(cls)
    return _hint_cache[cls]


def _tname(tp: Any) -> str:
    if tp is Any:
        return "any value"
    origin = get_origin(tp)
    if origin is Union:
        return " or ".join(_tname(a) for a in get_args(tp))
    if origin is Literal:
        return "one of " + ", ".join(repr(a) for a in get_args(tp))
    if origin in (list, List):
        return f"a list of {_tname(get_args(tp)[0])}" if get_args(tp) else "a list"
    if origin in (tuple, Tuple):
        return "a list"
    if origin in (dict, Dict):
        return "a mapping"
    if isinstance(tp, type):
        if issubclass(tp, Block):
            return f"a {tp.__name__} mapping"
        return {int: "an integer", float: "a number", str: "a string", bool: "true/false", type(None): "null"}.get(tp, tp.__name__)
    return str(tp)


def _join(path: str, key: Union[int, str]) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _suggest(key: str, options: Any) -> Optional[str]:
    m = difflib.get_close_matches(str(key), list(options), n=1, cutoff=0.6)
    return f"did you mean '{m[0]}'?" if m else None


def _check_meta(value: Any, meta: Mapping[str, Any], path: str, issues: List[ConfigIssue]) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for k, op, sym in (
            ("gt", lambda a, b: a > b, ">"),
            ("ge", lambda a, b: a >= b, ">="),
            ("lt", lambda a, b: a < b, "<"),
            ("le", lambda a, b: a <= b, "<="),
        ):
            if k in meta and not op(value, meta[k]):
                unit = f" {meta['unit']}" if meta.get("unit") else ""
                issues.append(ConfigIssue(path, f"must be {sym} {meta[k]}{unit}, got {value}"))
    if "choices" in meta and value not in meta["choices"]:
        issues.append(
            ConfigIssue(
                path,
                f"must be one of {list(meta['choices'])}, got {value!r}",
                _suggest(value, [str(c) for c in meta["choices"]]) if isinstance(value, str) else None,
            )
        )
    if isinstance(value, (list, tuple, dict, str)):
        if "min_len" in meta and len(value) < meta["min_len"]:
            issues.append(ConfigIssue(path, f"needs at least {meta['min_len']} item(s), got {len(value)}"))
        if "max_len" in meta and len(value) > meta["max_len"]:
            issues.append(ConfigIssue(path, f"allows at most {meta['max_len']} item(s), got {len(value)}"))


def _parse(tp: Any, value: Any, path: str, issues: List[ConfigIssue]) -> Any:
    """Validate/convert ``value`` to type ``tp``, appending problems to ``issues``."""
    if tp is Any:
        return value
    if type(value).__module__ == "numpy":  # arrays/scalars built in code: treat as plain lists / numbers
        value = value.tolist() if hasattr(value, "tolist") else value
    origin, args = get_origin(tp), get_args(tp)

    if origin is Union:
        if value is None and type(None) in args:
            return None
        cands = [a for a in args if a is not type(None)]
        best_issues, best_n = None, 10**9
        for a in cands:
            local: List[ConfigIssue] = []
            out = _parse(a, value, path, local)
            if not local:
                return out
            if len(local) < best_n:
                best_issues, best_n = local, len(local)
        if len(cands) == 1:
            issues.extend(best_issues or [])
        else:
            issues.append(ConfigIssue(path, f"expected {_tname(tp)}, got {type(value).__name__}"))
        return value

    if value is None:
        issues.append(ConfigIssue(path, f"expected {_tname(tp)}, got null"))
        return None

    if origin is Literal:
        if value not in args:
            issues.append(
                ConfigIssue(
                    path, f"must be one of {list(args)}, got {value!r}", _suggest(value, [str(a) for a in args]) if isinstance(value, str) else None
                )
            )
        return value

    if isinstance(tp, type) and issubclass(tp, Block):
        return tp._from_value(value, path, issues)

    if origin in (list, List, tuple, Tuple, typing.Sequence):
        if not isinstance(value, (list, tuple)):
            issues.append(ConfigIssue(path, f"expected {_tname(tp)}, got {type(value).__name__}"))
            return value
        if origin in (tuple, Tuple) and args and not (len(args) == 2 and args[1] is Ellipsis):
            if len(value) != len(args):
                issues.append(ConfigIssue(path, f"expected {len(args)} values, got {len(value)}"))
                return value
            return tuple(_parse(a, v, _join(path, i), issues) for i, (a, v) in enumerate(zip(args, value)))
        et = args[0] if args else Any
        out = [_parse(et, v, _join(path, i), issues) for i, v in enumerate(value)]
        return tuple(out) if origin in (tuple, Tuple) else out

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            issues.append(ConfigIssue(path, f"expected a mapping, got {type(value).__name__}"))
            return value
        vt = args[1] if len(args) == 2 else Any
        return {k: _parse(vt, v, _join(path, k), issues) for k, v in value.items()}

    if tp is bool:
        if not isinstance(value, bool):
            issues.append(ConfigIssue(path, f"expected true/false, got {value!r}"))
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not value.is_integer()):
            issues.append(ConfigIssue(path, f"expected an integer, got {value!r}"))
            return value
        return int(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(ConfigIssue(path, f"expected a number, got {value!r}"))
            return value
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            issues.append(ConfigIssue(path, f"expected a string, got {value!r}"))
        return value
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(value)
        except ValueError:
            issues.append(ConfigIssue(path, f"must be one of {[e.value for e in tp]}, got {value!r}"))
            return value
    return value


@dataclass
class Block:
    """Base class of every configuration section."""

    # per-subclass hooks ---------------------------------------------------
    @classmethod
    def _shorthand(cls, value: Any) -> Optional[dict]:
        """Convert a non-mapping shorthand (e.g. a bare string) into a mapping, or return ``None``."""
        return None

    def _validate(self, path: str, issues: List[ConfigIssue]) -> None:
        """Cross-field checks (override)."""

    # ----------------------------------------------------------------------
    @classmethod
    def _from_value(cls, value: Any, path: str, issues: List[ConfigIssue]) -> Any:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            sh = cls._shorthand(value)
            if sh is None:
                issues.append(ConfigIssue(path, f"expected {_tname(cls)}, got {type(value).__name__}"))
                return None
            value = sh
        hints = _hints(cls)
        flds = {f.name: f for f in fields(cls)}
        kwargs = {}
        n_before = len(issues)
        for k in value:
            if k not in flds:
                issues.append(ConfigIssue(_join(path, k), "unknown field", _suggest(k, flds)))
        for name, f in flds.items():
            if name in value:
                v = _parse(hints[name], value[name], _join(path, name), issues)
                _check_meta(v, f.metadata, _join(path, name), issues)
                kwargs[name] = v
            elif f.default is MISSING and f.default_factory is MISSING:
                issues.append(ConfigIssue(_join(path, name), "required field is missing"))
        try:
            obj = cls(**dict(kwargs.items()))
        except TypeError:
            return None
        if len(issues) == n_before:  # cross-field checks assume well-typed fields; report the type problems first
            obj._validate(path, issues)
        return obj

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Optional[str] = None) -> Any:
        """Validate ``data`` and build the config, raising :class:`ConfigError` listing every problem."""
        issues: List[ConfigIssue] = []
        obj = cls._from_value(data, "", issues)
        if issues:
            raise ConfigError(issues, source)
        return obj

    @classmethod
    def validate_dict(cls, data: Dict[str, Any]) -> List[ConfigIssue]:
        """Return the list of problems without raising."""
        issues: List[ConfigIssue] = []
        cls._from_value(data, "", issues)
        return issues

    def to_dict(self, skip_defaults: bool = False) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if skip_defaults:
                if f.default is not MISSING and v == f.default:
                    continue
                if f.default_factory is not MISSING and v == f.default_factory():
                    continue
            out[f.name] = _dump(v, skip_defaults)
        return out

    def replace(self, **changes: Any) -> Any:
        return dataclasses.replace(self, **changes)


def _dump(v: Any, skip_defaults: Any) -> Any:
    if isinstance(v, Block):
        return v.to_dict(skip_defaults)
    if isinstance(v, (list, tuple)):
        return [_dump(x, skip_defaults) for x in v]
    if isinstance(v, dict):
        return {k: _dump(x, skip_defaults) for k, x in v.items()}
    if isinstance(v, enum.Enum):
        return v.value
    return v


def parse(cls: Any, data: Dict[str, Any], source: Optional[str] = None) -> Any:
    return cls.from_dict(data, source)


# ---------------------------------------------------------------------------- JSON Schema
def _schema_for(tp: Any, defs: dict) -> dict:
    if tp is Any:
        return {}
    origin, args = get_origin(tp), get_args(tp)
    if origin is Union:
        return {"anyOf": [_schema_for(a, defs) for a in args]}
    if origin is Literal:
        return {"enum": list(args)}
    if isinstance(tp, type) and issubclass(tp, Block):
        name = tp.__name__
        if name not in defs:
            defs[name] = {}  # break recursion
            defs[name] = _block_schema(tp, defs)
        return {"$ref": f"#/$defs/{name}"}
    if origin in (list, List, typing.Sequence):
        return {"type": "array", "items": _schema_for(args[0], defs) if args else {}}
    if origin in (tuple, Tuple):
        return {"type": "array", "items": _schema_for(args[0], defs) if args else {}}
    if origin in (dict, Dict):
        return {"type": "object", "additionalProperties": _schema_for(args[1], defs) if len(args) == 2 else True}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if tp is type(None):
        return {"type": "null"}
    return {}


def _block_schema(cls: Any, defs: Any) -> dict:
    hints = _hints(cls)
    props, required = {}, []
    for f in fields(cls):
        s = _schema_for(hints[f.name], defs)
        m = f.metadata
        for k, js in (
            ("gt", "exclusiveMinimum"),
            ("ge", "minimum"),
            ("lt", "exclusiveMaximum"),
            ("le", "maximum"),
            ("min_len", "minItems"),
            ("max_len", "maxItems"),
        ):
            if k in m:
                s = {**s, js: m[k]}
        if "choices" in m:
            s = {**s, "enum": list(m["choices"])}
        if m.get("doc"):
            s = {**s, "description": m["doc"] + (f" [{m['unit']}]" if m.get("unit") else "")}
        if f.default is not MISSING:
            try:
                import json

                json.dumps(f.default)
                s = {**s, "default": f.default}
            except TypeError:
                pass
        else:
            if f.default_factory is MISSING:
                required.append(f.name)
        props[f.name] = s
    out = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        out["required"] = required
    if cls.__doc__:
        out["description"] = cls.__doc__.strip().split("\n")[0]
    return out


def json_schema(cls: Any, title: Optional[str] = None) -> dict:
    """JSON-Schema (draft 2020-12) for a :class:`Block` subclass."""
    defs: dict = {}
    top = _schema_for(cls, defs)
    ref = top["$ref"].split("/")[-1]
    doc = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title or cls.__name__, **defs[ref]}
    others = {k: v for k, v in defs.items() if k != ref}
    if others:
        doc["$defs"] = others
    return doc
