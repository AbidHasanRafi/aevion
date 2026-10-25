"""Loading, saving and hashing configuration files (YAML or JSON)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

from ..errors import ConfigError, ConfigIssue
from .blocks import EnvironmentConfig, SimulationConfig
from .migrate import migrate

PathLike = Union[str, Path]


_YAML_LOADER: Any = None


def yaml_loads(text: str) -> Any:
    """``yaml.safe_load`` that also reads ``3.5e9``, ``1e-3`` and ``2E6`` as floats.

    PyYAML follows YAML 1.1, where a float needs a dot *and* a signed exponent (``3.5e+9``); ``3.5e9`` silently becomes a *string*,
    a classic trap in engineering configurations. This loader is otherwise identical to ``safe_load`` (no tags, no code).
    """
    global _YAML_LOADER
    import re

    import yaml

    if _YAML_LOADER is None:

        class _Loader(yaml.SafeLoader):  # type: ignore[misc, name-defined]
            pass

        _Loader.add_implicit_resolver(
            "tag:yaml.org,2002:float", re.compile(r"^[-+]?(?:[0-9][0-9_]*\.?[0-9_]*|\.[0-9_]+)[eE][-+]?[0-9]+$"), list("-+0123456789.")
        )
        _YAML_LOADER = _Loader
    loader = _YAML_LOADER(text)  # a SafeLoader subclass: it cannot construct arbitrary Python objects
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf not in (".yaml", ".yml", ".json"):
        raise ConfigError([ConfigIssue("<file>", f"unsupported extension '{suf}'", "use .yaml, .yml or .json")], str(path))
    try:
        if suf == ".json":
            data = json.loads(text)
        else:
            data = yaml_loads(text)
    except Exception as ex:  # parse errors carry line information already
        raise ConfigError([ConfigIssue("<file>", f"cannot parse: {ex}")], str(path)) from ex
    if not isinstance(data, dict):
        raise ConfigError([ConfigIssue("<root>", f"top level must be a mapping, got {type(data).__name__}")], str(path))
    return data


def load_config(source: Union[PathLike, dict]) -> SimulationConfig:
    """Load, **migrate** and validate a configuration (file path or dict)."""
    name = None
    if isinstance(source, (str, Path)):
        name = str(source)
        data = _read(Path(source))
    else:
        data = dict(source)
    try:
        data = migrate(data)
    except ValueError as ex:
        raise ConfigError([ConfigIssue("schema_version", str(ex))], name) from ex
    except (TypeError, KeyError, AttributeError) as ex:  # a malformed legacy scenario
        raise ConfigError(
            [ConfigIssue("<root>", f"cannot migrate this configuration to the current schema: {type(ex).__name__}: {ex}")], name
        ) from ex
    cfg = SimulationConfig.from_dict(data, name)
    if isinstance(source, (str, Path)):  # not a schema field (so not hashed or dumped): lets relative input files resolve next to the config
        object.__setattr__(cfg, "_source_dir", str(Path(source).resolve().parent))
    return cfg


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dump_config(cfg: SimulationConfig, path: PathLike, skip_defaults: bool = False) -> Path:
    path = Path(path)
    d = cfg.to_dict(skip_defaults)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        path.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=None, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(d, indent=2, default=_default), encoding="utf-8")
    return path


def canonical_json(cfg: Union[SimulationConfig, dict]) -> str:
    d = cfg.to_dict() if isinstance(cfg, SimulationConfig) else cfg
    if isinstance(d, dict) and d.get("environment") == EnvironmentConfig().to_dict():  # configurations that do not use it keep their pre-0.5 hash
        d = {k: v for k, v in d.items() if k != "environment"}
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=_default)


def config_hash(cfg: Union[SimulationConfig, dict], length: int = 16) -> str:
    """Stable content hash of a configuration (independent of key order and file format)."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()[:length]
