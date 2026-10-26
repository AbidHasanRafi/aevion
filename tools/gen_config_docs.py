"""Generate the configuration reference inside DOCUMENTATION.md and docs/simulation.schema.json from the configuration dataclasses.

Only the text between the ``config-reference`` markers in DOCUMENTATION.md is rewritten.

Run from the repository root:  python tools/gen_config_docs.py   (``--check`` verifies without writing)
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from aevion.config import SimulationConfig, json_schema
from aevion.config import blocks as B
from aevion.config.schema import Block, _hints, _tname

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTATION = ROOT / "DOCUMENTATION.md"
START = "<!-- config-reference:start -->"
END = "<!-- config-reference:end -->"
ORDER = [
    B.SimulationConfig,
    B.EngineConfig,
    B.WorldConfig,
    B.WindConfig,
    B.VehicleConfig,
    B.InitialConfig,
    B.FormationConfig,
    B.ControllerConfig,
    B.TrajectoryConfig,
    B.SensorConfig,
    B.EstimatorConfig,
    B.NetworkConfig,
    B.SwarmConfig,
    B.CommModelConfig,
    B.FaultConfig,
    B.HookConfig,
    B.HealthConfig,
    B.LoggingConfig,
]


def default_repr(f):
    if f.default is not dataclasses.MISSING:
        v = f.default
        return "`inf`" if isinstance(v, float) and math.isinf(v) else f"`{json.dumps(v)}`" if v is not None else "`null`"
    if f.default_factory is not dataclasses.MISSING:
        v = f.default_factory()
        if isinstance(v, Block):
            return f"*{type(v).__name__}*"
        return f"`{json.dumps(v)}`"
    return "required"


def constraints(f):
    m = f.metadata
    out = []
    for k, sym in (("gt", ">"), ("ge", "≥"), ("lt", "<"), ("le", "≤")):
        if k in m:
            out.append(f"{sym} {m[k]}")
    if "choices" in m:
        out.append("one of " + ", ".join(map(str, m["choices"])))
    if "min_len" in m:
        out.append(f"len ≥ {m['min_len']}")
    return "; ".join(out)


def block_section(cls):
    hints = _hints(cls)
    lines = [
        f"#### `{cls.__name__}`",
        "",
        (cls.__doc__ or "").strip().split("\n")[0],
        "",
        "| field | type | default | constraints | description |",
        "|---|---|---|---|---|",
    ]
    for f in dataclasses.fields(cls):
        doc = f.metadata.get("doc", "")
        unit = f" [{f.metadata['unit']}]" if f.metadata.get("unit") else ""
        t = _tname(hints[f.name]).replace("|", "\\|")
        lines.append(f"| `{f.name}` | {t} | {default_repr(f)} | {constraints(f)} | {doc}{unit} |")
    if issubclass(cls, B.ComponentConfig) and cls.registry_kind:
        lines += [
            "",
            f"*Component type is chosen from the `{cls.registry_kind}` registry (`aevion registry {cls.registry_kind}`); remaining keys are the component's own parameters.*",
        ]
    return "\n".join(lines)


def main(argv=None):
    import sys

    from aevion.registry import registries

    argv = sys.argv[1:] if argv is None else argv
    parts = [
        "## Configuration reference (schema version 1)",
        "",
        "*Generated from the code by `tools/gen_config_docs.py`: do not edit by hand.*",
        "",
        "Write `schema_version: 1` at the top of hand-written files. Components accept a bare name (`controller: geometric`) or a mapping with a ",
        "`type` (alias `kind`) key plus parameters. Machine-readable version: [simulation.schema.json](docs/simulation.schema.json).",
        "",
    ]
    for cls in ORDER:
        parts += [block_section(cls), ""]
    parts += ["### Registered components", "", "```text", registries.describe(), "```"]
    section = "\n".join([START, *parts, END])
    schema = json.dumps(json_schema(SimulationConfig, "Aevion simulation configuration"), indent=2)
    schema_path = ROOT / "docs" / "simulation.schema.json"
    guide = DOCUMENTATION.read_text(encoding="utf-8")
    head, found, rest = guide.partition(START)
    if not found or END not in rest:
        print(f"{DOCUMENTATION.name} has no {START} ... {END} section", file=sys.stderr)
        return 1
    new_guide = head + section + rest.partition(END)[2]
    if "--check" in argv:
        stale = []
        if new_guide != guide:
            stale.append(DOCUMENTATION.name)
        if not schema_path.exists() or schema_path.read_text(encoding="utf-8") != schema:
            stale.append(schema_path.name)
        if stale:
            print(f"out of date: {', '.join(stale)}  (run: python tools/gen_config_docs.py)", file=sys.stderr)
            return 1
        print("generated docs are up to date")
        return 0
    DOCUMENTATION.write_text(new_guide, encoding="utf-8", newline="\n")
    schema_path.write_text(schema, encoding="utf-8")
    print("wrote the configuration reference in DOCUMENTATION.md and docs/simulation.schema.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
