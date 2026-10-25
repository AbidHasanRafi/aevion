"""Check or update the public-API snapshot (tests/golden/public_api.json).

python tools/api_snapshot.py            # report breaking changes and additions (exit 1 if any)
python tools/api_snapshot.py --update   # accept the current API as the new snapshot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aevion._api import compare, snapshot

PATH = Path(__file__).resolve().parents[1] / "tests" / "golden" / "public_api.json"


def main(argv: list[str]) -> int:
    new = snapshot()
    if "--update" in argv:
        PATH.write_text(json.dumps(new, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {PATH} ({sum(len(v) for v in new.values())} public names)")
        return 0
    old = json.loads(PATH.read_text(encoding="utf-8"))
    breaking, additions = compare(old, new)
    for b in breaking:
        print("BREAKING:", b)
    for a in additions:
        print("added:   ", a)
    if not breaking and not additions:
        print("public API unchanged")
    return 1 if (breaking or additions) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
