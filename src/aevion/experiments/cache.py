"""A content-addressed cache of trial results, so an interrupted experiment resumes where it stopped and a repeated one costs nothing.

A trial's key is the BLAKE2b hash of its canonical JSON description (the full configuration, the metric and the Aevion version), so *any* change to a trial
yields a new key and a stale result can never be returned for a changed configuration. Entries are written atomically (a temporary file in the same
directory, then ``os.replace``), so a crash cannot leave a half-written file; a damaged or unreadable entry is treated as a miss and recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

__all__ = ["canonical_json", "trial_key", "ResultCache"]


def _plain(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return repr(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    raise TypeError(f"cannot hash a value of type {type(o).__name__}: configurations must be plain data")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace, NumPy values converted) - the same data always gives the same text."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"))


def trial_key(spec: Dict[str, Any], salt: str = "") -> str:
    """32-hex-digit BLAKE2b digest of the canonical description (plus an optional ``salt`` such as a code version)."""
    return hashlib.blake2b((salt + "\n" + canonical_json(spec)).encode("utf-8"), digest_size=16).hexdigest()


class ResultCache:
    """Directory of JSON entries ``<dir>/<key[:2]>/<key>.json``; safe to share between processes (writes are atomic, entries immutable)."""

    def __init__(self, directory: Any) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = self.writes = self.corrupt = 0

    def path(self, key: str) -> Path:
        if not key or any(c not in "0123456789abcdef" for c in key):
            raise ValueError(f"not a cache key: {key!r}")
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self.path(key)
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
            if blob.get("key") != key or "row" not in blob:
                raise ValueError("entry does not belong to this key")
        except FileNotFoundError:
            self.misses += 1
            return None
        except (OSError, ValueError, json.JSONDecodeError):
            self.corrupt += 1
            self.misses += 1
            return None
        self.hits += 1
        return blob["row"]

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def put(self, key: str, row: Dict[str, Any]) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(dict(key=key, created=time.time(), row=json.loads(canonical_json(row))), sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self.writes += 1
        return p

    def keys(self) -> Iterator[str]:
        for p in sorted(self.dir.glob("??/*.json")):
            yield p.stem

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())

    def clear(self) -> int:
        n = 0
        for p in list(self.dir.glob("??/*.json")):
            p.unlink()
            n += 1
        return n

    def merge_from(self, other: "ResultCache") -> int:
        """Copy the entries of another cache (e.g. one per cluster node) that this one lacks; returns how many were added."""
        added = 0
        for k in other:
            if not self.has(k):
                row = other.get(k)
                if row is not None:
                    self.put(k, row)
                    added += 1
        return added

    def stats(self) -> Dict[str, int]:
        return dict(hits=self.hits, misses=self.misses, writes=self.writes, corrupt=self.corrupt, entries=len(self))
