"""Chunked, streaming, integrity-checked on-disk run store (``*.aevd`` directory, schema ``aevion.run/2``).

Layout::

    run.aevd/
      manifest.json                    channels, chunk lists + hashes, metadata, provenance, status
      channels/<safe-name>/000000.npy  one NumPy file per chunk of rows (time steps)

Design goals: **bounded memory** while writing (rows are buffered per channel and flushed in chunks),
**crash tolerance** (the manifest is replaced atomically after every flush, so a killed run still opens up to its
last flush with ``status == "incomplete"``), **lazy reads** (slicing reads only the needed chunks) and **integrity** (BLAKE2b per chunk, checked by :meth:`ArchiveReader.verify`).
No third-party dependency beyond NumPy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .._types import PathLike
from ..errors import ArchiveError

SCHEMA = "aevion.run/2"
MANIFEST = "manifest.json"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
MAX_MANIFEST_BYTES = 256 * 1024**2  # a manifest larger than this is not a plausible run description
MAX_CHUNK_BYTES = 4 * 1024**3
_DTYPE_KINDS = "biufc"  # bool, int, uint, float, complex: never object/string dtypes (those would need pickle)


def _safe(name: str) -> str:
    return _SAFE.sub("__", name).strip("_") or "channel"


def _hash_bytes(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ======================================================================================== writer
class ArchiveWriter:
    """Append rows to named channels; flush in chunks of ``chunk_steps`` rows.

    >>> w = ArchiveWriter("run.aevd", meta={"name": "demo"})
    >>> for t in range(1000): w.append("pos", np.zeros((1, 3)))
    >>> w.close()
    """

    def __init__(
        self, path: PathLike, meta: Optional[dict] = None, chunk_steps: int = 512, overwrite: bool = False, provenance: Optional[dict] = None
    ) -> None:
        self.path = Path(path)
        if self.path.exists():
            if not overwrite:
                raise ArchiveError(f"'{self.path}' already exists (pass overwrite=True to replace it)")
            shutil.rmtree(self.path)
        (self.path / "channels").mkdir(parents=True)
        self.chunk_steps = max(1, int(chunk_steps))
        self.meta = dict(meta or {})
        self.provenance = dict(provenance or {})
        self._buf: Dict[str, List[np.ndarray]] = {}
        self._info: Dict[str, dict] = {}
        self._dirs: Dict[str, str] = {}
        self._closed = False
        self._lock = threading.Lock()
        self.created = time.time()
        self._write_manifest("incomplete")

    # ------------------------------------------------------------------------- writing
    def _channel_dir(self, name: str) -> str:
        if name not in self._dirs:
            base = _safe(name)
            cand, i = base, 1
            while cand in self._dirs.values():
                i += 1
                cand = f"{base}-{i}"
            self._dirs[name] = cand
            (self.path / "channels" / cand).mkdir(exist_ok=True)
        return self._dirs[name]

    def append(self, name: str, row: Any) -> None:
        """Append one row (a time step) to channel ``name``. All rows of a channel must share shape and dtype."""
        if self._closed:
            raise ArchiveError("archive writer is closed")
        a = np.asarray(row)
        info = self._info.get(name)
        if info is None:
            info = self._info[name] = dict(dtype=a.dtype.str, tail=list(a.shape), length=0, chunks=[])
            self._buf[name] = []
        elif list(a.shape) != info["tail"]:
            raise ArchiveError(f"channel '{name}': row shape {list(a.shape)} != established shape {info['tail']}")
        elif a.dtype.str != info["dtype"]:
            a = a.astype(np.dtype(info["dtype"]))
        self._buf[name].append(a)
        if len(self._buf[name]) >= self.chunk_steps:
            self._flush_channel(name)
            self._write_manifest("incomplete")

    def append_many(self, rows: Dict[str, np.ndarray]) -> None:
        for k, v in rows.items():
            self.append(k, v)

    def append_block(self, name: str, block: Any) -> None:
        """Append many rows (axis 0) at once. Full chunks are written directly; a remainder stays buffered."""
        if self._closed:
            raise ArchiveError("archive writer is closed")
        a = np.asarray(block)
        if a.ndim == 0:
            raise ArchiveError("append_block needs at least one axis (rows)")
        if len(a) == 0:
            return
        info = self._info.get(name)
        if info is None:
            info = self._info[name] = dict(dtype=a.dtype.str, tail=list(a.shape[1:]), length=0, chunks=[])
            self._buf[name] = []
        elif list(a.shape[1:]) != info["tail"]:
            raise ArchiveError(f"channel '{name}': row shape {list(a.shape[1:])} != established shape {info['tail']}")
        if a.dtype.str != info["dtype"]:
            a = a.astype(np.dtype(info["dtype"]))
        i = 0
        # top up a partially filled buffer first so chunk boundaries stay regular
        while self._buf[name] and i < len(a):
            self._buf[name].append(a[i])
            i += 1
            if len(self._buf[name]) >= self.chunk_steps:
                self._flush_channel(name)
        while len(a) - i >= self.chunk_steps:
            self._write_chunk(name, a[i : i + self.chunk_steps])
            i += self.chunk_steps
        for row in a[i:]:
            self._buf[name].append(row)
        self._write_manifest("incomplete")

    def write_channel(self, name: str, data: Dict[str, Any]) -> None:
        """Write a whole array (rows along axis 0)."""
        self.append_block(name, data)

    def _write_chunk(self, name: str, arr: np.ndarray) -> None:
        info = self._info[name]
        d = self._channel_dir(name)
        idx = len(info["chunks"])
        rel = f"channels/{d}/{idx:06d}.npy"
        fp = self.path / rel
        tmp = fp.with_suffix(".tmp.npy")
        np.save(tmp, np.ascontiguousarray(arr), allow_pickle=False)
        os.replace(tmp, fp)
        info["chunks"].append(dict(file=rel, n=len(arr), hash=_hash_file(fp)))
        info["length"] += len(arr)

    def _flush_channel(self, name: str) -> None:
        buf = self._buf.get(name)
        if not buf:
            return
        self._write_chunk(name, np.stack(buf))
        self._buf[name] = []

    def flush(self) -> None:
        for name in list(self._buf):
            self._flush_channel(name)
        self._write_manifest("incomplete")

    # ---------------------------------------------------------------------------- close
    def _manifest(self, status: str) -> dict:
        return dict(
            schema=SCHEMA,
            status=status,
            created=_iso(self.created),
            meta=self.meta,
            provenance=self.provenance,
            chunk_steps=self.chunk_steps,
            channels={
                n: dict(dtype=i["dtype"], tail=i["tail"], length=i["length"], chunks=i["chunks"], dir=self._dirs.get(n))
                for n, i in self._info.items()
            },
        )

    def _write_manifest(self, status: str) -> None:
        with self._lock:
            _atomic_write(self.path / MANIFEST, json.dumps(self._manifest(status), default=_json_default).encode("utf-8"))

    def close(self, meta: Optional[dict] = None, provenance: Optional[dict] = None) -> Path:
        """Flush everything, merge extra metadata, and mark the archive complete."""
        if self._closed:
            return self.path
        for name in list(self._buf):
            self._flush_channel(name)
        if meta:
            self.meta.update(meta)
        if provenance:
            self.provenance.update(provenance)
        self._write_manifest("complete")
        self._closed = True
        return self.path

    def abort(self) -> None:
        """Flush what exists and leave the archive marked ``incomplete`` (used when a run fails)."""
        if not self._closed:
            for name in list(self._buf):
                self._flush_channel(name)
            self._write_manifest("incomplete")
            self._closed = True

    def __enter__(self) -> "ArchiveWriter":
        return self

    def __exit__(self, et: Any, ev: Any, tb: Any) -> None:
        if et is None:
            self.close()
        else:
            self.abort()


def _iso(ts: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()


# ======================================================================================== reader
class LazyChannel:
    """A channel on disk. Behaves like a read-only array: ``shape``, ``dtype``, ``len()``, slicing and
    ``np.asarray(...)``. Slicing along the first (time) axis reads only the chunks it needs."""

    def __init__(self, reader: "ArchiveReader", name: str, info: dict) -> None:
        self._r, self.name, self._info = reader, name, info
        self.dtype = np.dtype(info["dtype"])
        self.shape = (int(info["length"]),) + tuple(info["tail"])
        self.ndim = len(self.shape)
        self._starts = np.concatenate([[0], np.cumsum([c["n"] for c in info["chunks"]])]).astype(np.int64)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def size(self) -> Any:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> Any:
        return self.size * self.dtype.itemsize

    @property
    def n_chunks(self) -> Any:
        return len(self._info["chunks"])

    def _chunk(self, i: int) -> np.ndarray:
        return self._r._load_chunk(self.name, i)

    def _rows(self, start: int, stop: int, step: int = 1) -> np.ndarray:
        """Rows ``start:stop:step`` (non-negative, clipped) assembled from chunks."""
        if stop <= start:
            return np.empty((0,) + self.shape[1:], self.dtype)
        first = int(np.searchsorted(self._starts, start, side="right") - 1)
        parts = []
        i = first
        while i < self.n_chunks and self._starts[i] < stop:
            a, b = int(self._starts[i]), int(self._starts[i + 1])
            lo, hi = max(start, a) - a, min(stop, b) - a
            parts.append(np.asarray(self._chunk(i)[lo:hi]))
            i += 1
        out = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        return out[::step] if step != 1 else out

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            key = (key,)
        first, rest = (key[0] if key else slice(None)), key[1:]
        n = self.shape[0]
        if isinstance(first, (int, np.integer)):
            i = int(first)
            i = i + n if i < 0 else i
            if not 0 <= i < n:
                raise IndexError(f"index {first} out of range for channel '{self.name}' with {n} rows")
            row = self._rows(i, i + 1)[0]
            return row[rest] if rest else row
        if isinstance(first, slice):
            start, stop, step = first.indices(n)
            data = self.to_numpy()[first] if step < 0 else self._rows(start, stop, step)
            return data[(slice(None),) + rest] if rest else data
        idx = np.asarray(first)
        if idx.dtype == bool:
            idx = np.nonzero(idx)[0]
        if idx.size == 0:
            return np.empty((0,) + self.shape[1:], self.dtype)[(slice(None),) + rest]
        idx = np.where(idx < 0, idx + n, idx)
        if idx.min() < 0 or idx.max() >= n:
            raise IndexError(f"index out of range for channel '{self.name}'")
        lo, hi = int(idx.min()), int(idx.max()) + 1
        block = self._rows(lo, hi)
        return block[idx - lo][(slice(None),) + rest] if rest else block[idx - lo]

    def to_numpy(self) -> np.ndarray:
        if self.n_chunks == 0:
            return np.empty(self.shape, self.dtype)
        return self._rows(0, self.shape[0])

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        a = self.to_numpy()
        return a.astype(dtype) if dtype is not None else a

    def iter_chunks(self) -> Iterator[np.ndarray]:
        for i in range(self.n_chunks):
            yield np.asarray(self._chunk(i))

    def __repr__(self) -> str:
        return f"LazyChannel({self.name!r}, shape={self.shape}, dtype={self.dtype}, chunks={self.n_chunks})"


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_manifest(manifest: Any, where: str = "archive") -> None:
    """Reject a malformed or hostile manifest with :class:`ArchiveError` *before* any of it is used.

    Checks structure and types, numeric dtypes only, sane shapes, and that every chunk file is a safe relative path.
    """

    def bad(msg: str) -> ArchiveError:
        return ArchiveError(f"{where}: invalid manifest: {msg}")

    if not isinstance(manifest, dict):
        raise bad("top level must be an object")
    channels = manifest.get("channels")
    if not isinstance(channels, dict):
        raise bad("'channels' must be an object")
    for name, info in channels.items():
        if not isinstance(info, dict):
            raise bad(f"channel {name!r} must be an object")
        try:
            dt = np.dtype(info["dtype"])
        except Exception as ex:
            raise bad(f"channel {name!r} has an unusable dtype ({ex})") from ex
        if dt.kind not in _DTYPE_KINDS or dt.hasobject:
            raise bad(f"channel {name!r} has dtype {dt}; only numeric/bool dtypes are allowed")
        length, tail, chunks = info.get("length"), info.get("tail"), info.get("chunks")
        if not isinstance(length, int) or isinstance(length, bool) or length < 0 or length > 2**40:
            raise bad(f"channel {name!r} has an invalid length {length!r}")
        if not isinstance(tail, list) or len(tail) > 8 or not all(_is_int(d) and 0 <= d <= 2**31 for d in tail):
            raise bad(f"channel {name!r} has an invalid tail shape {tail!r}")
        if not isinstance(chunks, list):
            raise bad(f"channel {name!r}: 'chunks' must be a list")
        row_bytes = int(float(np.prod(tail, dtype=np.float64)) * dt.itemsize) if tail else dt.itemsize
        for c in chunks:
            if not isinstance(c, dict) or not isinstance(c.get("file"), str) or not _is_int(c.get("n")) or c["n"] < 0:
                raise bad(f"channel {name!r} has a malformed chunk entry")
            if not isinstance(c.get("hash", ""), str):
                raise bad(f"channel {name!r} has a malformed chunk hash")
            if c["n"] * row_bytes > MAX_CHUNK_BYTES:
                raise bad(f"channel {name!r} declares a chunk of {c['n'] * row_bytes:,} bytes (limit {MAX_CHUNK_BYTES:,})")
            safe_member_path(Path("."), c["file"])  # raises ArchiveError for traversal / absolute paths


class ArchiveReader:
    """Open a ``.aevd`` directory (or a packed ``.aevz`` file) for lazy reading."""

    def __init__(self, path: PathLike, max_open_chunks: int = 16) -> None:
        self.path = Path(path)
        self._zip: Optional[zipfile.ZipFile] = None
        self._cache: "OrderedDict[Tuple[str, int], np.ndarray]" = OrderedDict()
        self._max = max_open_chunks
        if self.path.is_file() and self.path.suffix == ".aevz":
            try:
                self._zip = zipfile.ZipFile(self.path)
                if self._zip.getinfo(MANIFEST).file_size > MAX_MANIFEST_BYTES:
                    raise ArchiveError(f"'{self.path}': manifest is implausibly large")
                raw = self._zip.read(MANIFEST)
            except (zipfile.BadZipFile, KeyError) as ex:
                raise ArchiveError(f"'{self.path}' is not a valid packed Aevion archive: {ex}") from ex
        else:
            mp = self.path / MANIFEST
            if not mp.exists():
                raise ArchiveError(f"'{self.path}' is not an Aevion run archive (no {MANIFEST})")
            if mp.stat().st_size > MAX_MANIFEST_BYTES:
                raise ArchiveError(f"'{self.path}': manifest is implausibly large")
            raw = mp.read_bytes()
        try:
            self.manifest = json.loads(raw.decode("utf-8"))
        except Exception as ex:
            raise ArchiveError(f"corrupt manifest in '{self.path}': {ex}") from ex
        schema = self.manifest.get("schema", "") if isinstance(self.manifest, dict) else ""
        if not isinstance(schema, str) or not re.fullmatch(r"aevion\.run/\d+", schema):
            raise ArchiveError(f"'{self.path}': unrecognised schema {schema!r}")
        if int(schema.split("/")[1]) > int(SCHEMA.split("/")[1]):
            raise ArchiveError(f"'{self.path}' uses schema {schema}, newer than this Aevion supports ({SCHEMA}); upgrade Aevion")
        validate_manifest(self.manifest, str(self.path))
        self.meta: dict = self.manifest.get("meta", {})
        self.provenance: dict = self.manifest.get("provenance", {})
        self.status: str = self.manifest.get("status", "incomplete")
        self.created: str = self.manifest.get("created", "")
        self.channels: Dict[str, LazyChannel] = {n: LazyChannel(self, n, i) for n, i in self.manifest["channels"].items()}

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def _load_chunk(self, name: str, i: int) -> np.ndarray:
        key = (name, i)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        info = self.manifest["channels"][name]
        rel = info["chunks"][i]["file"]
        try:
            if self._zip is not None:
                import io

                if self._zip.getinfo(rel).file_size > MAX_CHUNK_BYTES:
                    raise ArchiveError(f"chunk '{rel}' of channel '{name}' is implausibly large")
                arr = np.load(io.BytesIO(self._zip.read(rel)), allow_pickle=False)
            else:
                fp = safe_member_path(self.path, rel)
                if not fp.exists():
                    raise ArchiveError(f"missing chunk file '{rel}' of channel '{name}' in '{self.path}'")
                if fp.stat().st_size > MAX_CHUNK_BYTES:
                    raise ArchiveError(f"chunk '{rel}' of channel '{name}' is implausibly large")
                # Read the whole (bounded-size) chunk instead of memory-mapping it: a mmap keeps the file open, which on
                # Windows blocks deleting or moving the archive while it is still referenced.
                arr = np.load(fp, allow_pickle=False)
        except ArchiveError:
            raise
        except KeyError as ex:
            raise ArchiveError(f"missing chunk file '{rel}' of channel '{name}' in '{self.path}'") from ex
        except Exception as ex:  # corrupt .npy, pickle refused, truncated file, ...
            raise ArchiveError(f"cannot read chunk '{rel}' of channel '{name}': {type(ex).__name__}: {ex}") from ex
        expect = (int(info["chunks"][i]["n"]),) + tuple(info["tail"])
        if arr.shape != expect or arr.dtype != np.dtype(info["dtype"]):
            raise ArchiveError(f"chunk '{rel}' of channel '{name}' is {arr.dtype}{arr.shape}, manifest says {info['dtype']}{expect}")
        self._cache[key] = arr
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return arr

    def verify(self, deep: bool = True) -> List[str]:
        """Return a list of integrity problems (empty = fine). ``deep`` re-hashes every chunk."""
        problems: List[str] = []
        if not self.complete:
            problems.append(f"archive status is '{self.status}' (the run did not finish cleanly)")
        for name, info in self.manifest["channels"].items():
            total = 0
            for c in info["chunks"]:
                total += c["n"]
                if self._zip is not None:
                    try:
                        data = self._zip.read(c["file"])
                    except KeyError:
                        problems.append(f"{name}: missing {c['file']}")
                        continue
                    if deep and _hash_bytes(data) != c["hash"]:
                        problems.append(f"{name}: hash mismatch in {c['file']}")
                    continue
                fp = self.path / c["file"]
                if not fp.exists():
                    problems.append(f"{name}: missing {c['file']}")
                elif deep and _hash_file(fp) != c["hash"]:
                    problems.append(f"{name}: hash mismatch in {c['file']}")
            if total != info["length"]:
                problems.append(f"{name}: chunk rows ({total}) != recorded length ({info['length']})")
        return problems

    def close(self) -> None:
        self._cache.clear()
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "ArchiveReader":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


# ===================================================================================== utilities
def pack(src: PathLike, dst: Optional[PathLike] = None, compress: bool = True) -> Path:
    """Pack a ``.aevd`` directory into one shareable ``.aevz`` file (a ZIP; readable in place, no unpacking)."""
    src = Path(src)
    if not (src / MANIFEST).exists():
        raise ArchiveError(f"'{src}' is not a run archive directory")
    out = Path(dst) if dst else src.with_suffix(".aevz")
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(out, "w", mode) as z:
        for fp in sorted(src.rglob("*")):
            if fp.is_file():
                z.write(fp, fp.relative_to(src).as_posix())
    return out


def safe_member_path(root: Path, name: str) -> Path:
    """Resolve an archive member name under ``root`` or raise :class:`ArchiveError`.

    Blocks absolute paths, drive letters, ``..`` components, NUL bytes and anything that resolves outside ``root``.
    """
    norm = name.replace("\\", "/")
    if not norm or norm.startswith("/") or ":" in norm.split("/")[0] or chr(0) in norm:
        raise ArchiveError(f"unsafe path in archive: {name!r}")
    if any(part == ".." for part in norm.split("/")):
        raise ArchiveError(f"unsafe path in archive (parent traversal): {name!r}")
    base = root.resolve()
    target = (base / norm).resolve()
    if target != base and base not in target.parents:
        raise ArchiveError(f"unsafe path in archive (escapes the destination): {name!r}")
    return target


def unpack(src: PathLike, dst: Optional[PathLike] = None, max_bytes: int = 64 * 1024**3, max_ratio: float = 1000.0) -> Path:
    """Safely unpack a ``.aevz`` file.

    Every member is validated (no absolute paths, no ``..``, nothing outside ``dst``, no symlinks), the declared
    uncompressed size is capped at ``max_bytes``, and a compression ratio above ``max_ratio`` (a zip bomb) is refused.
    """
    src = Path(src)
    out = Path(dst) if dst else src.with_suffix(".aevd")
    if out.exists():
        raise ArchiveError(f"'{out}' already exists")
    with zipfile.ZipFile(src) as z:
        infos = z.infolist()
        total = sum(i.file_size for i in infos)
        if total > max_bytes:
            raise ArchiveError(f"archive would expand to {total / 1e9:.1f} GB (> limit {max_bytes / 1e9:.1f} GB)")
        for i in infos:
            if i.compress_size and i.file_size / max(i.compress_size, 1) > max_ratio and i.file_size > 1_000_000:
                raise ArchiveError(f"suspicious compression ratio for {i.filename!r} (possible zip bomb)")
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ArchiveError(f"symbolic links are not allowed in archives: {i.filename!r}")
        out.mkdir(parents=True)
        try:
            for i in infos:
                target = safe_member_path(out, i.filename)
                if i.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as fin, open(target, "wb") as fout:
                    shutil.copyfileobj(fin, fout, length=1 << 20)
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)  # never leave a half-extracted (possibly hostile) tree behind
            raise
    return out


def is_store(path: PathLike) -> bool:
    p = Path(path)
    return (p.is_dir() and (p / MANIFEST).exists()) or (p.is_file() and p.suffix == ".aevz")
