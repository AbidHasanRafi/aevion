"""Run archives: self-describing, reproducible records of a simulation.

A :class:`RunArchive` holds time-series channels (``name -> array``, shape ``(T, N, ...)``) plus a JSON
manifest (scenario/config, provenance, events). Two on-disk formats are supported:

* **v2 store** (``*.aevd`` directory, or packed ``*.aevz``): chunked, streamable, lazily readable, integrity-hashed
  (see :mod:`aevion.datasets.store`). Opened archives keep channels on disk; ``archive["pos"]`` materialises a
  channel (and caches it) while ``archive.lazy("pos")[a:b]`` reads only the needed rows.
* **v1 file** (``*.aev``, a compressed NPZ): the original single-file format, still fully supported.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .._types import PathLike
from ..errors import ArchiveError
from . import store as _store
from .store import ArchiveReader, ArchiveWriter, LazyChannel

SCHEMA_V1 = "aevion.run/1"
SCHEMA = _store.SCHEMA  # current (v2)


def environment_info() -> dict:
    import scipy

    from .. import __version__

    info = dict(
        aevion=__version__,
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
    )
    for mod in ("numba", "torch", "matplotlib", "gymnasium"):  # read from package metadata: importing torch/numba here would be slow and unsafe
        try:
            info[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            info[mod] = None
    return info


@dataclass
class RunArchive:
    channels: Dict[str, Any] = field(default_factory=dict)  # ndarray or LazyChannel
    meta: dict = field(default_factory=dict)
    source_path: Optional[Path] = None
    _cache: dict = field(default_factory=dict, repr=False)
    _reader: Optional[ArchiveReader] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """One line (the default dataclass repr would print every array)."""
        try:
            n, nv = self.n_steps, self.n_vehicles
            span = f"{float(self.t[0]):g}-{float(self.t[-1]):g} s" if n else "empty"
        except Exception:
            n, nv, span = 0, 0, "?"
        name = self.meta.get("name", "run")
        state = "complete" if self.complete else "incomplete"
        return f"RunArchive({name!r}: {n} steps x {nv} vehicle(s), {span}, {len(self.channels)} channels, {state})"

    # ------------------------------------------------------------------ access
    def __getitem__(self, k: str) -> Any:
        v = self.channels[k]
        if isinstance(v, LazyChannel):
            if k not in self._cache:
                self._cache[k] = v.to_numpy()
            return self._cache[k]
        return v

    def __contains__(self, k: str) -> bool:
        return k in self.channels

    def get(self, k: str, default: Any = None) -> Any:
        return self[k] if k in self.channels else default

    def lazy(self, k: str) -> Any:
        """The channel without materialising it: a :class:`LazyChannel` (sliceable) or the in-memory array."""
        return self.channels[k]

    def release(self, k: Optional[str] = None) -> None:
        """Drop cached materialised channels (all if ``k`` is None) to free memory."""
        if k is None:
            self._cache.clear()
        else:
            self._cache.pop(k, None)

    @property
    def is_lazy(self) -> bool:
        return any(isinstance(v, LazyChannel) for v in self.channels.values())

    @property
    def t(self) -> np.ndarray:
        return self["t"]

    @property
    def n_steps(self) -> int:
        return len(self.channels["t"])

    @property
    def n_vehicles(self) -> int:
        return self.channels["pos"].shape[1]

    @property
    def dt(self) -> float:
        t = self.t
        return float(np.median(np.diff(t))) if len(t) > 1 else 0.0

    @property
    def scenario(self) -> dict:
        return self.meta.get("scenario", {})

    @property
    def config(self) -> dict:
        """The v2 configuration dict, if recorded (``meta['config']``)."""
        return self.meta.get("config", {})

    @property
    def provenance(self) -> dict:
        return self.meta.get("provenance", {})

    @property
    def complete(self) -> bool:
        return self.meta.get("status", "complete") == "complete"

    @property
    def world(self) -> Any:
        from ..environments.world import World

        w = self.meta.get("world")
        return World.from_dict(w) if w else None

    @property
    def spec(self) -> Any:
        from ..vehicles.spec import VehicleSpec

        v = self.meta.get("vehicle")
        return VehicleSpec.from_dict(v) if v else None

    def euler_deg(self) -> np.ndarray:
        return np.degrees(self["euler"])

    def slice_time(self, t0: float, t1: float) -> "RunArchive":
        """An in-memory archive restricted to ``t0 <= t <= t1`` (side streams sliced by their own time)."""
        t = self.t
        i0, i1 = int(np.searchsorted(t, t0, "left")), int(np.searchsorted(t, t1, "right"))
        ch = {}
        for k, v in self.channels.items():
            if k.startswith("sensors/") and not k.endswith("/t") and f"{k}/t" in self.channels:
                st = self[f"{k}/t"]
                a, b = int(np.searchsorted(st, t0, "left")), int(np.searchsorted(st, t1, "right"))
                ch[k] = np.asarray(self.lazy(k)[a:b])
            elif k.startswith("sensors/") and k.endswith("/t"):
                a, b = int(np.searchsorted(self[k], t0, "left")), int(np.searchsorted(self[k], t1, "right"))
                ch[k] = self[k][a:b]
            elif len(v) == len(t):
                ch[k] = np.asarray(self.lazy(k)[i0:i1])
            else:
                ch[k] = self[k]
        return RunArchive(ch, dict(self.meta))

    # -------------------------------------------------------------------- I/O
    def save(self, path: PathLike, format: Optional[str] = None, chunk_steps: int = 512, overwrite: bool = True) -> Path:
        """Save. ``.aevd`` → v2 store (directory), ``.aevz`` → packed v2, ``.aev``/none → v1 NPZ file.

        ``format`` may force ``"v1"`` or ``"v2"``.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        fmt = format or ("v2" if suffix in (".aevd", ".aevz") else "v1")
        if fmt == "v1":
            if suffix == "":
                path = path.with_suffix(".aev")
            return self._save_v1(path)
        if suffix == ".aevz":
            tmp = path.with_suffix(".aevd")
            self._save_v2(tmp, chunk_steps, overwrite=True)
            out = _store.pack(tmp, path)
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            self.source_path = out
            return out
        if suffix == "":
            path = path.with_suffix(".aevd")
        out = self._save_v2(path, chunk_steps, overwrite)
        self.source_path = out
        return out

    def _save_v1(self, path: Path) -> Path:
        meta = dict(self.meta)
        meta["schema"] = SCHEMA_V1
        meta.setdefault("created", _dt.datetime.now(_dt.timezone.utc).isoformat())
        meta.setdefault("environment", environment_info())
        payload = {k: np.asarray(self[k]) for k in self.channels}
        payload["__meta__"] = np.frombuffer(json.dumps(meta, default=_store._json_default).encode("utf-8"), dtype=np.uint8)
        with open(path, "wb") as f:  # np.savez_compressed would append .npz otherwise
            np.savez_compressed(f, **payload)  # type: ignore[arg-type]
        self.source_path = path
        return path

    def _save_v2(self, path: Path, chunk_steps: int, overwrite: bool) -> Path:
        meta = dict(self.meta)
        prov = dict(meta.pop("provenance", {}) or {})
        if not prov:
            from .provenance import collect

            prov = collect(seed=self.scenario.get("seed"), backend=meta.get("backend"))
        meta.setdefault("status", "complete")
        w = ArchiveWriter(path, meta=meta, chunk_steps=chunk_steps, overwrite=overwrite, provenance=prov)
        try:
            for k in self.channels:
                v = self.lazy(k)
                if isinstance(v, LazyChannel):  # stream chunk by chunk (never the whole channel in RAM)
                    for chunk in v.iter_chunks():
                        w.append_block(k, chunk)
                else:
                    w.append_block(k, v)
        except BaseException:
            w.abort()
            raise
        return w.close()

    @classmethod
    def load(cls, path: PathLike) -> "RunArchive":
        """Open any supported archive (auto-detects v2 store / packed / v1 NPZ)."""
        path = Path(path)
        if not path.exists():
            raise ArchiveError(f"archive '{path}' does not exist")
        if _store.is_store(path):
            return cls.open_store(path)
        try:
            with np.load(path, allow_pickle=False) as z:
                if "__meta__" not in z.files:
                    raise ArchiveError(f"'{path}' is not an Aevion archive (no metadata)")
                meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
                channels = {k: z[k] for k in z.files if k != "__meta__"}
        except ArchiveError:
            raise
        except Exception as ex:
            raise ArchiveError(f"cannot read archive '{path}': {ex}") from ex
        a = cls(channels, meta)
        a.source_path = path
        return a

    @classmethod
    def open_store(cls, path: PathLike) -> "RunArchive":
        r = ArchiveReader(path)
        meta = dict(r.meta)
        meta["provenance"] = r.provenance
        meta.setdefault("status", r.status)  # a failed run records a specific reason ('diverged', 'failed') in meta
        meta.setdefault("created", r.created)
        a = cls(dict(r.channels), meta)
        a._reader = r
        a.source_path = Path(path)
        return a

    open = open_store  # short alias

    def verify_integrity(self, deep: bool = True) -> Any:
        """Hash-check a v2 store; v1 files report nothing to verify."""
        if self._reader is None:
            return []
        return self._reader.verify(deep)

    def to_hdf5(self, path: PathLike) -> None:
        import h5py

        with h5py.File(path, "w") as f:
            for k in self.channels:
                # HDF5 cannot have a dataset that is also a group: store 'sensors/imu' as 'sensors/imu/data'
                name = k + "/data" if any(o.startswith(k + "/") for o in self.channels) else k
                f.create_dataset(name, data=self[k], compression="gzip")
            f.attrs["meta"] = json.dumps(self.meta, default=_store._json_default)

    @classmethod
    def from_hdf5(cls, path: PathLike) -> Any:
        import h5py

        channels = {}
        with h5py.File(path, "r") as f:

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    channels[name[:-5] if name.endswith("/data") else name] = obj[()]

            f.visititems(visit)
            meta = json.loads(f.attrs["meta"])
        return cls(channels, meta)

    def to_csv(self, path: PathLike, vehicle: int = 0) -> None:
        """Flat CSV of the main scalar/vector channels for one vehicle (for spreadsheets / MATLAB / R)."""
        n = self.n_steps
        cols, names = [], []
        for k in self.channels:
            v = self.lazy(k)
            if k == "t":
                names.append("t")
                cols.append(self["t"])
            elif v.ndim >= 2 and v.shape[0] == n and v.shape[1] > vehicle and not k.startswith("sensors/"):
                x = np.asarray(v[:, vehicle]).reshape(n, -1)
                for j in range(x.shape[1]):
                    names.append(f"{k}[{j}]" if x.shape[1] > 1 else k)
                    cols.append(x[:, j])
        order = sorted(range(len(names)), key=lambda i: (names[i] != "t", names[i]))
        data = np.stack([cols[i] for i in order], 1)
        np.savetxt(path, data, delimiter=",", header=",".join(names[i] for i in order), comments="")

    def summary(self) -> str:
        sc = self.scenario
        cfg = self.config
        ctrl = sc.get("controller") or cfg.get("controller", {}).get("type")
        traj = (sc.get("trajectory") or cfg.get("trajectory") or {}).get("kind") or cfg.get("trajectory", {}).get("type")
        seed = sc.get("seed", cfg.get("seed"))
        veh = self.meta.get("vehicle", {}).get("name", "?")
        fmt = "v2 store" if self._reader is not None else "v1 file"
        lines = [
            f"RunArchive  [{fmt}]  schema={self.meta.get('schema', SCHEMA)}  created={str(self.meta.get('created', '?'))[:19]}  status={self.meta.get('status', 'complete')}",
            f"  vehicle={veh}  n={self.n_vehicles}  steps={self.n_steps}  duration={self.t[-1] - self.t[0]:.2f}s  record_dt={self.dt * 1e3:.1f}ms",
            f"  scenario: controller={ctrl}  trajectory={traj}  seed={seed}",
            f"  channels: {', '.join(f'{k}{tuple(v.shape[1:])}' for k, v in self.channels.items() if not k.startswith('sensors/'))}",
        ]
        ev = self.meta.get("events") or []
        if ev:
            lines.append(f"  events: {len(ev)} ({', '.join(sorted({e['type'] for e in ev}))})")
        p = self.provenance
        if p:
            git = p.get("git") or {}
            lines.append(
                f"  provenance: aevion {p.get('aevion')} · python {p.get('python')} · config {p.get('config_hash', '-')} · kernel {p.get('kernel')}"
                + (f" · git {str(git.get('commit'))[:8]}{'*' if git.get('dirty') else ''}" if git.get("commit") else "")
            )
        return "\n".join(lines)


# the json default lives in store; re-exported for backwards compatibility
_json_default = _store._json_default


class Recorder:
    """Collects per-step arrays in memory during a simulation and builds a :class:`RunArchive`."""

    def __init__(
        self, every: int = 1, vehicles: Optional[list] = None, channels: Optional[Iterable[str]] = None, collision_events: bool = True
    ) -> None:
        self.every, self.vehicles = max(1, int(every)), vehicles
        self.collision_events = collision_events
        self.keep = None if channels is None else set(channels) | {"t"}
        self._step = 0
        self._rows: Dict[str, list] = {}
        self._side: Dict[str, list] = {}
        self.events: list = []

    def _sel(self, x: Any) -> Any:
        x = np.asarray(x)
        return x if self.vehicles is None else x[self.vehicles]

    def _row(self, B: Any, state: Any, extras: Any) -> dict:
        from ..core import rotation as rot

        f = lambda x: self._sel(B.to_numpy(x)).astype(np.float64)  # noqa: E731
        row = dict(
            t=np.float64(state.t),
            pos=f(state.pos),
            vel=f(state.vel),
            quat=f(state.quat),
            omega=f(state.omega),
            euler=f(rot.to_euler(B, state.quat)),
            rotor=f(state.rotor),
            thrust=f(B.sum(state.thrust, axis=-1)),
            power=f(state.power),
            soc=f(state.soc),
            wind=f(state.wind),
            acc=f(state.acc),
        )
        if state.contact is not None:
            row["contact"] = self._sel(B.to_numpy(state.contact)).astype(np.int8)
        if state.collided is not None:
            col = self._sel(B.to_numpy(state.collided))
            row["collided"] = col.astype(np.int8)
            if self.collision_events:
                for i in np.nonzero(col)[0]:
                    self.events.append(dict(type="collision", t=float(state.t), vehicle=int(i)))
        for k, v in extras.items():
            row[k] = f(v) if hasattr(v, "shape") else np.asarray(v)
        return row

    def _emit(self, row: dict) -> None:
        for k, v in row.items():
            if self.keep is None or k in self.keep:
                self._rows.setdefault(k, []).append(v)

    def record(self, B: Any, state: Any, **extras: Any) -> None:
        """Record one step. ``extras`` are extra per-vehicle arrays (e.g. ``ref_pos=...``)."""
        if self._step % self.every == 0:
            self._emit(self._row(B, state, extras))
        self._step += 1

    def record_side(self, name: str, t: float, data: np.ndarray) -> None:
        """Record an irregularly sampled stream (sensor frames) with its own timestamps."""
        self._side.setdefault(name, []).append((float(t), np.asarray(data)))

    def add_event(self, ev: dict) -> None:
        self.events.append(ev)

    def build(self, meta: Optional[dict] = None) -> RunArchive:
        ch = {k: np.stack(v) for k, v in self._rows.items()}
        for name, items in self._side.items():
            ch[f"sensors/{name}/t"] = np.array([t for t, _ in items])
            ch[f"sensors/{name}"] = np.stack([d for _, d in items])
        m = dict(meta or {})
        m["events"] = self.events
        return RunArchive(ch, m)


class StreamRecorder(Recorder):
    """Like :class:`Recorder` but streams rows to an :class:`ArchiveWriter`: memory use is bounded by one chunk
    per channel regardless of run length. :meth:`build` finalises the store and returns a lazily-opened archive."""

    def __init__(
        self,
        path: PathLike,
        every: int = 1,
        vehicles: Optional[list] = None,
        channels: Optional[Iterable[str]] = None,
        chunk_steps: int = 512,
        meta: Optional[dict] = None,
        provenance: Optional[dict] = None,
        overwrite: bool = True,
        collision_events: bool = True,
    ) -> None:
        super().__init__(every, vehicles, channels, collision_events)
        self.writer = ArchiveWriter(path, meta=meta, chunk_steps=chunk_steps, overwrite=overwrite, provenance=provenance)
        self.path = Path(path)

    def _emit(self, row: dict) -> None:
        for k, v in row.items():
            if self.keep is None or k in self.keep:
                self.writer.append(k, v)

    def record_side(self, name: str, t: float, data: np.ndarray) -> None:
        if self.keep is None or f"sensors/{name}" in self.keep or f"sensors/{name}/t" in self.keep:
            self.writer.append(f"sensors/{name}/t", np.float64(t))
            self.writer.append(f"sensors/{name}", np.asarray(data))

    def build(self, meta: Optional[dict] = None, provenance: Optional[dict] = None) -> RunArchive:
        m = dict(meta or {})
        m["events"] = self.events
        m.setdefault("status", "complete")
        self.writer.close(meta=m, provenance=provenance)
        return RunArchive.open_store(self.path)

    def abort(self, meta: Optional[dict] = None) -> None:
        if meta:
            self.writer.meta.update(meta)
        self.writer.meta["events"] = self.events
        self.writer.abort()
