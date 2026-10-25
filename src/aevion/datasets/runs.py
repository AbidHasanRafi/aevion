"""A SQLite registry of runs: find, compare and manage archives without remembering paths.

The database lives in ``$AEVION_HOME/runs.db`` (default ``~/.aevion/runs.db``). It stores *pointers and summary
data* (name, path, config hash, seed, tags, headline metrics); the archives themselves stay where they are.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from ..errors import ArchiveError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    created     TEXT NOT NULL,
    registered  REAL NOT NULL,
    config_hash TEXT,
    aevion      TEXT,
    seed        INTEGER,
    duration    REAL,
    n_vehicles  INTEGER,
    status      TEXT,
    tags        TEXT NOT NULL DEFAULT '[]',
    metrics     TEXT NOT NULL DEFAULT '{}',
    notes       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_runs_cfg ON runs(config_hash);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created);
"""


def home() -> Path:
    return Path(os.environ.get("AEVION_HOME", Path.home() / ".aevion"))


@dataclass
class RunRecord:
    run_id: str
    name: str
    path: str
    created: str
    config_hash: Optional[str]
    aevion: Optional[str]
    seed: Optional[int]
    duration: Optional[float]
    n_vehicles: Optional[int]
    status: Optional[str]
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()

    def open(self) -> Any:
        from .archive import RunArchive

        return RunArchive.load(self.path)


class RunRegistry:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else home() / "runs.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        con = sqlite3.connect(self.path, timeout=10.0)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _rec(row: Any) -> RunRecord:
        return RunRecord(
            row["run_id"],
            row["name"],
            row["path"],
            row["created"],
            row["config_hash"],
            row["aevion"],
            row["seed"],
            row["duration"],
            row["n_vehicles"],
            row["status"],
            json.loads(row["tags"]),
            json.loads(row["metrics"]),
            row["notes"],
        )

    # ------------------------------------------------------------------------ write
    def register(
        self, archive_or_path: Any, tags: Sequence[str] = (), metrics: Optional[dict] = None, notes: str = "", compute_metrics: bool = True
    ) -> str:
        """Record a run. Accepts a :class:`RunArchive` that was saved (``.source_path``) or a path."""
        from .archive import RunArchive

        if isinstance(archive_or_path, (str, Path)):
            path = Path(archive_or_path)
            arch = RunArchive.load(path)
        else:
            arch = archive_or_path
            src = getattr(arch, "source_path", None)
            if src is None:
                raise ArchiveError("this archive has not been saved; save it first (archive.save(path)) or pass a path")
            path = Path(src)
        meta, prov = arch.meta, arch.meta.get("provenance", {})
        sc = meta.get("scenario", {})
        name = sc.get("name") or meta.get("config", {}).get("name") or Path(path).stem
        created = prov.get("created") or meta.get("created") or ""
        cfg_hash = prov.get("config_hash") or meta.get("config_hash")
        if metrics is None and compute_metrics:
            try:
                from .. import metrics as MT

                metrics = {k: float(v.mean()) for k, v in MT.flight_metrics(arch).items()}
            except Exception:
                metrics = {}
        run_id = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{(cfg_hash or 'nohash')[:8]}"
        with self._conn() as c:
            n = 0
            base = run_id
            while c.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
                n += 1
                run_id = f"{base}-{n}"
            c.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    name,
                    str(Path(path).resolve()),
                    created,
                    time.time(),
                    cfg_hash,
                    prov.get("aevion"),
                    prov.get("seed", sc.get("seed")),
                    float(arch.t[-1] - arch.t[0]) if arch.n_steps else 0.0,
                    arch.n_vehicles,
                    meta.get("status", "complete"),
                    json.dumps(list(tags)),
                    json.dumps(metrics or {}),
                    notes,
                ),
            )
        return run_id

    def tag(self, run_id: str, *tags: str, remove: bool = False) -> List[str]:
        rec = self.get(run_id)
        cur = set(rec.tags)
        cur = cur - set(tags) if remove else cur | set(tags)
        with self._conn() as c:
            c.execute("UPDATE runs SET tags=? WHERE run_id=?", (json.dumps(sorted(cur)), rec.run_id))
        return sorted(cur)

    def set_note(self, run_id: str, notes: str) -> None:
        rec = self.get(run_id)
        with self._conn() as c:
            c.execute("UPDATE runs SET notes=? WHERE run_id=?", (notes, rec.run_id))

    def update_metrics(self, run_id: str, metrics: dict) -> None:
        rec = self.get(run_id)
        rec.metrics.update(metrics)
        with self._conn() as c:
            c.execute("UPDATE runs SET metrics=? WHERE run_id=?", (json.dumps(rec.metrics), rec.run_id))

    def remove(self, run_id: str, delete_files: bool = False) -> None:
        rec = self.get(run_id)
        if delete_files and rec.exists:
            import shutil

            p = Path(rec.path)
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        with self._conn() as c:
            c.execute("DELETE FROM runs WHERE run_id=?", (rec.run_id,))

    def prune_missing(self) -> int:
        """Delete registry rows whose files no longer exist. Returns the number removed."""
        n = 0
        for rec in self.list(limit=10**9):
            if not rec.exists:
                with self._conn() as c:
                    c.execute("DELETE FROM runs WHERE run_id=?", (rec.run_id,))
                n += 1
        return n

    # ------------------------------------------------------------------------- read
    def get(self, run_id: str) -> RunRecord:
        """Look up by full id or unique prefix."""
        with self._conn() as c:
            rows = c.execute("SELECT * FROM runs WHERE run_id=? OR run_id LIKE ?", (run_id, run_id + "%")).fetchall()
        if not rows:
            raise ArchiveError(f"no registered run matches '{run_id}'")
        exact = [r for r in rows if r["run_id"] == run_id]
        if len(rows) > 1 and not exact:
            raise ArchiveError(f"'{run_id}' is ambiguous ({', '.join(r['run_id'] for r in rows[:5])}…)")
        return self._rec(exact[0] if exact else rows[0])

    def list(
        self, name: Optional[str] = None, tag: Optional[str] = None, config_hash: Optional[str] = None, since: Optional[str] = None, limit: int = 50
    ) -> List[RunRecord]:
        q = "SELECT * FROM runs WHERE 1=1"
        args: List[Any] = []
        if name:
            q += " AND name LIKE ?"
            args.append(f"%{name}%")
        if config_hash:
            q += " AND config_hash LIKE ?"
            args.append(config_hash + "%")
        if since:
            q += " AND created >= ?"
            args.append(since)
        q += " ORDER BY registered DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        recs = [self._rec(r) for r in rows]
        return [r for r in recs if tag in r.tags] if tag else recs

    def find_by_config(self, config_hash: str) -> List[RunRecord]:
        return self.list(config_hash=config_hash, limit=10**6)

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def table(self, records: Optional[Iterable[RunRecord]] = None, metric: Optional[str] = "rmse_pos") -> str:
        recs = list(records) if records is not None else self.list()
        head = f"{'run id':<26}{'name':<22}{'veh':>4}{'dur s':>8}{metric or '':>11}  tags"
        lines = [head, "-" * len(head)]
        for r in recs:
            m = f"{r.metrics[metric]:.4g}" if metric and metric in r.metrics else "-"
            miss = "" if r.exists else "  [missing]"
            lines.append(f"{r.run_id:<26}{r.name[:21]:<22}{(r.n_vehicles or 0):>4}{(r.duration or 0):>8.1f}{m:>11}  {','.join(r.tags)}{miss}")
        return "\n".join(lines)
