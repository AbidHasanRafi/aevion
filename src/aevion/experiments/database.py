"""A small SQLite database of experiment results, and leaderboards computed from it.

The standard library's ``sqlite3`` is the whole dependency. Rows are stored as JSON next to the columns queries need (method, condition, seed, status), and
every numeric metric is also written to a ``metrics`` table so "the best method by ``rmse_pos`` in the windy condition" is one SQL query. Results are
keyed by the trial's content hash, so adding the same trial twice (a re-run, or a merge of two databases) updates instead of duplicating.

:func:`leaderboard` ranks methods by a metric. Whether the best method is *really* better is a statistical question, so the board also reports each
method's Holm-corrected Mann–Whitney p-value against the leader and marks the methods that cannot be told apart from it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..analysis import stats as ST
from .engine import ExperimentResult

__all__ = ["ResultDB", "leaderboard", "format_leaderboard"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (name TEXT PRIMARY KEY, created REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS trials (
    experiment TEXT NOT NULL, key TEXT NOT NULL, method TEXT, condition TEXT, seed INTEGER, status TEXT, wall_s REAL, row TEXT NOT NULL,
    PRIMARY KEY (experiment, key)
);
CREATE TABLE IF NOT EXISTS metrics (experiment TEXT NOT NULL, key TEXT NOT NULL, name TEXT NOT NULL, value REAL, PRIMARY KEY (experiment, key, name));
CREATE INDEX IF NOT EXISTS metrics_by_name ON metrics (experiment, name);
"""


_BEST = (
    "SELECT t.row FROM metrics m JOIN trials t ON t.experiment=m.experiment AND t.key=m.key "
    "WHERE m.experiment=? AND m.name=? AND m.value IS NOT NULL ORDER BY m.value "
)
_BEST_MIN = _BEST + "ASC LIMIT 1"
_BEST_MAX = _BEST + "DESC LIMIT 1"


class ResultDB:
    def __init__(self, path: Any = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(_SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "ResultDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------------------------------------- writing
    def add_result(self, result: ExperimentResult, experiment: Optional[str] = None) -> int:
        """Store (or update) every row of an :class:`ExperimentResult`; returns the number of rows written."""
        name = experiment or result.name
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?)", (name, time.time(), json.dumps(result.meta, default=float)))
            n = 0
            for i, r in enumerate(result.rows):
                key = str(r.get("key") or f"row{i:08d}")
                self.con.execute(
                    "INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?,?,?)",
                    (
                        name,
                        key,
                        r.get("method"),
                        r.get("condition"),
                        r.get("seed"),
                        r.get("status", "ok"),
                        r.get("wall_s"),
                        json.dumps(r, default=float),
                    ),
                )
                self.con.execute("DELETE FROM metrics WHERE experiment=? AND key=?", (name, key))
                for k, v in r.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("seed", "wall_s"):
                        self.con.execute("INSERT INTO metrics VALUES (?,?,?,?)", (name, key, k, float(v) if np.isfinite(v) else None))
                n += 1
        return n

    def delete(self, experiment: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM trials WHERE experiment=?", (experiment,))
            self.con.execute("DELETE FROM metrics WHERE experiment=?", (experiment,))
            self.con.execute("DELETE FROM experiments WHERE name=?", (experiment,))

    def merge(self, other: "ResultDB") -> int:
        """Copy every experiment of ``other`` that this database lacks and every trial it lacks; returns the number of trials added."""
        added = 0
        for (name,) in other.con.execute("SELECT name FROM experiments"):
            have = {k for (k,) in self.con.execute("SELECT key FROM trials WHERE experiment=?", (name,))}
            res = other.result(name)
            new = [r for r in res.rows if str(r.get("key")) not in have]
            if new or name not in self.experiments():
                self.add_result(ExperimentResult(name, new, res.meta))
                added += len(new)
        return added

    # ---------------------------------------------------------------------------------------------- reading
    def experiments(self) -> List[str]:
        return [n for (n,) in self.con.execute("SELECT name FROM experiments ORDER BY created, name")]

    def rows(
        self, experiment: str, method: Optional[str] = None, condition: Optional[str] = None, status: Optional[str] = "ok"
    ) -> List[Dict[str, Any]]:
        q, args = "SELECT row FROM trials WHERE experiment=?", [experiment]
        for col, v in (("method", method), ("condition", condition), ("status", status)):
            if v is not None:
                q += f" AND {col}=?"
                args.append(v)
        return [json.loads(r) for (r,) in self.con.execute(q + " ORDER BY rowid", args)]

    def result(self, experiment: str) -> ExperimentResult:
        meta_row = self.con.execute("SELECT meta FROM experiments WHERE name=?", (experiment,)).fetchone()
        if meta_row is None:
            raise KeyError(f"no experiment named {experiment!r}")
        return ExperimentResult(experiment, self.rows(experiment), json.loads(meta_row[0]))

    def metric_values(self, experiment: str, metric: str, method: Optional[str] = None, condition: Optional[str] = None) -> np.ndarray:
        q = "SELECT m.value FROM metrics m JOIN trials t ON t.experiment=m.experiment AND t.key=m.key WHERE m.experiment=? AND m.name=? AND t.status='ok' AND m.value IS NOT NULL"
        args: List[Any] = [experiment, metric]
        if method is not None:
            q += " AND t.method=?"
            args.append(method)
        if condition is not None:
            q += " AND t.condition=?"
            args.append(condition)
        return np.array([v for (v,) in self.con.execute(q, args)], float)

    def best(self, experiment: str, metric: str, minimize: bool = True) -> Dict[str, Any]:
        """The single trial with the best value of ``metric``."""
        query = _BEST_MIN if minimize else _BEST_MAX
        r = self.con.execute(query, (experiment, metric)).fetchone()
        if r is None:
            raise KeyError(f"no values of {metric!r} in experiment {experiment!r}")
        return json.loads(r[0])

    def failures(self, experiment: str) -> List[Dict[str, Any]]:
        return self.rows(experiment, status="failed")

    def counts(self, experiment: str) -> Dict[str, int]:
        return dict(self.con.execute("SELECT status, COUNT(*) FROM trials WHERE experiment=? GROUP BY status", (experiment,)))


# ------------------------------------------------------------------------------------------------ leaderboards
def leaderboard(
    rows: Sequence[Dict[str, Any]], metric: str, lower_is_better: bool = True, group: str = "method", alpha: float = 0.05, level: float = 0.95
) -> List[Dict[str, Any]]:
    """Rank the groups (methods) by the mean of ``metric`` over all their trials.

    Each entry carries ``rank``, ``mean``, a confidence interval, ``n``, ``p_vs_best`` (Holm-corrected Mann–Whitney against the leader) and
    ``tied_with_best`` (``True`` when the difference to the leader is not significant at ``alpha``).
    """
    samples: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("status", "ok") != "ok" or metric not in r or not isinstance(r[metric], (int, float)) or not np.isfinite(r[metric]):
            continue
        samples.setdefault(str(r[group]), []).append(float(r[metric]))
    if not samples:
        raise ValueError(f"no finite values of {metric!r} to rank")
    means = {k: float(np.mean(v)) for k, v in samples.items()}
    order = sorted(means, key=means.get, reverse=not lower_is_better)  # type: ignore[arg-type]
    best = order[0]
    raw = {}
    for k in order[1:]:
        a, b = np.asarray(samples[best]), np.asarray(samples[k])
        try:
            raw[k] = float(ST.sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:  # identical samples
            raw[k] = 1.0
    adj = dict(zip(raw, ST.holm_correction(list(raw.values())))) if raw else {}
    out = []
    for i, k in enumerate(order):
        v = np.asarray(samples[k])
        m, lo, hi = ST.mean_ci(v, level) if len(v) > 1 else (float(v[0]), float(v[0]), float(v[0]))
        p = None if k == best else float(adj[k])
        out.append(
            dict(rank=i + 1, name=k, mean=m, ci_lo=lo, ci_hi=hi, n=len(v), p_vs_best=p, tied_with_best=(k == best) or (p is not None and p >= alpha))
        )
    return out


def format_leaderboard(board: Sequence[Dict[str, Any]], metric: str = "", fmt: str = "{:.4g}") -> str:
    w = max(len(b["name"]) for b in board)
    lines = [f"{metric}  (mean [95% CI], n; * = not distinguishable from the leader)"]
    for b in board:
        mark = "*" if b["tied_with_best"] and b["rank"] > 1 else " "
        lines.append(
            f" {b['rank']:>2}{mark} {b['name']:<{w}}  {fmt.format(b['mean'])} [{fmt.format(b['ci_lo'])}, {fmt.format(b['ci_hi'])}]  n={b['n']}"
        )
    return "\n".join(lines)
