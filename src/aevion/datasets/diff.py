"""Compare two run archives: configuration differences, channel-by-channel numerical differences, metric deltas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# keys that legitimately differ between otherwise identical runs
VOLATILE = {
    "created",
    "wall_s",
    "wall_time_s",
    "registered",
    "git",
    "numba_threads",
    "cpu_count",
    "platform",
    "machine",
    "package_set_hash",
    "n_packages",
}


def diff_dicts(a: Any, b: Any, path: str = "", ignore: Any = VOLATILE) -> List[tuple]:
    """Recursive structural diff. Returns ``[(path, a_value, b_value), ...]`` (missing side shown as ``<absent>``)."""
    out: List[tuple] = []
    ABSENT = "<absent>"
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in ignore:
                continue
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append((p, ABSENT, b[k]))
            elif k not in b:
                out.append((p, a[k], ABSENT))
            else:
                out.extend(diff_dicts(a[k], b[k], p, ignore))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [(path + ".<length>", len(a), len(b))]
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_dicts(x, y, f"{path}[{i}]", ignore))
        return out
    try:
        if isinstance(a, float) or isinstance(b, float):
            if np.isclose(float(a), float(b), rtol=1e-12, atol=0.0, equal_nan=True):
                return []
        elif a == b:
            return []
    except (TypeError, ValueError):
        pass
    return [(path, a, b)]


@dataclass
class ChannelDiff:
    name: str
    shape_a: tuple
    shape_b: tuple
    max_abs: float = 0.0
    rms: float = 0.0
    first_index: Optional[int] = None  # first row where |a-b| > tol
    first_time: Optional[float] = None
    identical: bool = True


@dataclass
class ArchiveDiff:
    meta_changes: List[tuple] = field(default_factory=list)
    channels: Dict[str, ChannelDiff] = field(default_factory=dict)
    only_a: List[str] = field(default_factory=list)
    only_b: List[str] = field(default_factory=list)
    metric_deltas: Dict[str, tuple] = field(default_factory=dict)
    tol: float = 0.0

    @property
    def identical(self) -> bool:
        return not self.only_a and not self.only_b and all(c.identical for c in self.channels.values())

    @property
    def config_identical(self) -> bool:
        return not self.meta_changes

    def summary(self, max_items: int = 12) -> str:
        L = []
        L.append(
            "Channels: "
            + ("IDENTICAL" if self.identical else "DIFFER")
            + f" (tol {self.tol:g})  ·  configuration: "
            + ("identical" if self.config_identical else f"{len(self.meta_changes)} change(s)")
        )
        if self.only_a:
            L.append(f"  only in A: {', '.join(self.only_a)}")
        if self.only_b:
            L.append(f"  only in B: {', '.join(self.only_b)}")
        bad = [c for c in self.channels.values() if not c.identical]
        for c in sorted(bad, key=lambda c: -c.max_abs)[:max_items]:
            when = f", first at t={c.first_time:.3f}s" if c.first_time is not None else ""
            shape = f"  shape {c.shape_a} vs {c.shape_b}" if c.shape_a != c.shape_b else ""
            L.append(f"  {c.name:<22} max|Δ|={c.max_abs:.3e}  rms={c.rms:.3e}{when}{shape}")
        if self.meta_changes:
            L.append("Configuration changes:")
            for p, x, y in self.meta_changes[:max_items]:
                L.append(f"  {p}: {_short(x)}  →  {_short(y)}")
            if len(self.meta_changes) > max_items:
                L.append(f"  … {len(self.meta_changes) - max_items} more")
        if self.metric_deltas:
            L.append("Metrics (A → B):")
            for k, (x, y) in self.metric_deltas.items():
                rel = f" ({100 * (y - x) / abs(x):+.1f}%)" if abs(x) > 1e-12 else ""
                L.append(f"  {k:<22} {x:12.5g} → {y:12.5g}{rel}")
        return "\n".join(L)


def _short(v: Any, n: int = 60) -> Any:
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def diff_archives(a: Any, b: Any, tol: float = 0.0, channels: Optional[List[str]] = None, with_metrics: bool = True) -> ArchiveDiff:
    """Compare two :class:`RunArchive` objects. ``tol`` is the absolute difference regarded as 'equal'."""
    d = ArchiveDiff(tol=tol)
    ma = {k: v for k, v in a.meta.items() if k in ("config", "scenario", "vehicle", "world", "events", "backend", "controller", "n_sub")}
    mb = {k: v for k, v in b.meta.items() if k in ("config", "scenario", "vehicle", "world", "events", "backend", "controller", "n_sub")}
    d.meta_changes = diff_dicts(ma, mb)
    ka, kb = set(a.channels), set(b.channels)
    d.only_a, d.only_b = sorted(ka - kb), sorted(kb - ka)
    t = a.get("t")
    for name in sorted(ka & kb):
        if channels is not None and name not in channels:
            continue
        la, lb = a.lazy(name), b.lazy(name)
        cd = ChannelDiff(name, tuple(la.shape), tuple(lb.shape))
        if tuple(la.shape) != tuple(lb.shape):
            cd.identical = False
            n = min(len(la), len(lb))
            if n and la.shape[1:] == lb.shape[1:]:
                x, y = np.asarray(la[:n]), np.asarray(lb[:n])
                cd.max_abs = float(np.nanmax(np.abs(x - y))) if x.size else 0.0
            d.channels[name] = cd
            continue
        x, y = np.asarray(a[name]), np.asarray(b[name])
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            with np.errstate(invalid="ignore"):  # inf - inf is handled explicitly below
                diff = np.abs(x.astype(float) - y.astype(float))
            both_nan = np.isnan(x.astype(float)) & np.isnan(y.astype(float))
            diff = np.where(both_nan, 0.0, diff)
            both_inf = np.isinf(x.astype(float)) & (x.astype(float) == y.astype(float))
            diff = np.where(both_inf, 0.0, diff)
        else:
            diff = np.abs(x.astype(float) - y.astype(float)) if x.dtype.kind in "iub" else (x != y).astype(float)
        if diff.size:
            cd.max_abs = float(np.nanmax(diff))
            cd.rms = float(np.sqrt(np.nanmean(diff**2)))
            bad = diff > tol
            if bad.any():
                cd.identical = False
                row = int(np.nonzero(bad.reshape(len(bad), -1).any(1))[0][0])
                cd.first_index = row
                if t is not None and len(t) == len(diff):
                    cd.first_time = float(t[row] - t[0])
        d.channels[name] = cd
    if with_metrics:
        try:
            from .. import metrics as MT

            ma_, mb_ = MT.flight_metrics(a), MT.flight_metrics(b)
            for k in ma_:
                if k in mb_:
                    xm, ym = float(np.mean(ma_[k])), float(np.mean(mb_[k]))
                    if xm != ym:
                        d.metric_deltas[k] = (xm, ym)
        except Exception:
            pass
    return d
