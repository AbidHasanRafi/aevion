"""Geofences and no-fly zones.

A zone is a region with a signed distance function (negative inside, in metres): :class:`SphereZone`, :class:`CylinderZone`,
:class:`BoxZone`, :class:`PolygonZone` (a prism over a simple polygon) and :class:`AltitudeBand`. A :class:`Geofence` pairs a zone
with a rule, ``inclusion`` (stay inside) or ``exclusion`` (stay out), and a safety ``margin``. :class:`GeofenceSet` evaluates many
fences for many vehicles at once; :class:`GeofenceCheck` plugs it into the engine's health monitor, emitting ``geofence.breach`` and
``geofence.clear`` events and recording a violation.

Distances are exact for spheres, boxes, vertical cylinders and altitude bands, and a true Euclidean distance for prisms. Only the
vehicle's state is checked; its own radius is not added (use ``margin``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..health.checks import HealthCheck, Violation

__all__ = [
    "Zone",
    "SphereZone",
    "CylinderZone",
    "BoxZone",
    "PolygonZone",
    "AltitudeBand",
    "Geofence",
    "GeofenceSet",
    "GeofenceCheck",
    "zone_from_dict",
    "ZONES",
]


class Zone:
    kind = "zone"

    def signed_distance(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


def _pts(p: Any) -> np.ndarray:
    a = np.asarray(p, float)
    return a.reshape(-1, 3) if a.ndim != 2 else a


class SphereZone(Zone):
    kind = "sphere"

    def __init__(self, center: Sequence[float], radius: float):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.center, self.radius = np.asarray(center, float).reshape(3), float(radius)

    def signed_distance(self, p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(_pts(p) - self.center, axis=1) - self.radius

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="sphere", center=self.center.tolist(), radius=self.radius)


class CylinderZone(Zone):
    """Vertical cylinder around ``center`` (x, y) between ``z_min`` and ``z_max`` (use infinities for an unbounded column)."""

    kind = "cylinder"

    def __init__(self, center: Sequence[float], radius: float, z_min: float = -np.inf, z_max: float = np.inf):
        if radius <= 0 or z_max <= z_min:
            raise ValueError("radius must be positive and z_max above z_min")
        self.center, self.radius, self.z_min, self.z_max = np.asarray(center, float).reshape(-1)[:2], float(radius), float(z_min), float(z_max)

    def signed_distance(self, p: np.ndarray) -> np.ndarray:
        q = _pts(p)
        dr = np.hypot(q[:, 0] - self.center[0], q[:, 1] - self.center[1]) - self.radius
        mid, half = 0.5 * (self.z_min + self.z_max), 0.5 * (self.z_max - self.z_min)
        dz = np.abs(q[:, 2] - mid) - half if np.isfinite(half) else np.full(len(q), -np.inf)
        return np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(dr, dz), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="cylinder", center=self.center.tolist(), radius=self.radius, z_min=_fin(self.z_min), z_max=_fin(self.z_max))


class BoxZone(Zone):
    kind = "box"

    def __init__(self, lo: Sequence[float], hi: Sequence[float]):
        self.lo, self.hi = np.asarray(lo, float).reshape(3), np.asarray(hi, float).reshape(3)
        if np.any(self.hi <= self.lo):
            raise ValueError("hi must exceed lo on every axis")

    def signed_distance(self, p: np.ndarray) -> np.ndarray:
        q = np.abs(_pts(p) - 0.5 * (self.lo + self.hi)) - 0.5 * (self.hi - self.lo)
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="box", lo=self.lo.tolist(), hi=self.hi.tolist())


class PolygonZone(Zone):
    """Prism over a simple polygon ``vertices`` (``(n, 2)``, any winding) from ``z_min`` to ``z_max``."""

    kind = "polygon"

    def __init__(self, vertices: Sequence[Sequence[float]], z_min: float = -np.inf, z_max: float = np.inf):
        v = np.asarray(vertices, float)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3 or not np.all(np.isfinite(v)):
            raise ValueError("vertices must be an (n, 2) array with n >= 3")
        if z_max <= z_min:
            raise ValueError("z_max must exceed z_min")
        self.vertices, self.z_min, self.z_max = v, float(z_min), float(z_max)

    def _sd2d(self, xy: np.ndarray) -> np.ndarray:
        """Signed 2-D distance to the polygon (Inigo Quilez's exact formulation): negative inside."""
        v = self.vertices
        n = len(v)
        d = np.full(len(xy), np.inf)
        s = np.ones(len(xy))
        j = n - 1
        for i in range(n):
            e = v[j] - v[i]
            w = xy - v[i]
            b = w - e * np.clip((w @ e) / (e @ e), 0.0, 1.0)[:, None]
            d = np.minimum(d, np.sum(b * b, axis=1))
            c = np.stack([xy[:, 1] >= v[i, 1], xy[:, 1] < v[j, 1], e[0] * w[:, 1] > e[1] * w[:, 0]])
            flip = np.all(c, axis=0) | np.all(~c, axis=0)
            s = np.where(flip, -s, s)
            j = i
        return s * np.sqrt(d)

    def signed_distance(self, p: np.ndarray) -> np.ndarray:
        q = _pts(p)
        d2 = self._sd2d(q[:, :2])
        mid, half = 0.5 * (self.z_min + self.z_max), 0.5 * (self.z_max - self.z_min)
        dz = np.abs(q[:, 2] - mid) - half if np.isfinite(half) else np.full(len(q), -np.inf)
        return np.hypot(np.maximum(d2, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(d2, dz), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="polygon", vertices=self.vertices.tolist(), z_min=_fin(self.z_min), z_max=_fin(self.z_max))


class AltitudeBand(Zone):
    """Everything between ``z_min`` and ``z_max`` (a ceiling when ``z_min`` is ``-inf``, a floor when ``z_max`` is ``inf``)."""

    kind = "altitude"

    def __init__(self, z_min: float = -np.inf, z_max: float = 120.0):
        if z_max <= z_min:
            raise ValueError("z_max must exceed z_min")
        self.z_min, self.z_max = float(z_min), float(z_max)

    def signed_distance(self, p: np.ndarray) -> np.ndarray:
        z = _pts(p)[:, 2]
        return np.maximum(self.z_min - z, z - self.z_max)

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="altitude", z_min=_fin(self.z_min), z_max=_fin(self.z_max))


def _fin(x: float) -> Optional[float]:
    return float(x) if np.isfinite(x) else None


ZONES = {c.kind: c for c in (SphereZone, CylinderZone, BoxZone, PolygonZone, AltitudeBand)}


def zone_from_dict(d: Dict[str, Any]) -> Zone:
    d = dict(d)
    kind = d.pop("type")
    if kind not in ZONES:
        raise ValueError(f"unknown zone type {kind!r}; known: {sorted(ZONES)}")
    for k in ("z_min", "z_max"):
        if k in d and d[k] is None:
            d[k] = -np.inf if k == "z_min" else np.inf
    return ZONES[kind](**d)


class Geofence:
    """A zone plus a rule. ``margin`` shrinks an inclusion zone / grows an exclusion zone so the warning fires before the boundary."""

    def __init__(self, zone: Zone, kind: str = "exclusion", margin: float = 0.0, name: str = ""):
        if kind not in ("inclusion", "exclusion"):
            raise ValueError("kind must be 'inclusion' or 'exclusion'")
        if margin < 0:
            raise ValueError("margin must be >= 0")
        self.zone, self.kind, self.margin, self.name = zone, kind, float(margin), name or f"{kind}_{zone.kind}"

    def clearance(self, p: np.ndarray) -> np.ndarray:
        """Distance to the rule's boundary, positive while the rule is satisfied (negative = breached), margin included."""
        sd = self.zone.signed_distance(p)
        return (-sd - self.margin) if self.kind == "inclusion" else (sd - self.margin)

    def to_dict(self) -> Dict[str, Any]:
        return dict(kind=self.kind, margin=self.margin, name=self.name, zone=self.zone.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Geofence":
        return cls(zone_from_dict(d["zone"]), d.get("kind", "exclusion"), d.get("margin", 0.0), d.get("name", ""))


class GeofenceSet:
    def __init__(self, fences: Sequence[Geofence]):
        self.fences: List[Geofence] = list(fences)
        names = [f.name for f in self.fences]
        if len(set(names)) != len(names):
            raise ValueError("geofence names must be unique (give each one a name=...)")

    def __len__(self) -> int:
        return len(self.fences)

    def clearances(self, positions: np.ndarray) -> np.ndarray:
        """``(N, K)`` clearance of every vehicle to every fence (positive = satisfied)."""
        p = _pts(positions)
        if not self.fences:
            return np.zeros((len(p), 0))
        return np.stack([f.clearance(p) for f in self.fences], axis=1)

    def breaches(self, positions: np.ndarray) -> np.ndarray:
        """``(N, K)`` boolean mask of breached fences."""
        return self.clearances(positions) < 0.0

    def nearest_boundary(self, positions: np.ndarray) -> np.ndarray:
        """``(N,)`` smallest clearance over all fences (``inf`` without fences): the quantity a planner wants to keep positive."""
        c = self.clearances(positions)
        return c.min(axis=1) if c.shape[1] else np.full(len(c), np.inf)

    def to_list(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.fences]

    @classmethod
    def from_list(cls, items: Sequence[Dict[str, Any]]) -> "GeofenceSet":
        return cls([Geofence.from_dict(d) for d in items])


class GeofenceCheck(HealthCheck):
    """Health check: breaching a geofence emits ``geofence.breach`` (once per entry) and ``geofence.clear`` on leaving, and records a violation.

    ``fences`` is a list of dictionaries (``{"kind": "exclusion", "name": "airport", "margin": 5, "zone": {"type": "cylinder", ...}}``).
    With ``stop=True`` the first breach is a *hard* violation, so a health policy of ``"stop"`` ends the run cleanly.
    """

    name = "geofence"

    def __init__(self, fences: Sequence[Dict[str, Any]] = (), stop: bool = False) -> None:
        self.set = GeofenceSet.from_list(fences)
        self.hard = bool(stop)
        self._inside: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._inside = None

    def check(self, state: Any, ctx: Any) -> List[Violation]:
        if not len(self.set):
            return []
        pos = ctx.B.to_numpy(state.pos) if ctx.B is not None else np.asarray(state.pos)
        breach = self.set.breaches(pos)
        was = self._inside if self._inside is not None else np.zeros_like(breach)
        emit = getattr(ctx, "emit", None)
        out: List[Violation] = []
        for i, k in zip(*np.nonzero(breach & ~was)):
            fence = self.set.fences[k]
            clearance = float(fence.clearance(pos[i : i + 1])[0])
            if emit is not None:
                emit("geofence.breach", int(i), fence=fence.name, clearance=clearance)
            out.append(Violation(self.name, f"breached geofence '{fence.name}' ({fence.kind})", int(i), "geofence", clearance, hard=self.hard))
        if emit is not None:
            for i, k in zip(*np.nonzero(~breach & was)):
                emit("geofence.clear", int(i), fence=self.set.fences[k].name)
        self._inside = breach
        return out
