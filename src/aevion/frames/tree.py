"""A transform tree (like ROS ``tf``): named frames, static and time-varying transforms, lookups."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..errors import AevionError
from .transform import Transform

WORLD = "world"
BODY = "body"


class FrameError(AevionError):
    """Unknown frame, cycle, or disconnected tree."""


class TransformTree:
    """Frames form a tree rooted at ``world``. Each non-root frame has one parent and a transform
    ``T_parent_frame``. Transforms may be static or provided by a callable ``f(t, vehicle) -> Transform``.

    >>> tree = TransformTree()
    >>> tree.add("body", "world", provider=lambda t, v: pose_of_vehicle(t, v))
    >>> tree.add("camera", "body", Transform.from_rpy([0, 20, 0], [0.1, 0, 0], degrees=True))
    >>> tree.lookup("world", "camera", t=1.0)       # T_world_camera
    """

    def __init__(self) -> None:
        self._parent: Dict[str, Optional[str]] = {WORLD: None}
        self._static: Dict[str, Transform] = {}
        self._dynamic: Dict[str, Callable] = {}

    # --------------------------------------------------------------- building
    def add(self, frame: str, parent: str, transform: Optional[Transform] = None, provider: Optional[Callable] = None) -> "TransformTree":
        if frame == WORLD:
            raise FrameError("'world' is the root frame and cannot be re-parented")
        if parent not in self._parent:
            raise FrameError(f"unknown parent frame '{parent}' for '{frame}' (known: {sorted(self._parent)})")
        if (transform is None) == (provider is None):
            raise FrameError(f"frame '{frame}': give exactly one of transform= or provider=")
        # cycle check: parent chain must not contain `frame`
        p: Optional[str] = parent
        while p is not None:
            if p == frame:
                raise FrameError(f"adding '{frame}' under '{parent}' would create a cycle")
            p = self._parent[p]
        self._parent[frame] = parent
        self._static.pop(frame, None)
        self._dynamic.pop(frame, None)
        if transform is not None:
            self._static[frame] = transform
        elif provider is not None:
            self._dynamic[frame] = provider
        return self

    def has(self, frame: str) -> bool:
        return frame in self._parent

    def frames(self) -> List[str]:
        return sorted(self._parent)

    def parent(self, frame: str) -> Optional[str]:
        self._require(frame)
        return self._parent[frame]

    def _require(self, frame: str) -> None:
        if frame not in self._parent:
            raise FrameError(f"unknown frame '{frame}' (known: {sorted(self._parent)})")

    def is_static(self, frame: str) -> bool:
        self._require(frame)
        return frame in self._static or frame == WORLD

    # ----------------------------------------------------------------- lookup
    def _local(self, frame: str, t: float, vehicle: int) -> Transform:
        if frame in self._static:
            return self._static[frame]
        return self._dynamic[frame](t, vehicle)

    def _chain_to_root(self, frame: str) -> List[str]:
        out: List[str] = []
        cur: Optional[str] = frame
        while cur is not None:
            out.append(cur)
            cur = self._parent[cur]
        return out

    def lookup(self, target: str, source: str, t: float = 0.0, vehicle: int = 0) -> Transform:
        """``T_target_source``: maps coordinates in ``source`` into ``target``."""
        self._require(target)
        self._require(source)

        def to_root(frame: str) -> Any:
            T = Transform.identity()
            for f in self._chain_to_root(frame)[:-1]:
                T = self._local(f, t, vehicle) @ T
            return T  # T_world_frame

        return to_root(target).inverse() @ to_root(source)

    def transform_points(self, points: Any, target: str, source: str, t: float = 0.0, vehicle: int = 0) -> Any:
        return self.lookup(target, source, t, vehicle).apply(points)

    # -------------------------------------------------------------- utilities
    def static_dict(self) -> dict:
        return {f: {"parent": self._parent[f], **T.to_dict()} for f, T in self._static.items()}

    @classmethod
    def from_static_dict(cls, d: dict) -> "TransformTree":
        tree = cls()
        pending = dict(d)
        while pending:  # order-independent insertion
            progressed = False
            for f, v in list(pending.items()):
                if v["parent"] in tree._parent:
                    tree.add(f, v["parent"], Transform.from_dict(v))
                    del pending[f]
                    progressed = True
            if not progressed:
                raise FrameError(f"frames with unknown parents: {sorted(pending)}")
        return tree

    def describe(self) -> str:
        children: Dict[Optional[str], List[str]] = {}
        for f, p in self._parent.items():
            children.setdefault(p, []).append(f)

        lines: List[str] = []

        def walk(f: Any, depth: Any) -> None:
            tag = "dynamic" if f in self._dynamic else ("static" if f in self._static else "root")
            extra = ""
            if f in self._static:
                T = self._static[f]
                extra = f"  t={T.t.round(3).tolist()} rpy_deg={T.rpy(True).round(1).tolist()}"
            lines.append("  " * depth + f"{f} [{tag}]{extra}")
            for c in sorted(children.get(f, [])):
                walk(c, depth + 1)

        walk(WORLD, 0)
        return "\n".join(lines)
