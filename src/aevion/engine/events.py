"""A small synchronous event bus used for simulation milestones (faults, collisions, mission progress, …)."""

from __future__ import annotations

import fnmatch
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


@dataclass
class Event:
    name: str  # dotted, e.g. "fault.start", "collision", "battery.low"
    t: float
    step: int
    vehicle: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict:
        """JSON-friendly form stored in the archive manifest (``type`` is the fault type for fault events)."""
        typ = self.data.get("fault") if self.name == "fault.start" else self.name
        rec = {"type": typ, "t": float(self.t), "name": self.name}
        if self.vehicle is not None:
            rec["vehicle"] = int(self.vehicle)
        for k, v in self.data.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                rec[k] = v
        return rec


Handler = Callable[[Event], None]


class EventBus:
    """Publish/subscribe with glob patterns (``"fault.*"``) and an optional bounded history."""

    def __init__(self, history: int = 100000) -> None:
        self._subs: List[Tuple[str, int, Handler]] = []
        self.history: Deque[Event] = deque(maxlen=history)
        self._counts: Dict[str, int] = {}

    def subscribe(self, pattern: str, handler: Handler, priority: int = 0) -> Callable[[], None]:
        """Register ``handler`` for events matching ``pattern``; returns an ``unsubscribe`` function."""
        entry = (pattern, priority, handler)
        self._subs.append(entry)
        self._subs.sort(key=lambda e: e[1])

        def unsubscribe() -> None:
            if entry in self._subs:
                self._subs.remove(entry)

        return unsubscribe

    def emit(self, name: str, t: float, step: int, vehicle: Optional[int] = None, **data: Any) -> Event:
        ev = Event(name, t, step, vehicle, data)
        self.history.append(ev)
        self._counts[name] = self._counts.get(name, 0) + 1
        for pattern, _, handler in list(self._subs):
            if fnmatch.fnmatchcase(name, pattern):
                handler(ev)
        return ev

    def events(self, pattern: str = "*") -> List[Event]:
        return [e for e in self.history if fnmatch.fnmatchcase(e.name, pattern)]

    def count(self, name: str) -> int:
        return self._counts.get(name, 0)

    def clear(self) -> None:
        self.history.clear()
        self._counts.clear()
