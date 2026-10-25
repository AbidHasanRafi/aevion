"""Radio-link failure detection (3GPP TS 38.133 / 38.331 timers, abstracted to the evaluation interval).

The physical layer indicates *out-of-sync* when the (filtered) downlink SINR is below ``Q_out`` (-8 dB, about 10 % PDCCH BLER)
and *in-sync* above ``Q_in`` (-6 dB, about 2 %). After ``N310`` consecutive out-of-sync indications timer ``T310`` starts; ``N311``
consecutive in-sync indications stop it; if it expires, radio-link failure (RLF) is declared and the UAV needs RRC
re-establishment (``recovery_s`` without service) before it can be served again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

__all__ = ["RadioLinkMonitor", "RlfEvent"]


@dataclass
class RlfEvent:
    t: float
    link: int
    kind: str  # "t310_start" | "t310_stop" | "rlf" | "recovered"


@dataclass
class RadioLinkMonitor:
    n_links: int
    q_out_db: float = -8.0
    q_in_db: float = -6.0
    n310: int = 1
    n311: int = 1
    t310_s: float = 1.0
    recovery_s: float = 0.5
    filter_s: float = 0.2  # L1 averaging time of the sync indications
    events: List[RlfEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.q_in_db < self.q_out_db:
            raise ValueError("q_in must be >= q_out")
        self.reset()

    def reset(self) -> None:
        n = self.n_links
        self._filt: Optional[np.ndarray] = None
        self._oos = np.zeros(n, int)
        self._ins = np.zeros(n, int)
        self._t310 = np.full(n, -1.0)  # remaining time, < 0 = stopped
        self._recover = np.zeros(n)  # remaining recovery time
        self.n_rlf = np.zeros(n, int)
        self.events = []

    @property
    def in_recovery(self) -> np.ndarray:
        return self._recover > 0

    def step(self, t: float, sinr_db: Any, dt: float) -> np.ndarray:
        """Update with the current SINR [dB]; returns a bool array: link is *unavailable* (recovering from RLF)."""
        s = np.broadcast_to(np.asarray(sinr_db, float), (self.n_links,))
        a = 1.0 - np.exp(-dt / max(self.filter_s, 1e-9))
        self._filt = s.copy() if self._filt is None else self._filt + a * (s - self._filt)
        out_of_sync = self._filt < self.q_out_db
        in_sync = self._filt > self.q_in_db

        recovering = self._recover > 0
        self._recover = np.maximum(self._recover - dt, 0.0)
        for i in np.nonzero(recovering & (self._recover <= 0))[0]:
            self.events.append(RlfEvent(t, int(i), "recovered"))

        active = ~recovering
        self._oos = np.where(active & out_of_sync, self._oos + 1, 0)
        self._ins = np.where(active & in_sync, self._ins + 1, 0)

        running = self._t310 >= 0
        start = active & ~running & (self._oos >= self.n310)
        for i in np.nonzero(start)[0]:
            self._t310[i] = self.t310_s
            self.events.append(RlfEvent(t, int(i), "t310_start"))
        stop = active & running & (self._ins >= self.n311)
        for i in np.nonzero(stop)[0]:
            self._t310[i] = -1.0
            self.events.append(RlfEvent(t, int(i), "t310_stop"))
        running = self._t310 >= 0
        self._t310 = np.where(running & active, self._t310 - dt, self._t310)
        fail = active & running & (self._t310 < 0)
        for i in np.nonzero(fail)[0]:
            self.n_rlf[i] += 1
            self._recover[i] = self.recovery_s
            self._t310[i] = -1.0
            self._oos[i] = self._ins[i] = 0
            self._filt[i] = max(self._filt[i], self.q_in_db + 1.0)  # fresh start after re-establishment
            self.events.append(RlfEvent(t, int(i), "rlf"))
        return self._recover > 0
