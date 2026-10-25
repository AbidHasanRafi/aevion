"""Named, independent, reproducible random streams.

Every stochastic component asks for its own stream by *name*. Streams derive from one master seed through
``numpy.random.SeedSequence`` so they are statistically independent, and adding or removing a component never
changes the numbers another component draws.
"""

from __future__ import annotations

import zlib
from typing import Dict

import numpy as np


class RngStreams:
    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._streams: Dict[str, np.random.Generator] = {}

    @staticmethod
    def _key(name: str) -> int:
        return zlib.crc32(name.encode("utf-8"))

    def stream(self, name: str) -> np.random.Generator:
        """The generator for ``name`` (created on first use; same object afterwards)."""
        if name not in self._streams:
            ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(self._key(name),))
            self._streams[name] = np.random.Generator(np.random.PCG64(ss))
        return self._streams[name]

    def fresh(self, name: str) -> np.random.Generator:
        """A new generator for ``name`` restarted from its initial state (does not affect :meth:`stream`)."""
        ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(self._key(name),))
        return np.random.Generator(np.random.PCG64(ss))

    def int_seed(self, name: str) -> int:
        """A 31-bit integer seed for libraries that want an int (e.g. torch)."""
        return int(np.random.SeedSequence(entropy=self.seed, spawn_key=(self._key(name),)).generate_state(1)[0] & 0x7FFFFFFF)

    def reset(self) -> None:
        self._streams.clear()
