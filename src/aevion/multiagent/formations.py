"""Formation offset generators: ``formation(n, spacing) -> (n, 3)`` offsets from the formation centre."""

from __future__ import annotations

import math

import numpy as np


def ring(n: int, spacing: float = 1.0) -> np.ndarray:
    """Vehicles on a circle; neighbouring vehicles are ``spacing`` apart (chord length)."""
    a = 2 * np.pi * np.arange(n) / max(n, 1)
    r = spacing / (2 * np.sin(np.pi / n)) if n > 2 else spacing / 2
    return np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], 1)


def line(n: int, spacing: float = 1.0) -> np.ndarray:
    return np.stack([np.zeros(n), (np.arange(n) - (n - 1) / 2) * spacing, np.zeros(n)], 1)


def column(n: int, spacing: float = 1.0) -> np.ndarray:
    """Single file along the direction of travel (x)."""
    return np.stack([-(np.arange(n) - (n - 1) / 2) * spacing, np.zeros(n), np.zeros(n)], 1)


def grid(n: int, spacing: float = 1.0) -> np.ndarray:
    k = math.ceil(math.sqrt(n))
    i, j = np.divmod(np.arange(n), k)
    return np.stack([(i - (k - 1) / 2) * spacing, (j - (k - 1) / 2) * spacing, np.zeros(n)], 1)


def vee(n: int, spacing: float = 1.0) -> np.ndarray:
    """V (arrow-head) formation, leader at the tip."""
    i = np.arange(n)
    side = np.where(i % 2 == 0, 1, -1)
    rank = (i + 1) // 2
    return np.stack([-rank * spacing, side * rank * spacing, np.zeros(n)], 1)


def sphere(n: int, spacing: float = 1.0) -> np.ndarray:
    """Roughly even points on a sphere (Fibonacci lattice) whose nearest-neighbour distance is about ``spacing``."""
    if n == 1:
        return np.zeros((1, 3))
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5**0.5) * i
    r = spacing * math.sqrt(n / (4 * math.pi)) * 1.6
    return np.stack([r * np.cos(th) * np.sin(phi), r * np.sin(th) * np.sin(phi), r * np.cos(phi)], 1)


FORMATIONS = {"ring": ring, "line": line, "column": column, "grid": grid, "vee": vee, "sphere": sphere}


def make_formation(kind: str, n: int, spacing: float = 1.0) -> np.ndarray:
    try:
        return FORMATIONS[kind](n, spacing)
    except KeyError:
        raise ValueError(f"unknown formation {kind!r}; choose from {sorted(FORMATIONS)}") from None
