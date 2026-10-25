"""Rigid transforms (rotation + translation) in NumPy, with quaternion algebra.

A :class:`Transform` ``T_parent_child`` maps points expressed in the *child* frame into the *parent*
frame: ``p_parent = R p_child + t``. Quaternions are ``[w, x, y, z]`` (Hamilton), as everywhere in Aevion.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

_EPS = 1e-12


def quat_normalize(q: Any) -> np.ndarray:
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if n < _EPS:
        raise ValueError("zero quaternion")
    return q / n


def quat_mul(a: Any, b: Any) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: Any) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_matrix(q: Any) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: Any) -> np.ndarray:
    """Shepperd's method (numerically stable for all rotations)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = quat_normalize(q)
    return q if q[0] >= 0 else -q


def quat_from_rpy(roll: Any, pitch: Any, yaw: Any) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_to_rpy(q: Any) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ]
    )


def quat_slerp(a: Any, b: Any, t: float) -> np.ndarray:
    a, b = quat_normalize(a), quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return quat_normalize(a + t * (b - a))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


class Transform:
    """``T_parent_child``: rotation (unit quaternion) and translation."""

    __slots__ = ("q", "t")

    def __init__(self, translation: ArrayLike = (0.0, 0.0, 0.0), rotation: ArrayLike = (1.0, 0.0, 0.0, 0.0)) -> None:
        t = np.asarray(translation, float).reshape(3)
        self.t = t
        self.q = quat_normalize(rotation)

    # -------------------------------------------------------------- constructors
    @classmethod
    def identity(cls) -> "Transform":
        return cls()

    @classmethod
    def from_rpy(cls, rpy: Sequence[float], translation: Sequence[float] = (0, 0, 0), degrees: bool = False) -> "Transform":
        r = np.radians(rpy) if degrees else np.asarray(rpy, float)
        return cls(translation, quat_from_rpy(*r))

    @classmethod
    def from_matrix(cls, M: Any) -> "Transform":
        M = np.asarray(M, float)
        if M.shape != (4, 4):
            raise ValueError("expected a 4x4 homogeneous matrix")
        return cls(M[:3, 3], matrix_to_quat(M[:3, :3]))

    @classmethod
    def from_dict(cls, d: dict) -> "Transform":
        if "rpy_deg" in d:
            return cls.from_rpy(d["rpy_deg"], d.get("translation", (0, 0, 0)), degrees=True)
        if "rpy" in d:
            return cls.from_rpy(d["rpy"], d.get("translation", (0, 0, 0)))
        return cls(d.get("translation", (0, 0, 0)), d.get("quaternion", (1, 0, 0, 0)))

    # ------------------------------------------------------------------ queries
    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.q)

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.rotation
        M[:3, 3] = self.t
        return M

    def rpy(self, degrees: bool = False) -> np.ndarray:
        r = quat_to_rpy(self.q)
        return np.degrees(r) if degrees else r

    def to_dict(self) -> dict:
        return {"translation": self.t.tolist(), "quaternion": self.q.tolist()}

    # --------------------------------------------------------------- operations
    def inverse(self) -> "Transform":
        qi = quat_conj(self.q)
        R = quat_to_matrix(qi)
        return Transform(-R @ self.t, qi)

    def compose(self, other: "Transform") -> "Transform":
        """``self ∘ other``: first apply ``other`` then ``self`` (``T_a_c = T_a_b @ T_b_c``)."""
        return Transform(self.rotation @ other.t + self.t, quat_mul(self.q, other.q))

    __matmul__ = compose

    def apply(self, points: Any) -> np.ndarray:
        """Transform points ``(..., 3)`` from the child frame to the parent frame."""
        p = np.asarray(points, float)
        return p @ self.rotation.T + self.t

    def rotate(self, vectors: Any) -> np.ndarray:
        """Rotate free vectors (no translation)."""
        return np.asarray(vectors, float) @ self.rotation.T

    def interpolate(self, other: "Transform", a: float) -> "Transform":
        return Transform((1 - a) * self.t + a * other.t, quat_slerp(self.q, other.q, a))

    def isclose(self, other: "Transform", atol: float = 1e-9) -> bool:
        return bool(np.allclose(self.t, other.t, atol=atol) and (np.allclose(self.q, other.q, atol=atol) or np.allclose(self.q, -other.q, atol=atol)))

    def __repr__(self) -> str:
        return f"Transform(t={np.round(self.t, 4).tolist()}, rpy_deg={np.round(self.rpy(True), 2).tolist()})"
