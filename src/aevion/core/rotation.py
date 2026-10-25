"""Batched quaternion / rotation utilities.

Conventions
-----------
* Quaternions are ``[w, x, y, z]`` (Hamilton), unit norm, rotating **body -> world**.
* World frame is ENU-like (``z`` up). Body frame is FLU (x forward, y left, z up).
* Euler angles are intrinsic Z-Y-X (yaw, pitch, roll).
* All functions are batched over leading dimensions and take a ``Backend``.
"""

from __future__ import annotations

import math


def qmul(B, a, b):
    """Hamilton product ``a ⊗ b``."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return B.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def qconj(B, q):
    return B.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def qnormalize(B, q):
    return q / B.maximum(B.norm(q, axis=-1, keepdims=True), 1e-12 * B.ones(1))


def to_matrix(B, q):
    """Rotation matrix (body -> world), shape ``(..., 3, 3)``."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r0 = B.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1)
    r1 = B.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1)
    r2 = B.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1)
    return B.stack([r0, r1, r2], axis=-2)


def rotate(B, q, v):
    """Rotate vectors ``v`` (body -> world) by ``q``."""
    return B.einsum("...ij,...j->...i", to_matrix(B, q), v)


def rotate_inv(B, q, v):
    """Rotate vectors ``v`` (world -> body) by ``q``."""
    return B.einsum("...ji,...j->...i", to_matrix(B, q), v)


def from_euler(B, roll, pitch, yaw):
    cr, sr = B.cos(roll * 0.5), B.sin(roll * 0.5)
    cp, sp = B.cos(pitch * 0.5), B.sin(pitch * 0.5)
    cy, sy = B.cos(yaw * 0.5), B.sin(yaw * 0.5)
    return B.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def to_euler(B, q):
    """Return ``(roll, pitch, yaw)`` stacked on the last axis."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = B.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = B.arcsin(2 * (w * y - z * x))
    yaw = B.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return B.stack([roll, pitch, yaw], axis=-1)


def from_axis_angle(B, axis, angle):
    n = B.norm(axis, axis=-1, keepdims=True)
    u = axis / B.maximum(n, 1e-12 * B.ones(1))
    h = angle[..., None] * 0.5
    return B.cat([B.cos(h), u * B.sin(h)], axis=-1)


def from_rotvec(B, rv):
    """Quaternion from a rotation vector (axis * angle), safe at zero."""
    ang = B.norm(rv, axis=-1, keepdims=True)
    half = ang * 0.5
    small = ang < 1e-8
    k = B.where(small, 0.5 - ang * ang / 48.0, B.sin(half) / B.maximum(ang, 1e-12 * B.ones(1)))
    return B.cat([B.cos(half), rv * k], axis=-1)


def identity(B, *batch):
    q = B.zeros(*batch, 4)
    q = q + B.asarray([1.0, 0.0, 0.0, 0.0])
    return q


def deriv(B, q, omega_body):
    """``q̇ = ½ q ⊗ [0, ω_body]``."""
    zero = omega_body[..., :1] * 0.0
    return 0.5 * qmul(B, q, B.cat([zero, omega_body], axis=-1))


def integrate(B, q, omega_body, dt):
    """Exact (exponential-map) attitude update for constant body rate over ``dt``."""
    return qnormalize(B, qmul(B, q, from_rotvec(B, omega_body * dt)))


def angle_between(B, qa, qb):
    """Geodesic angle (rad) between two attitudes."""
    d = B.abs(B.sum(qa * qb, axis=-1))
    return 2.0 * B.arccos(B.clip(d, 0.0, 1.0))


def hat(B, v):
    z = v[..., 0] * 0.0
    return B.stack(
        [
            B.stack([z, -v[..., 2], v[..., 1]], axis=-1),
            B.stack([v[..., 2], z, -v[..., 0]], axis=-1),
            B.stack([-v[..., 1], v[..., 0], z], axis=-1),
        ],
        axis=-2,
    )


def vee(B, m):
    return B.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], axis=-1)


def deg2rad(x):
    return x * math.pi / 180.0


def rad2deg(x):
    return x * 180.0 / math.pi
