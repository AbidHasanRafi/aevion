"""Camera lens models: map pixels to viewing rays (and back).

Camera frame: ``x`` forward, ``y`` left, ``z`` up (the body-frame convention). Pixel ``(u, v)`` has ``u`` to the right and ``v``
downward; the principal point is ``(cx, cy)`` and ``f`` is the focal length in pixels.

* :class:`PinholeLens`   rectilinear projection with optional Brown-Conrady distortion (``k1, k2, k3`` radial, ``p1, p2`` tangential).
* :class:`FisheyeLens`   ``equidistant`` (``r = f theta``), ``equisolid`` (``2 f sin(theta/2)``), ``stereographic`` (``2 f tan(theta/2)``)
  or ``orthographic`` (``f sin(theta)``); the field of view may reach and exceed 180 degrees.

``unproject`` returns the unit ray of every pixel and a validity mask (pixels outside a fisheye's image circle, or whose distortion
cannot be inverted, are invalid); ``project`` is the inverse. The tests check ``unproject(project(d)) == d``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["Lens", "PinholeLens", "FisheyeLens", "make_lens", "FISHEYE_MODELS"]

FISHEYE_MODELS = ("equidistant", "equisolid", "stereographic", "orthographic")


class Lens:
    kind = "lens"
    f: float  # focal length in pixels (set by the subclass)

    def __init__(self, width: int, height: int, fov_deg: float):
        if width < 2 or height < 2 or not 1.0 <= fov_deg < 360.0:
            raise ValueError("need width, height >= 2 and a horizontal field of view in [1, 360) degrees")
        self.W, self.H, self.fov = int(width), int(height), float(fov_deg)
        self.cx, self.cy = self.W / 2.0, self.H / 2.0

    def pixel_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        return u.reshape(-1), v.reshape(-1)

    def unproject(
        self, u: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def project(self, dirs: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class PinholeLens(Lens):
    kind = "pinhole"

    def __init__(self, width: int, height: int, fov_deg: float = 90.0, distortion: Optional[Tuple[float, ...]] = None):
        super().__init__(width, height, fov_deg)
        if fov_deg >= 180.0:
            raise ValueError("a pinhole lens needs a field of view below 180 degrees (use a fisheye lens)")
        self.f = 0.5 * self.W / math.tan(math.radians(self.fov) / 2.0)
        d = list(distortion) if distortion is not None else [0.0] * 5
        d = (d + [0.0] * 5)[:5]
        self.k1, self.k2, self.p1, self.p2, self.k3 = (float(x) for x in d)  # OpenCV order: k1, k2, p1, p2, k3
        self.distorted = any(abs(x) > 0 for x in (self.k1, self.k2, self.p1, self.p2, self.k3))

    def _distort(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r2 = x * x + y * y
        radial = 1.0 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
        return x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x), y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y

    def unproject(self, u: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if u is None:
            u, v = self.pixel_grid()
        assert u is not None and v is not None
        xd, yd = (u - self.cx) / self.f, (v - self.cy) / self.f
        x, y = xd.copy(), yd.copy()
        if self.distorted:  # fixed-point inversion of the distortion model
            for _ in range(25):
                dx, dy = self._distort(x, y)
                x, y = x + (xd - dx), y + (yd - dy)
            ex, ey = self._distort(x, y)
            ok = np.hypot(ex - xd, ey - yd) < 1e-6
        else:
            ok = np.ones(xd.shape, bool)
        d = np.stack([np.ones_like(x), -x, -y], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True), ok

    def project(self, dirs: np.ndarray) -> np.ndarray:
        d = np.asarray(dirs, float)
        x, y = -d[..., 1] / d[..., 0], -d[..., 2] / d[..., 0]
        xd, yd = self._distort(x, y) if self.distorted else (x, y)
        return np.stack([xd * self.f + self.cx, yd * self.f + self.cy], axis=-1)

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="pinhole", fov_deg=self.fov, distortion=[self.k1, self.k2, self.p1, self.p2, self.k3] if self.distorted else None)


class FisheyeLens(Lens):
    kind = "fisheye"

    def __init__(self, width: int, height: int, fov_deg: float = 180.0, model: str = "equidistant"):
        super().__init__(width, height, fov_deg)
        if model not in FISHEYE_MODELS:
            raise ValueError(f"fisheye model must be one of {FISHEYE_MODELS}")
        self.model = model
        half = math.radians(self.fov) / 2.0
        if model == "orthographic" and half > math.pi / 2:
            raise ValueError("an orthographic lens cannot exceed 180 degrees")
        self.f = 0.5 * self.W / self._radius_of(half)  # the image circle is inscribed in the image width

    def _radius_of(self, theta: Any) -> Any:
        """Image radius in units of the focal length for a ray ``theta`` off the optical axis."""
        t = np.asarray(theta, float)
        if self.model == "equidistant":
            return t
        if self.model == "equisolid":
            return 2.0 * np.sin(t / 2.0)
        if self.model == "stereographic":
            return 2.0 * np.tan(t / 2.0)
        return np.sin(t)

    def _theta_of(self, r: np.ndarray) -> np.ndarray:
        if self.model == "equidistant":
            return r
        if self.model == "equisolid":
            return 2.0 * np.arcsin(np.clip(r / 2.0, -1.0, 1.0))
        if self.model == "stereographic":
            return 2.0 * np.arctan(r / 2.0)
        return np.arcsin(np.clip(r, -1.0, 1.0))

    def unproject(self, u: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if u is None:
            u, v = self.pixel_grid()
        assert u is not None and v is not None
        dx, dy = u - self.cx, v - self.cy
        r = np.hypot(dx, dy) / self.f
        theta = self._theta_of(r)
        phi = np.arctan2(dy, dx)
        ok = np.hypot(dx, dy) <= 0.5 * self.W + 1e-9  # inside the image circle
        d = np.stack([np.cos(theta), -np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi)], axis=-1)
        return d, ok

    def project(self, dirs: np.ndarray) -> np.ndarray:
        d = np.asarray(dirs, float)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        theta = np.arccos(np.clip(d[..., 0], -1.0, 1.0))
        phi = np.arctan2(-d[..., 2], -d[..., 1])
        rho = self._radius_of(theta) * self.f
        return np.stack([self.cx + rho * np.cos(phi), self.cy + rho * np.sin(phi)], axis=-1)

    def to_dict(self) -> Dict[str, Any]:
        return dict(type="fisheye", fov_deg=self.fov, model=self.model)


def make_lens(spec: Any, width: int, height: int, fov_deg: float, distortion: Optional[Tuple[float, ...]] = None) -> Lens:
    """``"pinhole"`` | ``"equidistant"`` | ``"equisolid"`` | ``"stereographic"`` | ``"orthographic"`` | ``{"type": ..., ...}`` | a :class:`Lens`."""
    if isinstance(spec, Lens):
        return spec
    d = {"type": spec} if isinstance(spec, str) else dict(spec)
    kind = d.pop("type", "pinhole")
    d.setdefault("fov_deg", fov_deg)
    if kind == "pinhole":
        d.setdefault("distortion", distortion)
        return PinholeLens(width, height, **d)
    if kind == "fisheye":
        return FisheyeLens(width, height, **d)
    if kind in FISHEYE_MODELS:
        return FisheyeLens(width, height, model=kind, **d)
    raise ValueError(f"unknown lens {kind!r}; use pinhole, fisheye or one of {FISHEYE_MODELS}")
