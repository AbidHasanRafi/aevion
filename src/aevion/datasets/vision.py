"""Vision datasets from the ray-cast camera: PNG images, COCO and YOLO labels, and optical-flow ground truth.

The camera knows the exact geometry, so labels are exact rather than annotated:

* **Boxes** come from the per-pixel primitive ids (:func:`boxes_from_segmentation`) and cover only the visible part of an object.
* **Optical flow** (:func:`optical_flow`) back-projects every pixel with the depth image, views it from the next pose and projects it
  again; this is exact for a rigid static scene. A second depth image marks pixels occluded in the next frame. Moving obstacles are
  not rigid in the world, so their flow is wrong by their own motion, and the ``valid`` mask cannot detect this.
* **Images and depth** are written as PNG (:func:`write_png`, 8- and 16-bit, no image library needed); flow uses the Middlebury
  ``.flo`` format.

:class:`VisionRecorder` is an engine hook that produces all of it during a run. The images are stylised renderings, not photorealistic,
so expect a domain gap on real images.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..engine.hooks import Hook

__all__ = [
    "camera_pose",
    "depth_to_points",
    "project_points",
    "optical_flow",
    "boxes_from_segmentation",
    "object_table",
    "write_png",
    "read_png",
    "write_flo",
    "read_flo",
    "flow_to_rgb",
    "CocoWriter",
    "write_yolo_labels",
    "write_yolo_yaml",
    "VisionRecorder",
]


# --------------------------------------------------------------------------------------------- geometry
def camera_pose(cam: Any, pos: Any, quat: Any) -> Tuple[np.ndarray, np.ndarray]:
    """World position and camera-to-world rotation of a :class:`~aevion.sensors.ranging.Camera` on a vehicle at ``pos`` with attitude ``quat`` (w, x, y, z)."""
    w, x, y, z = (float(v) for v in np.asarray(quat, float))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    Rb = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rc2b = Ry
    if getattr(cam, "extrinsics", None) is not None:
        Rc2b = cam.extrinsics.rotate(np.eye(3)).T @ Ry
    return np.asarray(pos, float) + Rb @ np.asarray(cam.mount, float), Rb @ Rc2b


def depth_to_points(depth: np.ndarray, lens: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Camera-frame 3-D points ``(H, W, 3)`` and a validity mask: pinhole depth is z-depth along the optical axis, fisheye depth is range."""
    d = np.asarray(depth, float)
    dirs, ok = lens.unproject()
    dirs = dirs.reshape(lens.H, lens.W, 3)
    ok = ok.reshape(lens.H, lens.W) & np.isfinite(d)
    if lens.kind == "pinhole":
        pts = dirs * (d / np.maximum(dirs[..., 0], 1e-9))[..., None]
    else:
        pts = dirs * d[..., None]
    return np.where(ok[..., None], pts, 0.0), ok


def project_points(points_cam: np.ndarray, lens: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates ``(..., 2)`` of camera-frame points and a mask of those in front of the camera and inside the image."""
    p = np.asarray(points_cam, float)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    dirs = p / np.maximum(n, 1e-12)
    front = (p[..., 0] > 1e-6) if lens.kind == "pinhole" else (n[..., 0] > 1e-6)
    uv = lens.project(dirs if lens.kind != "pinhole" else p)
    inside = front & (uv[..., 0] >= 0) & (uv[..., 0] < lens.W) & (uv[..., 1] >= 0) & (uv[..., 1] < lens.H) & np.isfinite(uv).all(-1)
    return uv, inside


def optical_flow(
    depth1: np.ndarray,
    lens: Any,
    pose1: Tuple[np.ndarray, np.ndarray],
    pose2: Tuple[np.ndarray, np.ndarray],
    depth2: Optional[np.ndarray] = None,
    rel_tol: float = 0.03,
    abs_tol: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Ground-truth flow from frame 1 to frame 2 of a static scene.

    Returns ``flow`` ``(H, W, 2)`` in pixels (``u`` to the right, ``v`` downward), ``valid`` (a depth exists and the point lands inside frame 2) and, if
    ``depth2`` is given, ``occluded`` (the point is hidden behind something else in frame 2: its predicted depth exceeds the depth seen there).
    """
    o1, R1 = pose1
    o2, R2 = pose2
    pts, ok = depth_to_points(depth1, lens)
    world = o1 + pts @ R1.T
    p2 = (world - o2) @ R2
    uv2, inside = project_points(p2, lens)
    u, v = lens.pixel_grid()
    uv1 = np.stack([u, v], -1).reshape(lens.H, lens.W, 2)
    valid = ok & inside
    flow = np.where(valid[..., None], uv2 - uv1, 0.0)
    out: Dict[str, np.ndarray] = dict(flow=flow, valid=valid)
    if depth2 is not None:
        d2 = np.asarray(depth2, float)
        iu = np.clip(np.floor(uv2[..., 0]).astype(int), 0, lens.W - 1)
        iv = np.clip(np.floor(uv2[..., 1]).astype(int), 0, lens.H - 1)
        seen = d2[iv, iu]
        pred = p2[..., 0] if lens.kind == "pinhole" else np.linalg.norm(p2, axis=-1)
        out["occluded"] = valid & np.isfinite(seen) & (pred > seen * (1 + rel_tol) + abs_tol)
    return out


# --------------------------------------------------------------------------------------------- labels
def object_table(world: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Map primitive ids (boxes, spheres, cylinders, then mesh triangles) to object ids; returns ``(object_of_primitive, objects)`` with each object's ``category``."""
    objs: List[Dict[str, Any]] = []
    owner: List[int] = []
    for kind, n in (("box", len(world.box_colors)), ("sphere", len(world.sphere_colors)), ("cylinder", len(world.cyl_colors))):
        for i in range(n):
            owner.append(len(objs))
            objs.append(dict(category=kind, index=i))
    for mi, m in enumerate(world.meshes):
        oid = len(objs)
        objs.append(dict(category="mesh", index=mi))
        owner.extend([oid] * m.n_faces)
    return np.asarray(owner, int), objs


def boxes_from_segmentation(seg: np.ndarray, owner: Optional[np.ndarray] = None, min_pixels: int = 4) -> List[Dict[str, Any]]:
    """Tight boxes ``(x0, y0, x1, y1)`` (pixel edges, ``x1``/``y1`` exclusive) of the visible part of every object in a primitive-id image (``-1`` = background)."""
    s = np.asarray(seg)
    ids = np.where(s >= 0, s, -1).astype(int)
    if owner is not None:
        ok = (ids >= 0) & (ids < len(owner))
        ids = np.where(ok, owner[np.clip(ids, 0, max(len(owner) - 1, 0))], -1)
    out: List[Dict[str, Any]] = []
    for oid in np.unique(ids[ids >= 0]):
        m = ids == oid
        n = int(m.sum())
        if n < min_pixels:
            continue
        ys, xs = np.nonzero(m)
        out.append(dict(object=int(oid), x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()) + 1, y1=int(ys.max()) + 1, pixels=n))
    return out


# --------------------------------------------------------------------------------------------- file formats
def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Any, img: np.ndarray) -> Path:
    """Write ``(H, W)`` or ``(H, W, 3|4)`` ``uint8`` / ``(H, W)`` ``uint16`` data as a PNG (zlib-compressed, no filtering)."""
    a = np.asarray(img)
    if a.dtype == np.uint8 and a.ndim in (2, 3):
        depth = 8
    elif a.dtype == np.uint16 and a.ndim == 2:
        depth = 16
    else:
        raise ValueError("write_png takes uint8 (H, W[, 3|4]) or uint16 (H, W) data")
    if a.ndim == 2:
        color = 0
    elif a.shape[2] in (3, 4):
        color = 2 if a.shape[2] == 3 else 6
    else:
        raise ValueError("a colour image needs 3 or 4 channels")
    h, w = a.shape[:2]
    rows = a.astype(">u2").tobytes() if depth == 16 else a.tobytes()
    stride = len(rows) // h
    raw = b"".join(b"\x00" + rows[i * stride : (i + 1) * stride] for i in range(h))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )
    return p


def read_png(path: Any) -> np.ndarray:
    """Decode a non-interlaced 8/16-bit grey, RGB or RGBA PNG (all five scan-line filters)."""
    b = Path(path).read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos, idat, hdr = 8, b"", None
    while pos < len(b):
        n = struct.unpack(">I", b[pos : pos + 4])[0]
        tag, data = b[pos + 4 : pos + 8], b[pos + 8 : pos + 8 + n]
        if zlib.crc32(tag + data) & 0xFFFFFFFF != struct.unpack(">I", b[pos + 8 + n : pos + 12 + n])[0]:
            raise ValueError(f"corrupt PNG chunk {tag!r}")
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat += data
        pos += 12 + n
    if hdr is None:
        raise ValueError("PNG has no header")
    w, h, depth, color, _, _, interlace = hdr
    if interlace or depth not in (8, 16) or color not in (0, 2, 6):
        raise ValueError("unsupported PNG variant (need non-interlaced 8/16-bit grey, RGB or RGBA)")
    ch = {0: 1, 2: 3, 6: 4}[color]
    bpp = ch * depth // 8
    stride = w * bpp
    raw = zlib.decompress(idat)
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.uint8)
    for i in range(h):
        f = raw[i * (stride + 1)]
        line = np.frombuffer(raw[i * (stride + 1) + 1 : (i + 1) * (stride + 1)], np.uint8).astype(np.int32)
        if f == 0:
            cur = line
        elif f == 1:
            cur = line.copy()
            for k in range(bpp, stride):
                cur[k] = (cur[k] + cur[k - bpp]) & 255
        elif f == 2:
            cur = (line + prev) & 255
        elif f == 3:
            cur = line.copy()
            for k in range(stride):
                left = cur[k - bpp] if k >= bpp else 0
                cur[k] = (cur[k] + ((left + int(prev[k])) >> 1)) & 255
        elif f == 4:
            cur = line.copy()
            for k in range(stride):
                a = int(cur[k - bpp]) if k >= bpp else 0
                bb = int(prev[k])
                c = int(prev[k - bpp]) if k >= bpp else 0
                p = a + bb - c
                pa, pb, pc = abs(p - a), abs(p - bb), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (bb if pb <= pc else c)
                cur[k] = (cur[k] + pr) & 255
        else:
            raise ValueError(f"bad PNG filter {f}")
        out[i] = cur.astype(np.uint8)
        prev = out[i]
    img = np.frombuffer(out.tobytes(), ">u2").astype(np.uint16).reshape(h, w, ch) if depth == 16 else out.reshape(h, w, ch)
    return img[..., 0] if ch == 1 else img


def write_flo(path: Any, flow: np.ndarray) -> Path:
    """Middlebury ``.flo``: magic ``PIEH``, width, height (int32), then ``float32`` (u, v) pairs row by row."""
    f = np.asarray(flow, np.float32)
    if f.ndim != 3 or f.shape[2] != 2:
        raise ValueError("flow must be (H, W, 2)")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"PIEH" + struct.pack("<ii", f.shape[1], f.shape[0]) + f.astype("<f4").tobytes())
    return p


def read_flo(path: Any) -> np.ndarray:
    b = Path(path).read_bytes()
    if b[:4] != b"PIEH":
        raise ValueError("not a .flo file")
    w, h = struct.unpack("<ii", b[4:12])
    return np.frombuffer(b[12:], "<f4").reshape(h, w, 2).astype(np.float32)


def flow_to_rgb(flow: np.ndarray, max_mag: Optional[float] = None) -> np.ndarray:
    """Colour-wheel visualisation: hue = direction, saturation = magnitude (``uint8``)."""
    u, v = flow[..., 0], flow[..., 1]
    mag = np.hypot(u, v)
    m = float(max_mag if max_mag is not None else max(mag.max(), 1e-9))
    hue = (np.arctan2(-v, -u) / np.pi + 1.0) / 2.0
    sat = np.clip(mag / m, 0.0, 1.0)
    i = np.floor(hue * 6).astype(int) % 6
    f = hue * 6 - np.floor(hue * 6)
    p, q, t = 1 - sat, 1 - sat * f, 1 - sat * (1 - f)
    r = np.choose(i, [np.ones_like(f), q, p, p, t, np.ones_like(f)])
    g = np.choose(i, [t, np.ones_like(f), np.ones_like(f), q, p, p])
    b = np.choose(i, [p, p, t, np.ones_like(f), np.ones_like(f), q])
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


class CocoWriter:
    """Accumulates a COCO detection dataset (``images``, ``annotations``, ``categories``) and writes the JSON; the format is validated on save."""

    def __init__(self, categories: Sequence[str], description: str = "Aevion simulated dataset") -> None:
        self.categories = [dict(id=i + 1, name=n) for i, n in enumerate(categories)]
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.description = description

    def add_image(self, file_name: str, width: int, height: int, **extra: Any) -> int:
        iid = len(self.images) + 1
        self.images.append(dict(id=iid, file_name=file_name, width=int(width), height=int(height), **extra))
        return iid

    def add_annotation(
        self, image_id: int, category: str, x0: float, y0: float, x1: float, y1: float, area: Optional[float] = None, **extra: Any
    ) -> int:
        cid = next((c["id"] for c in self.categories if c["name"] == category), None)
        if cid is None:
            raise ValueError(f"unknown category {category!r}")
        w, h = float(x1 - x0), float(y1 - y0)
        if w <= 0 or h <= 0:
            raise ValueError("a box needs positive width and height")
        aid = len(self.annotations) + 1
        self.annotations.append(
            dict(
                id=aid,
                image_id=int(image_id),
                category_id=cid,
                bbox=[float(x0), float(y0), w, h],
                area=float(area if area is not None else w * h),
                iscrowd=0,
                **extra,
            )
        )
        return aid

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            info=dict(description=self.description, version="1.0"), images=self.images, annotations=self.annotations, categories=self.categories
        )

    def save(self, path: Any) -> Path:
        ids = {im["id"] for im in self.images}
        for a in self.annotations:
            if a["image_id"] not in ids:
                raise ValueError(f"annotation {a['id']} refers to a missing image")
            im = next(i for i in self.images if i["id"] == a["image_id"])
            x, y, w, h = a["bbox"]
            if x < -1e-6 or y < -1e-6 or x + w > im["width"] + 1e-6 or y + h > im["height"] + 1e-6:
                raise ValueError(f"annotation {a['id']} lies outside its image")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return p


def write_yolo_labels(path: Any, boxes: Sequence[Tuple[int, float, float, float, float]], width: int, height: int) -> Path:
    """One line per object, ``class cx cy w h`` normalised to ``[0, 1]`` (the YOLO text format); boxes are ``(class, x0, y0, x1, y1)`` in pixels."""
    lines = []
    for c, x0, y0, x1, y1 in boxes:
        lines.append(f"{int(c)} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


def write_yolo_yaml(path: Any, names: Sequence[str], train: str = "images", val: str = "images") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = [f"path: {p.parent.as_posix()}", f"train: {train}", f"val: {val}", f"nc: {len(names)}", "names:"] + [
        f"  {i}: {n}" for i, n in enumerate(names)
    ]
    p.write_text("\n".join(body) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------------------------- recorder
class VisionRecorder(Hook):
    """Engine hook that writes a labelled image dataset while the simulation runs.

    ``camera`` is the sensor name; it should have ``modes`` containing ``"depth"`` and ``"seg"`` (and ``"rgb"`` for images). Every ``every``-th camera
    sample is saved as ``images/NNNNNN.png`` (RGB) or the depth/segmentation images, with ``depth16/`` (millimetres, uint16), ``seg/`` (object id + 1,
    0 = background), ``flow/`` (flow from this frame to the next, ``.flo``), COCO ``annotations.json`` and YOLO ``labels/`` + ``data.yaml``. ``categories``
    maps object kinds (``box``, ``sphere``, ``cylinder``, ``mesh``) to class names; kinds that are missing are labelled with their own name.
    """

    name = "vision_recorder"
    priority = 96

    def __init__(
        self,
        out_dir: Any,
        camera: str = "camera",
        every: int = 1,
        max_frames: int = 500,
        vehicle: int = 0,
        categories: Optional[Dict[str, str]] = None,
        flow: bool = True,
        min_pixels: int = 12,
    ) -> None:
        self.out, self.cam_name, self.every, self.max_frames, self.vehicle = Path(out_dir), camera, max(int(every), 1), int(max_frames), int(vehicle)
        self.cat_map = categories or {}
        self.flow, self.min_pixels = flow, int(min_pixels)
        self.frames = 0
        self._seen = 0
        self._prev: Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], str]] = None
        self.coco: Optional[CocoWriter] = None
        self.class_names: List[str] = []
        self.owner: Optional[np.ndarray] = None
        self.objects: List[Dict[str, Any]] = []

    def on_reset(self, sim: Any) -> None:
        if sim.suite is None or self.cam_name not in sim.suite:
            raise ValueError(f"vision_recorder: the simulation has no sensor named {self.cam_name!r}")
        cam = sim.suite[self.cam_name]
        missing = [m for m in ("depth", "seg") if m not in cam.modes]
        if missing:
            raise ValueError(f"vision_recorder: the camera needs the modes {missing} (it has {list(cam.modes)})")
        self.frames, self._seen, self._prev = 0, 0, None
        self.owner, self.objects = object_table(sim.world)
        kinds = sorted({o["category"] for o in self.objects}) or ["object"]
        self.class_names = [self.cat_map.get(k, k) for k in kinds]
        self._kind_index = {k: i for i, k in enumerate(kinds)}
        self.coco = CocoWriter(self.class_names)
        for sub in ("depth16", "seg", "labels"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)

    def post_step(self, sim: Any) -> None:
        if self.frames >= self.max_frames or not sim.ctx.fresh or not sim.ctx.fresh.get(self.cam_name):
            return
        self._seen += 1
        if (self._seen - 1) % self.every:
            return
        cam = sim.suite[self.cam_name]
        data = sim.ctx.readings[self.cam_name]
        B = sim.B
        v = self.vehicle
        pos, quat = np.asarray(B.to_numpy(sim.state.pos))[v], np.asarray(B.to_numpy(sim.state.quat))[v]
        pose = camera_pose(cam, pos, quat)
        depth = np.asarray(B.to_numpy(data["depth"]))[v]
        seg = np.asarray(B.to_numpy(data["seg"]))[v].astype(int) if "seg" in data else None
        name = f"{self.frames:06d}"
        if "rgb" in data:
            write_png(self.out / "images" / f"{name}.png", np.clip(np.asarray(B.to_numpy(data["rgb"]))[v], 0, 255).astype(np.uint8))
        mm = np.where(np.isfinite(depth), np.clip(depth * 1000.0, 0, 65535), 0).astype(np.uint16)
        write_png(self.out / "depth16" / f"{name}.png", mm)
        if self._prev is not None and self.flow:
            pdepth, ppose, pname = self._prev
            f = optical_flow(pdepth, cam.lens, ppose, pose, depth)
            write_flo(self.out / "flow" / f"{pname}.flo", np.where(f["valid"][..., None], f["flow"], np.nan))
        self._prev = (depth, pose, name)
        assert self.coco is not None and self.owner is not None
        iid = self.coco.add_image(f"images/{name}.png", cam.W, cam.H)
        yolo = []
        if seg is not None:
            owner_img = np.where(seg >= 0, self.owner[np.clip(seg, 0, len(self.owner) - 1)] + 1, 0).astype(np.uint16)
            write_png(self.out / "seg" / f"{name}.png", owner_img)
            for b in boxes_from_segmentation(seg, self.owner, self.min_pixels):  # seg is not None here
                kind = self.objects[b["object"]]["category"]
                self.coco.add_annotation(
                    iid, self.class_names[self._kind_index[kind]], b["x0"], b["y0"], b["x1"], b["y1"], area=b["pixels"], object_id=b["object"]
                )
                yolo.append((self._kind_index[kind], b["x0"], b["y0"], b["x1"], b["y1"]))
        write_yolo_labels(self.out / "labels" / f"{name}.txt", yolo, cam.W, cam.H)
        self.frames += 1

    def on_end(self, sim: Any, archive: Any = None) -> None:
        if self.coco is not None:
            self.coco.save(self.out / "annotations.json")
            write_yolo_yaml(self.out / "data.yaml", self.class_names)
