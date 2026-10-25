"""Triangle meshes: construction, primitives, polygon extrusion and OBJ / STL / glTF-binary import.

A :class:`TriangleMesh` is plain data (vertices, faces, a colour). Ray casting, collision and distance queries live in
:mod:`aevion.environments.mesh_index`, and a :class:`~aevion.environments.world.World` owns meshes through ``add_mesh``.

The loaders are written for untrusted files: file size, vertex and triangle counts are capped, indices are validated, no code runs
and no external resources are fetched (glTF buffers must be embedded or inside the ``.gltf`` file's directory).

The world frame is z-up. Many CAD exports are y-up: pass ``up="y"`` to rotate them (glTF is y-up, so :meth:`TriangleMesh.from_glb`
defaults to it). Limits: extrusion supports simple polygons without holes (a courtyard becomes a solid block), and OBJ faces with
more than three vertices are fan-triangulated, which is only correct for convex faces.
"""

from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["TriangleMesh", "MeshError", "extrude_polygon", "MAX_FILE_BYTES", "MAX_TRIANGLES"]

MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TRIANGLES = 5_000_000
MAX_VERTICES = 15_000_000
_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # (x, y, z)_yup -> (x, -z, y)_zup


class MeshError(ValueError):
    """Invalid or unsupported mesh data."""


def _rotation_matrix(rotation_deg: Sequence[float]) -> np.ndarray:
    """Intrinsic z-y-x (yaw, pitch, roll) rotation from ``(roll, pitch, yaw)`` degrees."""
    r, p, y = (math.radians(a) for a in rotation_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class TriangleMesh:
    """``vertices`` ``(V, 3)`` float, ``faces`` ``(F, 3)`` int (counter-clockwise seen from outside), a name and an RGB colour in [0, 1]."""

    def __init__(self, vertices: Any, faces: Any, name: str = "mesh", color: Sequence[float] = (0.72, 0.72, 0.75)):
        v = np.ascontiguousarray(np.asarray(vertices, dtype=float))
        f = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
        if v.ndim != 2 or v.shape[1] != 3:
            raise MeshError(f"vertices must have shape (V, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise MeshError(f"faces must have shape (F, 3), got {f.shape}")
        if len(v) > MAX_VERTICES or len(f) > MAX_TRIANGLES:
            raise MeshError(f"mesh too large ({len(v)} vertices, {len(f)} triangles; limits {MAX_VERTICES}, {MAX_TRIANGLES})")
        if not np.all(np.isfinite(v)):
            raise MeshError("vertices contain NaN or inf")
        if len(f) and (f.min() < 0 or f.max() >= len(v)):
            raise MeshError("face indices out of range")
        c = tuple(float(x) for x in color)
        if len(c) != 3 or any(not 0.0 <= x <= 1.0 for x in c):
            raise MeshError("color must be three values in [0, 1]")
        self.vertices, self.faces, self.name, self.color = v, f, str(name), c

    # ---------------------------------------------------------------------------------------- properties
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> np.ndarray:
        """``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``."""
        if not len(self.vertices):
            return np.zeros((2, 3))
        return np.stack([self.vertices.min(0), self.vertices.max(0)])

    def triangles(self) -> np.ndarray:
        """``(F, 3, 3)`` triangle corner coordinates."""
        return self.vertices[self.faces]

    def face_normals(self) -> np.ndarray:
        t = self.triangles()
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-300)

    def areas(self) -> np.ndarray:
        t = self.triangles()
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)

    @property
    def surface_area(self) -> float:
        return float(self.areas().sum())

    @property
    def volume(self) -> float:
        """Signed volume by the divergence theorem (positive for outward-facing counter-clockwise faces of a closed mesh)."""
        t = self.triangles()
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def is_watertight(self) -> bool:
        """Every edge is shared by exactly two faces, traversed in opposite directions (a closed, consistently oriented surface)."""
        if not len(self.faces):
            return False
        e = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        key_fwd = e[:, 0] * self.n_vertices + e[:, 1]
        key_rev = e[:, 1] * self.n_vertices + e[:, 0]
        fwd, rev = np.sort(key_fwd), np.sort(key_rev)
        return bool(len(np.unique(key_fwd)) == len(key_fwd) and np.array_equal(fwd, rev))

    def degenerate_faces(self, tol: float = 1e-12) -> np.ndarray:
        """Indices of zero-area triangles (they never block a ray but may indicate a broken export)."""
        return np.nonzero(self.areas() <= tol)[0]

    # ------------------------------------------------------------------------------------------ transforms
    def transformed(
        self,
        scale: Union[float, Sequence[float]] = 1.0,
        rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
        translation: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "TriangleMesh":
        """Scale, then rotate by ``(roll, pitch, yaw)`` degrees, then translate. A mirroring scale flips the face winding."""
        s = np.broadcast_to(np.asarray(scale, float), (3,))
        v = (self.vertices * s) @ _rotation_matrix(rotation_deg).T + np.asarray(translation, float)
        faces = self.faces[:, [0, 2, 1]] if np.prod(np.sign(s)) < 0 else self.faces
        return TriangleMesh(v, faces, self.name, self.color)

    def with_color(self, color: Sequence[float]) -> "TriangleMesh":
        return TriangleMesh(self.vertices, self.faces, self.name, color)

    @classmethod
    def merge(cls, meshes: Sequence["TriangleMesh"], name: str = "merged") -> "TriangleMesh":
        if not meshes:
            return cls(np.zeros((0, 3)), np.zeros((0, 3), int), name)
        offs = np.cumsum([0] + [m.n_vertices for m in meshes[:-1]])
        return cls(np.concatenate([m.vertices for m in meshes]), np.concatenate([m.faces + o for m, o in zip(meshes, offs)]), name, meshes[0].color)

    # --------------------------------------------------------------------------------------- primitives
    @classmethod
    def box(cls, center: Sequence[float] = (0, 0, 0), size: Sequence[float] = (1, 1, 1), **kw: Any) -> "TriangleMesh":
        c, s = np.asarray(center, float), np.asarray(size, float) / 2.0
        corners = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float) * s + c
        f = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4), (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
        return cls(corners, f, **kw)

    @classmethod
    def cylinder(cls, center: Sequence[float] = (0, 0, 0), radius: float = 1.0, height: float = 1.0, segments: int = 24, **kw: Any) -> "TriangleMesh":
        """Vertical cylinder whose base centre is ``center``."""
        if segments < 3:
            raise MeshError("a cylinder needs at least 3 segments")
        a = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        ring = np.stack([radius * np.cos(a), radius * np.sin(a)], 1)
        c = np.asarray(center, float)
        v = (
            np.concatenate(
                [np.column_stack([ring, np.zeros(segments)]), np.column_stack([ring, np.full(segments, height)]), [[0, 0, 0], [0, 0, height]]]
            )
            + c
        )
        bc, tc, f = 2 * segments, 2 * segments + 1, []
        for i in range(segments):
            j = (i + 1) % segments
            f += [(i, j, segments + j), (i, segments + j, segments + i), (bc, j, i), (tc, segments + i, segments + j)]
        return cls(v, f, **kw)

    @classmethod
    def sphere(cls, center: Sequence[float] = (0, 0, 0), radius: float = 1.0, n_lat: int = 12, n_lon: int = 24, **kw: Any) -> "TriangleMesh":
        if n_lat < 2 or n_lon < 3:
            raise MeshError("a sphere needs n_lat >= 2 and n_lon >= 3")
        c = np.asarray(center, float)
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)[1:-1]
        lon = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
        ring = [[radius * np.cos(p) * np.cos(l), radius * np.cos(p) * np.sin(l), radius * np.sin(p)] for p in lat for l in lon]
        v = np.array([[0, 0, -radius]] + ring + [[0, 0, radius]], float) + c
        f = []
        for i in range(n_lon):
            f.append((0, 1 + (i + 1) % n_lon, 1 + i))
        for r in range(n_lat - 2):
            for i in range(n_lon):
                a, b = 1 + r * n_lon + i, 1 + r * n_lon + (i + 1) % n_lon
                c2, d = a + n_lon, b + n_lon
                f += [(a, b, d), (a, d, c2)]
        top = len(v) - 1
        for i in range(n_lon):
            base = 1 + (n_lat - 2) * n_lon
            f.append((top, base + i, base + (i + 1) % n_lon))
        return cls(v, f, **kw)

    @classmethod
    def extrude(cls, polygon_xy: Any, height: float, base_z: float = 0.0, **kw: Any) -> "TriangleMesh":
        return extrude_polygon(polygon_xy, height, base_z, **kw)

    # ----------------------------------------------------------------------------------------- serialise
    def to_dict(self) -> Dict[str, Any]:
        """Self-contained JSON-able form (float32 vertices and int32 faces as base64)."""
        return dict(
            name=self.name,
            color=list(self.color),
            n_vertices=self.n_vertices,
            n_faces=self.n_faces,
            vertices_b64=base64.b64encode(self.vertices.astype("<f8").tobytes()).decode("ascii"),
            faces_b64=base64.b64encode(self.faces.astype("<i4").tobytes()).decode("ascii"),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TriangleMesh":
        nv, nf = int(d["n_vertices"]), int(d["n_faces"])
        if nv > MAX_VERTICES or nf > MAX_TRIANGLES:
            raise MeshError("serialised mesh exceeds the size limits")
        v = np.frombuffer(base64.b64decode(d["vertices_b64"]), dtype="<f8")
        f = np.frombuffer(base64.b64decode(d["faces_b64"]), dtype="<i4")
        if v.size != nv * 3 or f.size != nf * 3:
            raise MeshError("serialised mesh arrays do not match the declared sizes")
        return cls(v.reshape(nv, 3), f.reshape(nf, 3), d.get("name", "mesh"), d.get("color", (0.72, 0.72, 0.75)))

    def __repr__(self) -> str:
        return f"TriangleMesh({self.name!r}, {self.n_vertices} vertices, {self.n_faces} faces)"

    # ---------------------------------------------------------------------------------------------- IO
    @classmethod
    def load(cls, path: Union[str, Path], up: Optional[str] = None, **kw: Any) -> "TriangleMesh":
        """Load by file extension (``.obj``, ``.stl``, ``.glb``, ``.gltf``)."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".obj":
            return cls.from_obj(p, up=up or "z", **kw)
        if ext == ".stl":
            return cls.from_stl(p, up=up or "z", **kw)
        if ext in (".glb", ".gltf"):
            return cls.from_gltf(p, up=up or "y", **kw)
        raise MeshError(f"unsupported mesh format {ext!r}; use .obj, .stl, .glb or .gltf")

    @classmethod
    def from_obj(
        cls, path: Union[str, Path], up: str = "z", name: Optional[str] = None, color: Sequence[float] = (0.72, 0.72, 0.75)
    ) -> "TriangleMesh":
        data = _read_capped(path)
        verts: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []
        for lineno, raw in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            if tok[0] == "v" and len(tok) >= 4:
                try:
                    verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
                except ValueError as e:
                    raise MeshError(f"{path}:{lineno}: bad vertex") from e
            elif tok[0] == "f" and len(tok) >= 4:
                idx = []
                for t in tok[1:]:
                    try:
                        i = int(t.split("/")[0])
                    except ValueError as e:
                        raise MeshError(f"{path}:{lineno}: bad face index {t!r}") from e
                    idx.append(i - 1 if i > 0 else len(verts) + i)  # OBJ indices are 1-based; negative ones count back
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
            if len(verts) > MAX_VERTICES or len(faces) > MAX_TRIANGLES:
                raise MeshError("OBJ file exceeds the size limits")
        if not faces:
            raise MeshError(f"{path}: no faces found")
        return cls(_apply_up(np.asarray(verts, float), up), faces, name or Path(path).stem, color)

    @classmethod
    def from_stl(
        cls, path: Union[str, Path], up: str = "z", weld: bool = True, name: Optional[str] = None, color: Sequence[float] = (0.72, 0.72, 0.75)
    ) -> "TriangleMesh":
        data = _read_capped(path)
        if len(data) >= 84:
            (n,) = struct.unpack_from("<I", data, 80)
            binary = len(data) == 84 + 50 * n
        else:
            binary = False
        if binary:
            if n > MAX_TRIANGLES:
                raise MeshError("STL file exceeds the size limits")
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            tri = rec["v"].astype(float).reshape(-1, 3)
        else:
            vals = []
            for line in data.decode("utf-8", "replace").splitlines():
                t = line.split()
                if len(t) == 4 and t[0].lower() == "vertex":
                    try:
                        vals.append((float(t[1]), float(t[2]), float(t[3])))
                    except ValueError as e:
                        raise MeshError(f"{path}: bad vertex {line!r}") from e
                    if len(vals) > 3 * MAX_TRIANGLES:
                        raise MeshError("STL file exceeds the size limits")
            if not vals or len(vals) % 3:
                raise MeshError(f"{path}: not a valid STL file")
            tri = np.asarray(vals, float)
        tri = _apply_up(tri, up)
        if weld:
            uniq, inv = np.unique(np.round(tri, 9), axis=0, return_inverse=True)
            return cls(uniq, inv.reshape(-1, 3), name or Path(path).stem, color)
        return cls(tri, np.arange(len(tri)).reshape(-1, 3), name or Path(path).stem, color)

    @classmethod
    def from_gltf(cls, path: Union[str, Path], up: str = "y", name: Optional[str] = None, color: Optional[Sequence[float]] = None) -> "TriangleMesh":
        """Triangle geometry of every mesh node of a ``.glb`` / ``.gltf`` (node transforms applied, materials ignored)."""
        p = Path(path)
        raw = _read_capped(p)
        if raw[:4] == b"glTF":
            doc, bins = _parse_glb(raw)
        else:
            try:
                doc = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise MeshError(f"{p}: not a glTF file") from e
            bins = [None]
        buffers = _gltf_buffers(doc, bins, p.parent)
        verts, faces, off = [], [], 0

        def node_matrix(n: Dict[str, Any]) -> np.ndarray:
            if "matrix" in n:
                return np.asarray(n["matrix"], float).reshape(4, 4).T
            m = np.eye(4)
            qx, qy, qz, qw = n.get("rotation", (0, 0, 0, 1))
            rot = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            m[:3, :3] = rot * np.asarray(n.get("scale", (1, 1, 1)), float)[None, :]
            m[:3, 3] = n.get("translation", (0, 0, 0))
            return m

        def visit(ni: int, parent: np.ndarray, depth: int) -> None:
            nonlocal off
            if depth > 64:
                raise MeshError("glTF node hierarchy too deep")
            node = doc["nodes"][ni]
            m = parent @ node_matrix(node)
            if "mesh" in node:
                for prim in doc["meshes"][node["mesh"]].get("primitives", []):
                    if prim.get("mode", 4) != 4:
                        continue
                    pos = _gltf_accessor(doc, buffers, prim["attributes"]["POSITION"]).astype(float)
                    idx = _gltf_accessor(doc, buffers, prim["indices"]).astype(np.int64).reshape(-1) if "indices" in prim else np.arange(len(pos))
                    if len(idx) % 3:
                        raise MeshError("glTF primitive index count is not a multiple of three")
                    verts.append(pos @ m[:3, :3].T + m[:3, 3])
                    faces.append(idx.reshape(-1, 3) + off)
                    off += len(pos)
            for c in node.get("children", []):
                visit(int(c), m, depth + 1)

        scenes = doc.get("scenes") or [{"nodes": list(range(len(doc.get("nodes", []))))}]
        roots = scenes[doc.get("scene", 0)].get("nodes", [])
        for r in roots:
            visit(int(r), np.eye(4), 0)
        if not faces:
            raise MeshError(f"{p}: no triangle meshes found")
        v = _apply_up(np.concatenate(verts), up)
        return cls(v, np.concatenate(faces), name or p.stem, color or (0.72, 0.72, 0.75))

    from_glb = from_gltf


def _apply_up(v: np.ndarray, up: str) -> np.ndarray:
    if up == "z":
        return v
    if up == "y":
        return v @ _Y_UP_TO_Z_UP.T
    raise MeshError("up must be 'z' or 'y'")


def _read_capped(path: Union[str, Path]) -> bytes:
    p = Path(path)
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise MeshError(f"{p}: file is {size / 1e6:.0f} MB, above the {MAX_FILE_BYTES // 2**20} MB limit")
    return p.read_bytes()


def _parse_glb(raw: bytes) -> Tuple[Dict[str, Any], List[Optional[bytes]]]:
    if len(raw) < 20:
        raise MeshError("GLB too short")
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    if version != 2 or length > len(raw):
        raise MeshError("unsupported or truncated GLB (need glTF 2.0)")
    off, doc, bin_chunk = 12, None, None
    while off + 8 <= length:
        clen, ctype = struct.unpack_from("<I4s", raw, off)
        body = raw[off + 8 : off + 8 + clen]
        if len(body) != clen:
            raise MeshError("truncated GLB chunk")
        if ctype == b"JSON":
            try:
                doc = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise MeshError("GLB JSON chunk is invalid") from e
        elif ctype == b"BIN\x00" and bin_chunk is None:
            bin_chunk = body
        off += 8 + clen + (-clen) % 4
    if doc is None:
        raise MeshError("GLB has no JSON chunk")
    return doc, [bin_chunk]


def _gltf_buffers(doc: Dict[str, Any], bins: List[Optional[bytes]], base: Path) -> List[bytes]:
    out: List[bytes] = []
    for i, b in enumerate(doc.get("buffers", [])):
        uri = b.get("uri")
        if uri is None:
            if i != 0 or bins[0] is None:
                raise MeshError("glTF buffer without data")
            out.append(bins[0])
        elif uri.startswith("data:"):
            try:
                out.append(base64.b64decode(uri.split(",", 1)[1]))
            except Exception as e:  # malformed data URI
                raise MeshError("bad data URI in glTF buffer") from e
        else:
            target = (base / uri).resolve()
            if base.resolve() not in target.parents:
                raise MeshError(f"glTF buffer {uri!r} lies outside the model's directory")
            out.append(_read_capped(target))
    return out


_COMPONENT = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _gltf_accessor(doc: Dict[str, Any], buffers: List[bytes], index: int) -> np.ndarray:
    try:
        acc = doc["accessors"][index]
        if "sparse" in acc:
            raise MeshError("sparse glTF accessors are not supported")
        dt = np.dtype(_COMPONENT[acc["componentType"]])
        nc = _NCOMP[acc["type"]]
        count = int(acc["count"])
        if "bufferView" not in acc:
            return np.zeros((count, nc), dt) if nc > 1 else np.zeros(count, dt)
        view = doc["bufferViews"][acc["bufferView"]]
        buf = buffers[view["buffer"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride") or dt.itemsize * nc
        if count > MAX_VERTICES or (count and start + stride * (count - 1) + dt.itemsize * nc > len(buf)):
            raise MeshError("glTF accessor reads outside its buffer")
        if stride == dt.itemsize * nc:
            arr = np.frombuffer(buf, dtype=dt, count=count * nc, offset=start).reshape(count, nc)
        else:
            raw = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + dt.itemsize * nc, offset=start)
            arr = (
                np.stack([np.frombuffer(raw[i * stride : i * stride + dt.itemsize * nc].tobytes(), dtype=dt) for i in range(count)])
                if count
                else np.zeros((0, nc), dt)
            )
        return arr if nc > 1 else arr.reshape(-1)
    except (KeyError, IndexError, TypeError) as e:
        raise MeshError(f"malformed glTF accessor {index}: {e!r}") from e


# --------------------------------------------------------------------------------------------- extrusion
def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ear_clip(p: np.ndarray) -> List[Tuple[int, int, int]]:
    """Triangulate a simple counter-clockwise polygon (indices into ``p``); raises on self-intersecting input."""
    n = len(p)
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3:
        ear = False
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = p[i0], p[i1], p[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:
                continue  # reflex or collinear corner
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                q = p[j]
                d1 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
                d2 = (c[0] - b[0]) * (q[1] - b[1]) - (c[1] - b[1]) * (q[0] - b[0])
                d3 = (a[0] - c[0]) * (q[1] - c[1]) - (a[1] - c[1]) * (q[0] - c[0])
                if d1 >= -1e-12 and d2 >= -1e-12 and d3 >= -1e-12:
                    inside = True
                    break
            if not inside:
                tris.append((i0, i1, i2))
                idx.pop(k)
                ear = True
                break
        guard += 1
        if not ear or guard > 4 * n * n:
            raise MeshError("polygon cannot be triangulated (self-intersecting or degenerate footprint)")
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def extrude_polygon(polygon_xy: Any, height: float, base_z: float = 0.0, **kw: Any) -> TriangleMesh:
    """Prism over a simple polygon (no holes) from ``base_z`` to ``base_z + height``; closed and consistently oriented."""
    p = np.asarray(polygon_xy, float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise MeshError("polygon must have shape (n, 2)")
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]  # drop a repeated closing vertex
    if len(p) < 3:
        raise MeshError("a polygon needs at least three distinct vertices")
    if not height > 0 or not np.all(np.isfinite(p)):
        raise MeshError("height must be positive and coordinates finite")
    area = _signed_area(p)
    if abs(area) < 1e-9:
        raise MeshError("polygon has zero area")
    if area < 0:
        p = p[::-1]
    n = len(p)
    verts = np.concatenate([np.column_stack([p, np.full(n, base_z)]), np.column_stack([p, np.full(n, base_z + height)])])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    for a, b, c in _ear_clip(p):
        faces.append((n + a, n + b, n + c))  # top, counter-clockwise seen from above
        faces.append((a, c, b))  # bottom, facing down
    return TriangleMesh(verts, faces, **kw)
