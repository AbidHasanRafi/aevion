"""Turn a :class:`~aevion.datasets.RunArchive` into the viewer's ``aevion.view/2`` data.

Three consumers share this module so they cannot drift apart:

* :func:`build_payload`  one JSON-able dict with everything embedded (single-file replay, Jupyter, ``--payload``)
* :class:`FrameSet`      random access to decimated windows of the channels (the HTTP API of ``aevion view``)
* :func:`layer_data`     the non-frame data: world, LiDAR scans, cells, radio environment maps, antenna lobes, cameras, waypoints

Time is always relative to the first recorded sample, so a viewer never has to know the absolute clock.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .protocol import encode_bundle

__all__ = ["FORMAT", "CORE_CHANNELS", "FrameSet", "build_payload", "layer_data", "derive_events", "json_safe", "decimate_indices"]

FORMAT = "aevion.view/2"
CORE_CHANNELS = ("pos", "quat", "vel", "euler", "rotor", "thrust", "power", "soc", "wind")
_SKIP_EXTRA = set(CORE_CHANNELS) | {"t", "ref_pos"}
_PRIORITY = (
    "link_sinr_db",
    "link_rate_mbps",
    "link_serving",
    "link_los",
    "link_outage",
    "link_rsrp_dbm",
    "link_n_handovers",
    "link_mcs",
    "link_bler",
)
MAX_EVENTS = 500
MAX_MESH_TRIANGLES = 250_000  # beyond this the viewer would stall: further meshes are left out and counted in world.meshes_skipped


# ------------------------------------------------------------------------------------------------ helpers
def b64(a: Any, dtype: Any = np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def decimate_indices(T: int, max_frames: int) -> np.ndarray:
    """At most ``max_frames`` indices spread evenly over ``range(T)``, always including the first and last frame."""
    if max_frames >= T or T <= 2:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, max(int(max_frames), 2)).round().astype(int))


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy values to Python and non-finite floats to ``None`` (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ------------------------------------------------------------------------------------------------ frames
class FrameSet:
    """The viewer's view of an archive's per-frame channels.

    Core channels keep their archive names (``pos``, ``quat``, ...), the reference path is ``ref`` and the tracking error ``err``; every other
    (T, N[, k]) numeric channel is an *extra* named ``x:<archive name>`` (``x:link_sinr_db``, ...).
    """

    def __init__(self, archive: Any, extras: Union[str, Sequence[str], None, bool] = "auto"):
        self.archive = archive
        self.T, self.N = int(archive.n_steps), int(archive.n_vehicles)
        self.time = np.asarray(archive.t, float) - float(archive.t[0])
        self.core: List[str] = [n for n in CORE_CHANNELS if n in archive]
        if "ref_pos" in archive:
            self.core += ["ref", "err"]
        self.extra: List[str] = self._pick_extras(extras)

    def _eligible(self, name: str) -> bool:
        if name in _SKIP_EXTRA or "/" in name or name not in self.archive:
            return False
        shape = self.archive.shape(name) if hasattr(self.archive, "shape") else np.shape(self.archive[name])
        return len(shape) in (2, 3) and tuple(shape[:2]) == (self.T, self.N)

    def _pick_extras(self, extras: Any) -> List[str]:
        if extras in (None, False):
            return []
        names = list(self.archive.channels)
        if extras != "auto" and extras is not True:
            names = [n for n in names if n in set(extras)]
        ok = [n for n in names if self._eligible(n)]
        rank = {n: i for i, n in enumerate(_PRIORITY)}
        return sorted(ok, key=lambda n: (rank.get(n, len(rank)), n))

    @property
    def names(self) -> List[str]:
        return self.core + [f"x:{n}" for n in self.extra]

    def shape(self, name: str, frames: Optional[int] = None) -> List[int]:
        T = self.T if frames is None else frames
        a = self.archive
        if name == "err":
            return [T, self.N]
        src = "ref_pos" if name == "ref" else name[2:] if name.startswith("x:") else name
        return [T, *a.shape(src)[1:]] if hasattr(a, "shape") else [T, *np.shape(a[src])[1:]]

    def array(self, name: str, idx: Union[np.ndarray, slice] = slice(None)) -> np.ndarray:
        a = self.archive
        if name == "err":
            return np.linalg.norm(a["pos"][idx] - a["ref_pos"][idx], axis=-1).astype(np.float32)
        src = "ref_pos" if name == "ref" else name[2:] if name.startswith("x:") else name
        out = np.asarray(a[src][idx])
        return np.nan_to_num(out.astype(np.float32), nan=0.0) if out.dtype.kind != "f" else out.astype(np.float32)

    def window(self, start: Optional[float], stop: Optional[float], max_frames: int) -> np.ndarray:
        """Indices of at most ``max_frames`` frames with ``start <= t <= stop`` (relative seconds)."""
        lo = 0 if start is None else int(np.searchsorted(self.time, start, side="left"))
        hi = self.T if stop is None else int(np.searchsorted(self.time, stop, side="right"))
        lo, hi = min(max(lo, 0), self.T - 1), max(min(hi, self.T), lo + 1)
        return lo + decimate_indices(hi - lo, max_frames)

    def bundle(self, names: Iterable[str], start: Optional[float] = None, stop: Optional[float] = None, max_frames: int = 4000) -> bytes:
        """The binary frame bundle for ``names`` over a time window (see :mod:`aevion.viewer.protocol`)."""
        idx = self.window(start, stop, max_frames)
        known = set(self.names)
        wanted = [n for n in dict.fromkeys(names) if n in known]
        t0 = float(self.time[idx[0]])
        arrays: Dict[str, np.ndarray] = {"t": (self.time[idx] - t0).astype(np.float32)}
        for n in wanted:
            arrays[n] = self.array(n, idx)
        return encode_bundle({"type": "frames", "t0": t0, "t1": float(self.time[idx[-1]]), "frames": len(idx)}, arrays)

    def chunk(self, i0: int, i1: int, names: Optional[Iterable[str]] = None) -> bytes:
        """Frames ``i0:i1`` at full rate (live streaming)."""
        sl = slice(i0, i1)
        t0 = float(self.time[i0])
        arrays: Dict[str, np.ndarray] = {"t": (self.time[sl] - t0).astype(np.float32)}
        for n in names or self.names:
            arrays[n] = self.array(n, sl)
        return encode_bundle({"type": "frames", "t0": t0, "t1": float(self.time[i1 - 1]), "frames": i1 - i0}, arrays)


# ------------------------------------------------------------------------------------------------ events
def derive_events(archive: Any, fs: FrameSet, limit: int = MAX_EVENTS) -> Tuple[List[dict], int]:
    """Recorded events plus the radio events implied by the link channels (handovers, radio-link failures).

    Returns ``(events sorted by time, number dropped to respect the limit)``.
    """
    t0 = float(archive.t[0])
    events = [dict(e, t=float(e["t"]) - t0) for e in (archive.meta.get("events") or []) if "t" in e]
    have = {e.get("type") for e in events}
    t = fs.time
    if "link_serving" in archive and "handover" not in have and not _is_mesh(archive):
        s = np.asarray(archive["link_serving"])
        k, v = np.nonzero(s[1:] != s[:-1])
        for kk, vv in zip(k, v):
            events.append({"type": "handover", "t": float(t[kk + 1]), "vehicle": int(vv), "from": int(s[kk, vv]), "to": int(s[kk + 1, vv])})
    if "link_rlf" in archive and "rlf" not in have:
        r = np.asarray(archive["link_rlf"]) > 0.5
        rise = np.zeros_like(r)
        rise[0], rise[1:] = r[0], r[1:] & ~r[:-1]
        for kk, vv in zip(*np.nonzero(rise)):
            events.append({"type": "rlf", "t": float(t[kk]), "vehicle": int(vv)})
    events.sort(key=lambda e: e["t"])
    dropped = max(0, len(events) - limit)
    if dropped:  # keep every non-handover event, thin the handovers evenly
        keep = [e for e in events if e["type"] != "handover"]
        ho = [e for e in events if e["type"] == "handover"]
        room = max(limit - len(keep), 0)
        ho = [ho[i] for i in np.linspace(0, len(ho) - 1, room).round().astype(int)] if room and ho else []
        events = sorted(keep + ho, key=lambda e: e["t"])[:limit]
    return events, dropped


def _is_mesh(archive: Any) -> bool:
    return "link_hops" in archive


# ------------------------------------------------------------------------------------------------ layers
def _world(world: Any) -> Optional[dict]:
    if world is None:
        return None
    wd = dict(
        boxes=world.boxes.tolist(),
        spheres=world.spheres.tolist(),
        cylinders=world.cylinders.tolist(),
        box_colors=world.box_colors.tolist(),
        sphere_colors=world.sphere_colors.tolist(),
        cyl_colors=world.cyl_colors.tolist(),
        bounds=world.bounds.tolist(),
    )
    if world.heightmap is not None:
        hm = world.heightmap
        step = max(1, math.ceil(max(hm.shape) / 160))
        hs = hm[::step, ::step]
        wd["heightmap"] = dict(origin=list(world.hm_origin), res=world.hm_res * step, rows=hs.shape[0], cols=hs.shape[1], h=b64(hs))
    meshes, budget, skipped = [], MAX_MESH_TRIANGLES, 0
    for m in getattr(world, "meshes", []) or []:
        if m.n_faces > budget:
            skipped += 1
            continue
        budget -= m.n_faces
        meshes.append(
            dict(
                name=m.name, color=list(m.color), n_vertices=m.n_vertices, n_faces=m.n_faces, vertices=b64(m.vertices), faces=b64(m.faces, np.uint32)
            )
        )
    if meshes or skipped:
        wd["meshes"] = meshes
        wd["meshes_skipped"] = skipped
    return wd


def _actors(archive: Any, idx: np.ndarray, time: np.ndarray) -> Optional[dict]:
    """Moving obstacles: their shapes and positions at the decimated frames."""
    world = archive.world
    if world is None or not getattr(world, "actors", None) or "actor_pos" not in archive:
        return None
    pos = np.asarray(archive["actor_pos"])[idx]
    shapes = [
        dict(shape=a.shape, radius=a.radius, size=a.size.tolist(), height=a.height, color=list(a.color) if a.color else None, name=a.name)
        for a, _, _ in world.actors
    ]
    return dict(shapes=shapes, t=b64(time[idx]), pos=b64(pos), n=len(shapes), frames=len(idx))


def _geofences(config: dict) -> Optional[list]:
    """Geofence definitions from ``environment.geofences`` and from ``health.checks`` of type ``geofence``."""
    out = list((config.get("environment") or {}).get("geofences") or [])
    for c in (config.get("health") or {}).get("checks") or []:
        if isinstance(c, dict) and c.get("type") == "geofence":
            out += list((c.get("params") or {}).get("fences") or [])
    return out or None


def _environment(world: Any) -> Optional[dict]:
    if world is None:
        return None
    env: Dict[str, Any] = {}
    if getattr(world, "origin", None) is not None:
        env["origin"] = world.origin.to_dict()
    if getattr(world, "utc", None):
        env["utc"] = world.utc
    if getattr(world, "weather", None) is not None:
        env["weather"] = world.weather.to_dict()
    return env or None


def _lidar_stream(archive: Any) -> Optional[str]:
    for sc in archive.config.get("sensors", []) or []:
        if isinstance(sc, dict) and sc.get("type") == "lidar":
            key = sc.get("name") or "lidar"
            if f"sensors/{key}" in archive:
                return key
    return "lidar" if "sensors/lidar" in archive else None


def _lidar(archive: Any, max_frames: int) -> Optional[dict]:
    key = _lidar_stream(archive)
    if key is None:
        return None
    lt, lr = np.asarray(archive[f"sensors/{key}/t"], float), archive[f"sensors/{key}"]
    li = decimate_indices(len(lt), max_frames)
    cfg: dict = {}
    try:
        from ..registry import registries

        params: dict = {}
        for sc in archive.config.get("sensors", []) or []:
            if isinstance(sc, dict) and sc.get("type") == "lidar" and (sc.get("name") or "lidar") == key:
                params = {k: v for k, v in sc.items() if k not in ("type", "name", "record")}
        lid = registries.sensors.create("lidar", **params)
        cfg = dict(
            channels=lid.channels,
            samples=lid.samples,
            v_fov=list(lid.v_fov),
            h_fov=lid.h_fov,
            mount=lid.mount.tolist(),
            max_range=lid.max_range,
            directions=lid.dirs_body.tolist() if lid.extrinsics is not None else None,
        )
    except Exception:  # a LiDAR the registry cannot rebuild is shown without geometry rather than failing the export
        cfg = {}
    sel = np.asarray(lr[li], float)
    sel = sel.reshape(len(li), int(archive.n_vehicles), -1)
    r = np.where(np.isfinite(sel), sel, -1.0)
    return dict(cfg=cfg, t=b64(lt[li] - float(archive.t[0])), ranges=b64(r), frames=len(li))


def _wireless(w: Optional[dict], mesh: bool = False) -> Optional[dict]:
    """Cell/outage description from a network's ``to_dict()`` (``mesh``: the run has mesh-hop channels instead of cells)."""
    if not w:
        return None
    out: Dict[str, Any] = dict(outage_db=w.get("outage_db", -6.0), name=w.get("type") or ("mesh" if mesh else "cellular"), freq_hz=w.get("freq_hz"))
    if "cells" in w:
        out["kind"] = "cells"
        out["cells"] = [
            dict(
                pos=c["pos"],
                yaw_deg=c.get("yaw_deg", 0.0),
                tx_power_dbm=c.get("tx_power_dbm"),
                omni=str((c.get("antenna") or {}).get("name", "")).lower() in ("isotropic", "omni", "dipole"),
                antenna=(c.get("antenna") or {}).get("name"),
            )
            for c in w["cells"]
        ]
    elif "bs" in w:
        out["kind"] = "cells"
        out["cells"] = [
            dict(
                pos=b["pos"],
                yaw_deg=0.0,
                tx_power_dbm=b.get("tx_power_dbm"),
                omni=str(b.get("antenna", "omni")).lower() in ("omni", "isotropic", "none"),
            )
            for b in w["bs"]
        ]
    else:
        out["kind"] = "mesh"
        out["cells"] = []
    return out


def _cameras(config: dict) -> Optional[list]:
    cams = []
    for sc in config.get("sensors", []) or []:
        if isinstance(sc, dict) and sc.get("type") == "camera":
            cams.append(
                dict(
                    name=sc.get("name") or "camera",
                    fov_deg=float(sc.get("fov_deg", 90.0)),
                    width=int(sc.get("width", 64)),
                    height=int(sc.get("height", 48)),
                    far=float(sc.get("far", 60.0)),
                    mount=list(sc.get("mount", (0.1, 0.0, 0.0))),
                    pitch_deg=float(sc.get("pitch_deg", 0.0)),
                )
            )
    return cams or None


def _planned(config: dict, scenario: Optional[dict] = None) -> Optional[dict]:
    traj = config.get("trajectory") or {}
    params = traj.get("params") if isinstance(traj, dict) else None
    pts = (params or {}).get("waypoints") or ((scenario or {}).get("trajectory", {}) or {}).get("waypoints")
    try:
        arr = np.asarray(pts, float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 1:
        return None
    return dict(points=arr[:200].tolist())


def _rebuild_network(archive: Any) -> Any:
    cfg = (archive.config or {}).get("network")
    if not isinstance(cfg, dict) or "type" not in cfg:
        return None
    from ..registry import registries

    params = (
        dict(cfg.get("params") or {}) if "params" in cfg else {k: v for k, v in cfg.items() if k not in ("type", "rate_hz")}
    )  # archived configs are flattened
    return registries.networks.create(cfg["type"], world=archive.world, seed=int(archive.config.get("seed", 0) or 0), **params)


def _extent(pos: Any, world: Any, pad: float = 0.15) -> Tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) covering the positions and the world, padded by ``pad`` of its size."""
    pos = np.asarray(pos).reshape(-1, 3)
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    if world is not None:
        b = np.asarray(world.bounds, float)
        lo, hi = np.minimum(lo, b[:2]), np.maximum(hi, b[3:5])
    w = np.maximum(hi - lo, 20.0)
    lo, hi = lo - pad * w, hi + pad * w
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _rem(net: Any, ext: Tuple[float, float, float, float], altitudes: Sequence[float], max_cells: int = 90) -> Optional[dict]:
    """SINR slices of the network over ``ext`` at each altitude, quantised to uint8 (a 90 x 90 slice is ~11 kB)."""
    if not hasattr(net, "large_scale_rsrp"):
        return None
    from ..wireless.maps.rem import radio_environment_map

    res = max(ext[1] - ext[0], ext[3] - ext[2]) / max_cells
    alts = sorted({round(float(a), 1) for a in altitudes})
    maps = [radio_environment_map(net, a, ext, res) for a in alts]
    lo = float(min(np.percentile(m.sinr_db, 1) for m in maps))
    hi = float(max(np.percentile(m.sinr_db, 99) for m in maps))
    hi = max(hi, lo + 1.0)
    slices = []
    for a, m in zip(alts, maps):
        q = np.clip(np.round((m.sinr_db - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
        slices.append(dict(altitude=a, xs=np.round(m.xs, 2).tolist(), ys=np.round(m.ys, 2).tolist(), vmin=lo, vmax=hi, u8=b64(q, np.uint8)))
    return dict(outage_db=float(getattr(net, "outage_sinr_db", -6.0)), slices=slices)


def _lobe(pattern_gain: Any, kind: str, step_deg: float = 5.0, **extra: Any) -> dict:
    az = np.arange(-180.0, 180.0, step_deg)
    el = np.arange(-90.0, 90.0 + 1e-9, step_deg)
    g = np.asarray(pattern_gain(np.deg2rad(az)[None, :], np.deg2rad(el)[:, None]), float)
    g = np.broadcast_to(g, (len(el), len(az)))
    return dict(kind=kind, n_az=len(az), n_el=len(el), az_deg=az.tolist(), el_deg=el.tolist(), gain_db=b64(g), peak_dbi=float(g.max()), **extra)


def _beams(net: Any, max_cells: int = 24) -> Optional[dict]:
    cells = getattr(net, "cells", None)
    if not cells or not hasattr(cells[0].get("antenna"), "gain_dbi"):
        return None
    lobes = [_lobe(c["antenna"].gain_dbi, "cell", cell=i) for i, c in enumerate(cells[:max_cells])]
    ua = getattr(net, "uav_antenna", None)
    if ua is not None:
        lobes.append(_lobe(ua.pattern.gain_dbi, "uav"))
    return dict(lobes=lobes)


def layer_data(
    archive: Any,
    *,
    lidar_max_frames: int = 400,
    rem: Union[bool, str] = "auto",
    beams: Union[bool, str] = "auto",
) -> Dict[str, Any]:
    """Everything that is not a per-frame channel. Optional layers that fail to build are skipped, never fatal."""
    wmeta = archive.meta.get("wireless") if "link_sinr_db" in archive else None
    out: Dict[str, Any] = dict(
        world=_world(archive.world),
        lidar=_lidar(archive, lidar_max_frames),
        wireless=_wireless(wmeta, _is_mesh(archive)),
        camera=_cameras(archive.config),
        planned=_planned(archive.config, archive.scenario),
        geofences=_geofences(archive.config),
    )
    if out["wireless"] and (rem or beams):
        try:
            net = _rebuild_network(archive)
        except Exception:
            net = None
        if net is not None and rem:
            try:
                z = np.asarray(archive["pos"])[..., 2].ravel()
                out["rem"] = _rem(net, _extent(archive["pos"], archive.world), np.quantile(z, np.linspace(0.1, 0.9, 3)))
            except Exception:
                out["rem"] = None
        if net is not None and beams:
            try:
                out["beams"] = _beams(net)
            except Exception:
                out["beams"] = None
    return out


# ------------------------------------------------------------------------------------------------ meta + payload
def _vehicle(spec: Any) -> Optional[dict]:
    if spec is None:
        return None
    return dict(
        name=spec.name,
        mass=spec.mass,
        rotors=spec.rotor_positions.tolist(),
        spin=spec.rotor_spin.tolist(),
        radius=spec.rotor_radius,
        w_max=spec.w_max,
        collision_radius=spec.collision_radius,
    )


def build_meta(archive: Any, fs: FrameSet, frames: int, title: Optional[str] = None) -> dict:
    events, dropped = derive_events(archive, fs)
    prov = archive.meta.get("provenance") or {}
    return dict(
        title=title or archive.scenario.get("name", "Aevion run"),
        n=fs.N,
        T=int(frames),
        M=int(archive["rotor"].shape[-1]),
        created=archive.meta.get("created", ""),
        controller=archive.scenario.get("controller", ""),
        trajectory=(archive.scenario.get("trajectory", {}) or {}).get("kind", ""),
        seed=archive.scenario.get("seed"),
        config_hash=archive.meta.get("config_hash") or prov.get("config_hash"),
        events=events,
        events_dropped=dropped,
        environment=_environment(archive.world),
        vehicle=_vehicle(archive.spec),
        backend=archive.meta.get("backend"),
        duration=float(fs.time[-1]),
        t0=0.0,
        t1=float(fs.time[-1]),
    )


def _run_entry(label: str, archive: Any, max_frames: int, color: Optional[int] = None, offset: float = 0.0) -> dict:
    fs = FrameSet(archive, extras=None)
    idx = decimate_indices(fs.T, max_frames)
    names = [n for n in ("pos", "quat", "vel", "euler", "rotor") if n in fs.core]
    chans = {n: b64(fs.array(n, idx)) for n in names}
    chans["t"] = b64(fs.time[idx])
    meta = dict(
        n=fs.N,
        T=len(idx),
        M=int(archive["rotor"].shape[-1]),
        shapes={n: fs.shape(n, len(idx)) for n in names},
        vehicle=_vehicle(archive.spec),
        title=archive.scenario.get("name", label),
    )
    entry: Dict[str, Any] = dict(label=label, meta=meta, channels=chans, offset=offset)
    if color is not None:
        entry["color"] = int(color)
    return entry


def build_payload(
    archive: Any,
    max_frames: int = 2400,
    lidar_max_frames: int = 400,
    *,
    title: Optional[str] = None,
    extras: Union[str, Sequence[str], None, bool] = "auto",
    extras_budget_mb: float = 48.0,
    rem: Union[bool, str] = "auto",
    beams: Union[bool, str] = "auto",
    runs: Optional[Sequence[Tuple[str, Any]]] = None,
    run_max_frames: int = 1200,
) -> dict:
    """Pack ``archive`` into a self-contained, JSON-serialisable ``aevion.view/2`` payload.

    ``runs`` is a list of ``(label, RunArchive)`` shown as ghost vehicles next to the main run (compare mode).
    ``extras_budget_mb`` bounds the optional channels; channels over budget are listed in ``meta['dropped_channels']``.
    """
    fs = FrameSet(archive, extras=extras)
    idx = decimate_indices(fs.T, max_frames)
    Td = len(idx)
    meta = build_meta(archive, fs, Td, title)
    payload: Dict[str, Any] = dict(format=FORMAT, meta=meta, t=b64(fs.time[idx]))
    shapes: Dict[str, List[int]] = {}
    for n in fs.core:
        payload[n] = b64(fs.array(n, idx))
        shapes[n] = fs.shape(n, Td)
    ex: Dict[str, Any] = {}
    budget = extras_budget_mb * 1e6
    dropped: List[str] = []
    for n in fs.extra:
        shp = fs.shape(f"x:{n}", Td)
        size = 4 * int(np.prod(shp))
        if size > budget:
            dropped.append(n)
            continue
        budget -= size
        ex[f"x:{n}"] = dict(b64=b64(fs.array(f"x:{n}", idx)), shape=shp)
    payload["extras"] = ex
    meta["shapes"] = shapes
    meta["dropped_channels"] = dropped
    payload.update(layer_data(archive, lidar_max_frames=lidar_max_frames, rem=rem, beams=beams))
    payload["actors"] = _actors(archive, idx, fs.time)
    if runs:
        palette = (0xFF4D6D, 0x3DDC97, 0xFFC857, 0x9B8CFF, 0x4CC9F0, 0xF78FB3)
        payload["runs"] = [_run_entry(lbl, a, run_max_frames, palette[i % len(palette)]) for i, (lbl, a) in enumerate(runs)]
    return json_safe(payload)


def server_meta(
    archive: Any, fs: FrameSet, *, title: Optional[str] = None, runs: Optional[Sequence[Tuple[str, Any]]] = None, **layer_opts: Any
) -> dict:
    """What ``GET /api/meta`` returns: the payload minus the frames, plus the channel list the frame API serves."""
    meta = build_meta(archive, fs, fs.T, title)
    meta["shapes"] = {n: fs.shape(n) for n in fs.names}
    out: Dict[str, Any] = dict(format=FORMAT, meta=meta, channels=fs.names, t0=0.0, t1=float(fs.time[-1]))
    out.update(layer_data(archive, **layer_opts))
    out["actors"] = _actors(archive, decimate_indices(fs.T, 4000), fs.time)
    if runs:
        out["runs"] = [_run_entry(lbl, a, 1200, 0xFF4D6D) for lbl, a in runs]
    return json_safe(out)
