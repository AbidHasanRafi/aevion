"""Optional exporters to analysis-friendly formats (Parquet via pyarrow, Zarr, CSV)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .._types import PathLike
from ..errors import ArchiveError

if TYPE_CHECKING:
    from .archive import RunArchive


def to_parquet(archive: "RunArchive", path: PathLike, vehicle: Optional[int] = 0, compression: str = "zstd") -> Path:
    """One Parquet file: a row per recorded step, a column per scalar/vector component (``pos_0, pos_1, …``).

    ``vehicle=None`` writes a long table with an extra ``vehicle`` column for all vehicles. The archive manifest is
    stored in the Parquet schema metadata under ``aevion``. Requires ``pyarrow``.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as ex:
        raise ArchiveError("Parquet export needs pyarrow: pip install pyarrow") from ex
    n = archive.n_steps
    t = np.asarray(archive["t"])
    vs = [vehicle] if vehicle is not None else list(range(archive.n_vehicles))
    frames = []
    for v in vs:
        cols = {"t": t}
        if vehicle is None:
            cols["vehicle"] = np.full(n, v, dtype=np.int32)
        for k in archive.channels:
            if k == "t" or k.startswith("sensors/"):
                continue
            ch = archive.lazy(k)
            if ch.ndim >= 2 and ch.shape[0] == n and ch.shape[1] > v:
                x = np.asarray(ch[:, v]).reshape(n, -1)
                for j in range(x.shape[1]):
                    cols[f"{k}_{j}" if x.shape[1] > 1 else k] = x[:, j]
        frames.append(pa.table(cols))
    table = pa.concat_tables(frames)
    meta = {b"aevion": json.dumps({k: archive.meta.get(k) for k in ("scenario", "config", "provenance", "events")}, default=str).encode("utf-8")}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    path = Path(path)
    pq.write_table(table, path, compression=compression)
    return path


def to_zarr(archive: "RunArchive", path: PathLike) -> Path:
    """Zarr group with one array per channel (requires ``zarr``)."""
    try:
        import zarr
    except ImportError as ex:
        raise ArchiveError("Zarr export needs zarr: pip install zarr") from ex
    path = Path(path)
    root = zarr.open_group(str(path), mode="w")
    for k in archive.channels:
        a = np.asarray(archive[k])
        root.create_dataset(k, data=a, chunks=(min(1024, max(1, a.shape[0])),) + a.shape[1:], overwrite=True)
    root.attrs["aevion_meta"] = json.dumps(archive.meta, default=str)
    return path


def from_parquet(path: PathLike) -> Any:
    """Read a Parquet export back into a pandas DataFrame (for notebooks)."""
    import pandas as pd

    return pd.read_parquet(path)
