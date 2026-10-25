"""Provenance: everything needed to know *exactly* what produced a result.

Collected: Aevion/Python/library versions, a content hash of the installed package set, the git commit (and
whether the working tree was dirty) of the Aevion source, hardware thread counts, the configuration hash, seed,
loaded plugins and the dynamics kernel. Deliberately **not** collected: user name, host name, paths, or
environment variables (so archives can be shared without leaking personal information).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=1)
def _package_set() -> tuple:
    names = []
    for d in md.distributions():
        try:
            names.append(f"{d.metadata['Name'].lower()}=={d.version}")
        except Exception:
            continue
    names.sort()
    h = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
    return h, len(names)


def package_versions(names: Any = ("numpy", "scipy", "numba", "torch", "matplotlib", "gymnasium", "stable_baselines3", "pyarrow", "h5py")) -> dict:
    out: Dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = md.version(n.replace("_", "-"))
        except md.PackageNotFoundError:
            out[n] = None
    return out


@lru_cache(maxsize=1)
def git_info() -> Optional[dict]:
    """Commit and dirty-flag of the repository containing the Aevion source, or ``None`` if not a git checkout."""
    here = Path(__file__).resolve().parent

    def run(*args: Any) -> Any:
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=3).stdout.strip()

    try:
        top = run("rev-parse", "--show-toplevel")
        if not top:
            return None
        # only report if Aevion itself lives in this repository
        if Path(top).resolve() not in here.parents and Path(top).resolve() != here:
            return None
        return dict(
            commit=run("rev-parse", "HEAD") or None, branch=run("rev-parse", "--abbrev-ref", "HEAD") or None, dirty=bool(run("status", "--porcelain"))
        )
    except Exception:
        return None


def collect(
    config: Any = None,
    seed: Optional[int] = None,
    kernel: Optional[str] = None,
    backend: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build the provenance record. ``config`` may be a config object/dict (hashed) or an already computed hash."""
    from .. import __version__
    from ..registry import records

    lock, count = _package_set()
    prov = dict(
        aevion=__version__,
        created=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        libraries=package_versions(),
        package_set_hash=lock,
        n_packages=count,
        seed=seed,
        kernel=kernel,
        backend=backend,
        git=git_info(),
        plugins=[dict(name=r.name, loaded=r.loaded, version=r.version) for r in records()],
    )
    try:
        import numba

        prov["numba_threads"] = getattr(numba.config, "NUMBA_NUM_THREADS", None)
    except Exception:
        pass
    if config is not None:
        if isinstance(config, str):
            prov["config_hash"] = config
        else:
            from ..config import config_hash

            prov["config_hash"] = config_hash(config)
    if extra:
        prov.update(extra)
    return prov
