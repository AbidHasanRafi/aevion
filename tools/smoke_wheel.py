"""Install a built wheel into a *fresh* virtual environment and check that it works outside the source tree.

python tools/smoke_wheel.py dist/aevion-0.1.0-py3-none-any.whl
python tools/smoke_wheel.py dist/*.whl --offline      # reuse this interpreter's site-packages for the dependencies

Catches what tests inside the repository cannot: files missing from the wheel (templates, viewer assets, py.typed),
undeclared dependencies, and imports that only work because the source tree is on the path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

CHECKS = r"""
import importlib.metadata as md, json, pathlib, sys
import aevion
from aevion import __version__
assert md.version("aevion") == __version__, (md.version("aevion"), __version__)
root = pathlib.Path(aevion.__file__).resolve().parent
assert "site-packages" in str(root) or "venv" in str(root).lower(), f"imported from the source tree: {root}"
assert (root / "py.typed").exists(), "py.typed missing from the wheel"
web = root / "viewer" / "web"
for rel in ("index.html", "css/aevion.css", "src/app.js", "vendor/three/three.module.min.js", "vendor/three/MANIFEST.json"):
    assert (web / rel).exists(), f"viewer asset missing from the wheel: {rel}"
assert len(list((root / "templates" / "plugin").rglob("*.tmpl"))) == 6, "plugin template missing"

from aevion.engine import Simulation
arch = Simulation({"duration": 0.5, "sensors": ["imu"]}).run()
assert arch.n_steps > 10 and arch.complete

import tempfile
with tempfile.TemporaryDirectory() as d:
    p = arch.save(pathlib.Path(d) / "run.aevd")
    from aevion.datasets import RunArchive
    assert RunArchive.open_store(p).verify_integrity() == []
    from aevion.viewer import export_html
    page = export_html(arch, pathlib.Path(d) / "r.html")
    assert page.stat().st_size > 500_000, "single-file replay export looks empty"
print("smoke ok", __version__)
"""


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--offline", action="store_true", help="no network: reuse the current environment's packages for dependencies")
    args = ap.parse_args(argv)
    if not args.wheel.exists():
        print(f"no such file: {args.wheel}", file=sys.stderr)
        return 2
    wheel = args.wheel.resolve()

    with tempfile.TemporaryDirectory(prefix="aevion-smoke-") as tmp:
        env_dir = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=args.offline, clear=True).create(env_dir)
        py = env_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        pip_args = ["-m", "pip", "install", "--disable-pip-version-check", "--force-reinstall", "--no-cache-dir"]
        if args.offline:
            pip_args += ["--no-deps", "--no-index"]
        run([py, *pip_args, wheel])
        work = Path(tmp) / "work"  # run from an unrelated directory so the repository cannot leak onto sys.path
        work.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONIOENCODING": "utf-8", "AEVION_HOME": str(work / "home")}
        run([py, "-I", "-c", CHECKS], cwd=work, env=env)
        run([py, "-m", "aevion.cli", "info"], cwd=work, env=env, stdout=subprocess.DEVNULL)
        run([py, "-m", "aevion.cli", "new-plugin", "smoke-ext", "--dest", work / "smoke-ext"], cwd=work, env=env, stdout=subprocess.DEVNULL)
    print("wheel smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
