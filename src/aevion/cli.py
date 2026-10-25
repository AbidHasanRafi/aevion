"""``aevion`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


# ------------------------------------------------------------------------------------ info
def _cmd_info(args):
    import platform

    import numpy as np

    from . import __version__
    from .analysis import style
    from .core.backend import cuda_available, torch_available
    from .dynamics import fast
    from .registry import records, registries

    print(f"Aevion {__version__}")
    print(f"  Python {platform.python_version()} · NumPy {np.__version__} · {platform.platform()}")
    print(f"  numba fast kernels : {'yes' if fast.HAVE_NUMBA else 'no  (pip install aevion[fast])'}")
    if fast.HAVE_NUMBA:
        import numba

        print(f"    numba {numba.__version__}, {getattr(numba.config, 'NUMBA_NUM_THREADS', '?')} threads")
    if torch_available():
        import torch

        print(f"  PyTorch            : {torch.__version__}  CUDA: {'yes (' + torch.cuda.get_device_name(0) + ')' if cuda_available() else 'no'}")
    else:
        print("  PyTorch            : not installed  (pip install aevion[torch])")
    # probe through package metadata, never by importing: a broken native extension must not be able to crash `info`
    import importlib.metadata as md

    for mod, dist in (
        ("gymnasium", "gymnasium"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("yaml", "PyYAML"),
        ("h5py", "h5py"),
        ("pyarrow", "pyarrow"),
        ("stable_baselines3", "stable-baselines3"),
    ):
        try:
            print(f"  {mod:<19}: {md.version(dist)}")
        except md.PackageNotFoundError:
            print(f"  {mod:<19}: not installed")
    print(f"  figure font        : {style.describe()['font']}")
    n = sum(len(r) for r in registries.all().values())
    print(
        f"  components         : {n} registered across {len([r for r in registries.all().values() if len(r)])} registries · {len(records())} plugin(s)"
    )


def _cmd_vehicles(args):
    from .vehicles.spec import PRESETS

    print(f"{'preset':<14}{'mass':>7}{'rotors':>7}{'T/W':>7}{'hover rpm':>11}{'hover W':>9}{'endurance':>11}")
    for name, f in PRESETS.items():
        v = f()
        print(
            f"{name:<14}{v.mass:>6.3f}k{v.n_rotors:>7}{v.thrust_to_weight():>7.2f}{v.hover_rotor_speed() * 9.549:>11.0f}{v.hover_power():>9.1f}{v.hover_endurance_min():>9.1f} m"
        )


# ------------------------------------------------------------------------------- components
def _cmd_registry(args):
    from .registry import registries

    if args.kind:
        if args.kind not in registries.KINDS:
            raise SystemExit(f"unknown registry '{args.kind}'. Kinds: {', '.join(registries.KINDS)}")
        print(getattr(registries, args.kind).describe())
    else:
        print(registries.describe())


def _cmd_plugins(args):
    from .registry import records

    recs = records()
    if not recs:
        print("no plugins installed (entry-point group 'aevion.plugins')")
    for r in recs:
        print(f"  {'ok ' if r.loaded else 'ERR'}  {r.name:<20} {r.version or '':<10} {r.target}" + (f"\n        {r.error}" if r.error else ""))


# --------------------------------------------------------------------------------- configs
def _cmd_validate(args):
    from .config import ConfigError, config_hash, load_config
    from .safety import ResourceLimitError

    bad = 0
    limits = _limits(args)
    for p in args.files:
        try:
            if limits is not None:
                from .safety import load_config_checked

                cfg = load_config_checked(p, limits)
            else:
                cfg = load_config(p)
            print(f"OK   {p}  (v{cfg.schema_version}, hash {config_hash(cfg)}, {cfg.n_steps} steps, {cfg.vehicle.count} vehicle(s))")
        except (ConfigError, ResourceLimitError) as e:
            bad += 1
            print(f"FAIL {p}\n{e}", file=sys.stderr)
    sys.exit(1 if bad else 0)


def _cmd_schema(args):
    from .config import SimulationConfig, json_schema

    text = json.dumps(json_schema(SimulationConfig, "Aevion simulation configuration"), indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


def _cmd_new_plugin(args):
    from .scaffold import create_plugin, normalise_names

    root = Path(args.dest) if args.dest else Path.cwd() / normalise_names(args.name)["dist_name"]
    files = create_plugin(args.name, root, args.description, args.force)
    print(f"created {len(files)} files in {root}")
    for f in files:
        print(f"  {f.relative_to(root)}")
    print(f'next:  cd {root} && pip install -e ".[test]" && pytest')


def _cmd_init(args):
    """Write a documented starter configuration."""
    from .config import SimulationConfig, dump_config

    cfg = SimulationConfig.from_dict(
        {
            "name": args.name,
            "duration": 15.0,
            "seed": 0,
            "world": {"type": "forest", "n_trees": 25, "half_extent": 20},
            "wind": {"type": "composite", "models": [{"type": "constant", "velocity": [2.0, 1.0, 0.0]}, {"type": "dryden", "w20": 4.0}]},
            "vehicle": {"model": "generic_1kg"},
            "controller": {"type": "geometric", "ki_pos": 3.0},
            "trajectory": {"type": "figure8", "center": [0, 0, 3.0], "size": 3.0, "period": 9.0},
            "sensors": ["imu", "gps", "baro", "mag", {"type": "lidar", "record": True}],
            "faults": [{"type": "gps_outage", "t_start": 6.0, "t_end": 9.0}],
            "logging": {"record_every": 4, "sensors": ["imu"]},
        }
    )
    out = Path(args.out)
    dump_config(cfg, out)
    print(f"wrote {out}  (edit it, then:  aevion validate {out}  &&  aevion run {out})")


def _prepare_output(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _finish(archive, out: Path, args):
    from .analysis.report import build_report, insights

    out.mkdir(parents=True, exist_ok=True)
    if archive.source_path is None or not Path(archive.source_path).exists():
        fmt = getattr(args, "format", "v2")
        p = archive.save(out / ("run.aevd" if fmt == "v2" else "run.aev"))
    else:
        p = Path(archive.source_path)
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    print(f"  archive  : {p}  ({size / 1024:.0f} KB)")
    if getattr(args, "register", False):
        from .datasets import RunRegistry

        rid = RunRegistry().register(p)
        print(f"  registered as run {rid}")
    if not getattr(args, "no_report", False):
        r = build_report(archive, out / "report", offline_replay=getattr(args, "offline", False))
        print(f"  report   : {r}")
        for i in insights(archive):
            print(f"    [{i['level']:<4}] {i['title']}: {i['text']}")
        if getattr(args, "open", False):
            webbrowser.open((out / "report" / "replay.html").resolve().as_uri())


def _scenario_from_args(args):
    from .scenario import Scenario

    traj = dict(kind=args.trajectory)
    if args.trajectory == "circle":
        traj.update(center=[0, 0, 3.0], radius=2.5, omega=0.9)
    elif args.trajectory == "figure8":
        traj.update(center=[0, 0, 3.0], size=3.0, period=9.0)
    elif args.trajectory == "waypoints":
        traj.update(points=[[0, 0, 1], [6, 0, 3], [6, 6, 4], [0, 6, 2], [0, 0, 1]], speed=2.0)
    else:
        traj.update(pos=[0, 0, 2.0])
    wind = None
    if args.wind > 0:
        wind = dict(
            type="CompositeWind",
            models=[
                dict(type="ConstantWind", velocity=[args.wind, args.wind * 0.4, 0.0]),
                dict(type="DrydenTurbulence", w20=max(args.wind, 1.0), severity=1.0),
            ],
        )
    sensors = ["imu", "gps", "baro", "mag"] + (["lidar"] if args.world != "empty" else [])
    return Scenario(
        name=args.name,
        vehicle=args.vehicle,
        world=None if args.world == "empty" else args.world,
        wind=wind,
        controller=args.controller,
        controller_kwargs=dict(ki_pos=3.0) if (args.controller == "geometric" and args.wind > 0) else {},
        trajectory=traj,
        duration=args.duration,
        n=args.n,
        formation=dict(kind="ring", spacing=1.5) if args.n > 1 else None,
        seed=args.seed,
        sensors=sensors,
        record_sensors=["imu"] + (["lidar"] if "lidar" in sensors else []),
        record_every=4,
    )


def _cmd_demo(args):
    sc = _scenario_from_args(args)
    print(f"Running '{sc.name}': {args.vehicle}, {args.controller}, {args.trajectory}, world={args.world}, {args.duration:g}s ...")
    t0 = time.perf_counter()
    arch = sc.run(progress=True)
    print(f"  simulated {args.duration:g}s of flight in {time.perf_counter() - t0:.1f}s")
    _finish(arch, Path(args.out), args)


def _limits(args):
    """Resource limits for ``--untrusted`` (None when the flag is absent)."""
    if not getattr(args, "untrusted", False):
        return None
    from .registry import plugins
    from .safety import ResourceLimits

    if any(r.loaded for r in plugins.records()):
        print("note: third-party plugins are loaded; set AEVION_DISABLE_PLUGINS=1 to keep their code out of untrusted runs", file=sys.stderr)
    return ResourceLimits.untrusted()


def _run_sim(sim):
    from .engine.hooks import HookError
    from .safety import ResourceLimitError

    try:
        return sim.run(progress=True)
    except HookError as ex:
        if isinstance(ex.__cause__, ResourceLimitError):
            raise ex.__cause__ from None
        raise


def _cmd_run(args):
    from .config import load_config
    from .engine import Simulation

    limits = _limits(args)
    out = _prepare_output(args)
    if limits is not None:
        from .safety import WallClockLimit, load_config_checked

        limits = limits.replace(output_root=out.resolve())
        cfg = load_config_checked(args.scenario, limits)
    else:
        cfg = load_config(args.scenario)  # YAML/JSON, schema v2 or legacy scenario (migrated)
    if args.backend:
        cfg.engine.backend = args.backend
    if args.stream:
        cfg.logging.mode, cfg.logging.path = "stream", str(out / "run.aevd")
    if limits is not None:
        from .safety import check_config

        check_config(cfg, limits)
    print(f"Running '{cfg.name}' ({cfg.duration:g} s, {cfg.vehicle.count} vehicle(s), {cfg.n_steps} steps) ...")
    t0 = time.perf_counter()
    hooks = [WallClockLimit(limits.max_wall_s)] if limits is not None and limits.max_wall_s else []
    sim = Simulation(cfg, hooks=hooks)
    arch = _run_sim(sim)
    print(
        f"  done in {time.perf_counter() - t0:.1f}s  (kernel: {sim.dyn.kernel.name})"
        + (f"  (stopped early: {sim.stop_reason})" if sim.stop_reason else "")
    )
    if args.profile:
        print(sim.scheduler.report())
    _finish(arch, out, args)


# ------------------------------------------------------------------------------------ archives
def _open(path):
    from .datasets import RunArchive

    return RunArchive.load(path)


def _size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        if not (16 <= w <= 7680 and 16 <= h <= 4320):
            raise ValueError
    except ValueError:
        raise SystemExit(f"error: --size must look like 1280x720 (got {text!r})") from None
    return w, h


def _replay_runs(args):
    """(main archive, ghosts, title) from the positional archive and --compare/--label."""
    from .viewer.compare import label_runs

    main = _open(args.archive)
    if not args.compare:
        return main, None, args.label[0] if args.label else None
    arch = [main] + [_open(p) for p in args.compare]
    if args.label and len(args.label) != len(arch):
        raise SystemExit(f"error: {len(args.label)} --label values for {len(arch)} runs")
    labelled = label_runs(arch, args.label or None)
    return labelled[0][1], labelled[1:], f"{labelled[0][0]} vs {', '.join(n for n, _ in labelled[1:])}"


def _cmd_replay(args):
    a, runs, title = _replay_runs(args)
    base = Path(args.archive)
    if args.screenshot or args.video:
        from .viewer import render_video, screenshot

        common = dict(size=_size(args.size), camera=args.camera, theme=args.theme or "dark", runs=runs)
        if args.screenshot:
            p = screenshot(a, args.screenshot, t=args.t, ui=not args.no_ui, **common)
            print(f"wrote {p}")
        if args.video:
            p = render_video(a, args.video, fps=args.fps, speed=args.speed, sweep_deg=args.sweep, ui=args.ui, progress=True, **common)
            print(f"wrote {p}")
        return
    if args.serve:
        from .viewer import ViewerServer

        srv = ViewerServer(a, runs=runs, title=title, host=args.host, port=args.port, theme=args.theme)
        print(f"Aevion viewer: {srv.url}  (Ctrl+C to stop)")
        if args.open:
            webbrowser.open(srv.url)
        srv.serve_forever()
        return
    from .viewer import export_html

    out = Path(args.out or base.with_suffix(".html"))
    p = export_html(a, out, title=title, runs=runs, theme=args.theme)
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    if args.open:
        webbrowser.open(p.resolve().as_uri())


def _cmd_live(args):
    from .config import load_config
    from .viewer import serve_live

    cfg = load_config(args.scenario)
    arch = serve_live(
        cfg,
        port=args.port,
        host=args.host,
        speed=None if args.speed == 0 else args.speed,
        open_browser=args.open,
        wait_for_client=not args.no_wait,
        keep_serving=args.keep,
        theme=args.theme,
    )
    if arch is not None and args.save:
        print(f"saved {arch.save(args.save)}")


def _cmd_edit(args):
    from .viewer import open_editor, write_editor

    p = open_editor(args.out) if args.open else write_editor(args.out or "scene_editor.html")
    print(f"wrote {p}; draw a scene, download scene.json, then: aevion run scene.json")


def _cmd_report(args):
    from .analysis.report import build_report

    print(build_report(_open(args.archive), args.out, offline_replay=args.offline))


def _cmd_inspect(args):
    from . import metrics

    a = _open(args.archive)
    print(a.summary())
    for k, v in metrics.flight_metrics(a).items():
        print(f"  {k:<24} {float(v.mean()):12.5g}")


def _cmd_verify(args):
    from .scenario import verify

    r = verify(_open(args.archive), tol=args.tol)
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)


def _cmd_check(args):
    """Integrity check of a v2 archive (hashes every chunk)."""
    a = _open(args.archive)
    problems = a.verify_integrity(deep=not args.quick)
    if not problems:
        print(f"OK  {args.archive}: {len(a.channels)} channels, {a.n_steps} steps, integrity verified" + (" (shallow)" if args.quick else ""))
        sys.exit(0)
    print(f"PROBLEMS in {args.archive}:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)


def _cmd_diff(args):
    from .datasets import diff_archives

    d = diff_archives(_open(args.a), _open(args.b), tol=args.tol)
    print(d.summary())
    sys.exit(0 if d.identical and (d.config_identical or not args.strict) else 1)


def _cmd_convert(args):
    a = _open(args.src)
    out = a.save(args.dst, chunk_steps=args.chunk)
    print(f"wrote {out}")


def _cmd_pack(args):
    from .datasets import pack

    p = pack(args.archive, args.out)
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")


def _cmd_unpack(args):
    from .datasets import unpack

    print(f"wrote {unpack(args.archive, args.out)}")


def _cmd_runs(args):
    from .datasets import RunRegistry

    reg = RunRegistry()
    if args.action == "list":
        print(reg.table(reg.list(name=args.name, tag=args.tag, limit=args.limit), args.metric))
    elif args.action == "add":
        print(reg.register(args.target, tags=args.tags or ()))
    elif args.action == "show":
        r = reg.get(args.target)
        print(
            json.dumps(
                dict(
                    run_id=r.run_id,
                    name=r.name,
                    path=r.path,
                    exists=r.exists,
                    created=r.created,
                    config_hash=r.config_hash,
                    aevion=r.aevion,
                    seed=r.seed,
                    duration=r.duration,
                    vehicles=r.n_vehicles,
                    tags=r.tags,
                    notes=r.notes,
                    metrics=r.metrics,
                ),
                indent=2,
            )
        )
    elif args.action == "tag":
        print(reg.tag(args.target, *args.tags, remove=args.remove))
    elif args.action == "rm":
        reg.remove(args.target, delete_files=args.delete_files)
        print("removed")
    elif args.action == "prune":
        n = reg.prune_missing()
        print(f"removed {n} stale entr{'y' if n == 1 else 'ies'}")
    elif args.action == "find":
        for r in reg.find_by_config(args.target):
            print(r.run_id, r.path)


def _cmd_bench(args):
    import numpy as np

    from .core.backend import cuda_available, torch_available
    from .dynamics import Multirotor, fast

    def bench(label, **kw):
        sim = Multirotor("generic_1kg", n=args.n, dt=0.005, **kw)
        s = sim.reset()
        cmd = sim.hover_command()
        for _ in range(5):
            s = sim.step(s, cmd)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            s = sim.step(s, cmd)
        if sim.B.name == "torch" and sim.B.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(
            f"  {label:<26} {args.n * args.steps / dt / 1e6:8.3f} M vehicle-steps/s   ({dt / args.steps * 1e3:.2f} ms per step of {args.n} vehicles)"
        )

    print(f"Physics throughput, N={args.n} vehicles, RK4, 4 rotors")
    bench("numpy (reference)", backend="numpy", kernel="reference")
    if fast.HAVE_NUMBA:
        bench("numpy + numba kernel", backend="numpy", kernel="numba")
    if torch_available():
        bench("torch cpu", backend="torch:cpu", kernel="reference")
        if cuda_available():
            bench("torch cuda", backend="cuda", kernel="reference")
    from .environments.world import World
    from .sensors.raycast import cast_rays

    w = World.random_forest(100, seed=1)
    d = np.random.default_rng(0).normal(size=(65536, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    cast_rays(w, [0, 0, 2.0], d[:1000], 60.0)
    t0 = time.perf_counter()
    cast_rays(w, [0, 0, 2.0], d, 60.0)
    print(f"  ray casting (100 cylinders) {65536 / (time.perf_counter() - t0) / 1e6:8.2f} M rays/s")


# ------------------------------------------------------------------------------------ experiments, suites, datasets
def _experiment_summary(res, metric):
    import numpy as np

    names = res.metric_names()
    metric = metric if metric in names else next((m for m in ("rmse_pos",) if m in names), names[0] if names else "")
    print(
        f"{res.name}: {res.meta['trials']} trials ({res.meta['cached']} cached, {res.meta['computed']} computed, {res.meta['failed']} failed) in {res.meta['wall_s']:.1f} s"
    )
    if metric:
        for m in res.methods:
            for c in res.conditions:
                v = res.values(metric, m, c)
                if len(v):
                    print(f"  {m:<16} {c:<14} {metric} = {np.mean(v):.4g}  (n={len(v)})")


def _cmd_experiment(args):
    from .experiments import ConfigExperiment, ProcessRunner, ResultCache, ResultDB, SerialRunner, slurm_script

    exp = ConfigExperiment.load_spec(args.spec)
    if args.action == "slurm":
        text = slurm_script(
            args.spec,
            args.cache,
            args.shards,
            job_name=exp.name.replace("/", "-"),
            partition=args.partition,
            time=args.time,
            mem=args.mem,
            cpus=args.cpus,
        )
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {args.out}: submit with `sbatch {args.out}`")
        else:
            print(text)
        return
    cache = ResultCache(args.cache) if args.cache else None
    db = ResultDB(args.db) if args.db else None
    runner = ProcessRunner(args.workers) if args.workers > 1 else SerialRunner()
    shard = (args.shard, args.of) if args.of > 1 else None
    res = exp.run(runner=runner, cache=cache, db=db, shard=shard, progress=True)
    _experiment_summary(res, args.metric)
    if args.csv:
        res.to_csv(args.csv)
        print(f"wrote {args.csv}")
    if db is not None:
        db.close()
    if res.meta["failed"]:
        raise SystemExit(1)


def _cmd_suite(args):
    from .experiments import ProcessRunner, ResultCache, ResultDB, SerialRunner, run_suite
    from .experiments.suites import SUITES, get_suite

    if args.action == "list":
        for n, st in SUITES.items():
            print(f"{n:<14} {len(st.scenarios)} scenarios ({', '.join(st.scenarios)}); ranked by {st.primary}")
            print(f"               {st.description}")
        return
    suite = get_suite(args.name)
    methods = {c: {"controller": c} for c in args.controllers}
    runner = ProcessRunner(args.workers) if args.workers > 1 else SerialRunner()
    out = run_suite(suite, methods, runner=runner, cache=ResultCache(args.cache) if args.cache else None, db=ResultDB(args.db) if args.db else None)
    print(out.table())
    if args.out:
        out.save(args.out)
        print(f"wrote {args.out}")


def _cmd_vision(args):
    from .config import load_config
    from .datasets.vision import VisionRecorder
    from .engine import Simulation

    cfg = load_config(args.config)
    rec = VisionRecorder(args.out, camera=args.camera, every=args.every, max_frames=args.max_frames)
    Simulation(cfg, hooks=[rec]).run()
    print(f"wrote {rec.frames} labelled frames to {args.out} (images, depth16, seg, flow, annotations.json, labels/, data.yaml)")


def _cmd_methods(args):
    from .analysis.appendix import methods_appendix
    from .config import load_config

    cfg = load_config(args.config)
    print(methods_appendix(config=cfg.to_dict()).text(args.format))


# ------------------------------------------------------------------------------------ parser
def _out_flags(p):
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--offline", action="store_true", help="accepted for compatibility; the replay is always self-contained")
    p.add_argument("--open", action="store_true", help="open the replay in a browser")
    p.add_argument("--register", action="store_true", help="add the run to the run registry (~/.aevion/runs.db)")
    p.add_argument("--format", choices=["v1", "v2"], default="v2", help="archive format (v2 = chunked store)")


def build_parser():
    ap = argparse.ArgumentParser(prog="aevion", description="Aevion: UAV simulation for research")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="versions, accelerators and registered components").set_defaults(fn=_cmd_info)
    sub.add_parser("vehicles", help="list vehicle presets").set_defaults(fn=_cmd_vehicles)
    p = sub.add_parser("registry", help="list registered components (controllers, sensors, winds, faults, …)")
    p.add_argument("kind", nargs="?")
    p.set_defaults(fn=_cmd_registry)
    sub.add_parser("plugins", help="list installed plugins").set_defaults(fn=_cmd_plugins)

    p = sub.add_parser("validate", help="validate configuration files and report every problem")
    p.add_argument("files", nargs="+")
    p.add_argument("--untrusted", action="store_true", help="also enforce the strict untrusted-configuration resource limits")
    p.set_defaults(fn=_cmd_validate)
    p = sub.add_parser("schema", help="print/write the JSON Schema of the configuration format")
    p.add_argument("-o", "--out")
    p.set_defaults(fn=_cmd_schema)
    p = sub.add_parser("new-plugin", help="scaffold an installable plugin package (sensor + hook + tests)")
    p.add_argument("name", help="distribution name, e.g. my-aevion-ext")
    p.add_argument("--dest", help="output directory (default: ./<name>)")
    p.add_argument("--description")
    p.add_argument("--force", action="store_true", help="write into an existing non-empty directory")
    p.set_defaults(fn=_cmd_new_plugin)
    p = sub.add_parser("init", help="write a starter configuration file")
    p.add_argument("out", nargs="?", default="simulation.yaml")
    p.add_argument("--name", default="my_simulation")
    p.set_defaults(fn=_cmd_init)

    d = sub.add_parser("demo", help="run a demo flight, then write archive + report + 3-D replay")
    d.add_argument("--vehicle", default="generic_1kg")
    d.add_argument("--controller", default="geometric", choices=["geometric", "cascaded", "lqr"])
    d.add_argument("--trajectory", default="figure8", choices=["hover", "circle", "figure8", "waypoints"])
    d.add_argument("--world", default="forest", choices=["empty", "forest", "city", "corridor", "hills"])
    d.add_argument("--wind", type=float, default=2.0, help="mean wind speed [m/s] (0 = calm)")
    d.add_argument("--duration", type=float, default=15.0)
    d.add_argument("-n", type=int, default=1, help="number of vehicles")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--name", default="demo")
    d.add_argument("--out", default="aevion_demo")
    _out_flags(d)
    d.set_defaults(fn=_cmd_demo)

    r = sub.add_parser("run", help="run a configuration file (YAML/JSON; schema v2 or legacy scenario)")
    r.add_argument("scenario")
    r.add_argument("--out", default="aevion_run")
    r.add_argument("--backend")
    r.add_argument("--stream", action="store_true", help="stream the archive to disk while running (bounded memory)")
    r.add_argument("--profile", action="store_true", help="print per-task timing")
    r.add_argument("--untrusted", action="store_true", help="strict resource limits and built-in components only (for configs from unknown sources)")
    _out_flags(r)
    r.set_defaults(fn=_cmd_run)

    p = sub.add_parser("replay", help="interactive 3-D replay of an archive: HTML file, local server, comparison, screenshot or video")
    p.add_argument("archive")
    p.add_argument("--out", help="HTML output path (default: next to the archive)")
    p.add_argument("--offline", action="store_true", help="accepted for compatibility; the replay is always self-contained")
    p.add_argument("--open", action="store_true", help="open the result in a browser")
    p.add_argument("--serve", action="store_true", help="serve it from a local web server instead of writing a file (best for long runs)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--compare", nargs="+", metavar="ARCHIVE", help="more runs shown as ghosts next to the first")
    p.add_argument("--label", action="append", help="name of each run (repeat; main run first)")
    p.add_argument("--theme", choices=["dark", "light", "print"])
    p.add_argument("--screenshot", metavar="PNG", help="render one frame to a PNG (needs Chrome/Edge/Chromium)")
    p.add_argument("--t", type=float, help="time of the screenshot in seconds (default: middle of the run)")
    p.add_argument("--no-ui", action="store_true", help="screenshot without the charts and panels")
    p.add_argument("--video", metavar="FILE", help=".mp4/.webm (ffmpeg), .gif (Pillow) or a directory for PNG frames")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--speed", type=float, default=1.0, help="simulated seconds per video second")
    p.add_argument("--sweep", type=float, default=0.0, help="turn the orbit camera by this many degrees over the clip")
    p.add_argument("--camera", choices=["orbit", "chase", "top", "fpv"], default="orbit")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--ui", action="store_true", help="include charts and panels in the video")
    p.set_defaults(fn=_cmd_replay)
    p = sub.add_parser("live", help="run a configuration and watch it live in the browser")
    p.add_argument("scenario")
    p.add_argument("--speed", type=float, default=1.0, help="simulated seconds per wall second (0 = as fast as possible)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--open", action="store_true", help="open a browser")
    p.add_argument("--no-wait", action="store_true", help="start immediately instead of waiting for a browser to connect")
    p.add_argument("--keep", action="store_true", help="keep serving after the run ends")
    p.add_argument("--theme", choices=["dark", "light", "print"])
    p.add_argument("--save", metavar="ARCHIVE", help="also save the finished run")
    p.set_defaults(fn=_cmd_live)
    p = sub.add_parser("edit", help="write the in-browser scene editor (exports a configuration for `aevion run`)")
    p.add_argument("--out", help="HTML path (default: scene_editor.html)")
    p.add_argument("--open", action="store_true")
    p.set_defaults(fn=_cmd_edit)
    p = sub.add_parser("report", help="figures, tables and insights for an archive")
    p.add_argument("archive")
    p.add_argument("--out", default="aevion_report")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(fn=_cmd_report)
    p = sub.add_parser("inspect", help="summary and metrics of an archive")
    p.add_argument("archive")
    p.set_defaults(fn=_cmd_inspect)
    p = sub.add_parser("verify", help="re-simulate an archive from its configuration and check it reproduces")
    p.add_argument("archive")
    p.add_argument("--tol", type=float, default=1e-9)
    p.set_defaults(fn=_cmd_verify)
    p = sub.add_parser("check", help="verify the integrity (hashes) of a v2 archive")
    p.add_argument("archive")
    p.add_argument("--quick", action="store_true", help="skip re-hashing chunk data")
    p.set_defaults(fn=_cmd_check)
    p = sub.add_parser("diff", help="compare two archives (configuration, channels, metrics)")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--strict", action="store_true", help="also fail if the configurations differ")
    p.set_defaults(fn=_cmd_diff)
    p = sub.add_parser("convert", help="convert an archive (.aev ⇄ .aevd ⇄ .aevz) by output extension")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--chunk", type=int, default=512)
    p.set_defaults(fn=_cmd_convert)
    p = sub.add_parser("pack", help="pack a .aevd directory into one shareable .aevz file")
    p.add_argument("archive")
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_pack)
    p = sub.add_parser("unpack", help="unpack a .aevz file")
    p.add_argument("archive")
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_unpack)

    p = sub.add_parser("runs", help="manage the run registry")
    p.add_argument("action", choices=["list", "add", "show", "tag", "rm", "prune", "find"])
    p.add_argument("target", nargs="?")
    p.add_argument("--name")
    p.add_argument("--tag")
    p.add_argument("--tags", nargs="*")
    p.add_argument("--metric", default="rmse_pos")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--remove", action="store_true")
    p.add_argument("--delete-files", action="store_true")
    p.set_defaults(fn=_cmd_runs)

    p = sub.add_parser("bench", help="benchmark simulation throughput on this machine")
    p.add_argument("-n", type=int, default=4096)
    p.add_argument("--steps", type=int, default=100)
    p.set_defaults(fn=_cmd_bench)

    p = sub.add_parser(
        "experiment", help="run an experiment specification (grid of simulation configurations) with caching, sharding and a result database"
    )
    p.add_argument("action", choices=["run", "slurm"], help="run it here, or write a Slurm job-array script")
    p.add_argument("spec", help="experiment JSON written by ConfigExperiment.save_spec")
    p.add_argument("--cache", help="result cache directory (makes runs resumable)")
    p.add_argument("--db", help="SQLite result database to store the rows in")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1, help="number of shards (this process runs shard --shard)")
    p.add_argument("--metric", default="rmse_pos", help="metric to print")
    p.add_argument("--csv", help="write the rows as CSV")
    p.add_argument("--shards", type=int, default=8, help="(slurm) number of array tasks")
    p.add_argument("--partition")
    p.add_argument("--time", default="01:00:00")
    p.add_argument("--mem", default="4G")
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("-o", "--out", help="(slurm) write the script here")
    p.set_defaults(fn=_cmd_experiment)

    p = sub.add_parser("suite", help="benchmark suites: compare controllers on fixed scenarios and rank them")
    p.add_argument("action", choices=["list", "run"])
    p.add_argument("name", nargs="?", default="tracking")
    p.add_argument("--controllers", nargs="+", default=["geometric", "cascaded"], help="registered controller names to compare")
    p.add_argument("--cache")
    p.add_argument("--db")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", help="write the leaderboard JSON")
    p.set_defaults(fn=_cmd_suite)

    p = sub.add_parser("vision-dataset", help="fly a configuration with a camera and write a labelled image dataset (PNG, COCO, YOLO, depth, flow)")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="camera")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=500)
    p.set_defaults(fn=_cmd_vision)

    p = sub.add_parser("methods", help="print an auto-generated methods appendix for a configuration")
    p.add_argument("config")
    p.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    p.set_defaults(fn=_cmd_methods)
    return ap


def main(argv=None):
    _utf8()
    args = build_parser().parse_args(argv)
    from .errors import AevionError, ConfigError

    try:
        args.fn(args)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    except AevionError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
