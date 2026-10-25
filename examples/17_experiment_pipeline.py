"""An experiment pipeline: which controller parameters matter, what is the best setting, and which method wins.

Sobol sensitivity of the tracking error to the controller bandwidth and the trajectory radius, Bayesian optimisation of the bandwidth, a cached controller
comparison with a leaderboard, and the methods appendix - the workflow of a small simulation study.

    python examples/17_experiment_pipeline.py --out experiment_pipeline
"""

import argparse
from pathlib import Path

import numpy as np

from aevion.analysis.appendix import methods_appendix
from aevion.engine import Simulation
from aevion.experiments import ConfigExperiment, ResultCache, ResultDB, bayes_opt, format_leaderboard, leaderboard, sobol_indices

BASE = {
    "duration": 3.0,
    "seed": 1,
    "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.7}},
    "wind": {"type": "constant", "velocity": [3.0, 0.0, 0.0]},
}


def tracking_error(wn: float, radius: float, omega: float) -> float:
    cfg = {
        **BASE,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": radius, "omega": omega}},
        "controller": {"type": "geometric", "params": {"wn_pos": wn}},
    }
    run = Simulation(cfg).run()
    late = run.t > 1.0
    return float(np.linalg.norm(run["pos"][late, 0] - run["ref_pos"][late, 0], axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="experiment_pipeline")
    ap.add_argument("--samples", type=int, default=16, help="Sobol base sample size (a power of two); the study runs samples x (d + 2) simulations")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. sensitivity: which of bandwidth, radius and angular rate drives the tracking error?
    names = ["wn", "radius", "omega"]
    box = [(2.0, 14.0), (1.0, 3.0), (0.3, 1.0)]
    res = sobol_indices(lambda x: tracking_error(*x), box, n=args.samples, seed=0)
    print(f"sensitivity (Sobol, n = {args.samples}, {args.samples * 5} simulations):")
    for i in res.ranking():
        print(
            f"  {names[i]:<7} S1 = {res.s1[i]:6.2f}   ST = {res.st[i]:6.2f}   (bootstrap ST interval [{res.st_ci[i, 0]:.2f}, {res.st_ci[i, 1]:.2f}])"
        )

    # 2. optimisation: the best position-loop bandwidth for the nominal circle
    bo = bayes_opt(lambda x: tracking_error(x[0], 2.0, 0.7), [(2.0, 14.0)], n_init=4, n_iter=6, seed=0)
    print(
        f"optimisation: best bandwidth {bo.x_best[0]:.2f} rad/s gives a mean error of {bo.y_best:.4f} m after {len(bo.y)} simulations (a weak 2 rad/s loop: {tracking_error(2.0, 2.0, 0.7):.4f} m)"
    )

    # 3. comparison: controllers x conditions x seeds, cached and stored
    exp = ConfigExperiment(
        "controllers",
        {"duration": 3.0, "trajectory": BASE["trajectory"]},
        methods={"geometric": {"controller": "geometric"}, "cascaded": {"controller": "cascaded"}, "lqr": {"controller": "lqr"}},
        conditions={"calm": {}, "windy": {"wind": {"type": "constant", "velocity": [4.0, 0.0, 0.0]}}},
        seeds=[0, 1, 2],
        metric_params={"skip": 1.0},
    )
    cache = ResultCache(out / "cache")
    first = exp.run(cache=cache)
    again = exp.run(cache=cache)
    print(
        f"comparison: {first.meta['trials']} trials ({first.meta['computed']} simulated); a repeat simulated {again.meta['computed']} and read {again.meta['cached']} from the cache"
    )
    with ResultDB(out / "results.db") as db:
        db.add_result(first)
    print(format_leaderboard(leaderboard(first.rows, "rmse_pos"), "rmse_pos"))

    # 4. the methods appendix for the paper
    app = methods_appendix(config=exp.base, result=first, experiment=exp, metrics=["rmse_pos", "energy_wh", "max_tilt_deg"])
    (out / "methods.md").write_text(app.markdown(), encoding="utf-8")
    (out / "methods.tex").write_text(app.latex(), encoding="utf-8")
    print(f"appendix  : {out / 'methods.md'} and methods.tex")


if __name__ == "__main__":
    main()
