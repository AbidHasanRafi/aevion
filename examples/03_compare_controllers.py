"""Rigorous controller comparison: 3 methods × 2 conditions × 10 seeds with randomised plants."""

import matplotlib

matplotlib.use("Agg")
from aevion.analysis import figures_stats as FS
from aevion.analysis import style
from aevion.experiments import Experiment
from aevion.scenario import Scenario

base = Scenario(trajectory=dict(kind="circle", center=[0, 0, 2.5], radius=1.5, omega=1.0), duration=10.0, record_every=5)
exp = Experiment(
    "controller_comparison",
    base,
    methods={"Geometric": {"controller": "geometric"}, "Cascaded PID": {"controller": "cascaded"}, "LQR": {"controller": "lqr"}},
    conditions={"calm": {}, "turbulent": {"wind": {"type": "DrydenTurbulence", "w20": 6.0, "severity": 1.0}}},
    seeds=range(10),
    randomization={"mass": (0.85, 1.15), "kT": (0.92, 1.08), "wind_speed": (0, 3)},
    skip=3.0,
)
res = exp.run(workers=4)  # trials are independent → parallel processes
print(res.table_text("rmse_pos"))
for row in res.compare("rmse_pos", condition="calm", baseline="Geometric"):
    print(f"{row['a']} vs {row['b']}: Welch p(Holm) = {row['t_p_holm']:.3g}, Cohen d = {row['cohens_d']:.2f}")
res.to_csv("controllers.csv")
print(res.to_latex("rmse_pos", caption="Tracking RMSE (m), mean $\\pm$ std over 10 seeds."))

style.use_style()
for name, fig in FS.experiment_figures(res, "rmse_pos", condition="calm", baseline="Geometric").items():
    style.save(fig, f"fig_{name}", formats=("pdf", "png"))
