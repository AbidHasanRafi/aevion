"""Figure-eight through a forest in turbulence → archive → 3-D replay + research-grade report."""

from aevion.analysis.report import build_report
from aevion.scenario import Scenario, verify

sc = Scenario(
    name="figure8_forest",
    vehicle="generic_1kg",
    world={"kind": "forest", "n_trees": 40, "half_extent": 25},
    wind={
        "type": "CompositeWind",
        "models": [{"type": "ConstantWind", "velocity": [2.0, 1.0, 0.0]}, {"type": "DrydenTurbulence", "w20": 5.0, "severity": 1.0}],
    },
    controller="geometric",
    controller_kwargs={"ki_pos": 3.0},
    trajectory=dict(kind="figure8", center=[0, 0, 3.0], size=3.0, period=9.0),
    sensors=["imu", "gps", "baro", "mag", "lidar"],
    record_sensors=["imu", "lidar"],
    duration=20.0,
    seed=7,
)
archive = sc.run(progress=True)
print(archive.summary())
archive.save("figure8_forest.aev")
print("report:", build_report(archive, "report_figure8"))  # open report_figure8/report.html and replay.html
print("bit-exact reproduction:", verify(archive)["ok"])
