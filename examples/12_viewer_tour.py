"""A tour of the viewer: replay file, comparison, screenshot, and the live stream, from one scenario.

Run it, then open the HTML files in a browser. The screenshot step needs Chrome, Edge or Chromium and is skipped without one.
"""

from aevion.scenario import Scenario
from aevion.viewer import compare_html, export_html, find_browser

base = dict(
    trajectory=dict(kind="figure8", center=[0, 0, 3.0], size=3.0, period=9.0),
    world={"kind": "forest", "n_trees": 40, "half_extent": 25},
    sensors=["imu", "lidar"],
    record_sensors=["lidar"],
    duration=12.0,
    record_every=4,
)
calm = Scenario(**base).run()
gusty = Scenario(**base, wind={"type": "ConstantWind", "velocity": [3.0, 1.0, 0.0]}).run()

print(export_html(calm, "tour_replay.html", title="Figure-eight in a forest"))  # one self-contained file
print(compare_html([calm, gusty], "tour_compare.html", labels=["calm", "gusty"]))  # ghosts + difference chart

if find_browser():
    from aevion.viewer import screenshot

    print(screenshot(calm, "tour_figure.png", theme="print", size=(1600, 900), ui=False))  # white-page figure

# Live: run a scenario file and watch it in the browser while it flies (Ctrl+C to stop):
#   aevion live examples/scenarios/wireless_city.yaml --open
# Scene editor: aevion edit --open   then   aevion run scene.json
