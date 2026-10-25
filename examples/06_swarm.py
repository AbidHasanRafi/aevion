"""Six-drone ring formation: how packet loss and delay degrade decentralised collision avoidance."""

from aevion import metrics
from aevion.scenario import Scenario

base = dict(
    n=6,
    formation=dict(kind="ring", spacing=1.2),
    trajectory=dict(kind="circle", center=[0, 0, 3.0], radius=4.0, omega=0.5),
    duration=14.0,
    record_every=10,
)
for name, sw in [
    ("no avoidance", None),
    ("ideal communication", dict(avoid_radius=1.4, gain=0.9)),
    ("40 % loss, 0.3 s delay", dict(avoid_radius=1.4, gain=0.9, comm=dict(loss=0.4, delay_s=0.3, rate_hz=5))),
]:
    a = Scenario(swarm=sw, **base).run()
    m = metrics.flight_metrics(a)
    print(f"{name:26s} min separation {m['min_separation'][0]:.2f} m   RMSE {m['rmse_pos'].mean() * 100:5.1f} cm")
a.save("swarm.aev")  # then: aevion replay swarm.aev --open
