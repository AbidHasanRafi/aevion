"""Cellular-connected UAV through a city: SINR along the path with geometric line-of-sight."""

import matplotlib

matplotlib.use("Agg")
from aevion.analysis import figures as F
from aevion.analysis import style
from aevion.scenario import Scenario
from aevion.wireless import AirToGroundChannel, BaseStation, CellularNetwork

net = CellularNetwork(
    [BaseStation((-60, -60, 25)), BaseStation((60, 60, 25)), BaseStation((0, 0, 30), antenna="isotropic")],
    AirToGroundChannel("urban", los_mode="geometric"),
)
# Compare flying low (between buildings) and high (above them)
for alt in (15.0, 80.0):
    sc = Scenario(
        name=f"city_{int(alt)}m",
        world={"kind": "city", "blocks": [3, 3]},
        wireless=net.to_dict(),
        trajectory=dict(kind="waypoints", points=[[-70, -70, alt], [70, -70, alt], [70, 70, alt], [-70, 70, alt]], speed=8.0),
        duration=45.0,
        record_every=10,
    )
    a = sc.run()
    sinr, los = a["link_sinr_db"][:, 0], a["link_los"][:, 0]
    print(
        f"altitude {alt:4.0f} m: median SINR {sinr.mean():5.1f} dB, LoS {los.mean() * 100:3.0f} %, mean rate {a['link_rate_mbps'].mean():5.1f} Mbit/s"
    )
    style.save(F.fig_link(a), f"link_{int(alt)}m", formats=("png",))

net2 = CellularNetwork.from_dict(a.meta["wireless"], world=a.world)
style.save(F.fig_coverage_map(net2, 50.0, (-100, 100, -100, 100), 4.0, path=a["pos"][:, 0]), "coverage_50m", formats=("pdf", "png"))
