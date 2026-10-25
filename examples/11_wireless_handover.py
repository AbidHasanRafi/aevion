"""Cellular-connected UAV crossing between two 5G cells: path loss, fading, handover, link adaptation and a jamming event.

python examples/11_wireless_handover.py [--out DIR] [--duration S] [--no-figures]

Writes the link-stack figure, the radio environment map with the flight path, an SINR CDF and the metrics to DIR.
"""

import argparse
from pathlib import Path

import numpy as np

from aevion.engine import Simulation
from aevion.metrics import flight_metrics
from aevion.wireless.mac_network import NrCellularNetwork
from aevion.wireless.maps import radio_environment_map

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="wireless_demo")
ap.add_argument("--duration", type=float, default=60.0)
ap.add_argument("--no-figures", action="store_true")
args = ap.parse_args()

SITES = [{"pos": [0, 0, 25], "sectors": 1}, {"pos": [600, 0, 25], "sectors": 1}]
NETWORK = {
    "sites": SITES,
    "propagation": {"model": "tr36777", "scenario": "uma_av"},
    "freq_hz": 3.5e9,
    "bandwidth_mhz": 100,
    "scs_khz": 30,
    "background_users": 3,
    "scheduler": "proportional_fair",
    "handover": {"hysteresis_db": 3.0, "time_to_trigger_s": 0.16},
}
t_jam = (0.62 * args.duration, 0.72 * args.duration)
config = {
    "name": "wireless_handover",
    "seed": 11,
    "duration": args.duration,
    "vehicle": {"initial": {"pos": [-100, 0, 60]}},
    "trajectory": {"type": "waypoints", "params": {"points": [[-100, 0, 60], [700, 0, 60]], "speed": 800.0 / args.duration}},
    "network": {"type": "nr_cellular", "rate_hz": 20, "params": NETWORK},
    "faults": [{"type": "network_jamming", "t_start": t_jam[0], "t_end": t_jam[1], "params": {"pos": [450, 40, 5], "power_dbm": 45.0}}],
}

archive = Simulation(config).run()
m = flight_metrics(archive)
print(archive)
print(
    f"handovers: {int(m['link_handovers'][0])}   outage: {m['link_outage_frac'][0] * 100:.1f} %   mean goodput: {m['link_mean_rate_mbps'][0]:.0f} Mbit/s"
)
t = archive.t
jam = (t > t_jam[0]) & (t < t_jam[1])
clean = Simulation({**config, "faults": []}).run()  # the same flight without the jammer (same seed: identical fading and shadowing)
drop = clean["link_sinr_db"][jam, 0] - archive["link_sinr_db"][jam, 0]
print(
    f"the jammer costs a median {np.median(drop):.1f} dB of SINR and {np.mean(clean['link_rate_mbps'][jam, 0] - archive['link_rate_mbps'][jam, 0]):.1f} Mbit/s of goodput"
)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
archive.save(out / "run.aevd")
if not args.no_figures:
    import matplotlib

    matplotlib.use("Agg")
    from aevion.analysis import figures as F

    net = NrCellularNetwork(**{**NETWORK, "seed": config["seed"]})
    rem = radio_environment_map(net, 60.0, (-200, 800, -200, 200), 10.0)
    for name, fig in {
        "link_stack": F.fig_link_stack(archive),
        "rem": F.fig_rem(rem, net, archive["pos"][:, 0, :2]),
        "sinr_cdf": F.fig_outage_cdf(archive),
    }.items():
        fig.savefig(out / f"{name}.png", dpi=200)
        fig.savefig(out / f"{name}.pdf")
    print("figures written to", out.resolve())
