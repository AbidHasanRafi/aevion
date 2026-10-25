"""Helpers shared by the viewer tests: synthetic payloads (no simulation needed) and browser discovery."""

from __future__ import annotations

import numpy as np

from aevion.viewer.payload import FORMAT, b64


def synthetic_payload(n: int = 8, T: int = 200, duration: float = 10.0, rotors: int = 4, seed: int = 0, link: bool = False) -> dict:
    """A valid ``aevion.view/2`` payload of ``n`` vehicles flying circles of different radius, built directly with numpy."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, T)
    phase = rng.uniform(0, 2 * np.pi, n)
    radius = rng.uniform(6, 30, n)
    omega = rng.uniform(0.2, 0.6, n)
    ang = omega[None, :] * t[:, None] + phase[None, :]
    pos = np.stack([radius * np.cos(ang), radius * np.sin(ang), 5 + 2 * np.sin(0.5 * ang)], axis=-1)
    vel = np.stack([-radius * omega * np.sin(ang), radius * omega * np.cos(ang), np.zeros_like(ang)], axis=-1)
    yaw = ang + np.pi / 2
    quat = np.zeros((T, n, 4))
    quat[..., 0], quat[..., 3] = np.cos(yaw / 2), np.sin(yaw / 2)
    euler = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], axis=-1)
    arrays = dict(
        pos=pos,
        quat=quat,
        vel=vel,
        euler=euler,
        rotor=np.full((T, n, rotors), 400.0),
        thrust=np.full((T, n), 10.0),
        power=np.full((T, n), 200.0 + 20 * np.sin(t)[:, None]),
        soc=np.clip(1 - t[:, None] / (4 * duration), 0, 1) * np.ones((1, n)),
        wind=np.zeros((T, n, 3)),
    )
    payload: dict = {"format": FORMAT, "t": b64(t)}
    shapes = {}
    for k, v in arrays.items():
        payload[k] = b64(v)
        shapes[k] = list(v.shape)
    extras = {}
    if link:
        sinr = 10 + 6 * np.sin(0.7 * t)[:, None] + rng.normal(0, 1, (T, n))
        extras = {
            "x:link_sinr_db": dict(b64=b64(sinr), shape=[T, n]),
            "x:link_serving": dict(b64=b64(np.zeros((T, n))), shape=[T, n]),
            "x:link_outage": dict(b64=b64(sinr < 6), shape=[T, n]),
        }
    payload["extras"] = extras
    payload["meta"] = dict(
        title=f"synthetic {n}x{T}",
        n=n,
        T=T,
        M=rotors,
        shapes=shapes,
        events=[{"type": "handover", "t": duration / 2, "vehicle": 0}],
        duration=duration,
        t0=0.0,
        t1=duration,
        controller="synthetic",
        trajectory="circles",
        vehicle=dict(
            name="synthetic",
            mass=1.0,
            rotors=[[0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0], [0, -0.2, 0]][:rotors],
            spin=[1, -1, 1, -1][:rotors],
            radius=0.1,
            w_max=700.0,
        ),
    )
    payload["world"] = dict(boxes=[], spheres=[], cylinders=[], box_colors=[], sphere_colors=[], cyl_colors=[], bounds=[-40, -40, 0, 40, 40, 30])
    if link:
        payload["wireless"] = dict(kind="cells", outage_db=6.0, cells=[dict(pos=[0, 0, 20], yaw_deg=0, omni=True)])
    return payload


def sim_archive(**kw):
    """A short real run (1 vehicle by default); ``kw`` are Scenario fields."""
    from aevion.scenario import Scenario

    base = dict(trajectory=dict(kind="circle", center=[0, 0, 2.5], radius=1.2, omega=1.0), duration=3.0, record_every=4)
    base.update(kw)
    return Scenario(**base).run()


NR_CONFIG = {
    "duration": 8.0,
    "seed": 2,
    "world": {"type": "city", "params": {"blocks": [3, 3]}},
    "trajectory": {"type": "circle", "params": {"center": [150, 0, 60.0], "radius": 120.0, "omega": 0.3}},
    "network": {
        "type": "nr_cellular",
        "rate_hz": 10,
        "params": {"sites": [{"pos": [0, 0, 25], "sectors": 3}, {"pos": [300, 0, 25], "sectors": 3}], "background_users": 1, "los_mode": "geometric"},
    },
}
