"""Physical effects that act on the vehicle from outside its rigid-body model: tethered loads, proximity aerodynamics, battery.

Each effect is an engine :class:`~aevion.engine.hooks.Hook`. Before every physics step the force/torque hooks write into the
vehicle's *external load* array (``dyn.params.ext``: world force, body torque, thrust multiplier), which both dynamics kernels
apply; after the step the battery hook replaces the simple state-of-charge integration by the detailed pack model. They are
configured like any hook::

    hooks:
      - {type: slung_load, params: {mass: 0.3, length: 1.5}}
      - {type: proximity_effects}
      - {type: battery_model, params: {capacity_ah: 4.0, cells: 3}}
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from ..core.backend import NumpyBackend
from ..vehicles.battery_pack import BatteryPack, PackSpec
from .hooks import Hook

if TYPE_CHECKING:
    from .simulation import Simulation

__all__ = ["SlungLoadHook", "ProximityEffectsHook", "BatteryModelHook", "tether_load_step"]

G0 = 9.80665


def _begin(sim: "Simulation") -> None:
    """The first effect of a step clears the external loads, the others add to them (whatever their priorities)."""
    if getattr(sim, "_effects_step", -1) != sim.k:
        sim.dyn.params.clear_external()
        sim._effects_step = sim.k  # type: ignore[attr-defined]


def _add_ext(
    sim: "Simulation", force: Optional[np.ndarray] = None, torque: Optional[np.ndarray] = None, thrust_scale: Optional[np.ndarray] = None
) -> None:
    B = sim.B
    ext = np.array(B.to_numpy(sim.dyn.params.ext), copy=True)
    if force is not None:
        ext[:, 0:3] += force
    if torque is not None:
        ext[:, 3:6] += torque
    if thrust_scale is not None:
        ext[:, 6] *= thrust_scale
    sim.dyn.params.ext = B.asarray(ext)


def tether_load_step(
    p_load: np.ndarray,
    v_load: np.ndarray,
    p_att0: np.ndarray,
    v_att: np.ndarray,
    dt: float,
    mass: float,
    length: float,
    stiffness: float,
    damping: float,
    drag: float,
    g: float = G0,
    ground_z: Optional[np.ndarray] = None,
    substeps: Optional[int] = None,
) -> tuple:
    """Advance an elastic tether load by ``dt`` while its attachment point moves linearly from ``p_att0`` with velocity ``v_att``.

    The rope pulls only when stretched: ``T = max(0, k (|d| - L) + c d|d|/dt)`` along the rope. Returns ``(p_load, v_load,
    mean_tension_force_on_the_attachment (n, 3), mean_tension (n,))``. The load integrates with a semi-implicit scheme over
    enough substeps for the rope's own frequency ``sqrt(k / m)``.
    """
    n_sub = substeps or max(1, math.ceil(dt / (0.15 * math.sqrt(mass / max(stiffness, 1e-9)))))
    h = dt / n_sub
    p, v = p_load.copy(), v_load.copy()
    f_sum = np.zeros_like(p)
    t_sum = np.zeros(len(p))
    gvec = np.array([0.0, 0.0, -g])
    for k in range(n_sub):
        pa = p_att0 + v_att * (k * h)
        d = p - pa
        ell = np.linalg.norm(d, axis=1)
        u = d / np.maximum(ell, 1e-9)[:, None]
        stretch = ell - length
        rate = np.sum(u * (v - v_att), axis=1)
        T = np.maximum(stiffness * stretch + damping * rate, 0.0) * (stretch > 0)
        speed = np.linalg.norm(v, axis=1)
        F = -T[:, None] * u + mass * gvec - drag * speed[:, None] * v
        v = v + (F / mass) * h
        p = p + v * h
        if ground_z is not None:
            below = p[:, 2] < ground_z
            p[below, 2] = ground_z[below]
            v[below, 2] = np.maximum(v[below, 2], 0.0)
            v[below, 0:2] *= 0.98  # sliding friction on the ground
        f_sum += T[:, None] * u
        t_sum += T
    return p, v, f_sum / n_sub, t_sum / n_sub


class SlungLoadHook(Hook):
    """A payload hanging from the vehicle on an elastic rope (tension only, so the rope goes slack in free fall).

    The load is a point mass ``mass`` [kg] on a rope of rest length ``length`` [m] attached at ``attach`` (body frame). The
    rope is a spring-damper (``stiffness`` [N/m], ``damping_ratio`` of critical): the default stiffness gives 1 cm of stretch
    under the load's own weight. The tension pulls the vehicle (a world force through its centre, plus the torque of the offset
    attachment) and the load swings, so a flying vehicle sees the pendulum disturbance a real slung load causes. Quadratic air
    drag acts on the load in still air. Recorded channels: ``load_pos`` and ``load_tension``.
    """

    name = "slung_load"
    priority = 20

    def __init__(
        self,
        mass: float = 0.3,
        length: float = 1.5,
        stiffness: Optional[float] = None,
        damping_ratio: float = 0.15,
        drag_coefficient: float = 0.5,
        drag_area: float = 0.01,
        attach: Sequence[float] = (0.0, 0.0, -0.05),
    ) -> None:
        if mass <= 0 or length <= 0:
            raise ValueError("mass and length must be positive")
        self.m, self.L = float(mass), float(length)
        self.k = float(stiffness) if stiffness is not None else self.m * G0 / 0.01
        self.zeta = float(damping_ratio)
        self.c = 2.0 * self.zeta * math.sqrt(self.k * self.m)
        self.drag = 0.5 * 1.225 * float(drag_coefficient) * float(drag_area)
        self.attach = np.asarray(attach, float)
        self.p_load: Any = None
        self.v_load: Any = None

    def _attachment(self, sim: "Simulation") -> tuple:
        B = sim.B
        st = sim.state
        pos, vel, q, om = (B.to_numpy(x) for x in (st.pos, st.vel, st.quat, st.omega))
        from ..core import rotation as rot

        R = rot.to_matrix(NumpyBackend(), q)
        r_w = np.einsum("nij,j->ni", R, self.attach)
        om_w = np.einsum("nij,nj->ni", R, om)
        return pos + r_w, vel + np.cross(om_w, r_w), R, r_w

    def on_reset(self, sim: "Simulation") -> None:
        pa, va, _, _ = self._attachment(sim)
        self.p_load = pa - np.array([0.0, 0.0, self.L + self.m * G0 / self.k])  # hanging at rest, stretched by its own weight
        self.v_load = va.copy()
        sim.ctx.extras["load_pos"] = sim.B.asarray(self.p_load.copy())  # (so the very first recorded row has them too)
        sim.ctx.extras["load_tension"] = sim.B.asarray(np.full(len(pa), self.m * sim.world.gravity.g))

    def pre_step(self, sim: "Simulation") -> None:
        _begin(sim)
        if self.p_load is None:
            self.on_reset(sim)
        pa, va, R, r_w = self._attachment(sim)
        gz = None
        try:
            gz = np.asarray(sim.world.ground_height(NumpyBackend(), self.p_load[:, 0], self.p_load[:, 1]), float)
        except Exception:  # a world without a ground model: no contact
            gz = None
        self.p_load, self.v_load, f_on_vehicle, tension = tether_load_step(
            self.p_load, self.v_load, pa, va, sim.dt, self.m, self.L, self.k, self.c, self.drag, sim.world.gravity.g, gz
        )
        torque_w = np.cross(r_w, f_on_vehicle)
        torque_b = np.einsum("nji,nj->ni", R, torque_w)
        _add_ext(sim, force=f_on_vehicle, torque=torque_b)
        sim.ctx.extras["load_pos"] = sim.B.asarray(self.p_load.copy())
        sim.ctx.extras["load_tension"] = sim.B.asarray(tension)


class ProximityEffectsHook(Hook):
    """Representative ceiling and wall effects on the rotors (ground effect is built into the dynamics).

    * **Ceiling**: a surface above the rotor plane at height ``h`` raises the thrust at a given speed,
      ``T / T∞ = 1 / (1 - κ_c (R / 4h)²)`` (the form of the ground-effect law of Cheeseman & Bennett, with a ceiling constant
      ``κ_c`` between 1.5 and 2 fitting the measurements of Sanchez-Cuevas et al., 2017), capped at ``max_gain``.
    * **Wall**: a vertical surface at distance ``d`` pulls the vehicle toward it, ``F = κ_w T (R / 4d)²`` (the lateral suction
      measured by the same authors is of this order), along the wall normal.

    The constants are representative, not fitted to a particular vehicle; set them to your measurements. The surfaces are
    whatever the world contains (boxes, meshes, terrain), found by ray casting.
    """

    name = "proximity_effects"
    priority = 25

    def __init__(
        self, ceiling_kappa: float = 1.8, wall_kappa: float = 0.6, max_gain: float = 1.5, range_factor: float = 4.0, wall_rays: int = 4
    ) -> None:
        self.kc, self.kw, self.max_gain, self.range_factor = float(ceiling_kappa), float(wall_kappa), float(max_gain), float(range_factor)
        ang = np.linspace(0.0, 2.0 * math.pi, int(wall_rays), endpoint=False)
        self.dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)

    def _compute(self, sim: "Simulation") -> tuple:
        from ..sensors.raycast import cast_rays

        B = sim.B
        R = sim.spec.rotor_radius
        reach = self.range_factor * R
        pos = B.to_numpy(sim.state.pos)
        n = len(pos)
        up = cast_rays(sim.world, pos, np.tile([0.0, 0.0, 1.0], (n, 1)), reach)
        h = np.asarray(up.t, float)
        gain = np.where(
            h < reach * 0.999, np.minimum(1.0 / np.maximum(1.0 - self.kc * (R / (4.0 * np.maximum(h, 0.3 * R))) ** 2, 1e-3), self.max_gain), 1.0
        )
        thrust = np.sum(B.to_numpy(sim.state.thrust), axis=1)
        force = np.zeros((n, 3))
        for d in self.dirs:
            hit = cast_rays(sim.world, pos, np.tile(d, (n, 1)), reach)
            dist = np.maximum(np.asarray(hit.t, float), 0.3 * R)
            near = np.asarray(hit.t, float) < reach * 0.999
            mag = np.where(near, self.kw * thrust * (R / (4.0 * dist)) ** 2, 0.0)
            force += mag[:, None] * d[None, :] / len(self.dirs) * 2.0
        return gain, force

    def on_reset(self, sim: "Simulation") -> None:
        gain, _ = self._compute(sim)
        sim.ctx.extras["ceiling_gain"] = sim.B.asarray(gain.copy())

    def pre_step(self, sim: "Simulation") -> None:
        _begin(sim)
        gain, force = self._compute(sim)
        _add_ext(sim, force=force, thrust_scale=gain)
        sim.ctx.extras["ceiling_gain"] = sim.B.asarray(gain.copy())


class BatteryModelHook(Hook):
    """Replace the dynamics' simple state-of-charge integration by a :class:`~aevion.vehicles.battery_pack.BatteryPack`.

    After each step the electrical power the rotors drew (``state.power``) is turned into a pack current, the pack advances
    (Peukert capacity, polarisation, heating, ageing), and the pack's state of charge is written back to ``state.soc`` - so the
    low-battery events and the empty-pack thrust cut-off follow the detailed model. Recorded: ``battery_voltage``,
    ``battery_temp_c``, ``battery_current`` and ``battery_soh``.
    """

    name = "battery_model"
    priority = 80

    def __init__(self, ambient_c: float = 25.0, initial_temperature_c: Optional[float] = None, **spec: Any) -> None:
        self._spec_kw = {k: v for k, v in spec.items() if v is not None}
        self.ambient, self.t0 = float(ambient_c), initial_temperature_c
        self.pack: Optional[BatteryPack] = None

    def _make(self, sim: "Simulation") -> BatteryPack:
        bat = sim.spec.battery
        kw = dict(cells=bat.cells, capacity_ah=bat.capacity_wh / (bat.cells * 3.7), r0=max(bat.internal_resistance, 1e-3))
        kw.update(self._spec_kw)
        return BatteryPack(PackSpec(**kw), sim.n, soc=1.0, temperature_c=self.ambient if self.t0 is None else self.t0)

    def on_reset(self, sim: "Simulation") -> None:
        self.pack = self._make(sim)
        self.pack.reset(np.asarray(sim.B.to_numpy(sim.state.soc), float), self.ambient if self.t0 is None else self.t0)
        B = sim.B
        for name, val in (
            ("battery_voltage", self.pack.voltage),
            ("battery_temp_c", self.pack.temperature),
            ("battery_current", self.pack.current),
            ("battery_soh", self.pack.retention),
        ):
            sim.ctx.extras[name] = B.asarray(np.array(val, copy=True))

    def post_step(self, sim: "Simulation") -> None:
        if self.pack is None:
            self.on_reset(sim)
        assert self.pack is not None
        B = sim.B
        out = self.pack.step(sim.dt, power_w=np.asarray(B.to_numpy(sim.state.power), float), ambient_c=self.ambient)
        sim.state.soc = B.asarray(self.pack.soc.copy())
        sim.ctx.extras["battery_voltage"] = B.asarray(out["voltage"].copy())
        sim.ctx.extras["battery_temp_c"] = B.asarray(out["temperature"].copy())
        sim.ctx.extras["battery_current"] = B.asarray(out["current"].copy())
        sim.ctx.extras["battery_soh"] = B.asarray(out["soh"].copy())
