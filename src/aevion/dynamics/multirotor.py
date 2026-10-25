"""Batched 6-DOF multirotor dynamics.

Equations of motion (world frame ENU, body frame FLU, ``R = R(q)`` body to world)::

    p_dot = v
    m v_dot = m g + R (F_thrust + F_drag + F_rotor_drag)
    q_dot = 1/2 q ⊗ [0, ω]
    J ω_dot = τ_thrust + τ_reaction + τ_gyro + τ_damp − ω × (J ω + h_r)
    w_dot_i = (w_cmd_i − w_i) / τ_m            (first-order motor lag, optional asymmetry)

with, for rotor ``i`` of spin ``s_i``::

    T_i = kT ρ/ρ0 η_i w_i^2 f_inflow f_ground            (thrust)
    Q_i = kQ ρ/ρ0 η_i w_i^2                              (reaction torque)
    τ_thrust = Σ r_i × (T_i ẑ)     τ_reaction = −Σ s_i (Q_i + J_r w_dot_i) ẑ
    h_r = Σ s_i J_r w_i ẑ          τ_gyro = −ω × h_r

All quantities are tensors of shape ``(N, ...)`` for ``N`` independent vehicles on the chosen
:class:`~aevion.core.backend.Backend`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..core import rotation as rot
from ..core.backend import get_backend
from ..core.constants import RHO0
from ..core.integrators import get_integrator
from ..environments.world import World
from ..vehicles.spec import BatchParams, VehicleSpec, get_vehicle
from . import fast as _fast
from . import kernels as _kernels


@dataclass
class State:
    pos: Any
    vel: Any
    quat: Any
    omega: Any  # body angular rate [rad/s]
    rotor: Any  # rotor speeds [rad/s], (N, M)
    soc: Any  # battery state of charge (N,)
    energy: Any  # consumed electrical energy [J] (N,)
    t: float = 0.0
    # diagnostics produced by the last step (not part of the integrated state)
    acc: Any = None  # world-frame linear acceleration (N, 3)
    f_body: Any = None  # specific force felt by an accelerometer (N, 3)
    thrust: Any = None  # per-rotor thrust (N, M)
    power: Any = None  # electrical power draw (N,)
    wind: Any = None  # world-frame wind at the vehicle (N, 3)
    contact: Any = None  # bool (N,)
    collided: Any = None  # bool (N,) obstacle contact during the step
    impact_speed: Any = None  # normal speed at ground impact this step (N,)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def clone(self, B=None):
        def c(x):
            if x is None or isinstance(x, float):
                return x
            return x.clone() if hasattr(x, "clone") else np.array(x, copy=True)

        return State(**{k: c(getattr(self, k)) for k in self.__dataclass_fields__})

    def detach(self):
        def d(x):
            return x.detach() if hasattr(x, "detach") else x

        return State(**{k: d(getattr(self, k)) for k in self.__dataclass_fields__})

    def to_numpy(self, B) -> dict:
        out = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            out[k] = v if v is None or isinstance(v, float) else B.to_numpy(v)
        return out

    def euler(self, B):
        return rot.to_euler(B, self.quat)

    def select(self, idx):
        """Index a subset of vehicles (used for partial resets)."""
        kw: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            kw[k] = v if v is None or isinstance(v, float) else v[idx]
        return State(**kw)


class Multirotor:
    """Batched multirotor simulator.

    Parameters
    ----------
    vehicle : VehicleSpec or preset name
    n : number of parallel vehicles
    backend : ``"numpy"`` (default), ``"torch"``, ``"cuda"`` or a Backend instance
    dt : integration step [s]
    integrator : ``"rk4"`` (default), ``"midpoint"``, ``"semi_implicit"``, ``"euler"``
    world : :class:`~aevion.environments.world.World` (gravity, atmosphere, wind, obstacles)
    """

    def __init__(
        self,
        vehicle="generic_1kg",
        n: int = 1,
        backend="numpy",
        dt: float = 0.005,
        integrator: str = "rk4",
        world: Optional[World] = None,
        seed: int = 0,
        dtype=None,
        ground_offset: float = 0.0,
        contact_ang_damping: float = 25.0,
        kernel: str = "auto",
        verify_kernel: bool = True,
    ):
        self.spec: VehicleSpec = get_vehicle(vehicle)
        self.B = get_backend(backend, dtype)
        self.n = int(n)
        self.dt = float(dt)
        self.integrator_name = integrator
        self._integrate = get_integrator(integrator)
        self.world = world if world is not None else World()
        self.seed = seed
        self.ground_offset = ground_offset
        self.contact_ang_damping = contact_ang_damping
        self.params = BatchParams(self.B, self.spec, self.n)
        self.world.reset(self.B, self.n, seed)
        self._aux: Optional[Dict[str, Any]] = None
        self._first = True
        # Kernel selection: the generic path (_step_reference) is the model definition; "numba" is a fused CPU
        # re-implementation guarded by a parity check (see dynamics/kernels.py). kernel="reference" forces the reference.
        if _fast.HAVE_NUMBA and self.B.name == "numpy":
            self._prep_fast()
        self.kernel = _kernels.select(self, kernel, verify_kernel)
        self.refresh_substeps()
        for w in self.spec.validate():
            import warnings

            warnings.warn(f"[{self.spec.name}] {w}", stacklevel=2)

    def refresh_substeps(self):
        """Choose the number of internal sub-steps so the stiff motor lag stays inside the
        integrator's stability region (explicit methods need ``dt_sub`` ≲ a few time constants).
        A 0.5 safety factor leaves room for randomised (shorter) time constants."""
        spec = self.spec
        taus = [spec.motor_tau] + ([spec.motor_tau_down] if spec.motor_tau_down else [])
        tau_min = 0.5 * max(min(taus), 1e-6)
        limit = (1.5 if self.integrator_name in ("rk4", "midpoint") else 0.5) * tau_min
        self.n_sub = max(1, math.ceil(self.dt / limit - 1e-9))

    # ------------------------------------------------------------------ fast path
    def _prep_fast(self):
        """Pack scalar settings for the compiled kernel (call again if the world/gravity changes)."""
        w, sp = self.world, self.spec
        gm = w.gravity
        kind = {"constant": 0, "custom": 0, "altitude": 1, "wgs84": 2}[gm.kind]
        g_lat = gm.latitude_g() if gm.kind == "wgs84" else gm.g
        has_hm = w.heightmap is not None
        self._sc = np.array(
            [
                sp.rotor_inertia,
                sp.rotor_radius,
                sp.inflow_gain,
                1.0 if sp.ground_effect else 0.0,
                kind,
                gm.g,
                g_lat,
                gm.altitude_ref,
                1.0 if w.atmosphere.enabled else 0.0,
                w.atmosphere.base_altitude,
                w.atmosphere.temp_offset,
                1.0 if has_hm else 0.0,
                w.hm_origin[0],
                w.hm_origin[1],
                w.hm_res,
                sp.battery.motor_efficiency,
                sp.battery.avionics_power,
                w.atmosphere.pressure_sl,
                w.atmosphere.humidity,
            ],
            dtype=float,
        )
        self._ones_n = np.ones(self.n)

    def _step_fast(self, state, cmd, dt):
        B, P, spec, w = self.B, self.params, self.spec, self.world
        N, M = self.n, spec.n_rotors
        c = np.ascontiguousarray
        bat = spec.battery
        scale = self._ones_n
        if bat.voltage_effects:
            ocv = bat.ocv(state.soc)
            pwr = state.power if state.power is not None else np.full(N, 10.0)
            disc = np.maximum(ocv * ocv - 4.0 * bat.internal_resistance * pwr, 0.0)
            scale = 0.5 * (ocv + np.sqrt(disc)) / bat.v_full
            scale = np.where(state.soc <= 0.0, 0.0, scale)
        wind = w.wind.sample(B, state.pos, state.vel, state.t, dt)
        hm = w.heightmap if w.heightmap is not None else np.zeros((2, 2))
        out = dict(
            pos=np.empty((N, 3)),
            vel=np.empty((N, 3)),
            quat=np.empty((N, 4)),
            omega=np.empty((N, 3)),
            rotor=np.empty((N, M)),
            soc=np.empty(N),
            energy=np.empty(N),
            acc=np.empty((N, 3)),
            fb=np.empty((N, 3)),
            thrust=np.empty((N, M)),
            power=np.empty(N),
            contact=np.empty(N, dtype=np.bool_),
            collided=np.empty(N, dtype=np.bool_),
            impact=np.empty(N),
        )
        _fast.step_kernel(
            c(state.pos),
            c(state.vel),
            c(state.quat),
            c(state.omega),
            c(state.rotor),
            c(state.soc),
            c(state.energy),
            c(cmd, dtype=float),
            c(wind),
            scale,
            c(P.mass[:, 0] + P.payload_mass[:, 0]),  # payload (a fault or a randomised carry) is part of the moving mass
            c(P.inertia),
            c(P.inertia_inv),
            c(P.kT),
            c(P.kQ),
            c(P.tau_up),
            c(P.tau_down),
            c(P.rotor_health),
            c(P.drag_lin),
            c(P.drag_quad),
            c(P.rotor_drag[:, 0]),
            c(P.ang_damp),
            c(P.com_offset),
            c(P.rotor_pos),
            c(P.rotor_spin),
            c(hm),
            c(P.ext),
            self._sc,
            float(P.w_min),
            c(P.w_max),
            dt,
            _fast.METHOD_IDS[self.integrator_name],
            bat.capacity_wh,
            spec.restitution,
            spec.ground_friction,
            self.contact_ang_damping,
            self.ground_offset,
            spec.collision_radius,
            c(w.boxes),
            c(w.spheres),
            c(w.cylinders),
            bool(w.has_obstacles),
            out["pos"],
            out["vel"],
            out["quat"],
            out["omega"],
            out["rotor"],
            out["soc"],
            out["energy"],
            out["acc"],
            out["fb"],
            out["thrust"],
            out["power"],
            out["contact"],
            out["collided"],
            out["impact"],
        )
        if self.world.meshes:  # the fused kernel only knows the primitives: resolve mesh contacts afterwards
            out["pos"], out["vel"], out["collided"] = self.world.mesh_response(
                out["pos"], out["vel"], out["collided"], self.spec.collision_radius, self.spec.restitution
            )
        return State(
            pos=out["pos"],
            vel=out["vel"],
            quat=out["quat"],
            omega=out["omega"],
            rotor=out["rotor"],
            soc=out["soc"],
            energy=out["energy"],
            t=state.t + dt,
            acc=out["acc"],
            f_body=out["fb"],
            thrust=out["thrust"],
            power=out["power"],
            wind=wind,
            contact=out["contact"],
            collided=out["collided"],
            impact_speed=out["impact"],
        )

    # ------------------------------------------------------------------ setup
    @property
    def n_rotors(self) -> int:
        return self.spec.n_rotors

    def hover_speed(self):
        g = self.world.gravity.g
        return float(np.sqrt(self.spec.mass * g / (self.spec.n_rotors * self.spec.kT)))

    def reset(self, pos=None, yaw=None, euler=None, vel=None, omega=None, hover_rotors=True, soc=1.0) -> State:
        """Create an initial state. ``pos`` is ``(3,)`` or ``(N, 3)``; defaults to 1 m up."""
        B, n, M = self.B, self.n, self.spec.n_rotors
        self.params.clear_external()  # effects (tethers, proximity) start from nothing in every episode
        p = B.ones(n, 3) * B.asarray([0.0, 0.0, 1.0] if pos is None else pos)
        v = B.zeros(n, 3) + (0.0 if vel is None else B.asarray(vel))
        w = B.zeros(n, 3) + (0.0 if omega is None else B.asarray(omega))
        if euler is not None:
            e = B.ones(n, 3) * B.asarray(euler)
            q = rot.from_euler(B, e[..., 0], e[..., 1], e[..., 2])
        else:
            yy = B.zeros(n) + (0.0 if yaw is None else B.asarray(yaw))
            q = rot.from_euler(B, B.zeros(n), B.zeros(n), yy)
        hs = self.hover_speed() if hover_rotors else 0.0
        rotors = B.ones(n, M) * hs
        s = State(
            pos=p,
            vel=v,
            quat=q,
            omega=w,
            rotor=rotors,
            soc=B.ones(n) * soc,
            energy=B.zeros(n),
            t=0.0,
        )
        self.world.reset(B, n, self.seed)
        self.world.advance(0.0)
        return self._diagnose(s, rotors)

    def _diagnose(self, s: State, cmd):
        """Fill diagnostic fields for ``s`` without advancing time."""
        B = self.B
        w = B.zeros(s.n, 3)
        s.wind = w
        y = (s.pos, s.vel, s.quat, s.omega, s.rotor)
        self._first = True
        self._deriv(y, cmd, w)
        a = self._aux
        assert a is not None  # filled by the _deriv call above
        s.acc, s.f_body, s.thrust, s.power = a["acc"], a["f_body"], a["thrust"], a["power"]
        s.contact = self._contact_mask(s.pos)
        s.collided = B.zeros(s.n) > 1.0
        s.impact_speed = B.zeros(s.n)
        return s

    def hover_command(self):
        return self.B.ones(self.n, self.n_rotors) * self.hover_speed()

    # --------------------------------------------------------------- dynamics
    def _contact_mask(self, pos):
        gz = self.world.ground_height(self.B, pos[..., 0], pos[..., 1]) + self.ground_offset
        return pos[..., 2] <= gz + 1e-3

    def _deriv(self, y, cmd, wind):
        B, P, spec = self.B, self.params, self.spec
        pos, vel, q, om, w = y
        R = rot.to_matrix(B, q)
        z = pos[..., 2]
        g = self.world.gravity.magnitude(B, z)
        rho_ratio = (self.world.atmosphere.density(B, z) / RHO0)[..., None]  # (N,1)

        # air-relative velocity in body axes
        v_rel_b = B.einsum("nji,nj->ni", R, vel - wind)
        vc = v_rel_b[..., 2:3]  # climb rate relative to air along thrust axis

        # rotor thrust and aerodynamic reaction torque
        w2 = w * w
        eff = rho_ratio * P.rotor_health
        f_in = 1.0
        if spec.inflow_gain != 0.0:
            f_in = B.clip(1.0 - spec.inflow_gain * vc / (B.maximum(w, 50.0 * B.ones(1)) * spec.rotor_radius), 0.0, 1.5)
        f_ge = 1.0
        if spec.ground_effect:
            agl = z - self.world.ground_height(B, pos[..., 0], pos[..., 1])
            ratio = spec.rotor_radius / (4.0 * B.maximum(agl, 0.5 * spec.rotor_radius * B.ones(1)))
            f_ge = B.clip(1.0 / (1.0 - (ratio * ratio)[..., None] + 1e-6), 1.0, 1.5)
        T = P.kT * eff * w2 * f_in * f_ge * P.ext[:, 6:7]  # (N,M); ext[:, 6] is the thrust multiplier of proximity effects
        Q = P.kQ * eff * w2  # (N,M)
        Tsum = B.sum(T, axis=-1, keepdims=True)  # (N,1)

        # forces (body frame)
        zero = Tsum * 0.0
        F_thrust = B.cat([zero, zero, Tsum], axis=-1)
        speed = B.norm(v_rel_b, axis=-1, keepdims=True)
        F_drag = -(P.drag_lin + P.drag_quad * speed) * v_rel_b * rho_ratio
        if spec.rotor_drag != 0.0:
            wsum = B.sum(w, axis=-1, keepdims=True)
            F_rotor = -P.rotor_drag * wsum * B.cat([v_rel_b[..., 0:1], v_rel_b[..., 1:2], zero], axis=-1)
        else:
            F_rotor = 0.0
        F_b = F_thrust + F_drag + F_rotor
        m = P.effective_mass()
        f_body = F_b / m
        gvec = B.stack([z * 0.0, z * 0.0, -g], -1)
        acc = gvec + B.einsum("nij,nj->ni", R, f_body) + P.ext[:, 0:3] / m  # + external force (world frame)

        # rotor dynamics
        up = cmd > w
        tau = B.where(up, P.tau_up, P.tau_down)
        wdot = (cmd - w) / tau

        # torques (relative to the centre of mass)
        rx = P.rotor_pos[:, 0][None, :] - P.com_offset[:, 0:1]
        ry = P.rotor_pos[:, 1][None, :] - P.com_offset[:, 1:2]
        tau_x = B.sum(ry * T, axis=-1)
        tau_y = -B.sum(rx * T, axis=-1)
        s = P.rotor_spin[None, :]
        tau_z = -B.sum(s * (Q + P.J_r * wdot), axis=-1)
        tau_thrust = B.stack([tau_x, tau_y, tau_z], -1)
        h_z = B.sum(s * P.J_r * w, axis=-1)
        h_r = B.stack([h_z * 0.0, h_z * 0.0, h_z], -1)
        tau_gyro = -B.cross(om, h_r)
        tau = tau_thrust + tau_gyro - P.ang_damp * om + P.ext[:, 3:6]  # + external torque (body frame)
        Jom = B.einsum("nij,nj->ni", P.inertia, om)
        omdot = B.einsum("nij,nj->ni", P.inertia_inv, tau - B.cross(om, Jom))

        qdot = rot.deriv(B, q, om)

        if self._first:
            p_mech = B.sum(Q * w, axis=-1)
            bat = spec.battery
            self._aux = dict(
                acc=acc,
                f_body=f_body,
                thrust=T,
                power=p_mech / bat.motor_efficiency + bat.avionics_power,
            )
            self._first = False
        return (vel, acc, qdot, omdot, wdot)

    def _project(self, y):
        pos, vel, q, om, w = y
        return (pos, vel, rot.qnormalize(self.B, q), om, w)

    # ------------------------------------------------------------------- step
    def step(self, state: State, cmd) -> State:
        """Advance one ``dt``. ``cmd`` are rotor-speed commands in rad/s, shape ``(N, M)``.

        The command is held constant over the step (zero-order hold). If the motor lag is stiff
        relative to ``dt`` the step is automatically split into ``n_sub`` internal sub-steps.
        """
        B = self.B
        if self.world.actors:
            self.world.advance(float(state.t))  # moving obstacles are placed at the time the step starts
        cmd = B.asarray(cmd)
        if cmd.ndim == 1:
            cmd = cmd[None, :] * B.ones(self.n, 1)
        if self.n_sub == 1:
            return self._step_core(state, cmd, self.dt)
        h = self.dt / self.n_sub
        impact, collided = None, None
        for _ in range(self.n_sub):
            state = self._step_core(state, cmd, h)
            impact = state.impact_speed if impact is None else B.maximum(impact, state.impact_speed)
            collided = state.collided if collided is None else (collided | state.collided)
        state.impact_speed, state.collided = impact, collided
        return state

    @property
    def fast(self) -> bool:
        """True when the fused Numba kernel is in use."""
        return self.kernel.name == "numba"

    def _step_core(self, state: State, cmd, dt: float) -> State:
        return self.kernel.step(self, state, cmd, dt)

    def _step_reference(self, state: State, cmd, dt: float) -> State:
        B, P, spec = self.B, self.params, self.spec

        bat = spec.battery
        if bat.voltage_effects:
            ocv = bat.ocv(state.soc)
            r_i = bat.internal_resistance
            pwr = state.power if state.power is not None else B.zeros(self.n) + 10.0
            disc = B.maximum(ocv * ocv - 4.0 * r_i * pwr, 0.0 * ocv)
            v_t = 0.5 * (ocv + B.sqrt(disc))
            scale = (v_t / bat.v_full)[..., None]
            w_hi = P.w_max * scale
            depleted = (state.soc <= 0.0)[..., None]
            w_hi = B.where(depleted, w_hi * 0.0, w_hi)
        else:
            w_hi = P.w_max
        cmd = B.minimum(B.maximum(cmd, P.w_min * B.ones(1)), w_hi)
        cmd = B.where(P.rotor_health <= 0.0, 0.0 * cmd, cmd)

        wind = self.world.wind.sample(B, state.pos, state.vel, state.t, dt)
        y0 = (state.pos, state.vel, state.quat, state.omega, state.rotor)
        self._first = True
        y1 = self._integrate(lambda y: self._deriv(y, cmd, wind), y0, dt, self._project)
        aux = self._aux
        assert aux is not None
        pos, vel, q, om, w = y1

        # ground contact (inelastic with restitution + Coulomb friction)
        gz = self.world.ground_height(B, pos[..., 0], pos[..., 1]) + self.ground_offset
        below = pos[..., 2] <= gz
        vz = vel[..., 2]
        impact = B.where(below & (vz < 0.0), -vz, 0.0 * vz)
        new_vz = B.where(below & (vz < 0.0), -spec.restitution * vz, vz)
        pos = B.stack([pos[..., 0], pos[..., 1], B.where(below, gz, pos[..., 2])], -1)
        speed_xy = B.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2 + 1e-18)
        g0 = self.world.gravity.g
        fric = B.maximum(1.0 - spec.ground_friction * g0 * dt / speed_xy, 0.0 * speed_xy)
        fric = B.where(below, fric, 1.0 + 0.0 * fric)
        vel = B.stack([vel[..., 0] * fric, vel[..., 1] * fric, new_vz], -1)
        om = om * B.where(below, (0.0 * vz + np.exp(-self.contact_ang_damping * dt)), 1.0 + 0.0 * vz)[..., None]

        # obstacle collisions
        collided = B.zeros(self.n) > 1.0
        if self.world.has_obstacles:
            hit, depth, nrm = self.world.collide(B, pos, spec.collision_radius)
            vn = B.sum(vel * nrm, axis=-1, keepdims=True)
            pos = pos + nrm * (depth[..., None] * hit[..., None])
            into = hit[..., None] & (vn < 0.0)
            vel = B.where(into, vel - (1.0 + spec.restitution) * vn * nrm, vel)
            collided = hit

        # battery bookkeeping (explicit Euler on the step-start power)
        e_step = aux["power"] * dt
        energy = state.energy + e_step
        soc = B.maximum(state.soc - e_step / (bat.capacity_wh * 3600.0), 0.0 * state.soc)

        new = State(
            pos=pos,
            vel=vel,
            quat=q,
            omega=om,
            rotor=w,
            soc=soc,
            energy=energy,
            t=state.t + dt,
            acc=aux["acc"],
            f_body=aux["f_body"],
            thrust=aux["thrust"],
            power=aux["power"],
            wind=wind,
            contact=self._contact_mask(pos),
            collided=collided,
            impact_speed=impact,
        )
        return new

    def step_n(self, state: State, cmd, n: int) -> State:
        for _ in range(int(n)):
            state = self.step(state, cmd)
        return state

    # ------------------------------------------------------------- utilities
    def kinetic_energy(self, s: State):
        B, P = self.B, self.params
        trans = 0.5 * P.effective_mass()[..., 0] * B.sum(s.vel * s.vel, axis=-1)
        Jom = B.einsum("nij,nj->ni", P.inertia, s.omega)
        return trans + 0.5 * B.sum(s.omega * Jom, axis=-1)

    def angular_momentum_world(self, s: State):
        B, P = self.B, self.params
        Jom = B.einsum("nij,nj->ni", P.inertia, s.omega)
        return rot.rotate(B, s.quat, Jom)

    def mixer_pinv(self):
        """Pseudo-inverse of the allocation matrix (M x 4): ``w^2 = pinv @ [T, tx, ty, tz]``."""
        return np.linalg.pinv(self.spec.allocation_matrix())

    def info(self) -> dict:
        s = self.spec
        return dict(
            vehicle=s.name,
            mass=s.mass,
            rotors=s.n_rotors,
            backend=self.B.name,
            device=self.B.device,
            dt=self.dt,
            integrator=self.integrator_name,
            thrust_to_weight=s.thrust_to_weight(self.world.gravity.g),
            hover_rotor_speed=self.hover_speed(),
            hover_endurance_min=s.hover_endurance_min(self.world.gravity.g),
        )
