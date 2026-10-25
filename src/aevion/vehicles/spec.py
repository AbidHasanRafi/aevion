"""Multirotor vehicle specification, presets and batched parameters.

Units are SI throughout. Rotor speeds are in **rad/s**. Thrust and drag-torque
coefficients follow the usual quadratic law::

    T_i = kT * w_i^2        [N]
    Q_i = kQ * w_i^2        [N m]
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..core.constants import G0


@dataclass
class BatteryModel:
    """Equivalent-circuit LiPo pack: open-circuit voltage curve plus series resistance."""

    capacity_wh: float = 20.0
    cells: int = 3
    internal_resistance: float = 0.02  # pack series resistance [ohm]
    avionics_power: float = 3.0  # constant load [W]
    motor_efficiency: float = 0.75  # electrical -> mechanical (motor + ESC)
    voltage_effects: bool = False  # if True, available rotor speed scales with terminal voltage

    @property
    def v_full(self):
        return 4.2 * self.cells

    @property
    def v_empty(self):
        return 3.3 * self.cells

    def ocv(self, soc):
        """Open-circuit voltage of the pack vs state of charge in [0, 1]."""
        return self.cells * (3.3 + 0.9 * soc**0.8)


@dataclass
class VehicleSpec:
    name: str = "quad"
    mass: float = 1.0
    inertia: Sequence[float] = (0.01, 0.01, 0.02)  # principal moments [kg m^2] (diag), or 9 values for full 3x3
    rotor_positions: Any = ()  # (M, 3) body frame (FLU)
    rotor_spin: Any = ()  # +1 = CCW seen from above, -1 = CW
    kT: float = 1e-5
    kQ: float = 1.6e-7
    motor_tau: float = 0.02  # first-order lag [s]
    motor_tau_down: Optional[float] = None  # slower spin-down if set
    w_min: float = 0.0
    w_max: float = 1000.0
    rotor_inertia: float = 0.0  # J_r of motor+prop [kg m^2]
    rotor_radius: float = 0.12  # propeller radius [m] (ground effect / inflow)
    # aerodynamics (all optional; zero disables)
    drag_lin: Sequence[float] = (0.0, 0.0, 0.0)  # body-axis linear drag [N s/m]
    drag_quad: Sequence[float] = (0.0, 0.0, 0.0)  # body-axis quadratic drag [N s^2/m^2]
    rotor_drag: float = 0.0  # induced (rotor) drag [N s/m per rad/s ... see dynamics]
    angular_damping: Sequence[float] = (0.0, 0.0, 0.0)  # aerodynamic rate damping [N m s]
    inflow_gain: float = 0.0  # thrust loss with climb rate (dimensionless)
    ground_effect: bool = False
    # structure / contact
    collision_radius: float = 0.15
    restitution: float = 0.0
    ground_friction: float = 0.6
    battery: BatteryModel = field(default_factory=BatteryModel)
    description: str = ""
    reference: str = ""

    # ------------------------------------------------------------------ derived
    def __post_init__(self):
        self.rotor_positions = np.asarray(self.rotor_positions, dtype=float).reshape(-1, 3)
        self.rotor_spin = np.asarray(self.rotor_spin, dtype=float).reshape(-1)
        if len(self.rotor_positions) != len(self.rotor_spin):
            raise ValueError("rotor_positions and rotor_spin must have the same length")
        if len(self.rotor_spin) < 3:
            raise ValueError("a multirotor needs at least 3 rotors")
        inertia = np.asarray(self.inertia, dtype=float)
        if inertia.size == 3:
            inertia = np.diag(inertia)
        self.inertia_matrix = inertia.reshape(3, 3)
        if np.any(np.linalg.eigvalsh(self.inertia_matrix) <= 0):
            raise ValueError("inertia matrix must be positive definite")

    @property
    def n_rotors(self) -> int:
        return len(self.rotor_spin)

    def allocation_matrix(self) -> np.ndarray:
        """Matrix ``A`` (4 x M) with ``[T, tau_x, tau_y, tau_z]^T = A @ w^2``."""
        x, y = self.rotor_positions[:, 0], self.rotor_positions[:, 1]
        A = np.stack(
            [
                np.full(self.n_rotors, self.kT),
                y * self.kT,
                -x * self.kT,
                -self.rotor_spin * self.kQ,
            ]
        )
        return A

    def hover_rotor_speed(self, g: float = G0) -> float:
        return math.sqrt(self.mass * g / (self.n_rotors * self.kT))

    def max_thrust(self) -> float:
        return self.n_rotors * self.kT * self.w_max**2

    def thrust_to_weight(self, g: float = G0) -> float:
        return self.max_thrust() / (self.mass * g)

    def hover_power(self, g: float = G0) -> float:
        """Ideal electrical hover power [W] from the drag-torque model."""
        w = self.hover_rotor_speed(g)
        return self.n_rotors * self.kQ * w**3 / self.battery.motor_efficiency + self.battery.avionics_power

    def hover_endurance_min(self, g: float = G0) -> float:
        return 60.0 * self.battery.capacity_wh / self.hover_power(g)

    def validate(self, g: float = G0) -> list[str]:
        """Return a list of human-readable warnings about physical plausibility."""
        out = []
        if self.thrust_to_weight(g) < 1.2:
            out.append(f"thrust-to-weight {self.thrust_to_weight(g):.2f} < 1.2: cannot climb meaningfully")
        if self.hover_rotor_speed(g) > self.w_max:
            out.append("cannot hover: hover rotor speed exceeds w_max")
        if abs(float(np.sum(self.rotor_spin))) > 1e-9:
            out.append("net rotor spin is non-zero: yaw authority will be biased")
        A = self.allocation_matrix()
        if np.linalg.matrix_rank(A) < 4:
            out.append("allocation matrix has rank < 4: vehicle is not fully controllable in (T, roll, pitch, yaw)")
        return out

    def copy(self, **changes) -> "VehicleSpec":
        new = copy.deepcopy(self)
        for k, v in changes.items():
            if not hasattr(new, k):
                raise AttributeError(f"VehicleSpec has no field {k!r}")
            setattr(new, k, v)
        new.__post_init__()
        return new

    def to_dict(self) -> dict:
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, BatteryModel):
                v = dataclasses.asdict(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, tuple):
                v = list(v)
            d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleSpec":
        d = dict(d)
        if isinstance(d.get("battery"), dict):
            d["battery"] = BatteryModel(**d["battery"])
        return cls(**d)

    # ----------------------------------------------------------------- presets
    @classmethod
    def quad_x(cls, arm=0.2, **kw) -> "VehicleSpec":
        """Quadrotor in X configuration. ``arm`` is the motor-to-centre distance."""
        d = arm / math.sqrt(2.0)
        pos = [[d, -d, 0], [-d, d, 0], [d, d, 0], [-d, -d, 0]]
        spin = [1, 1, -1, -1]
        return cls(rotor_positions=pos, rotor_spin=spin, **kw)

    @classmethod
    def quad_plus(cls, arm=0.2, **kw) -> "VehicleSpec":
        pos = [[arm, 0, 0], [0, arm, 0], [-arm, 0, 0], [0, -arm, 0]]
        spin = [1, -1, 1, -1]
        return cls(rotor_positions=pos, rotor_spin=spin, **kw)

    @classmethod
    def hexa(cls, arm=0.3, **kw) -> "VehicleSpec":
        ang = np.deg2rad(np.arange(6) * 60.0)
        pos = np.stack([arm * np.cos(ang), arm * np.sin(ang), np.zeros(6)], 1)
        spin = [1, -1, 1, -1, 1, -1]
        return cls(rotor_positions=pos, rotor_spin=spin, **kw)

    @classmethod
    def octo(cls, arm=0.4, **kw) -> "VehicleSpec":
        ang = np.deg2rad(np.arange(8) * 45.0 + 22.5)
        pos = np.stack([arm * np.cos(ang), arm * np.sin(ang), np.zeros(8)], 1)
        spin = [1, -1, 1, -1, 1, -1, 1, -1]
        return cls(rotor_positions=pos, rotor_spin=spin, **kw)

    @classmethod
    def from_components(
        cls,
        propeller: Any = "10x4.5",
        motor: Any = None,
        battery_cells: int = 3,
        battery_capacity_wh: float = 44.4,
        mass: float = 1.0,
        arm: float = 0.2,
        layout: str = "quad_x",
        esc: Any = None,
        inertia: Optional[Sequence[float]] = None,
        name: str = "",
        **kw: Any,
    ) -> "VehicleSpec":
        """A vehicle whose rotor model is *derived from its components*: propeller geometry (blade-element momentum theory), a
        BLDC motor and ESC on a battery of ``battery_cells``. The thrust and torque coefficients, the maximum rotor speed on the
        battery voltage, the motor time constant, the inflow gain and the motor efficiency come from those parts instead of
        being typed in. ``propeller`` is a designation such as ``"10x4.5"`` or a
        :class:`~aevion.vehicles.propeller.Propeller`; ``motor`` a :class:`~aevion.vehicles.motor.BLDCMotor` (default: a 920 kv
        motor sized for 10-inch propellers). Extra keyword arguments override fields of the specification."""
        from .motor import BLDCMotor, MotorPropSystem
        from .propeller import Propeller, get_propeller

        prop = propeller if isinstance(propeller, Propeller) else get_propeller(str(propeller))
        mot = (
            motor
            if motor is not None
            else BLDCMotor(kv=920.0, resistance=0.09, i0=0.6, max_current=40.0, rotor_inertia=1.5e-5, name="generic 920 kv")
        )
        sys_ = MotorPropSystem(mot, prop, esc)
        n_rot = {"quad_x": 4, "quad_plus": 4, "hexa": 6, "octo": 8}[layout]
        v_bus = 3.7 * battery_cells
        p = sys_.spec_parameters(mass, n_rot, v_bus)
        J = np.asarray(inertia if inertia is not None else (0.26 * mass * arm**2, 0.26 * mass * arm**2, 0.50 * mass * arm**2), float)
        bat = BatteryModel(capacity_wh=battery_capacity_wh, cells=battery_cells, motor_efficiency=min(max(p["motor_efficiency"], 0.3), 0.95))
        fields = dict(
            name=name or f"{layout}_{prop.name or 'custom'}",
            mass=mass,
            inertia=tuple(J),
            kT=p["kT"],
            kQ=p["kQ"],
            motor_tau=p["motor_tau"],
            w_max=p["w_max"],
            rotor_inertia=p["rotor_inertia"],
            rotor_radius=p["rotor_radius"],
            inflow_gain=p["inflow_gain"],
            ground_effect=True,
            battery=bat,
            description=f"derived from a {prop.name or 'custom'} propeller (BEMT) and {mot.name or 'a BLDC motor'}",
        )
        fields.update(kw)
        return getattr(cls, layout)(arm=arm, **fields)

    @classmethod
    def crazyflie(cls) -> "VehicleSpec":
        """Bitcraze Crazyflie 2.x (27 g). Parameters approximate those of Förster (2015)
        and Landry (2015), converted from RPM^2 to (rad/s)^2."""
        rpm2 = (60.0 / (2 * math.pi)) ** 2
        return cls.quad_x(
            arm=0.046,
            name="crazyflie",
            mass=0.027,
            inertia=(1.4e-5, 1.4e-5, 2.17e-5),
            kT=3.16e-10 * rpm2,
            kQ=7.94e-12 * rpm2,
            motor_tau=0.03,
            w_max=2300.0,
            rotor_inertia=2.0e-8,
            rotor_radius=0.023,
            drag_lin=(3.0e-5, 3.0e-5, 3.0e-5),
            collision_radius=0.06,
            battery=BatteryModel(capacity_wh=0.925, cells=1, internal_resistance=0.1, avionics_power=0.6, motor_efficiency=0.6),
            description="27 g nano quadrotor",
            reference="Förster, 'System identification of the Crazyflie 2.0 nano quadrocopter', 2015",
        )

    @classmethod
    def racing_5inch(cls) -> "VehicleSpec":
        """250 mm class FPV racing quad (≈0.6 kg, thrust-to-weight ≈ 6)."""
        return cls.quad_x(
            arm=0.11,
            name="racing_5inch",
            mass=0.6,
            inertia=(2.6e-3, 2.8e-3, 4.5e-3),
            kT=1.53e-6,
            kQ=1.8e-8,
            motor_tau=0.015,
            w_max=2400.0,
            rotor_inertia=1.5e-6,
            rotor_radius=0.064,
            drag_lin=(0.02, 0.02, 0.03),
            drag_quad=(0.01, 0.01, 0.02),
            collision_radius=0.12,
            battery=BatteryModel(capacity_wh=22.0, cells=4, internal_resistance=0.025, avionics_power=4.0),
            description="5-inch freestyle/racing quadrotor",
        )

    @classmethod
    def generic_1kg(cls) -> "VehicleSpec":
        """Generic 1 kg research quadrotor, T/W≈2.2. Good default for control studies."""
        return cls.quad_x(
            arm=0.20,
            name="generic_1kg",
            mass=1.0,
            inertia=(0.0105, 0.0105, 0.0198),
            kT=1.0e-5,
            kQ=1.6e-7,
            motor_tau=0.025,
            w_max=750.0,
            rotor_inertia=5e-5,
            rotor_radius=0.127,
            drag_lin=(0.10, 0.10, 0.15),
            drag_quad=(0.02, 0.02, 0.04),
            rotor_drag=1e-4,
            angular_damping=(1e-3, 1e-3, 2e-3),
            inflow_gain=0.0,
            ground_effect=True,
            collision_radius=0.25,
            battery=BatteryModel(capacity_wh=44.4, cells=3, internal_resistance=0.03, avionics_power=5.0),
            description="1 kg quadrotor, 10-inch props",
        )

    @classmethod
    def dji_like_2kg(cls) -> "VehicleSpec":
        """≈2 kg camera drone (illustrative parameters, not an exact DJI model)."""
        return cls.quad_x(
            arm=0.25,
            name="camera_2kg",
            mass=2.0,
            inertia=(0.021, 0.021, 0.038),
            kT=1.8e-5,
            kQ=2.4e-7,
            motor_tau=0.03,
            w_max=700.0,
            rotor_inertia=1.0e-4,
            rotor_radius=0.12,
            drag_lin=(0.2, 0.2, 0.3),
            drag_quad=(0.04, 0.04, 0.08),
            rotor_drag=2e-4,
            angular_damping=(2e-3, 2e-3, 4e-3),
            ground_effect=True,
            collision_radius=0.35,
            battery=BatteryModel(capacity_wh=89.0, cells=4, internal_resistance=0.04, avionics_power=8.0),
            description="2 kg camera quadrotor",
        )

    @classmethod
    def hex_3kg(cls) -> "VehicleSpec":
        return cls.hexa(
            arm=0.32,
            name="hex_3kg",
            mass=3.0,
            inertia=(0.045, 0.045, 0.080),
            kT=1.6e-5,
            kQ=2.2e-7,
            motor_tau=0.03,
            w_max=800.0,
            rotor_inertia=1.2e-4,
            rotor_radius=0.15,
            drag_lin=(0.25, 0.25, 0.35),
            drag_quad=(0.05, 0.05, 0.1),
            ground_effect=True,
            collision_radius=0.45,
            battery=BatteryModel(capacity_wh=133.0, cells=6, internal_resistance=0.05, avionics_power=10.0),
            description="3 kg hexacopter (single-rotor-failure tolerant)",
        )


PRESETS = {
    "crazyflie": VehicleSpec.crazyflie,
    "racing_5inch": VehicleSpec.racing_5inch,
    "generic_1kg": VehicleSpec.generic_1kg,
    "camera_2kg": VehicleSpec.dji_like_2kg,
    "hex_3kg": VehicleSpec.hex_3kg,
}


def get_vehicle(name_or_spec) -> VehicleSpec:
    if isinstance(name_or_spec, VehicleSpec):
        return name_or_spec
    try:
        return PRESETS[name_or_spec]()
    except KeyError:
        raise ValueError(f"unknown vehicle {name_or_spec!r}; presets: {sorted(PRESETS)}") from None


class BatchParams:
    """Per-environment parameters as backend tensors (supports domain randomization).

    Anything randomisable has a leading batch dimension ``N``. Use
    :meth:`randomize` to multiply selected parameters by per-env random factors.
    """

    def __init__(self, B, spec: VehicleSpec, n: int):
        self.B, self.spec, self.n = B, spec, int(n)
        M = spec.n_rotors
        f = B.asarray
        ones = lambda *s: B.ones(*s)  # noqa: E731
        self.mass = ones(n, 1) * spec.mass
        J = f(spec.inertia_matrix)
        self.inertia = ones(n, 3, 3) * J
        self.inertia_inv = ones(n, 3, 3) * f(np.linalg.inv(spec.inertia_matrix))
        self.kT = ones(n, M) * spec.kT
        self.kQ = ones(n, M) * spec.kQ
        self.tau_up = ones(n, M) * spec.motor_tau
        self.tau_down = ones(n, M) * (spec.motor_tau_down if spec.motor_tau_down else spec.motor_tau)
        self.w_min = spec.w_min
        self.w_max = ones(n, M) * spec.w_max
        self.rotor_health = ones(n, M)  # 1 = healthy, 0 = failed, in between = degraded
        self.drag_lin = ones(n, 3) * f(spec.drag_lin)
        self.drag_quad = ones(n, 3) * f(spec.drag_quad)
        self.rotor_drag = ones(n, 1) * spec.rotor_drag
        self.ang_damp = ones(n, 3) * f(spec.angular_damping)
        self.rotor_pos = f(spec.rotor_positions)  # (M,3), shared
        self.rotor_spin = f(spec.rotor_spin)  # (M,)
        self.J_r = spec.rotor_inertia
        self.com_offset = B.zeros(n, 3)  # centre-of-mass offset from geometric origin [m]
        self.payload_mass = B.zeros(n, 1)
        # external loads, set every step by effect hooks: world force (3), body torque (3), thrust multiplier (1)
        self.ext = B.zeros(n, 7)
        self.clear_external()

    def clear_external(self):
        """No external force or torque, thrust multiplier 1."""
        self.ext[:] = 0.0
        self.ext[:, 6] = 1.0

    def randomize(self, gen, **ranges):
        """Multiply parameters by ``U(1-r, 1+r)`` (relative) per environment.

        Example: ``params.randomize(gen, mass=0.1, kT=0.05, tau=0.2)``.
        """
        B = self.B
        n = self.n

        def fac(r, shape):
            return 1.0 + r * (2.0 * B.rand(shape, gen) - 1.0)

        if "mass" in ranges:
            self.mass = self.mass * fac(ranges["mass"], (n, 1))
        if "inertia" in ranges:
            k = fac(ranges["inertia"], (n, 1, 1))
            self.inertia = self.inertia * k
            self.inertia_inv = self.inertia_inv / k
        if "kT" in ranges:
            self.kT = self.kT * fac(ranges["kT"], (n, 1)) * fac(ranges.get("kT_individual", 0.0), self.kT.shape)
        if "kQ" in ranges:
            self.kQ = self.kQ * fac(ranges["kQ"], (n, 1))
        if "tau" in ranges:
            k = fac(ranges["tau"], (n, 1))
            self.tau_up = self.tau_up * k
            self.tau_down = self.tau_down * k
        if "drag" in ranges:
            k = fac(ranges["drag"], (n, 1))
            self.drag_lin = self.drag_lin * k
            self.drag_quad = self.drag_quad * k
        if "w_max" in ranges:
            self.w_max = self.w_max * fac(ranges["w_max"], (n, 1))
        if "com" in ranges:  # absolute offset in metres, uniform in +-r
            self.com_offset = ranges["com"] * (2.0 * B.rand((n, 3), gen) - 1.0)
        return self

    def set_rotor_failure(self, env_idx, rotor_idx, severity: float = 1.0):
        """Degrade rotor ``rotor_idx`` in environment(s) ``env_idx`` (1.0 = total loss)."""
        self.rotor_health[env_idx, rotor_idx] = 1.0 - severity

    def effective_mass(self):
        return self.mass + self.payload_mass
