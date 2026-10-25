"""Brushless motor, ESC and the motor-propeller system.

A permanent-magnet DC-equivalent BLDC model (the standard small-UAV model, e.g. Gong et al. 2018)::

    V = I R + K_e ω          back-EMF constant K_e = 1 / K_v  [V s/rad]
    Q_motor = K_t (I - I₀)   torque constant K_t = K_e (SI units), no-load current I₀
    P_elec = V I,   P_mech = Q ω,   η = P_mech / P_elec

A motor driven against a propeller load ``Q_load(ω)`` settles where ``K_t (V - K_e ω) / R - K_t I₀ = Q_load(ω)``. For the usual
quadratic load ``Q_load = k_q ω²`` this is a quadratic in ``ω`` with a closed-form root, which the tests check. The ESC is an ideal
buck converter with an efficiency: the duty cycle scales the bus voltage and the bus current is ``I_bus = duty * I / η_esc``, so the
battery sees electrical rather than mechanical power.

:class:`MotorPropSystem` joins a motor, an ESC and a :class:`~aevion.vehicles.propeller.Propeller`.
:meth:`MotorPropSystem.spec_parameters` returns what a simulation needs: thrust and torque coefficients, maximum rotor speed on a given
battery voltage, motor time constant and electrical efficiency at hover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .propeller import RHO_SL, Propeller, rotor_parameters

__all__ = ["BLDCMotor", "ESC", "MotorPropSystem", "MotorPoint"]


@dataclass
class MotorPoint:
    """A steady operating point of the motor at one bus voltage and duty cycle."""

    omega: float  # rotor speed [rad/s]
    current: float  # motor phase current [A]
    bus_current: float  # battery current [A]
    p_electrical: float  # drawn from the battery [W]
    p_mechanical: float  # shaft power [W]
    torque: float  # shaft torque [N m]
    efficiency: float  # shaft power / battery power (motor and ESC)
    saturated: bool = False  # the current limit is active


@dataclass
class BLDCMotor:
    """Brushless motor: ``kv`` [rpm/V], winding resistance ``resistance`` [ohm], no-load current ``i0`` [A]."""

    kv: float
    resistance: float
    i0: float = 0.5
    max_current: float = 40.0
    rotor_inertia: float = 1.0e-5  # motor rotor alone, plus the propeller's when combined
    name: str = ""

    def __post_init__(self) -> None:
        if self.kv <= 0 or self.resistance <= 0 or self.i0 < 0 or self.max_current <= 0:
            raise ValueError("kv, resistance and max_current must be positive and i0 non-negative")

    @property
    def ke(self) -> float:
        """Back-EMF (= torque) constant [V s/rad = N m/A]."""
        return 60.0 / (2.0 * math.pi * self.kv)

    kt = ke

    def current(self, voltage: float, omega: float) -> float:
        return (voltage - self.ke * omega) / self.resistance

    def torque(self, voltage: float, omega: float) -> float:
        return self.ke * (self.current(voltage, omega) - self.i0)

    def no_load_speed(self, voltage: float) -> float:
        """Speed at which the motor produces no shaft torque (``I = I₀``) [rad/s]."""
        return (voltage - self.i0 * self.resistance) / self.ke

    def stall_current(self, voltage: float) -> float:
        return voltage / self.resistance

    def max_efficiency(self, voltage: float) -> float:
        """The classic ``η_max = (1 - sqrt(I₀ R / V))²``, reached at ``I = sqrt(I₀ V / R)``."""
        return (1.0 - math.sqrt(self.i0 * self.resistance / voltage)) ** 2

    def max_efficiency_current(self, voltage: float) -> float:
        return math.sqrt(self.i0 * voltage / self.resistance)

    def operating_point(self, v_bus: float, duty: float, load: Callable[[float], float], esc_efficiency: float = 0.97) -> MotorPoint:
        """Steady state against a load torque ``load(ω)`` (any increasing function), found by bisection on the speed."""
        v = float(np.clip(duty, 0.0, 1.0)) * v_bus
        if v <= self.i0 * self.resistance:
            return MotorPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        lo, hi = 0.0, self.no_load_speed(v)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.torque(v, mid) - load(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        w = 0.5 * (lo + hi)
        i = self.current(v, w)
        sat = i > self.max_current
        if sat:  # the ESC limits the current: the speed where the torque at I_max balances the load
            q_lim = self.ke * (self.max_current - self.i0)
            lo, hi = 0.0, w
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if load(mid) < q_lim:
                    lo = mid
                else:
                    hi = mid
            w = 0.5 * (lo + hi)
            i = self.max_current
            v = self.ke * w + i * self.resistance  # the voltage the converter actually delivers
        q = load(w)
        p_mech = q * w
        p_motor_in = v * i
        p_bus = p_motor_in / esc_efficiency
        return MotorPoint(w, i, p_bus / v_bus if v_bus > 0 else 0.0, p_bus, p_mech, q, p_mech / p_bus if p_bus > 1e-12 else 0.0, sat)

    def time_constant(self, v_bus: float, duty: float, load_coefficient: float, total_inertia: Optional[float] = None) -> float:
        """Mechanical time constant of the spin-up, linearised at the steady point for a load ``k_q ω²``:
        ``τ = J / (K_t K_e / R + 2 k_q ω)``."""
        pt = self.operating_point(v_bus, duty, lambda w: load_coefficient * w * w)
        j = total_inertia if total_inertia is not None else self.rotor_inertia
        return j / (self.ke * self.ke / self.resistance + 2.0 * load_coefficient * pt.omega)


@dataclass
class ESC:
    """Electronic speed controller: efficiency, throttle curve ``duty = min_duty + (1 - min_duty) throttle^expo``."""

    efficiency: float = 0.97
    min_duty: float = 0.0
    expo: float = 1.0

    def duty(self, throttle: float) -> float:
        t = float(np.clip(throttle, 0.0, 1.0))
        return 0.0 if t <= 0.0 else self.min_duty + (1.0 - self.min_duty) * t**self.expo

    def throttle(self, duty: float) -> float:
        d = float(np.clip(duty, 0.0, 1.0))
        if d <= self.min_duty:
            return 0.0
        return ((d - self.min_duty) / (1.0 - self.min_duty)) ** (1.0 / self.expo)


class MotorPropSystem:
    """A motor driving a propeller through an ESC, powered from a battery of nominal voltage ``v_bus``."""

    def __init__(self, motor: BLDCMotor, prop: Propeller, esc: Optional[ESC] = None, rho: float = RHO_SL) -> None:
        self.motor, self.prop, self.esc, self.rho = motor, prop, esc or ESC(), rho
        self.params = rotor_parameters(prop, rho=rho)
        self.kT, self.kQ = self.params["kT"], self.params["kQ"]
        self.inertia = motor.rotor_inertia + self.params["rotor_inertia"]

    def _load(self, w: float) -> float:
        return self.kQ * w * w

    def at_duty(self, v_bus: float, duty: float) -> MotorPoint:
        return self.motor.operating_point(v_bus, duty, self._load, self.esc.efficiency)

    def at_throttle(self, v_bus: float, throttle: float) -> MotorPoint:
        return self.at_duty(v_bus, self.esc.duty(throttle))

    def thrust(self, omega: float) -> float:
        return self.kT * omega * omega

    def max_speed(self, v_bus: float) -> float:
        return self.at_duty(v_bus, 1.0).omega

    def duty_for_speed(self, v_bus: float, omega: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.at_duty(v_bus, mid).omega < omega:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def hover(self, mass: float, n_rotors: int, v_bus: float, g: float = 9.80665) -> Dict[str, float]:
        """Operating point that lifts ``mass``: rotor speed, duty, current, electrical power and efficiency per rotor."""
        w = math.sqrt(mass * g / (n_rotors * self.kT))
        duty = self.duty_for_speed(v_bus, w)
        pt = self.at_duty(v_bus, duty)
        return dict(
            omega=pt.omega,
            rpm=pt.omega * 60.0 / (2.0 * math.pi),
            duty=duty,
            throttle=self.esc.throttle(duty),
            current=pt.bus_current,
            p_electrical=pt.p_electrical,
            efficiency=pt.efficiency,
            total_power=n_rotors * pt.p_electrical,
            thrust_to_weight=n_rotors * self.thrust(self.max_speed(v_bus)) / (mass * g),
        )

    def spec_parameters(self, mass: float, n_rotors: int, v_bus: float, g: float = 9.80665) -> Dict[str, float]:
        """Coefficients for :class:`~aevion.vehicles.spec.VehicleSpec`: ``kT, kQ, w_max, motor_tau, rotor_inertia, rotor_radius,
        inflow_gain`` and the motor efficiency at hover (``motor_efficiency``)."""
        h = self.hover(mass, n_rotors, v_bus, g)
        tau = self.motor.time_constant(v_bus, h["duty"], self.kQ, self.inertia)
        return dict(
            kT=self.kT,
            kQ=self.kQ,
            w_max=self.max_speed(v_bus),
            motor_tau=tau,
            rotor_inertia=self.inertia,
            rotor_radius=self.prop.radius,
            inflow_gain=self.params["inflow_gain"],
            motor_efficiency=h["efficiency"],
        )
