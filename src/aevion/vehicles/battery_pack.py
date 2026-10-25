"""A lithium-polymer battery pack with Peukert capacity, a first-order polarisation circuit, heating and ageing.

The standard equivalent circuit (Plett, *Battery Management Systems*, 2015), extended with the effects that matter for flight-time studies:

* **Open-circuit voltage** ``OCV(soc)`` of ``cells`` in series.
* **Series resistance** ``R0`` and one **RC pair** ``(R1, C1)`` for slow voltage relaxation: ``V = OCV - I R0(T) - V1`` with
  ``dV1/dt = (I R1 - V1) / (R1 C1)``.
* **Peukert's law**: above the rated current ``I_r = Q_nom / H`` the delivered capacity shrinks. Depletion runs at
  ``I (I / I_r)^(k-1) / Q``, so a constant-current discharge lasts ``t = H (Q_nom / (I H))^k`` hours.
* **Temperature**: a lumped thermal mass heated by ``I² R0 + V1² / R1`` and cooled to ambient through ``hA``; a cold pack has higher
  resistance and less usable capacity.
* **Ageing**: capacity fades with the square root of equivalent full cycles, and resistance grows with the fade.

Power demand is converted to current by solving ``P = V I`` for the smaller root; a demand the pack cannot supply (negative
discriminant) is reported as a brown-out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["PackSpec", "BatteryPack"]


@dataclass
class PackSpec:
    cells: int = 3
    capacity_ah: float = 4.0  # at the rated discharge time
    r0: float = 0.025  # series resistance of the whole pack at 25 C [ohm]
    r1: float = 0.012
    c1: float = 1200.0  # polarisation time constant r1 c1 = 14 s
    peukert_k: float = 1.04
    rated_hours: float = 1.0
    mass_kg: float = 0.30
    specific_heat: float = 1000.0  # J / (kg K)
    h_a: float = 0.45  # convective conductance to the air [W / K]
    r_temp_coeff: float = 0.02  # R0(T) = R0 exp(coeff (25 - T))
    cap_temp_coeff: float = 0.006  # usable capacity loses this fraction per K below 25 C
    cycle_fade: float = 0.010  # capacity retention = 1 - cycle_fade sqrt(EFC)
    resistance_growth: float = 2.5  # R0 grows by this factor times the capacity loss
    v_cell_min: float = 3.0  # cutoff per cell under load

    def ocv(self, soc: Any) -> Any:
        """Open-circuit voltage of the pack: a LiPo-like curve, ``cells * (3.3 + 0.9 soc^0.8)`` (flat plateau, steep ends)."""
        s = np.clip(soc, 0.0, 1.0)
        return self.cells * (3.3 + 0.9 * s**0.8)

    @property
    def rated_current(self) -> float:
        return self.capacity_ah / self.rated_hours

    @property
    def energy_wh(self) -> float:
        return self.capacity_ah * self.cells * 3.7


class BatteryPack:
    """``n`` independent packs (one per vehicle), advanced together."""

    def __init__(self, spec: Optional[PackSpec] = None, n: int = 1, soc: float = 1.0, temperature_c: float = 25.0) -> None:
        self.spec = spec or PackSpec()
        self.n = int(n)
        self.reset(soc, temperature_c)

    def reset(self, soc: Any = 1.0, temperature_c: float = 25.0) -> None:
        n = self.n
        self.soc = np.full(n, 1.0) * np.asarray(soc, float)
        self.temperature = np.full(n, float(temperature_c))
        self.v1 = np.zeros(n)
        self.efc = np.zeros(n)  # equivalent full cycles
        self.voltage = self.spec.ocv(self.soc)
        self.current = np.zeros(n)
        self.brownout = np.zeros(n, bool)
        self.heat_w = np.zeros(n)

    # ------------------------------------------------------------------------------------------ state-dependent parameters
    @property
    def retention(self) -> np.ndarray:
        """Remaining capacity fraction from ageing (state of health)."""
        return np.maximum(1.0 - self.spec.cycle_fade * np.sqrt(self.efc), 0.1)

    def resistance(self) -> np.ndarray:
        """Series resistance at the present temperature and age [ohm]."""
        s = self.spec
        age = 1.0 + s.resistance_growth * (1.0 - self.retention)
        return s.r0 * age * np.exp(s.r_temp_coeff * (25.0 - self.temperature))

    def usable_capacity_ah(self) -> np.ndarray:
        s = self.spec
        cold = np.clip(1.0 - s.cap_temp_coeff * np.maximum(25.0 - self.temperature, 0.0), 0.3, 1.0)
        return s.capacity_ah * self.retention * cold

    # ------------------------------------------------------------------------------------------ stepping
    def current_for_power(self, power_w: Any) -> np.ndarray:
        """Discharge current that delivers ``power_w`` at the terminals; sets the brown-out flags."""
        P = np.asarray(power_w, float)
        v0 = self.spec.ocv(self.soc) - self.v1
        R = self.resistance()
        disc = v0 * v0 - 4.0 * R * P
        self.brownout = disc < 0.0
        i = (v0 - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * R)
        return np.where(self.brownout, v0 / (2.0 * R), i)

    def step(self, dt: float, power_w: Any = None, current_a: Any = None, ambient_c: float = 25.0) -> Dict[str, np.ndarray]:
        """Advance ``dt`` seconds with either a power demand [W] or a current [A] (positive = discharge)."""
        s = self.spec
        if (power_w is None) == (current_a is None):
            raise ValueError("give exactly one of power_w and current_a")
        i = self.current_for_power(power_w) if power_w is not None else np.asarray(current_a, float) * np.ones(self.n)
        R = self.resistance()
        # Peukert: the capacity drained per ampere-second grows with the current
        ratio = np.maximum(np.abs(i), 1e-9) / s.rated_current
        i_eff = i * np.where(i > 0, ratio ** (s.peukert_k - 1.0), 1.0)
        self.soc = np.clip(self.soc - i_eff * dt / (3600.0 * self.usable_capacity_ah()), 0.0, 1.0)
        # polarisation voltage (exact exponential update for a constant current over the step)
        r1 = s.r1 * np.exp(s.r_temp_coeff * (25.0 - self.temperature))
        tau = r1 * s.c1
        self.v1 = i * r1 + (self.v1 - i * r1) * np.exp(-dt / tau)
        # heat and temperature
        self.heat_w = i * i * R + self.v1**2 / r1
        c_th = s.mass_kg * s.specific_heat
        self.temperature = self.temperature + dt * (self.heat_w - s.h_a * (self.temperature - ambient_c)) / c_th
        self.efc = self.efc + np.abs(i) * dt / (2.0 * 3600.0 * s.capacity_ah)
        self.current = i
        self.voltage = s.ocv(self.soc) - i * R - self.v1
        return dict(voltage=self.voltage, current=i, soc=self.soc, temperature=self.temperature, soh=self.retention, brownout=self.brownout)

    def empty(self) -> np.ndarray:
        """Under the cutoff voltage (per cell) or out of charge."""
        return (self.voltage < self.spec.v_cell_min * self.spec.cells) | (self.soc <= 0.0)

    # ------------------------------------------------------------------------------------------ analytic references
    def peukert_runtime_hours(self, current_a: float) -> float:
        """Runtime of a constant-current discharge to empty from full, ``H (Q / (I H))^k`` (Peukert 1897)."""
        s = self.spec
        return s.rated_hours * (s.capacity_ah / (current_a * s.rated_hours)) ** s.peukert_k

    def thermal_steady_state(self, current_a: float, ambient_c: float = 25.0) -> float:
        """Steady temperature of a constant current: ``T_amb + I² (R0 + R1) / (hA)`` (the polarisation resistor dissipates
        ``V1² / R1 = I² R1`` once settled; resistances at that temperature, solved iteratively), for a fresh pack."""
        s = self.spec
        t = ambient_c
        for _ in range(80):
            r = (s.r0 + s.r1) * math.exp(s.r_temp_coeff * (25.0 - t))
            t = ambient_c + (current_a**2 * r) / s.h_a
        return t
