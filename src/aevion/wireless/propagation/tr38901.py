"""3GPP TR 38.901 (v16) large-scale path-loss models for terrestrial links: UMa, UMi-street-canyon, RMa and InH-mixed-office.

Implemented from Table 7.4.1-1 (path loss, shadow-fading spread) and Table 7.4.2-1 (LoS probability). Conventions as in
the standard: ``fc`` in GHz, distances in metres, ``d_BP' = 4 h'_BS h'_UT f_c / c`` with the effective environment height
``h_E = 1 m`` (the deterministic branch of the standard's random ``h_E`` for UMa/UMi), and the NLoS path loss is
``max(PL_LoS, PL'_NLoS)``.

The coefficients are transcribed from the published tables; they are cross-checked by hand-computed reference values in
``tests/test_wireless_propagation.py`` but have not been compared against a third-party implementation, so verify them
against the standard before quoting absolute numbers in a paper.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from ..units import C_LIGHT
from .base import LinkGeometry, PathLossModel, _clip_d, _log10, breakpoint_distance

SCENARIOS = ("uma", "umi", "rma", "inh")
DEFAULT_H_BS = {"uma": 25.0, "umi": 10.0, "rma": 35.0, "inh": 3.0}
_D2D_RANGE = {"uma": (10.0, 5000.0), "umi": (10.0, 5000.0), "rma": (10.0, 10000.0), "inh": (1.0, 150.0)}
_FC_RANGE = {"uma": (0.5, 100.0), "umi": (0.5, 100.0), "rma": (0.5, 30.0), "inh": (0.5, 100.0)}
# terminal heights for which the formulas are specified (UAVs above these need TR 36.777)
_H_UT_RANGE = {"uma": (1.5, 22.5), "umi": (1.5, 22.5), "rma": (1.0, 10.0), "inh": (1.0, 2.5)}


def _los_prob_uma(d2d: np.ndarray, h_ut: np.ndarray) -> np.ndarray:
    d = np.maximum(d2d, 1e-6)
    c_prime = np.where(h_ut <= 13.0, 0.0, np.where(h_ut <= 23.0, np.maximum((h_ut - 13.0) / 10.0, 0.0) ** 1.5, 1.0))
    p = (18.0 / d + np.exp(-d / 63.0) * (1.0 - 18.0 / d)) * (1.0 + c_prime * 1.25 * (d / 100.0) ** 3 * np.exp(-d / 150.0))
    return np.where(d <= 18.0, 1.0, np.clip(p, 0.0, 1.0))


def _los_prob_umi(d2d: np.ndarray) -> np.ndarray:
    d = np.maximum(d2d, 1e-6)
    return np.where(d <= 18.0, 1.0, 18.0 / d + np.exp(-d / 36.0) * (1.0 - 18.0 / d))


def _los_prob_rma(d2d: np.ndarray) -> np.ndarray:
    return np.where(d2d <= 10.0, 1.0, np.exp(-(d2d - 10.0) / 1000.0))


def _los_prob_inh_mixed(d2d: np.ndarray) -> np.ndarray:
    d = np.maximum(d2d, 1e-6)
    return np.where(d <= 1.2, 1.0, np.where(d < 6.5, np.exp(-(d - 1.2) / 4.7), np.exp(-(d - 6.5) / 32.6) * 0.32))


def _rma_los_pl1(d3d: np.ndarray, fc: float, h: float) -> np.ndarray:
    return (
        20.0 * _log10(40.0 * math.pi * d3d * fc / 3.0)
        + np.minimum(0.03 * h**1.72, 10.0) * _log10(d3d)
        - np.minimum(0.044 * h**1.72, 14.77)
        + 0.002 * math.log10(h) * d3d
    )


class TR38901(PathLossModel):
    """TR 38.901 path loss. ``scenario`` is one of ``uma``, ``umi``, ``rma``, ``inh``.

    Parameters
    ----------
    h_building, street_width_m
        RMa only (average building height and street width; defaults 5 m and 20 m as in the standard).
    """

    def __init__(self, freq_hz: float = 3.5e9, scenario: str = "uma", h_building: float = 5.0, street_width_m: float = 20.0):
        super().__init__(freq_hz)
        scenario = scenario.lower()
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.scenario = scenario
        self.h_building, self.W = float(h_building), float(street_width_m)
        self.name = f"tr38901_{scenario}"
        self.d2d_range_m = _D2D_RANGE[scenario]
        self.freq_range_ghz = _FC_RANGE[scenario]
        self.h_ut_range_m = _H_UT_RANGE[scenario]

    # ------------------------------------------------------------------------------------ path loss
    def _los_pl(self, g: LinkGeometry) -> np.ndarray:
        fc, d3, d2 = self.fc_ghz, _clip_d(g.d3d), g.d2d
        s = self.scenario
        if s in ("uma", "umi"):
            dbp = breakpoint_distance(g.h_bs, g.h_ut, self.freq_hz)
            a, b, c = (28.0, 22.0, 9.0) if s == "uma" else (32.4, 21.0, 9.5)
            pl1 = a + b * _log10(d3) + 20.0 * _log10(fc)
            pl2 = a + 40.0 * _log10(d3) + 20.0 * _log10(fc) - c * _log10(dbp**2 + (g.h_bs - g.h_ut) ** 2)
            return np.where(d2 <= dbp, pl1, pl2)
        if s == "rma":
            dbp = 2.0 * math.pi * np.maximum(g.h_bs, 0.1) * np.maximum(g.h_ut, 0.1) * self.freq_hz / C_LIGHT
            pl1 = _rma_los_pl1(d3, fc, self.h_building)
            d_bp3 = np.hypot(dbp, g.h_bs - g.h_ut)
            pl_bp = _rma_los_pl1(d_bp3, fc, self.h_building)
            pl2 = pl_bp + 40.0 * _log10(d3 / d_bp3)
            return np.where(d2 <= dbp, pl1, pl2)
        return 32.4 + 17.3 * _log10(d3) + 20.0 * _log10(fc)  # inh

    def _nlos_prime(self, g: LinkGeometry) -> np.ndarray:
        fc, d3 = self.fc_ghz, _clip_d(g.d3d)
        s = self.scenario
        if s == "uma":
            return 13.54 + 39.08 * _log10(d3) + 20.0 * _log10(fc) - 0.6 * (g.h_ut - 1.5)
        if s == "umi":
            return 35.3 * _log10(d3) + 22.4 + 21.3 * _log10(fc) - 0.3 * (g.h_ut - 1.5)
        if s == "rma":
            h, W, hb = self.h_building, self.W, np.maximum(g.h_bs, 1.0)
            return (
                161.04
                - 7.1 * math.log10(W)
                + 7.5 * math.log10(h)
                - (24.37 - 3.7 * (h / hb) ** 2) * _log10(hb)
                + (43.42 - 3.1 * _log10(hb)) * (_log10(d3) - 3.0)
                + 20.0 * _log10(fc)
                - (3.2 * _log10(11.75 * np.maximum(g.h_ut, 0.1)) ** 2 - 4.97)
            )
        return 38.3 * _log10(d3) + 17.30 + 24.9 * _log10(fc)  # inh

    def path_loss_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        pl_los = self._los_pl(g)
        pl_nlos = np.maximum(pl_los, self._nlos_prime(g))
        return np.where(np.asarray(los, bool), pl_los, pl_nlos)

    # ---------------------------------------------------------------------------------- LoS prob / shadowing
    def los_probability(self, g: LinkGeometry) -> np.ndarray:
        s = self.scenario
        if s == "uma":
            return _los_prob_uma(g.d2d, g.h_ut)
        if s == "umi":
            return _los_prob_umi(g.d2d)
        if s == "rma":
            return _los_prob_rma(g.d2d)
        return _los_prob_inh_mixed(g.d2d)

    def shadowing_std_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        los = np.asarray(los, bool)
        s = self.scenario
        sig_l: Any
        if s == "uma":
            sig_l, sig_n = 4.0, 6.0
        elif s == "umi":
            sig_l, sig_n = 4.0, 7.82
        elif s == "inh":
            sig_l, sig_n = 3.0, 8.03
        else:  # rma: 4 dB below the breakpoint, 6 dB above (LoS); 8 dB NLoS
            dbp = 2.0 * math.pi * np.maximum(g.h_bs, 0.1) * np.maximum(g.h_ut, 0.1) * self.freq_hz / C_LIGHT
            sig_l = np.where(g.d2d <= dbp, 4.0, 6.0)
            sig_n = 8.0
        return np.where(los, sig_l, sig_n).astype(float) * np.ones_like(g.d3d)

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "scenario": self.scenario}


class TR36777(PathLossModel):
    """TR 36.777 (Annex B) aerial-vehicle models: ``uma_av``, ``umi_av``, ``rma_av``.

    For UAV heights up to 22.5 m the terrestrial TR 38.901 model of the same environment applies (as in the report);
    between 22.5 m and 300 m the aerial-specific path loss, LoS probability and shadowing are used. Heights above 300 m are
    outside the model and are reported by :meth:`validity` (values are still returned, clamped to 300 m for the
    height-dependent terms).
    """

    H_SPLIT = 22.5
    H_MAX = 300.0

    def __init__(self, freq_hz: float = 2.0e9, scenario: str = "uma_av"):
        super().__init__(freq_hz)
        scenario = scenario.lower()
        if scenario not in ("uma_av", "umi_av", "rma_av"):
            raise ValueError("scenario must be 'uma_av', 'umi_av' or 'rma_av'")
        self.scenario = scenario
        self._ground = TR38901(freq_hz, scenario[:3])
        self.name = f"tr36777_{scenario}"
        self.freq_range_ghz = (0.5, 6.0)
        self.d2d_range_m = (10.0, 5000.0 if scenario != "rma_av" else 10000.0)
        self.h_ut_range_m = (1.5, self.H_MAX)

    def _h(self, g: LinkGeometry) -> np.ndarray:
        return np.clip(g.h_ut, 1.5, self.H_MAX)

    def validity(self, g: LinkGeometry) -> np.ndarray:
        return super().validity(g) & (g.h_ut <= self.H_MAX)

    def path_loss_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        los = np.asarray(los, bool)
        fc, d3, h = self.fc_ghz, _clip_d(g.d3d), self._h(g)
        s = self.scenario
        lfc = 20.0 * _log10(fc)
        if s == "uma_av":
            pl_los = 28.0 + 22.0 * _log10(d3) + lfc
            pl_nlos = -17.5 + (46.0 - 7.0 * _log10(h)) * _log10(d3) + 20.0 * _log10(40.0 * math.pi * fc / 3.0)
            pl_nlos = np.maximum(pl_los, pl_nlos)
        elif s == "umi_av":
            pl_los = 30.9 + (22.25 - 0.5 * _log10(h)) * _log10(d3) + lfc
            pl_nlos = np.maximum(pl_los, 32.4 + (43.2 - 7.0 * _log10(h)) * _log10(d3) + lfc)
        else:
            k = 20.0 * _log10(40.0 * math.pi * fc / 3.0)
            pl_los = np.maximum(23.9 - 1.8 * _log10(h), 20.0) * _log10(d3) + k
            pl_nlos = np.maximum(pl_los, -12.0 + (35.0 - 5.3 * _log10(h)) * _log10(d3) + k)
        aerial = np.where(los, pl_los, pl_nlos)
        return np.where(g.h_ut <= self.H_SPLIT, self._ground.path_loss_db(g, los), aerial)

    def los_probability(self, g: LinkGeometry) -> np.ndarray:
        h, d = self._h(g), np.maximum(g.d2d, 1e-6)
        s = self.scenario
        if s == "uma_av":
            p1 = 4300.0 * _log10(h) - 3800.0
            d1 = np.maximum(460.0 * _log10(h) - 700.0, 18.0)
        elif s == "umi_av":
            p1 = 233.98 * _log10(h) - 0.95
            d1 = np.maximum(294.05 * _log10(h) - 432.94, 18.0)
        else:
            p1 = np.maximum(15021.0 * _log10(h) - 16053.0, 1000.0)
            d1 = np.maximum(1350.0 * _log10(h) - 1602.0, 18.0)
        p_mid = np.where(d <= d1, 1.0, d1 / d + np.exp(-d / np.maximum(p1, 1e-6)) * (1.0 - d1 / d))
        p_air = np.where(g.h_ut > 100.0, 1.0, np.clip(p_mid, 0.0, 1.0))
        return np.where(g.h_ut <= self.H_SPLIT, self._ground.los_probability(g), p_air)

    def shadowing_std_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        los, h = np.asarray(los, bool), self._h(g)
        s = self.scenario
        if s == "uma_av":
            sig_l, sig_n = 4.64 * np.exp(-0.0066 * h), 6.0
        elif s == "umi_av":
            sig_l, sig_n = np.maximum(5.0 * np.exp(-0.01 * h), 2.0), 8.0
        else:
            sig_l, sig_n = 4.2 * np.exp(-0.0046 * h), 6.0
        aerial = np.where(los, sig_l, sig_n).astype(float) * np.ones_like(g.d3d)
        return np.where(g.h_ut <= self.H_SPLIT, self._ground.shadowing_std_db(g, los), aerial)

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "scenario": self.scenario, "h_range_m": (1.5, self.H_MAX)}


def valid_range(model: PathLossModel) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return model.freq_range_ghz, model.d2d_range_m
