"""Al-Hourani air-to-ground model as a :class:`PathLossModel`.

``PL = FSPL(d3d) + eta`` with environment-dependent excess losses and an elevation-angle LoS probability
(A. Al-Hourani, S. Kandeepan, S. Lardner, IEEE WCL 3(6), 2014). Valid for UAV altitudes of roughly 20-3000 m over
the specified urban classes; the numbers come from the paper's fit at 2 GHz and are used unchanged at other
frequencies, which is a common but approximate practice.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..channel import ENVIRONMENTS, SHADOWING_STD_DB, los_probability
from .base import LinkGeometry, PathLossModel, fspl_db


class AlHourani(PathLossModel):
    name = "al_hourani"
    freq_range_ghz: Tuple[float, float] = (0.7, 6.0)
    d2d_range_m: Tuple[float, float] = (1.0, 20000.0)

    def __init__(self, freq_hz: float = 2.0e9, env: str = "urban"):
        super().__init__(freq_hz)
        if env not in ENVIRONMENTS:
            raise ValueError(f"env must be one of {sorted(ENVIRONMENTS)}")
        self.env = env

    def path_loss_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        _, _, eta_los, eta_nlos = ENVIRONMENTS[self.env]
        return fspl_db(g.d3d, self.freq_hz) + np.where(np.asarray(los, bool), eta_los, eta_nlos)

    def los_probability(self, g: LinkGeometry) -> np.ndarray:
        return los_probability(g.elevation_deg, self.env)

    def shadowing_std_db(self, g: LinkGeometry, los: Any) -> np.ndarray:
        return np.where(np.asarray(los, bool), SHADOWING_STD_DB["los"], SHADOWING_STD_DB["nlos"]).astype(float) * np.ones_like(g.d3d)
