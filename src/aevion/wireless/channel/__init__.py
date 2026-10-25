"""Channel models.

The original air-to-ground channel (``air_to_ground``) is re-exported here, so ``from aevion.wireless.channel import ...`` keeps
working; the MIMO, blockage, delay-spread and mobility models live next to it.
"""

from .air_to_ground import (
    ENVIRONMENTS,
    SHADOWING_STD_DB,
    AirToGroundChannel,
    doppler_hz,
    elevation_deg,
    fspl_db,
    los_probability,
    rician_k_db,
    sector_gain_dbi,
    thermal_noise_dbm,
)
from .blockage import BlockageProcess, blockage_probability, blockage_rate, self_blockage_loss_db
from .delay_spread import coherence_bandwidth_hz, delay_spread_params, exponential_pdp, frequency_response, rms_delay_spread, sample_delay_spread_s
from .mimo import (
    condition_number_db,
    ergodic_capacity,
    exponential_correlation,
    geometric_channel,
    mimo_capacity,
    rayleigh_channel,
    water_filling_capacity,
)
from .mobility import LinkKinematics, link_kinematics

__all__ = [
    "ENVIRONMENTS",
    "SHADOWING_STD_DB",
    "AirToGroundChannel",
    "BlockageProcess",
    "LinkKinematics",
    "blockage_probability",
    "blockage_rate",
    "coherence_bandwidth_hz",
    "condition_number_db",
    "delay_spread_params",
    "doppler_hz",
    "elevation_deg",
    "ergodic_capacity",
    "exponential_correlation",
    "exponential_pdp",
    "frequency_response",
    "fspl_db",
    "geometric_channel",
    "link_kinematics",
    "los_probability",
    "mimo_capacity",
    "rayleigh_channel",
    "rician_k_db",
    "rms_delay_spread",
    "sample_delay_spread_s",
    "sector_gain_dbi",
    "self_blockage_loss_db",
    "thermal_noise_dbm",
    "water_filling_capacity",
]
