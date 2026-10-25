"""Wireless: channel models, antennas, PHY/MAC abstraction, mobility management and network-level models for UAVs.

The package is organised by layer; every subpackage has its own ``__all__``:

=================  ==========================================================================================
``propagation``    path-loss models (free space, two-ray, log-distance, Al-Hourani, 3GPP TR 38.901 / 36.777),
                   knife-edge diffraction, image-method ray tracing
``fading``         Rayleigh/Rician/Nakagami, time-correlated Jakes fading, correlated shadowing, Doppler statistics
``antennas``       patterns, arrays, codebooks, beamforming, UAV-attitude-dependent gain
``channel``        air-to-ground channel, MIMO, blockage, delay spread, mobility kinematics
``phy``            NR numerology, MCS/CQI tables, BLER, link adaptation + HARQ, OFDM effective SINR
``mac_network``    schedulers, handover, radio-link failure, relay, mesh routing, CSMA, integrated NR network
``ris``            reconfigurable intelligent surfaces
``ntn``            satellite links
``maps``           radio environment maps, coverage, map reconstruction
``security``       interference/jamming and GNSS spoofing threat models, matching faults
``metrics``        outage, capacity, age of information, latency
``interop``        ns-3 mobility/link traces, OBJ geometry export
=================  ==========================================================================================

The original names (``AirToGroundChannel``, ``CellularNetwork``, ...) remain available from here.
"""

from .channel import (
    ENVIRONMENTS,
    AirToGroundChannel,
    doppler_hz,
    elevation_deg,
    fspl_db,
    los_probability,
    rician_k_db,
    sector_gain_dbi,
    thermal_noise_dbm,
)
from .network import AerialBaseStation, BaseStation, CellularNetwork, LinkReport

__all__ = [
    "ENVIRONMENTS",
    "AirToGroundChannel",
    "doppler_hz",
    "elevation_deg",
    "fspl_db",
    "los_probability",
    "rician_k_db",
    "sector_gain_dbi",
    "thermal_noise_dbm",
    "AerialBaseStation",
    "BaseStation",
    "CellularNetwork",
    "LinkReport",
]
