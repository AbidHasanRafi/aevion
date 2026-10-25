"""Antenna patterns, phased arrays, beam codebooks, beamforming and UAV-attitude-dependent gain."""

from .arrays import PhasedArray, UniformLinearArray, UniformPlanarArray, direction_vector
from .beamforming import BeamManager, BeamState, mmse_weights, mrt_weights, pointing_loss_db, zf_weights
from .codebooks import Codebook, dft_codebook, grid_codebook
from .patterns import (
    GAIN_FLOOR_DBI,
    AntennaPattern,
    GaussianBeam,
    HalfWaveDipole,
    Isotropic,
    Parabolic,
    Patch3GPP,
    SectorTR36814,
    ShortDipole,
    off_boresight_angle,
)
from .uav_orientation import UavAntenna, direction_in_antenna_frame, quat_to_matrices, rpy_to_matrix

PATTERNS = {
    "isotropic": Isotropic,
    "short_dipole": ShortDipole,
    "half_wave_dipole": HalfWaveDipole,
    "patch_3gpp": Patch3GPP,
    "sector_36814": SectorTR36814,
    "gaussian_beam": GaussianBeam,
    "parabolic": Parabolic,
}

__all__ = [
    "GAIN_FLOOR_DBI",
    "PATTERNS",
    "AntennaPattern",
    "BeamManager",
    "BeamState",
    "Codebook",
    "GaussianBeam",
    "HalfWaveDipole",
    "Isotropic",
    "Parabolic",
    "Patch3GPP",
    "PhasedArray",
    "SectorTR36814",
    "ShortDipole",
    "UavAntenna",
    "UniformLinearArray",
    "UniformPlanarArray",
    "dft_codebook",
    "direction_in_antenna_frame",
    "direction_vector",
    "grid_codebook",
    "mmse_weights",
    "mrt_weights",
    "off_boresight_angle",
    "pointing_loss_db",
    "quat_to_matrices",
    "rpy_to_matrix",
    "zf_weights",
]
