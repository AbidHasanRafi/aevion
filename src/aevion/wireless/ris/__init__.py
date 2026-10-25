"""Reconfigurable intelligent surfaces: geometry, cascaded channel and phase configuration."""

from .cascaded_channel import ris_channel, ris_received_power_dbm, scaling_summary
from .phase_opt import optimal_phases, quantization_loss_db, quantize_phases
from .surface import RisPanel

__all__ = ["RisPanel", "optimal_phases", "quantization_loss_db", "quantize_phases", "ris_channel", "ris_received_power_dbm", "scaling_summary"]
