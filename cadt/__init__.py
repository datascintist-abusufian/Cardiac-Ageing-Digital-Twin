"""Cardiac Ageing Digital Twin (cadt).

Cell-level Ageing State Vector -> tissue parameters -> 2D tissue simulation.
"""

from .ageing_state import AgeingStateVector, load_state, save_state
from .mapping import map_to_tissue, map_with_uncertainty, describe_rules
from .tissue_sim import simulate, fibrosis_mask

__all__ = [
    "AgeingStateVector", "load_state", "save_state",
    "map_to_tissue", "map_with_uncertainty", "describe_rules",
    "simulate", "fibrosis_mask",
]
__version__ = "0.1.0"
