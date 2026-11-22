"""Cardiac Ageing Digital Twin (cadt).

Raw cell data -> features -> Ageing State Vector -> tissue parameters -> 2D tissue simulation.
"""

from .ageing_state import AgeingStateVector, load_state, save_state
from .mapping import map_to_tissue, map_with_uncertainty, describe_rules
from .tissue_sim import simulate, fibrosis_mask
from .motion import motion_features
from .nuclei import nuclear_features
from .calcium import calcium_features
from .scoring import YoungReference, build_asv, score_sample, explain

__all__ = [
    "AgeingStateVector", "load_state", "save_state",
    "map_to_tissue", "map_with_uncertainty", "describe_rules",
    "simulate", "fibrosis_mask",
    "motion_features", "nuclear_features", "calcium_features",
    "YoungReference", "build_asv", "score_sample", "explain",
]
__version__ = "0.2.0"
