"""mmhcl_plus.contrast — Rev5.2: Added HybridLossBalancer."""

from .uncertainty_balancer import UncertaintyLossBalancer
from .gradnorm import GradNormLossBalancer
from .hybrid_balancer import HybridLossBalancer
from .losses import (
    bpr_loss,
    chunked_info_nce_loss,
    info_nce_loss,
    temperature_free_info_nce_loss,
    vicreg_loss,
)
from .neighbor_pairs import build_neighbor_layer_pairs
from .soft_byol import soft_byol_alignment

__all__ = [
    "bpr_loss", "chunked_info_nce_loss", "info_nce_loss",
    "temperature_free_info_nce_loss", "vicreg_loss",
    "soft_byol_alignment", "UncertaintyLossBalancer",
    "GradNormLossBalancer", "HybridLossBalancer",
]
