"""
MMHCL+ Ablation package --- Revision 5.2

Central registry of every Q1-style ablation variant described in
``mmhcl_plus_ablation_guide_full_translation.tex``.  Each variant toggles
a specific architectural component so that the marginal contribution can
be estimated on Recall@K / NDCG@K / Precision@K.
"""
from __future__ import annotations

from .ablation_config import (
    AblationVariant,
    REGISTRY,
    apply_to_args,
    available_variants,
    get,
    summarise,
)

__all__ = [
    "AblationVariant",
    "REGISTRY",
    "apply_to_args",
    "available_variants",
    "get",
    "summarise",
]
