"""
Ablation configuration registry for MMHCL+ (Revision 5.2).

This module mirrors the specification in
``mmhcl_plus_ablation_guide_full_translation.tex``:

  * A-series : component-level ablations (A0 -- A8)
  * B-series : hyperparameter sensitivity (g-layer depth sweep)
  * C-series : alternative loss balancers (uncertainty / gradnorm / fixed)

Each :class:`AblationVariant` is a pure data container; it carries the
flags needed by ``main_mmhcl_plus.py`` to gate optional components
(Neighbor-Layer CL, SVD spectral augmentation, CL warmup ramp, FAISS
delay, Hutchinson Dirichlet, Soft BYOL cross-branch, ego-final anchor,
etc.) and to configure the projector dimension, the ``g``-layer depth,
and the loss balancer mode.

The :func:`apply_to_args` helper copies the variant's settings onto an
``argparse.Namespace`` (or any attribute-bearing object), so the outer
training loop does not need to know about the registry.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
#  Core dataclass
# ---------------------------------------------------------------------------
@dataclass
class AblationVariant:
    """One row of the Q1-style ablation table.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"A0_full"`` or ``"B2_g2"``.
    enable_neighbor_layer_cl : bool
        If ``False``, the u2u (VICReg) and i2i (Chunked InfoNCE) terms
        are zeroed out, reverting toward vanilla MMHCL.
    enable_svd_spectral : bool
        Toggle the SVD spectral filtering of the incidence matrix
        (popularity debias).
    vicreg_projector_dim, vicreg_hidden_dim : int
        Output / hidden dimensions of the expanded projector
        (``ExpandedProjector``).  The Rev5.2 default is 4096 / 1024.
    enable_cl_warmup_ramp : bool
        If ``False``, CL losses activate abruptly at the end of the
        BPR warmup instead of linearly ramping.
    ramp_epochs : int
        Number of epochs for the linear 0 -> 1 ramp.
    enable_delayed_faiss : bool
        If ``False``, FAISS hard negatives are available from epoch 0.
    delay_hard_negs_epoch : int
        Epoch at which FAISS hard negatives become active when
        :attr:`enable_delayed_faiss` is ``True``.
    enable_dirichlet : bool
        Toggle the Hutchinson-estimated Dirichlet energy regulariser.
    enable_ego_final_anchor : bool
        Re-enable the Rev5.1 ego-final VICReg anchor (evaluates the
        "paradox" claim that it conflicts with Neighbor-Layer CL).
    enable_soft_byol_cross : bool
        Toggle the Soft BYOL cross-branch alignment loss.
    g_layers : int
        Neighbor-layer depth used by ``build_neighbor_layer_pairs``.
    svd_top_k : int
        Number of top singular values zeroed during SVD filtering.
    balancer_type : str
        One of ``{"hybrid", "uncertainty", "gradnorm", "fixed"}``.
    balancer_switch_epoch : int
        Epoch at which the hybrid balancer transitions from
        Uncertainty Weighting to GradNorm.
    warmup_epochs : int
        BPR-only warmup length.
    n_tasks : int
        Number of loss components seen by the balancer.  Kept as a
        dataclass field for completeness; in practice ``main_mmhcl_plus``
        always instantiates the balancer with 5 slots (plus one for the
        ego-final anchor when ``enable_ego_final_anchor`` is ``True``).
    notes : str
        Free-form human-readable description of the variant.
    """

    name: str
    enable_neighbor_layer_cl: bool = True
    enable_svd_spectral: bool = True
    vicreg_projector_dim: int = 4096
    vicreg_hidden_dim: int = 1024
    enable_cl_warmup_ramp: bool = True
    ramp_epochs: int = 20
    enable_delayed_faiss: bool = True
    delay_hard_negs_epoch: int = 50
    enable_dirichlet: bool = True
    enable_ego_final_anchor: bool = False
    enable_soft_byol_cross: bool = True
    g_layers: int = 2
    svd_top_k: int = 10
    balancer_type: str = "hybrid"
    balancer_switch_epoch: int = 40
    balancer_blend_epochs: int = 20
    warmup_epochs: int = 15
    n_tasks: int = 5
    notes: str = ""

    # -- convenience ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def clone(self, **overrides: Any) -> "AblationVariant":
        new = copy.deepcopy(self)
        for key, value in overrides.items():
            if not hasattr(new, key):
                raise AttributeError(f"AblationVariant has no field '{key}'")
            setattr(new, key, value)
        return new


# ---------------------------------------------------------------------------
#  Registry (mirrors Table "Proposed ablation matrix" in the guide)
# ---------------------------------------------------------------------------
REGISTRY: dict[str, AblationVariant] = {
    # ================ A-series: component ablations =====================
    "A0_full": AblationVariant(
        name="A0_full",
        notes="Full MMHCL+ (Rev5.2 reference baseline).",
    ),
    "A1_no_nlcl": AblationVariant(
        name="A1_no_nlcl",
        enable_neighbor_layer_cl=False,
        n_tasks=3,
        notes="Disable u2u + i2i Neighbor-Layer CL (back to MMHCL).",
    ),
    "A2_no_svd": AblationVariant(
        name="A2_no_svd",
        enable_svd_spectral=False,
        notes="Disable SVD spectral filtering (popularity debias).",
    ),
    "A3_small_proj": AblationVariant(
        name="A3_small_proj",
        vicreg_projector_dim=1024,
        vicreg_hidden_dim=512,
        notes="Shrink VICReg projector 4096 -> 1024 (Dim. Paradox).",
    ),
    "A4_no_ramp": AblationVariant(
        name="A4_no_ramp",
        enable_cl_warmup_ramp=False,
        ramp_epochs=0,
        notes="Abrupt CL activation after warmup (CL activation shock).",
    ),
    "A5_no_delay": AblationVariant(
        name="A5_no_delay",
        enable_delayed_faiss=False,
        delay_hard_negs_epoch=0,
        notes="FAISS hard negatives active from epoch 0.",
    ),
    "A6_no_dirichlet": AblationVariant(
        name="A6_no_dirichlet",
        enable_dirichlet=False,
        n_tasks=4,
        notes="Remove Hutchinson Dirichlet energy term.",
    ),
    "A7_ego_final": AblationVariant(
        name="A7_ego_final",
        enable_ego_final_anchor=True,
        n_tasks=6,
        notes="Re-enable ego-final VICReg anchor (6 tasks; paradox test).",
    ),
    "A8_no_cross": AblationVariant(
        name="A8_no_cross",
        enable_soft_byol_cross=False,
        n_tasks=4,
        notes="Remove Soft BYOL cross-branch alignment.",
    ),
    # ================ B-series: g-layer depth ==========================
    "B1_g1": AblationVariant(
        name="B1_g1",
        g_layers=1,
        notes="Neighbor-layer depth g = 1.",
    ),
    "B2_g2": AblationVariant(
        name="B2_g2",
        g_layers=2,
        notes="Neighbor-layer depth g = 2 (DPI-safe, default).",
    ),
    "B3_g3": AblationVariant(
        name="B3_g3",
        g_layers=3,
        notes="Neighbor-layer depth g = 3 (beyond DPI safe zone).",
    ),
    # ================ C-series: balancer alternatives ==================
    "C1_uncertainty": AblationVariant(
        name="C1_uncertainty",
        balancer_type="uncertainty",
        notes="Uncertainty Weighting only (Kendall et al. 2018).",
    ),
    "C2_gradnorm": AblationVariant(
        name="C2_gradnorm",
        balancer_type="gradnorm",
        notes="GradNorm only (Chen et al. 2018).",
    ),
    "C3_fixed": AblationVariant(
        name="C3_fixed",
        balancer_type="fixed",
        notes="Uniform fixed task weights (lower-bound baseline).",
    ),
}


# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------
def available_variants() -> list[str]:
    """Return the names of every registered ablation variant."""
    return list(REGISTRY.keys())


def get(variant_name: str) -> AblationVariant:
    """Return a *deep copy* of the variant with the given name."""
    if variant_name not in REGISTRY:
        raise KeyError(
            f"Unknown ablation variant '{variant_name}'. "
            f"Available: {available_variants()}"
        )
    return copy.deepcopy(REGISTRY[variant_name])


# Mapping from AblationVariant fields to argparse attribute names.
# Any field not listed is considered purely documentary (e.g. ``name``,
# ``notes``, ``n_tasks`` -- the last is derived from the toggles).
_FIELD_TO_ARG: dict[str, str] = {
    "enable_neighbor_layer_cl": "enable_neighbor_layer_cl",
    "enable_svd_spectral": "use_svd_filtering",
    "vicreg_projector_dim": "projector_out_dim",
    "vicreg_hidden_dim": "projector_hidden_dim",
    "enable_cl_warmup_ramp": "enable_cl_warmup_ramp",
    "ramp_epochs": "cl_ramp_epochs",
    "enable_delayed_faiss": "enable_delayed_faiss",
    "delay_hard_negs_epoch": "delay_hard_negs_epoch",
    "enable_dirichlet": "enable_dirichlet",
    "enable_ego_final_anchor": "enable_ego_final_anchor",
    "enable_soft_byol_cross": "enable_soft_byol_cross",
    "g_layers": "g_layers",
    "svd_top_k": "svd_top_k",
    "balancer_type": "balancer_type",
    "balancer_switch_epoch": "balancer_transition_epoch",
    "balancer_blend_epochs": "balancer_blend_epochs",
    "warmup_epochs": "warmup_epochs",
}


def apply_to_args(variant: AblationVariant, args: Any) -> Any:
    """Copy a variant's configuration onto an ``argparse``-like object.

    Boolean flags are converted to ``int`` (0 / 1) to stay compatible
    with the existing ``argparse`` entries which use ``type=int``.
    """
    for vfield, argname in _FIELD_TO_ARG.items():
        value = getattr(variant, vfield)
        if isinstance(value, bool):
            value = int(value)
        setattr(args, argname, value)

    # Keep balancer <-> use_hybrid_balancer in sync for backward compat.
    if variant.balancer_type == "hybrid":
        setattr(args, "use_hybrid_balancer", 1)
    else:
        setattr(args, "use_hybrid_balancer", 0)

    setattr(args, "ablation_variant", variant.name)
    setattr(args, "ablation_target", variant.name)
    return args


def summarise() -> str:
    """Return a human-readable table of every registered variant."""
    rows = ["ID              | Notes"]
    rows.append("-" * 80)
    for name, variant in REGISTRY.items():
        rows.append(f"{name:15s} | {variant.notes}")
    return "\n".join(rows)


if __name__ == "__main__":  # quick inspection
    print(summarise())
