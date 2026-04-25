"""
Loss functions for MMHCL+ --- Revision 5.2 (Acceleration Guide §B + §C)

Changes from Rev44:
  - REMOVED: barlow_twins_loss (replaced by vicreg_loss)
  - ADDED:   vicreg_loss --- VICReg (Bardes et al., 2022) for u2u & ego-final
  - UPDATED: chunked_info_nce_loss --- added hard_negatives parameter for FAISS mining
  - KEPT:    bpr_loss, info_nce_loss, temperature_free_info_nce_loss (unchanged)

Rev5.2 speedup patches (mmhcl_rev52_speedup_analysis_en):
  * §B1 Lazy covariance: ``vicreg_loss(..., compute_cov=False)`` skips the
    O(D^2) covariance materialisation entirely.  Caller toggles it every
    other epoch -> ~10 % wall-clock saving with negligible NDCG impact
    (cov term carries weight 1.0 vs 25.0 for sim/var).
  * §B2 bf16 matmul: on RTX 5090 / Blackwell the covariance ``z^T z``
    is computed in bfloat16 (same dynamic range as fp32, native tensor
    cores) and immediately upcast back to fp32 for the off-diagonal sum.
  * §C torch.compile: ``vicreg_loss`` is wrapped with the soft-byol
    helper at module load time, fusing mean/var/relu/off-diagonal into a
    single Inductor kernel on Linux while keeping eager mode on Windows.

Loss landscape (5 objectives in Rev5.2 base; +1 ego_final for A7 ablation):
  L_BPR        --- main recommendation task
  L_u2u        --- vicreg_loss (neighbor-layer CL on user hypergraph)
  L_i2i        --- chunked_info_nce_loss + FAISS hard negatives
  L_align      --- soft_byol_alignment (cross-branch, from soft_byol.py)
  L_Dir        --- Dirichlet energy (from regularizers/dirichlet.py)
  L_ego_final  --- vicreg_loss (ego Layer 0 <-> final Layer L)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return all off-diagonal elements of a square matrix as a 1-D tensor."""
    n, m = x.shape
    assert n == m, "off_diagonal expects a square matrix"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------------------------------------------------------------------------
# VICReg (u2u branch + ego-final anchor --- Stage 1 & Stage 2)
# ---------------------------------------------------------------------------
def _bf16_matmul_cov(z_centered: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute ``(zᵀ z) * scale`` via bf16 tensor cores, upcast to fp32.

    Acceleration Guide §B2: RTX 5090 / Blackwell SM120 exposes native
    bfloat16 tensor cores with the same dynamic range as float32, so the
    [D, D] covariance product cannot overflow as it would with fp16.
    The result is immediately promoted back to fp32 for the
    off-diagonal sum, so downstream numerics are unaffected.
    """
    z_bf = z_centered.to(torch.bfloat16)
    cov = (z_bf.T @ z_bf).float() * scale
    return cov


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    soft_weights: torch.Tensor | None = None,
    compute_cov: bool = True,
) -> torch.Tensor:
    """
    VICReg Loss (Bardes et al., ICLR 2022).

    Eliminates the O(B*D^2) global cross-correlation matrix bottleneck of
    Barlow Twins. With D=4096 (Rev5.2 expanded projector), the variance
    and covariance terms still fit comfortably on a 32 GB GPU because the
    covariance is computed in bfloat16 (Acceleration Guide §B2).

    Three terms computed along the BATCH dimension:
      1. Invariance (sim):  MSE between paired embeddings z1, z2
      2. Variance   (var):  Hinge loss pushing std of each dimension >= 1
      3. Covariance (cov):  Penalizes off-diagonal entries of the covariance matrix
                            (skipped when ``compute_cov=False``)

    Args:
        z1, z2:      [B, D] projected embeddings from ExpandedProjector.
        sim_weight:  lambda for invariance term (default 25.0 per VICReg paper).
        var_weight:  mu for variance term (default 25.0).
        cov_weight:  nu for covariance term (default 1.0).
        soft_weights: Optional [B] per-sample importance from W_ema.
        compute_cov:  When ``False`` the covariance penalty is skipped
                      entirely.  Acceleration Guide §B1: caller toggles
                      this every other epoch (``epoch % 2 == 0``) to
                      recover ~10 % wall-clock at no measurable NDCG cost
                      (cov_weight=1.0 vs sim+var=25.0+25.0=50.0).

    Returns:
        Scalar loss.
    """
    # -- 1. Invariance term (MSE) --
    if soft_weights is not None:
        repr_loss = (
            F.mse_loss(z1, z2, reduction="none").mean(dim=1) * soft_weights
        ).mean()
    else:
        repr_loss = F.mse_loss(z1, z2)

    # Acceleration Guide §3 — VICReg variance+covariance in fp32 (variance)
    # and bf16 (covariance matmul, then upcast).  Disabling autocast +
    # casting to float32 guarantees numerical safety for the variance hinge
    # regardless of whether the caller is using fp16, bf16, or no AMP.
    with torch.amp.autocast("cuda", enabled=False):
        z1_f = z1.float()
        z2_f = z2.float()

        # -- 2. Variance term: push std of each dimension above 1 --
        z1_f = z1_f - z1_f.mean(dim=0)
        z2_f = z2_f - z2_f.mean(dim=0)

        std_loss = torch.mean(F.relu(1.0 - torch.sqrt(z1_f.var(dim=0) + 1e-4))) + \
                   torch.mean(F.relu(1.0 - torch.sqrt(z2_f.var(dim=0) + 1e-4)))

        # -- 3. Covariance term: decorrelate off-diagonal entries --
        if compute_cov:
            N, D = z1_f.size()
            scale = 1.0 / max(N - 1, 1)
            # §B2: bf16 matmul on Blackwell tensor cores; upcast for sum.
            if z1_f.is_cuda:
                cov_z1 = _bf16_matmul_cov(z1_f, scale)
                cov_z2 = _bf16_matmul_cov(z2_f, scale)
            else:
                cov_z1 = (z1_f.T @ z1_f) * scale
                cov_z2 = (z2_f.T @ z2_f) * scale

            cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(D) + \
                       off_diagonal(cov_z2).pow_(2).sum().div(D)
        else:
            cov_loss = z1_f.new_zeros(())

    return sim_weight * repr_loss + var_weight * std_loss + cov_weight * cov_loss


# ---------------------------------------------------------------------------
# Chunked InfoNCE with FAISS Hard Negatives (i2i --- Stage 1)
# ---------------------------------------------------------------------------
def chunked_info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    tau: float = 0.2,
    chunk_size: int = 512,
    dynamic_weights: torch.Tensor | None = None,
    hard_negatives: torch.Tensor | None = None,
    hard_neg_weight: float = 0.5,
) -> torch.Tensor:
    """
    Memory-safe chunked InfoNCE with FAISS-guided hard negatives.

    Rev5.1: Re-purposes FAISS LSH indices to sample structure-aware
    hard negatives. Modality-similar items lacking historical co-occurrence
    serve as potent negative signals, sharpening the NDCG decision boundary.

    Args:
        q:               [B, d]  online (query) embeddings.
        k:               [N, d]  target (key) embeddings.
        tau:             Temperature (follows annealing schedule from caller).
        chunk_size:      Number of query rows per chunk.
        dynamic_weights: Optional [B, N] float W_ema log-prior weights.
        hard_negatives:  Optional [B, K_neg, d] FAISS-mined hard negatives.
        hard_neg_weight: Scalar weight for hard negative log-prior (default 0.5).
                         Conservative value prevents hard negatives from
                         dominating the loss before warm-up completes.

    Returns:
        Scalar mean cross-entropy loss.
    """
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    n = q.size(0)
    labels = torch.arange(n, device=q.device)

    # Augment key pool with hard negatives if provided
    if hard_negatives is not None:
        hard_neg_flat = F.normalize(
            hard_negatives.reshape(-1, hard_negatives.size(-1)), p=2, dim=-1
        )
        k_augmented = torch.cat([k, hard_neg_flat], dim=0)
    else:
        k_augmented = k

    losses: list[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_q = q[start:end]
        logits = chunk_q @ k_augmented.T / tau

        if dynamic_weights is not None:
            w_chunk = dynamic_weights[start:end].to(q.device)
            if hard_negatives is not None:
                n_hard = k_augmented.size(0) - k.size(0)
                hard_w = torch.full(
                    (w_chunk.size(0), n_hard), hard_neg_weight,
                    device=q.device,
                )
                w_chunk = torch.cat([w_chunk, hard_w], dim=1)
            logits = logits + torch.log(w_chunk.clamp_min(1e-8))

        losses.append(F.cross_entropy(logits, labels[start:end]))

    return torch.stack(losses).mean()


def temperature_free_info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE with tau = 1.0 (temperature-free baseline)."""
    return chunked_info_nce_loss(
        q, k, tau=1.0, chunk_size=max(256, min(2048, q.size(0)))
    )


def info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    tau: float = 0.2,
) -> torch.Tensor:
    """Full-matrix InfoNCE (safe for small N, used in unit tests)."""
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    logits = q @ k.T / tau
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# BPR loss (main recommendation task)
# ---------------------------------------------------------------------------
def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Standard Bayesian Personalised Ranking loss."""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores).clamp_min(1e-8)).mean()


# ---------------------------------------------------------------------------
# Acceleration Guide §C — torch.compile wrap for ``vicreg_loss``
# ---------------------------------------------------------------------------
# The Inductor backend fuses ``mean``/``var``/``relu(1 - sqrt(...))`` and the
# off-diagonal sum-of-squares into a single kernel, removing the temporary
# 4096×4096 covariance materialisation in eager mode. The helper from
# ``soft_byol`` already handles platform-specific fallbacks (Windows skips
# Triton inductor, ``MMHCL_DISABLE_TORCH_COMPILE`` opt-out).
# Importing inside the module-level scope at the bottom avoids circular
# imports because ``soft_byol`` does not depend on ``losses``.
from .soft_byol import _maybe_torch_compile as _maybe_torch_compile  # noqa: E402

vicreg_loss = _maybe_torch_compile(vicreg_loss)
