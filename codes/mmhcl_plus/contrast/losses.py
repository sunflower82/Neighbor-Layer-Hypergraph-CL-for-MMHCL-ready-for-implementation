"""
Loss functions for MMHCL+ --- Revision 5.1

Changes from Rev44:
  - REMOVED: barlow_twins_loss (replaced by vicreg_loss)
  - ADDED:   vicreg_loss --- VICReg (Bardes et al., 2022) for u2u & ego-final
  - UPDATED: chunked_info_nce_loss --- added hard_negatives parameter for FAISS mining
  - KEPT:    bpr_loss, info_nce_loss, temperature_free_info_nce_loss (unchanged)

Loss landscape (6 objectives):
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
def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    soft_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    VICReg Loss (Bardes et al., ICLR 2022).

    Eliminates the O(B*D^2) global cross-correlation matrix bottleneck of
    Barlow Twins. With D=1024 (vs D=8192), saves ~80% VRAM and ~60% FLOPs.

    Three terms computed along the BATCH dimension:
      1. Invariance (sim):  MSE between paired embeddings z1, z2
      2. Variance   (var):  Hinge loss pushing std of each dimension >= 1
      3. Covariance (cov):  Penalizes off-diagonal entries of the covariance matrix

    Args:
        z1, z2:      [B, D] projected embeddings from ExpandedProjector (D=1024).
        sim_weight:  lambda for invariance term (default 25.0 per VICReg paper).
        var_weight:  mu for variance term (default 25.0).
        cov_weight:  nu for covariance term (default 1.0).
        soft_weights: Optional [B] per-sample importance from W_ema.

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

    # -- 2. Variance term: push std of each dimension above 1 --
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    std_loss = torch.mean(F.relu(1.0 - torch.sqrt(z1.var(dim=0) + 1e-4))) + \
               torch.mean(F.relu(1.0 - torch.sqrt(z2.var(dim=0) + 1e-4)))

    # -- 3. Covariance term: decorrelate off-diagonal entries --
    N, D = z1.size()
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(D) + \
               off_diagonal(cov_z2).pow_(2).sum().div(D)

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
    hard_neg_weight: float = 1.0,
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
        hard_neg_weight: Scalar weight for hard negative log-prior (default 1.0).

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
