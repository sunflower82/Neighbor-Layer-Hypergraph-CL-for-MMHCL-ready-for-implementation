"""
Loss functions for MMHCL+ — TEX Section 4, Steps 1–3.

Implemented functions
---------------------
barlow_twins_loss          — Barlow Twins with optional per-sample soft weighting
                             (u2u branch, TEX §4.4 code snippet).
chunked_info_nce_loss      — Memory-safe chunked InfoNCE with optional dynamic
                             per-sample weights (i2i branch, TEX §4.4).
temperature_free_info_nce_loss — Convenience wrapper (τ = 1.0).
info_nce_loss              — Baseline full-matrix InfoNCE (used for small batches).
bpr_loss                   — Standard Bayesian Personalised Ranking loss.

Soft weighting notes (from NLGCL+ §3.3 and TEX §4.2)
------------------------------------------------------
For the u2u / Barlow Twins branch:
  soft_weights [B] re-scales each sample's contribution to the cross-correlation
  matrix by multiplying its embedding before normalisation.  High-similarity pairs
  receive larger gradients; noise receives near-zero weight.

For the i2i / InfoNCE branch:
  dynamic_weights [B, N] is applied as an additive log-temperature correction
  to the raw dot-product logits before softmax:
      logits'[b, j] = (q[b] · k[j] / τ) + log(w[b, j])
  This is equivalent to treating w[b, j] as a prior probability that sample j is
  a valid positive/hard-negative, following the adaptive InfoNCE literature.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _align_chunk_size_tensor_cores(chunk_size: int, n: int) -> int:
    """Snap chunk size to a multiple of 64 for Tensor Core throughput (report §low-prio)."""
    if n <= 0:
        return 1
    chunk_size = max(1, min(chunk_size, n))
    aligned = (chunk_size // 64) * 64
    if aligned >= 64:
        return aligned
    return min(64, n)


def _maybe_torch_compile(fn):
    """
    JIT-compile dense loss kernels (MMHCL+ Optimization Report — Opt. 1).

    Set environment variable MMHCL_DISABLE_TORCH_COMPILE=1 to force eager mode.
    """
    if os.environ.get("MMHCL_DISABLE_TORCH_COMPILE", "").strip():
        return fn
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return fn
    try:
        return compile_fn(fn, dynamic=True, fullgraph=False)
    except Exception:
        return fn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return all off-diagonal elements of a square matrix as a 1-D tensor."""
    n, m = x.shape
    assert n == m, "off_diagonal expects a square matrix"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------------------------------------------------------------------------
# Barlow Twins (u2u branch — Stage 1 intra-branch CL)
# ---------------------------------------------------------------------------


def _barlow_twins_loss_impl(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambd: float = 5e-3,
    soft_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Barlow Twins loss with optional per-sample soft weighting (TEX §4.4).

    When `soft_weights` is provided, each row of z1 and z2 is multiplied by
    sqrt(w_i) before building the cross-correlation matrix.  This up-weights
    semantically similar pairs (high W_ema[i, i+]) and down-weights noisy ones,
    implementing Adaptive Sample Weighting (ASW) from NLGCL+ §3.3.

    The expanded projector (d=64 → D=8192) applied upstream ensures that the
    cross-correlation matrix is well-conditioned despite the small embedding dim.

    Args:
        z1, z2:       [B, D] projected embeddings (after ExpandedProjector).
        lambd:        Off-diagonal penalty coefficient.
        soft_weights: Optional [B] float tensor of per-sample importance weights.
                      Values are L1-normalised internally so the total scale of
                      the loss is preserved regardless of how weights are produced.

    Returns:
        Scalar loss.
    """
    if soft_weights is not None:
        w = soft_weights.float().to(z1.device)
        w = w / w.sum().clamp_min(1e-8) * z1.size(0)  # preserve scale
        # Apply sqrt so that the cross-corr entry c[i,j] is weighted by w_i
        w_sqrt = w.sqrt().unsqueeze(-1)  # [B, 1]
        z1 = z1 * w_sqrt
        z2 = z2 * w_sqrt

    # Batch normalise
    z1 = (z1 - z1.mean(0)) / (z1.std(0).clamp_min(1e-9))
    z2 = (z2 - z2.mean(0)) / (z2.std(0).clamp_min(1e-9))

    # Cross-correlation matrix  [D, D]
    c = torch.mm(z1.T, z2) / z1.size(0)

    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag_term = off_diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag_term


barlow_twins_loss = _maybe_torch_compile(_barlow_twins_loss_impl)


# ---------------------------------------------------------------------------
# Chunked InfoNCE (i2i branch — Stage 1 intra-branch CL)
# ---------------------------------------------------------------------------


def _chunked_info_nce_loss_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    tau: float = 0.2,
    chunk_size: int = 1024,
    dynamic_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Memory-safe chunked InfoNCE with optional dynamic per-sample weights.

    The i2i branch uses this loss (TEX §4.4) because VRAM fragmentation from
    materialising the full [N, N] logit matrix causes OOM on 24 GB GPUs when
    N > ~5 000.  Chunking along the query axis keeps peak VRAM at O(chunk × N).

    Adaptive sample weighting (NLGCL+ §3.3):
        If `dynamic_weights` is provided (shape [B, N]), it is interpreted as
        log-prior W_ema[b, j] and added to the scaled dot-product before softmax:
            logits'[b, j] = q[b]·k[j] / τ + log(w[b, j] + ε)
        This way W_ema > 1 near known positives encourages the model to pull them
        closer, while near-zero weights on irrelevant negatives suppress their
        gradients — a direct implementation of Eq. 12–13 in NLGCL+.

    Args:
        q:               [B, d] online (query) embeddings — L2-normalised inside.
        k:               [N, d] target (key) embeddings — L2-normalised inside.
        tau:             Temperature; follows the annealing schedule from the caller.
        chunk_size:      Number of query rows processed per chunk.
        dynamic_weights: Optional [B, N] float weight matrix from W_ema.

    Returns:
        Scalar mean cross-entropy loss.
    """
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    n = q.size(0)
    chunk_size = _align_chunk_size_tensor_cores(chunk_size, n)
    labels = torch.arange(n, device=q.device)
    losses: list[torch.Tensor] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_q = q[start:end]  # [C, d]
        logits = chunk_q @ k.T / tau  # [C, N]

        if dynamic_weights is not None:
            # Log-prior correction (adaptive sample weighting)
            w_chunk = dynamic_weights[start:end].to(q.device)  # [C, N]
            logits = logits + torch.log(w_chunk.clamp_min(1e-8))

        losses.append(F.cross_entropy(logits, labels[start:end]))

    return torch.stack(losses).mean()


chunked_info_nce_loss = _maybe_torch_compile(_chunked_info_nce_loss_impl)


def temperature_free_info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE with τ = 1.0 (temperature-free baseline)."""
    n = q.size(0)
    cs = max(256, min(2048, n))
    cs = _align_chunk_size_tensor_cores(cs, n)
    return chunked_info_nce_loss(q, k, tau=1.0, chunk_size=cs)


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
