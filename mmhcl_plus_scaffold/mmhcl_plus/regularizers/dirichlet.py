"""
Dirichlet energy regularisation — TEX Section 3, Corollary 1 and Eq. (3).

     L_Dir = tr( E^(l)^T (I - Theta) E^(l) )

Applied to the final hypergraph embedding layer to penalise over-smoothing.
At mini-batch scale, only the rows for `batch_idx` are extracted, reducing
the cost from O(N^2 d) to O(B * d_bar_v) per batch (TEX §3.3).

Note: the Laplacian block returned by `laplacian_block_getter` may live on CPU
(it is often pre-computed once and fetched lazily).  We move it to the same
device as E_batch before multiplying.
"""

from __future__ import annotations

import torch


def dirichlet_energy_batch(
    E_batch: torch.Tensor,
    lap_block: torch.Tensor,
) -> torch.Tensor:
    """
    Compute  tr( E^T L E )  for a batch sub-block of the Laplacian.

    Args:
        E_batch:   [B, d] float embedding sub-matrix.
        lap_block: [B, B] (I - Theta) sub-block (sparse or dense).
                   Automatically moved to E_batch.device.

    Returns:
        Scalar Dirichlet energy.
    """
    # Ensure the Laplacian is on the same device as the embeddings
    if lap_block.device != E_batch.device:
        lap_block = lap_block.to(E_batch.device)

    if lap_block.is_sparse:
        prod = torch.sparse.mm(lap_block, E_batch)   # [B, d]
    else:
        prod = lap_block @ E_batch                    # [B, d]

    return torch.trace(E_batch.T @ prod)              # scalar


def dirichlet_energy_minibatch(
    E: torch.Tensor,
    batch_idx: torch.Tensor,
    laplacian_block_getter,
) -> torch.Tensor:
    """
    Extract the batch sub-block and compute the Dirichlet energy.

    Args:
        E:                       [N, d] full embedding matrix (or [B, d] batch).
        batch_idx:               1-D LongTensor — row indices into E.
        laplacian_block_getter:  callable(batch_idx) → [B, B] Laplacian block
                                 (sparse or dense, may be on CPU).

    Returns:
        Scalar Dirichlet energy for the mini-batch.
    """
    E_batch = E[batch_idx]
    L_batch = laplacian_block_getter(batch_idx)
    return dirichlet_energy_batch(E_batch, L_batch)
