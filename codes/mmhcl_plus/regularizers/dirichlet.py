"""
Dirichlet energy regularisation — TEX Section 3, Corollary 1 and Eq. (3).

     L_Dir = tr( E^(l)^T (I - Theta) E^(l) )

Applied to the final hypergraph embedding layer to penalise over-smoothing.
At mini-batch scale, only the rows for `batch_idx` are extracted, reducing
the cost from O(N^2 d) to O(B * d_bar_v) per batch (TEX §3.3).

Two API styles are provided:

  1. **Laplacian-block API** (``dirichlet_energy_batch`` / ``dirichlet_energy_minibatch``):
     Caller supplies a pre-extracted (I-Θ) sub-block.  Used by the scaffold
     demo trainer (``TwoStageTrainer``).

  2. **Sparse-adjacency API** (``sparse_dirichlet_energy`` / ``sparse_dirichlet_energy_batch``):
     Caller supplies the full sparse normalised propagation operator Θ and a
     batch index; the function computes  (||E_B||² − ⟨E_B, (Θ@E)[B]⟩) / B
     in one sparse matmul.  Used by the production trainer
     (``MMHCLPlusTrainer`` in main_mmhcl_plus.py).
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
        prod = torch.sparse.mm(lap_block, E_batch)  # [B, d]
    else:
        prod = lap_block @ E_batch  # [B, d]

    return torch.trace(E_batch.T @ prod)  # scalar


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


# ---------------------------------------------------------------------------
# Sparse-adjacency API (used by main_mmhcl_plus.py production trainer)
# ---------------------------------------------------------------------------


def sparse_dirichlet_energy(emb: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Full-graph Dirichlet energy:  tr(E^T (I-Θ) E) / N.

    Avoids materialising the dense Laplacian by expanding the trace:
        (||E||_F^2  −  tr(E^T Θ E)) / N

    Args:
        emb: [N, d] full embedding matrix (on GPU).
        adj: [N, N] sparse normalised propagation operator Θ (on GPU).

    Returns:
        Scalar Dirichlet energy.
    """
    n = emb.size(0)
    adj_emb = torch.sparse.mm(adj, emb)
    return ((emb * emb).sum() - (emb * adj_emb).sum()) / n


def sparse_dirichlet_energy_batch(
    emb: torch.Tensor,
    adj: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Mini-batch Dirichlet energy (TEX §3.3, Corollary 1):

        tr( E_B^T (I − Θ) E_B ) / B

    Subsamples only the ``batch_idx`` rows from Θ@E after a single sparse
    matmul on the full matrix, reducing the trace cost to O(B·d) while the
    propagation remains O(nnz·d).

    Args:
        emb:       [N, d] full embedding matrix (on GPU).
        adj:       [N, N] sparse normalised propagation operator Θ (on GPU).
        batch_idx: [B] LongTensor of node indices for the current mini-batch.

    Returns:
        Scalar Dirichlet energy for the mini-batch.
    """
    E_batch = emb[batch_idx]
    adj_emb_batch = torch.sparse.mm(adj, emb)[batch_idx]
    n_batch = batch_idx.size(0)
    return ((E_batch * E_batch).sum() - (E_batch * adj_emb_batch).sum()) / n_batch
