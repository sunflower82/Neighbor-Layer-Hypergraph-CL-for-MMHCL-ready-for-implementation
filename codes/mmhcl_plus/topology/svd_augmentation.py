"""
SVD-Based Spectral Augmentation for Hypergraphs --- Revision 5.1

TEX Rev5.1 Section 2.2, Corollary 2.1:
    By performing SVD on the incidence matrix H and truncating the top-K
    principal singular values, we obtain the filtered matrix H_tilde.
    Contrasting embeddings derived from H_tilde against those from the
    original H forces the GNN to capture true collaborative signals
    independent of macroscopic popularity bridges.
"""

from __future__ import annotations

import torch
import scipy.sparse as sp
import numpy as np


def svd_filter_incidence(
    H: sp.spmatrix | torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Apply truncated SVD filtering to the incidence matrix H.

    Removes the top-K singular values (which encode popularity-dominated
    spectral directions) to produce a filtered H_tilde that emphasises
    long-tail collaborative signals.

    Args:
        H:     Incidence matrix (n_nodes x n_edges), sparse or dense.
        top_k: Number of top singular values to remove.

    Returns:
        H_tilde: Filtered incidence matrix as a dense torch.Tensor.
    """
    if sp.issparse(H):
        H_dense = H.toarray()
    elif isinstance(H, torch.Tensor):
        H_dense = H.detach().cpu().numpy()
    else:
        H_dense = np.asarray(H)

    H_dense = H_dense.astype(np.float32)

    # Full SVD (for small-to-medium scale hypergraphs)
    U, S, Vt = np.linalg.svd(H_dense, full_matrices=False)

    # Zero out the top-K singular values (popularity-dominated directions)
    k = min(top_k, len(S))
    S_filtered = S.copy()
    S_filtered[:k] = 0.0

    # Reconstruct the filtered incidence matrix
    H_filtered = U @ np.diag(S_filtered) @ Vt

    return torch.from_numpy(H_filtered).float()


def svd_filter_sparse(
    adj_matrix: torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Apply SVD filtering directly to a sparse adjacency/propagation matrix.

    For production use with large sparse matrices. Uses torch.svd_lowrank
    for efficiency.

    Args:
        adj_matrix: Sparse or dense torch.Tensor.
        top_k:      Number of top singular components to remove.

    Returns:
        Filtered matrix (sparse COO format if input was sparse).
    """
    if adj_matrix.is_sparse:
        dense = adj_matrix.to_dense()
    else:
        dense = adj_matrix

    # Use low-rank SVD approximation for efficiency
    rank = min(top_k + 50, min(dense.shape))
    U, S, V = torch.svd_lowrank(dense.float(), q=rank)

    # Zero out the top-K components
    k = min(top_k, S.size(0))
    S_filtered = S.clone()
    S_filtered[:k] = 0.0

    # Reconstruct: original - top_k_component = filtered
    reconstructed = U @ torch.diag(S_filtered) @ V.T

    if adj_matrix.is_sparse:
        return reconstructed.to_sparse_coo()
    return reconstructed
