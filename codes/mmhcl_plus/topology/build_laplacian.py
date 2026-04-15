"""
Hypergraph Laplacian builder — TEX §3, Eq. (2)–(3).

Constructs the normalised Laplacian  L = I − Θ  from a user–item interaction
(incidence) matrix, where the diffusion operator is:

    Θ = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}

Used by the Dirichlet Energy regularisation term (``regularizers/dirichlet.py``)
and the TwoStageTrainer scaffold.

The production trainer (``main_mmhcl_plus.py``) can also consume the Laplacian
through the Laplacian-block API in ``dirichlet_energy_minibatch()``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_hypergraph_laplacian(
    interaction_matrix: sp.spmatrix,
    w_e: np.ndarray | None = None,
    use_structural_weight: bool = True,
) -> sp.csr_matrix:
    """
    Build the hypergraph Laplacian  L = I − Θ  in CSR format.

    Parameters
    ----------
    interaction_matrix : sp.spmatrix
        Incidence matrix  H  of shape (n_nodes, n_edges).  For a user–item
        bipartite hypergraph this is typically the binary interaction matrix.
    w_e : np.ndarray or None
        Edge-weight vector of length n_edges.  If provided, used directly.
        If None, behaviour depends on ``use_structural_weight``.
    use_structural_weight : bool
        When ``w_e`` is None and this flag is True (default), automatically
        compute  W_e(e,e) = 1 / log(1 + |e|)  as specified by Rev5.1 Eq. (1).
        Set to False for uniform (all-ones) weights.

    Returns
    -------
    sp.csr_matrix
        Laplacian  L = I − D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}
        in CSR sparse format.
    """
    H = interaction_matrix.tocsc()
    n_nodes, n_edges = H.shape

    # Degree vectors
    d_v = np.asarray(H.sum(axis=1)).flatten()  # node degrees
    d_e = np.asarray(H.sum(axis=0)).flatten()  # hyperedge degrees

    # Inverse (sqrt) degree vectors — zero out missing entries
    d_v_inv_sqrt = np.where(d_v > 0, 1.0 / np.sqrt(d_v), 0.0)
    d_e_inv = np.where(d_e > 0, 1.0 / d_e, 0.0)

    if w_e is None:
        if use_structural_weight:
            # TEX Rev5.1 Eq. (1): W_e(e,e) = 1 / log(1 + |e|)
            # Logarithmic decay provides softer penalization for massive
            # hyperedges compared to linear inverse 1/(1+|e|).
            w_e = 1.0 / np.log1p(d_e.astype(np.float64)).clip(min=1e-8)
        else:
            w_e = np.ones(n_edges, dtype=np.float64)

    D_v_inv_sqrt = sp.diags(d_v_inv_sqrt)
    D_e_inv = sp.diags(d_e_inv)
    W_e_mat = sp.diags(w_e)

    # Θ = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}
    Theta = D_v_inv_sqrt @ H @ W_e_mat @ D_e_inv @ H.T @ D_v_inv_sqrt

    # L = I − Θ
    L = sp.eye(n_nodes, format="csr") - Theta
    return L.tocsr()
