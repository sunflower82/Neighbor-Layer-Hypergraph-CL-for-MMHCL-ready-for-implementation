"""
Spectral radius tracking for the MMHCL+ hypergraph diffusion operator.

This module is described in Section "Diagnostic package structure" of
``mmhcl_plus_ablation_guide_full_translation.tex``.

Key relations (Rev5.2 §2.5):

  * The hypergraph Laplacian is ``L = I - Theta`` where
        Theta = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}.
  * ``lambda_max(Theta) = 1 - lambda_min(L)`` controls the rate of
    oversmoothing.  Dirichlet regularisation pushes ``lambda_min(L)``
    away from zero, which equivalently pulls ``lambda_max(Theta)``
    away from 1.
  * The spectral gap ``lambda_2(L)`` is a connectivity / smoothing
    indicator that we can log once per epoch.

The implementation uses ARPACK (``scipy.sparse.linalg.eigsh``) so
that large sparse Laplacians are handled without dense expansion.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
#  Core functional API
# ---------------------------------------------------------------------------
def _as_sparse(matrix: Any) -> sp.csr_matrix:
    """Best-effort conversion to a SciPy CSR matrix."""
    if sp.issparse(matrix):
        return matrix.tocsr()
    # Torch tensor path
    try:
        import torch  # local import to keep NumPy-only callers lightweight

        if isinstance(matrix, torch.Tensor):
            if matrix.is_sparse:
                coo = matrix.detach().coalesce().cpu()
                indices = coo.indices().numpy()
                values = coo.values().numpy()
                return sp.coo_matrix(
                    (values, (indices[0], indices[1])),
                    shape=tuple(coo.shape),
                ).tocsr()
            return sp.csr_matrix(matrix.detach().cpu().numpy())
    except ImportError:
        pass
    return sp.csr_matrix(np.asarray(matrix))


def compute_spectral_radius(
    laplacian: Any,
    method: str = "arpack",
    tol: float = 1e-6,
) -> float:
    """Return ``lambda_max(Theta) = 1 - lambda_min(L)``.

    Parameters
    ----------
    laplacian
        The hypergraph Laplacian ``L``.  Accepts SciPy sparse, NumPy
        dense, or a ``torch.Tensor`` (sparse or dense).
    method
        ``"arpack"`` (ARPACK via ``scipy.sparse.linalg.eigsh``) or
        ``"dense"`` (NumPy ``eigvalsh``; only for small matrices).
    tol
        Convergence tolerance for ARPACK.
    """
    L = _as_sparse(laplacian)
    n = L.shape[0]
    if n <= 1:
        return 0.0
    if method == "dense" or n <= 64:
        eigvals = np.linalg.eigvalsh(L.toarray())
        lam_min = float(eigvals.min())
    else:
        # SM = smallest magnitude; L is PSD so lam_min is non-negative.
        eigvals = eigsh(L, k=1, which="SM", tol=tol, return_eigenvectors=False)
        lam_min = float(eigvals.min())
    return float(1.0 - lam_min)


def compute_spectral_gap(laplacian: Any, tol: float = 1e-6) -> float:
    """Return ``lambda_2(L)`` (second-smallest eigenvalue of L)."""
    L = _as_sparse(laplacian)
    n = L.shape[0]
    if n <= 2:
        return 0.0
    if n <= 64:
        eigvals = np.sort(np.linalg.eigvalsh(L.toarray()))
        return float(eigvals[1])
    eigvals = eigsh(L, k=2, which="SM", tol=tol, return_eigenvectors=False)
    eigvals = np.sort(eigvals)
    return float(eigvals[1])


def compute_layer_spectral_radius(
    layer_embeddings: list[np.ndarray],
) -> list[float]:
    """Estimate the empirical contraction factor between consecutive layers.

    For adjacent ``(H_l, H_{l+1})`` we solve ``H_{l+1} ≈ P H_l`` in the
    least-squares sense and report the top singular value of ``P``.
    Values near 1 indicate oversmoothing; values near 0 indicate that
    successive layers are nearly decorrelated.
    """
    radii: list[float] = []
    for a, b in zip(layer_embeddings[:-1], layer_embeddings[1:], strict=False):
        A = np.asarray(a)
        B = np.asarray(b)
        if A.size == 0 or B.size == 0:
            radii.append(float("nan"))
            continue
        # Solve min || P A.T - B.T ||_F^2, so P.T = A (A^T A)^-1 A^T B (ugly).
        # Easier: compute singular values of A^+ B (right multiplication).
        try:
            pinv = np.linalg.pinv(A, rcond=1e-8)
            P = pinv @ B
            _, s, _ = np.linalg.svd(P, full_matrices=False)
            radii.append(float(s[0]) if s.size else float("nan"))
        except Exception:
            radii.append(float("nan"))
    return radii


def track_spectral_radius_per_epoch(
    interaction_matrix: Any,
    svd_top_k: int = 10,
    tol: float = 1e-6,
) -> dict[str, float]:
    """One-shot spectral comparison before / after SVD filtering.

    Builds the symmetric normalised Laplacian from ``interaction_matrix``
    (treated as the bipartite incidence matrix) and returns a dict with
    the spectral radius + gap for the raw and SVD-filtered operators.
    """
    M = _as_sparse(interaction_matrix).astype(np.float64)
    # Symmetric normalised Laplacian of H @ H.T (a common surrogate).
    H = M
    deg_row = np.asarray(H.sum(axis=1)).flatten()
    deg_col = np.asarray(H.sum(axis=0)).flatten()
    d_row = sp.diags(1.0 / np.sqrt(np.clip(deg_row, 1e-8, None)))
    d_col = sp.diags(1.0 / np.clip(deg_col, 1e-8, None))
    Theta = d_row @ H @ d_col @ H.T @ d_row
    L = sp.eye(Theta.shape[0]) - Theta

    result: dict[str, float] = {
        "lambda_max_theta_raw": compute_spectral_radius(L, tol=tol),
        "spectral_gap_raw": compute_spectral_gap(L, tol=tol),
    }

    if svd_top_k > 0 and min(H.shape) > svd_top_k:
        # Rough SVD filtering: zero out the top-k singular values.
        from scipy.sparse.linalg import svds

        u, s, vt = svds(H.astype(np.float64), k=svd_top_k)
        low_rank = sp.csr_matrix((u * s) @ vt)
        H_filtered = (H - low_rank).tocsr()
        deg_row_f = np.asarray(H_filtered.sum(axis=1)).flatten()
        deg_col_f = np.asarray(H_filtered.sum(axis=0)).flatten()
        d_row_f = sp.diags(1.0 / np.sqrt(np.clip(np.abs(deg_row_f), 1e-8, None)))
        d_col_f = sp.diags(1.0 / np.clip(np.abs(deg_col_f), 1e-8, None))
        Theta_f = d_row_f @ H_filtered @ d_col_f @ H_filtered.T @ d_row_f
        L_f = sp.eye(Theta_f.shape[0]) - Theta_f
        result["lambda_max_theta_svd"] = compute_spectral_radius(L_f, tol=tol)
        result["spectral_gap_svd"] = compute_spectral_gap(L_f, tol=tol)

    return result


# ---------------------------------------------------------------------------
#  Tracker (training-loop integration)
# ---------------------------------------------------------------------------
@dataclass
class SpectralRadiusTracker:
    """Stateful tracker for training-loop integration.

    Typical usage::

        tracker = SpectralRadiusTracker(laplacian=L)
        for epoch in range(n_epochs):
            train_step(...)
            tracker.log_laplacian(epoch)
            if epoch % 10 == 0:
                tracker.log_layer_contraction(epoch, layer_embeddings)
        df = tracker.to_dataframe()
    """

    laplacian: Any | None = None
    tol: float = 1e-6
    laplacian_history: list[dict[str, float]] = field(default_factory=list)
    layer_history: list[dict[str, Any]] = field(default_factory=list)

    def update_laplacian(self, laplacian: Any) -> None:
        self.laplacian = laplacian

    def log_laplacian(self, epoch: int) -> dict[str, float]:
        if self.laplacian is None:
            raise ValueError(
                "SpectralRadiusTracker: Laplacian not set. "
                "Call .update_laplacian() or pass it to __init__."
            )
        row = {
            "epoch": int(epoch),
            "lambda_max_theta": compute_spectral_radius(self.laplacian, tol=self.tol),
            "spectral_gap": compute_spectral_gap(self.laplacian, tol=self.tol),
        }
        self.laplacian_history.append(row)
        return row

    def log_layer_contraction(
        self,
        epoch: int,
        layer_embeddings: list[np.ndarray],
    ) -> dict[str, Any]:
        radii = compute_layer_spectral_radius(layer_embeddings)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "layer_radii": radii,
            "mean_radius": float(np.nanmean(radii)) if radii else float("nan"),
        }
        self.layer_history.append(row)
        return row

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("to_dataframe() requires pandas") from exc
        lap = pd.DataFrame(self.laplacian_history)
        lay = pd.DataFrame(self.layer_history)
        return lap, lay

    def plot(self, path: str | None = None) -> None:
        """Minimal matplotlib visualisation (best-effort; no-op if missing)."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print(
                "[SpectralRadiusTracker] matplotlib not installed; "
                "skipping plot()"
            )
            return
        if not self.laplacian_history:
            return
        epochs = [r["epoch"] for r in self.laplacian_history]
        lam = [r["lambda_max_theta"] for r in self.laplacian_history]
        gap = [r["spectral_gap"] for r in self.laplacian_history]
        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        ax[0].plot(epochs, lam, marker="o")
        ax[0].set_ylabel("lambda_max(Theta)")
        ax[0].axhline(1.0, color="r", linestyle="--", alpha=0.6)
        ax[1].plot(epochs, gap, marker="s", color="orange")
        ax[1].set_ylabel("spectral gap lambda_2(L)")
        ax[1].set_xlabel("epoch")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
        else:
            plt.show()
        plt.close(fig)
