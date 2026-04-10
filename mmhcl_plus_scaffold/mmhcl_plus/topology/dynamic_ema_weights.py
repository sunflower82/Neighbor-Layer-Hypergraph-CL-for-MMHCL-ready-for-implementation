"""
Dynamic EMA weight matrix (W_ema) management — TEX Section 4, Step 1.

Design rationale (from TEX §4.2):
  The weight matrix W provides per-sample importance for the contrastive loss
  (Adaptive Sample Weighting, ASW, from NLGCL+).  Two failure modes exist:
    (a) A fully static W computed from raw features before training may diverge
        from the representation space the model gradually learns.
    (b) A continuous full-matrix update is O(N²) and cannot fit in VRAM.

  Solution — Two-tier W architecture:
    Tier 1  (static seed):  W is pre-computed once from raw multimodal features
                             via summed cosine similarity across modalities.
                             Cost: O(|M| · N² · d), done offline.
    Tier 2  (lazy EMA):     Every `update_interval` epochs, only the rows that
                             correspond to the current mini-batch are refreshed
                             using current student embeddings, blended with the
                             previous value via EMA:
                               W^(t)[idx] = α·W^(t-T)[idx] + (1-α)·sim(emb[idx])
                             The matrix lives on CPU (fp16) to avoid VRAM pressure;
                             batch rows are asynchronously transferred via
                             pin_memory + non_blocking=True.

Soft Topology-Aware Purification (TEX §3.4):
  After the initial W is built from cosine similarity, `apply_soft_topology()`
  applies continuous relaxation to mitigate popular-item bridges:
      w_ij = Softmax( Percentile(sim_ij) · Jaccard_proxy_ij · BridgePenalty_j )
  where:
    - Percentile: shifts scores below a quantile threshold to zero
    - Jaccard_proxy: we approximate J̃(N_i,N_j) ≈ sim_ij (cosine ≈ neighborhood overlap)
    - BridgePenalty_j = 1 / log(1 + degree_j)  (suppresses high-degree popular nodes)
  A full ANN/FAISS-based Jaccard computation can optionally replace the proxy
  by passing `neighbor_sets` (shape [N, k]) to `apply_soft_topology`.

Two standalone helper functions (update_ema_teacher, update_w_ema) are kept
for backward compatibility with existing call sites.

User-side WEMAManager:
  For users we have no raw feature files.  Instead, we derive user features as
  the mean-pooled item features across each user's training interactions.
  `build_user_features_from_interactions()` performs this aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .faiss_index import ANNIndex


# ---------------------------------------------------------------------------
# Standalone helpers (backward-compatible)
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema_teacher(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float = 0.99,
) -> None:
    """Polyak / exponential moving-average update of teacher parameters.

    ξ ← β·ξ + (1-β)·θ     (Algorithm 1, Step 11 in TEX §4.4)
    """
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


@torch.no_grad()
def update_w_ema(
    prev_w: torch.Tensor,
    embed_sim: torch.Tensor,
    alpha: float = 0.9,
) -> torch.Tensor:
    """Element-wise EMA blend of two similarity matrices."""
    return alpha * prev_w + (1.0 - alpha) * embed_sim


# ---------------------------------------------------------------------------
# User-feature builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_user_features_from_interactions(
    train_items: dict[int, list[int]],
    item_features_list: list[torch.Tensor],
    n_users: int,
    n_items: int,
) -> list[torch.Tensor]:
    """
    Derive per-user multimodal features by mean-pooling the item features
    of each user's training interactions (NLGCL+ Eq. 10).

    Parameters
    ----------
    train_items : dict {user_id: [item_ids]}
        Training interaction set (from data_generator.train_items).
    item_features_list : list of Tensor [n_items_file, feat_dim]
        One tensor per modality (e.g. image, text).
    n_users : int
    n_items : int

    Returns
    -------
    user_features_list : list of Tensor [n_users, feat_dim]
        One aggregated user-feature tensor per modality.
    """
    user_features_list: list[torch.Tensor] = []

    for item_feats in item_features_list:
        feat_dim = item_feats.size(-1)
        # Clip item_feats rows to n_items (feature files may have extra rows)
        item_feats_clipped = item_feats[:n_items].float().cpu()

        user_feats = torch.zeros(n_users, feat_dim, dtype=torch.float32)
        for uid, iids in train_items.items():
            if uid >= n_users:
                continue
            valid = [i for i in iids if i < n_items]
            if valid:
                user_feats[uid] = item_feats_clipped[valid].mean(dim=0)

        user_features_list.append(user_feats)

    return user_features_list


# ---------------------------------------------------------------------------
# WEMAManager — Two-tier W architecture + Soft Topology Purification
# ---------------------------------------------------------------------------

class WEMAManager:
    """
    Manages the dynamic ASW weight matrix W_ema.

    Lifecycle
    ---------
    1.  Call :meth:`precompute_from_raw` once **before** training begins.
        Pass a list of raw feature tensors (one per modality).
    2.  Optionally call :meth:`apply_soft_topology` immediately after step 1
        to enable the soft topology-aware purification (TEX §3.4).
    3.  During each training step, call :meth:`get_batch_weights` to obtain
        the W rows for the current batch (returned on the target device).
    4.  At the end of each epoch, call :meth:`step_update` to blend
        current embeddings into W for the batch nodes (runs only every
        `update_interval` epochs).

    Memory layout
    -------------
    W is stored as a fp16 CPU tensor of shape [n_nodes, n_nodes].
    Only the requested rows are moved to GPU per step (lazy fetch).
    """

    def __init__(
        self,
        n_nodes: int,
        alpha: float = 0.9,
        update_interval: int = 5,
    ) -> None:
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.update_interval = update_interval
        # Initialise to uniform (identity-like) similarity — zero off-diag
        self.W: torch.Tensor = torch.zeros(n_nodes, n_nodes, dtype=torch.float16)

    # ------------------------------------------------------------------
    # Tier 1 — offline pre-computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precompute_from_raw(self, raw_features_list: list[torch.Tensor]) -> None:
        """
        Compute W from raw multimodal features (NLGCL+ Eq. 11).

            W[s, t] = (1/|M|) · Σ_m  cos_sim(x^m_s, x^m_t)

        Args:
            raw_features_list: List of [n_nodes, feat_dim] float tensors.
                                Each entry corresponds to one modality.
        """
        if not raw_features_list:
            return

        acc = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float32)
        for feats in raw_features_list:
            feats_f = feats.float().cpu()
            normed = F.normalize(feats_f, p=2, dim=-1)
            acc += normed @ normed.T  # [N, N]

        acc /= max(len(raw_features_list), 1)
        # Clamp to [0, 1] (cosine similarity is in [-1, 1]; negatives are noise)
        acc.clamp_(min=0.0, max=1.0)
        self.W = acc.half()

    # ------------------------------------------------------------------
    # Soft Topology-Aware Purification (TEX §3.4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_soft_topology(
        self,
        percentile: float = 0.8,
        temp: float = 0.2,
        degrees: torch.Tensor | None = None,
        neighbor_sets: torch.Tensor | None = None,
    ) -> None:
        """
        Transform W in-place from raw cosine similarity to soft topology weights.

        Implements the formula from TEX §3.4:
            w_ij = Softmax( Percentile(ANN(i,j)) · J̃(N_i,N_j) · BridgePenalty_j )

        where:
          - Percentile(sim_ij): shifts scores below the `percentile` quantile to 0,
                                 providing continuous relaxation of a hard threshold.
          - J̃(N_i, N_j):        sampled Jaccard coefficient. If `neighbor_sets` is
                                 provided (shape [N, k], from an ANN search), the true
                                 Jaccard is computed.  Otherwise, cosine sim is used as
                                 a fast proxy (both measure neighborhood overlap).
          - BridgePenalty_j:    1 / log(1 + degree_j) — suppresses high-degree popular
                                 nodes that act as "popular-item bridges".

        Parameters
        ----------
        percentile : float in (0, 1)
            Quantile threshold; similarities below this are zeroed out.
        temp : float
            Softmax temperature for the final row-wise normalisation.
        degrees : Tensor [n_nodes] | None
            Node degrees (e.g. from hypergraph adjacency).  If None, degree
            is estimated from the column sums of W.
        neighbor_sets : Tensor [n_nodes, k] int64 | None
            ANN neighbor indices per node.  If provided, enables exact sampled
            Jaccard computation.  If None, cosine similarity is used as proxy.
        """
        W_f = self.W.float()  # [N, N] working copy

        # 1. Percentile-based continuous relaxation
        thresh = torch.quantile(W_f, percentile, dim=-1, keepdim=True)  # [N, 1]
        W_shifted = (W_f - thresh).clamp(min=0.0)                        # [N, N]

        # 2. Jaccard-like term
        if neighbor_sets is not None and neighbor_sets.shape[0] == self.n_nodes:
            # Exact sampled Jaccard between k-NN sets
            W_jaccard = _batch_jaccard(neighbor_sets, self.n_nodes)      # [N, N]
        else:
            # Fast proxy: cosine similarity already captures neighborhood overlap
            W_jaccard = W_f

        # 3. Bridge penalty
        if degrees is not None:
            deg = degrees.float().cpu().clamp(min=0.0)
        else:
            deg = W_f.sum(dim=0)   # approximate degree from column sums
        bridge_penalty = 1.0 / (1.0 + deg.log1p()).unsqueeze(0)  # [1, N]

        # 4. Combine and normalize per row
        W_raw = W_shifted * W_jaccard * bridge_penalty           # [N, N]
        W_soft = torch.softmax(W_raw / max(temp, 1e-6), dim=-1)  # [N, N]

        self.W = W_soft.half()

    # ------------------------------------------------------------------
    # Lazy row fetch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_batch_weights(
        self,
        idx: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Return W[idx] on `device`, transferred asynchronously (non_blocking).

        Args:
            idx:    1-D LongTensor of node indices for the current batch.
            device: Target device (e.g. cuda).  If None, returns CPU tensor.

        Returns:
            Float32 tensor of shape [len(idx), n_nodes].
        """
        rows = self.W[idx.cpu()].float()  # [B, N], fp32
        if device is not None and str(device) != 'cpu':
            rows = rows.pin_memory()
            rows = rows.to(device, non_blocking=True)
        return rows

    # ------------------------------------------------------------------
    # Tier 2 — lazy EMA update (batch-level)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_update(
        self,
        embeddings: torch.Tensor,
        idx: torch.Tensor,
        epoch: int,
    ) -> None:
        """
        Periodically refresh W[idx] rows using current student embeddings.

        Only executes when ``epoch % update_interval == 0``.
        The update blends the new batch-vs-batch cosine similarity with the
        stored W rows (batch-vs-batch block only):

            W^(t)[idx, idx] = α · W^(t-T)[idx, idx] + (1-α) · sim(emb[idx])

        Args:
            embeddings: [B, dim] student embeddings for the batch nodes.
            idx:        Batch node indices (1-D LongTensor).
            epoch:      Current training epoch (0-indexed).
        """
        if epoch % self.update_interval != 0:
            return

        emb_cpu = embeddings.detach().float().cpu()
        normed = F.normalize(emb_cpu, p=2, dim=-1)

        # Batch × batch cosine similarity block
        sim_block = normed @ normed.T  # [B, B]
        sim_block.clamp_(min=0.0, max=1.0)

        idx_cpu = idx.cpu()
        # Extract existing block
        old_block = self.W[idx_cpu.unsqueeze(1), idx_cpu.unsqueeze(0)].float()

        # EMA blend
        new_block = self.alpha * old_block + (1.0 - self.alpha) * sim_block

        # Write back in fp16
        self.W[idx_cpu.unsqueeze(1), idx_cpu.unsqueeze(0)] = new_block.half()


# ---------------------------------------------------------------------------
# Internal helper — batch Jaccard computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batch_jaccard(neighbor_sets: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    Compute a dense Jaccard similarity matrix from ANN neighbor sets.

    Parameters
    ----------
    neighbor_sets : int64 Tensor [N, k]
        ANN neighbor indices per node (from ANNIndex.search).
    n_nodes : int

    Returns
    -------
    J : float32 Tensor [N, N]
        J[i, j] = |N_i ∩ N_j| / |N_i ∪ N_j|   (sampled Jaccard).

    Note: For large N this is O(N² · k); only call for N ≤ ~10 k.
    """
    N, k = neighbor_sets.shape
    # Build binary membership matrix: M[i, v] = 1 if v in N_i
    M = torch.zeros(N, n_nodes, dtype=torch.float32)
    for i in range(N):
        M[i, neighbor_sets[i]] = 1.0

    dot = M @ M.T                        # [N, N]  intersection size
    row_sum = M.sum(dim=1, keepdim=True) # [N, 1]  |N_i|
    union = row_sum + row_sum.T - dot    # [N, N]  union size
    return dot / union.clamp_min(1.0)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_item_wema(
    n_items: int,
    item_feat_paths: list[str],
    alpha: float = 0.9,
    update_interval: int = 5,
    percentile: float = 0.8,
    purification_temp: float = 0.2,
    item_degrees: torch.Tensor | None = None,
    logger=None,
) -> WEMAManager | None:
    """
    Build and initialise a WEMAManager for the item side.

    Returns None if no feature files are found.
    """
    import numpy as np
    import os

    raw_feats: list[torch.Tensor] = []
    for fp in item_feat_paths:
        if not os.path.exists(fp):
            continue
        arr = np.load(fp).astype(np.float32)
        raw_feats.append(torch.from_numpy(arr)[:n_items])
        if logger:
            logger.logging(
                "[WEMAManager] Loaded item features: %s  shape=%s" % (fp, arr.shape)
            )

    if not raw_feats:
        return None

    mgr = WEMAManager(n_nodes=n_items, alpha=alpha, update_interval=update_interval)
    mgr.precompute_from_raw(raw_feats)
    mgr.apply_soft_topology(
        percentile=percentile,
        temp=purification_temp,
        degrees=item_degrees,
    )
    if logger:
        logger.logging(
            "[WEMAManager] Item W_ema ready: %d items, %d modalities, "
            "alpha=%.2f, update_interval=%d, purification_percentile=%.2f"
            % (n_items, len(raw_feats), alpha, update_interval, percentile)
        )
    return mgr


def build_user_wema(
    n_users: int,
    n_items: int,
    train_items: dict[int, list[int]],
    item_feat_paths: list[str],
    alpha: float = 0.9,
    update_interval: int = 5,
    percentile: float = 0.8,
    purification_temp: float = 0.2,
    user_degrees: torch.Tensor | None = None,
    logger=None,
) -> WEMAManager | None:
    """
    Build and initialise a WEMAManager for the user side.

    User features are derived by mean-pooling item features over each user's
    training interactions (NLGCL+ Eq. 10).

    Returns None if no feature files are found.
    """
    import numpy as np
    import os

    item_feat_list: list[torch.Tensor] = []
    for fp in item_feat_paths:
        if not os.path.exists(fp):
            continue
        arr = np.load(fp).astype(np.float32)
        item_feat_list.append(torch.from_numpy(arr))
        if logger:
            logger.logging(
                "[WEMAManager-user] Loaded item features for user pooling: "
                "%s  shape=%s" % (fp, arr.shape)
            )

    if not item_feat_list:
        return None

    user_feats = build_user_features_from_interactions(
        train_items, item_feat_list, n_users, n_items
    )

    mgr = WEMAManager(n_nodes=n_users, alpha=alpha, update_interval=update_interval)
    mgr.precompute_from_raw(user_feats)
    mgr.apply_soft_topology(
        percentile=percentile,
        temp=purification_temp,
        degrees=user_degrees,
    )
    if logger:
        logger.logging(
            "[WEMAManager-user] User W_ema ready: %d users, alpha=%.2f, "
            "update_interval=%d, purification_percentile=%.2f"
            % (n_users, alpha, update_interval, percentile)
        )
    return mgr
