"""
FAISS-Guided Hard Negative Mining --- Revision 5.1

TEX Rev5.1 Section 2.4:
    Re-purposes FAISS LSH indices to sample structure-aware hard negatives.
    Modality-similar items lacking historical co-occurrence serve as potent
    negative signals, sharpening the NDCG decision boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def mine_hard_negatives_faiss(
    query_embs: torch.Tensor,
    all_embs: torch.Tensor,
    interaction_mask: torch.Tensor | None = None,
    n_hard_neg: int = 10,
    pool_k: int = 64,
) -> torch.Tensor:
    """
    Mine hard negatives using FAISS-style nearest neighbor search.

    For each query, find pool_k nearest neighbors in embedding space,
    then filter out items the user has interacted with (using interaction_mask).
    The remaining modality-similar but non-interacted items are hard negatives.

    Args:
        query_embs:      [B, d] query embeddings (items in the current batch).
        all_embs:        [N, d] all item embeddings.
        interaction_mask: Optional [B, N] binary mask where 1 = interacted
                         (these are excluded from hard negatives).
        n_hard_neg:      Number of hard negatives to return per query.
        pool_k:          Size of the initial candidate pool from ANN search.

    Returns:
        hard_negatives: [B, n_hard_neg, d] hard negative embeddings.
    """
    B, d = query_embs.shape
    device = query_embs.device

    # L2 normalize for cosine similarity search
    q_norm = F.normalize(query_embs.detach(), p=2, dim=-1)
    all_norm = F.normalize(all_embs.detach(), p=2, dim=-1)

    # Compute similarity scores (dot product = cosine sim after normalization)
    sim = q_norm @ all_norm.T  # [B, N]

    # Mask out self-similarities and interacted items
    # Set interacted items to -inf so they won't be selected
    if interaction_mask is not None:
        sim = sim.masked_fill(interaction_mask.bool().to(device), float("-inf"))

    # Get top pool_k candidates per query
    _, topk_indices = sim.topk(pool_k, dim=-1)  # [B, pool_k]

    # Select n_hard_neg from the pool (take the closest non-interacted items)
    hard_neg_indices = topk_indices[:, :n_hard_neg]  # [B, n_hard_neg]

    # Gather hard negative embeddings
    hard_negatives = all_embs[hard_neg_indices]  # [B, n_hard_neg, d]

    return hard_negatives


def build_interaction_mask(
    batch_items: torch.Tensor,
    train_items: dict[int, list[int]],
    n_items: int,
    batch_users: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    Build a binary interaction mask for hard negative filtering.

    Args:
        batch_items: [B] item indices in the current batch.
        train_items: Dict mapping user_id -> list of interacted item_ids.
        n_items:     Total number of items.
        batch_users: [B] user indices corresponding to each batch item.

    Returns:
        mask: [B, n_items] binary tensor where 1 = interacted (to exclude).
              Returns None if batch_users is not provided.
    """
    if batch_users is None:
        return None

    B = len(batch_items)
    mask = torch.zeros(B, n_items, dtype=torch.bool)

    user_list = batch_users.cpu().tolist() if isinstance(batch_users, torch.Tensor) else batch_users
    for i, uid in enumerate(user_list):
        interacted = train_items.get(uid, [])
        if interacted:
            mask[i, interacted] = True

    return mask
