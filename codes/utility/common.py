"""
common.py — Shared utilities used by both main.py and main_mmhcl_plus.py.

Canonical implementations of helpers that were previously duplicated across
multiple training scripts.  All training entry points should import from here
rather than defining their own copies.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import scipy.sparse as sp
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + all GPUs)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_experiment_paths(a: argparse.Namespace) -> tuple[str, str, str]:
    """Compute experiment directory paths (path_name, path, record_path)."""
    pn: str = (
        f"uu_ii={a.User_layers}_{a.Item_layers}"
        f"_{a.user_loss_ratio}_{a.item_loss_ratio}"
        f"_topk={a.topk}_t={a.temperature}"
        f"_regs={a.regs}_dim={a.embed_size}"
        f"_seed={a.seed}_{a.ablation_target}"
    )
    p: str = f"../{a.dataset}/{pn}/"
    rp: str = f"../{a.dataset}/MM/"
    return pn, p, rp


def lr_decay_schedule(epoch: int, base: float = 0.96, divisor: float = 50) -> float:
    """Smooth exponential LR decay: lr(epoch) = base^(epoch / divisor)."""
    return base ** (epoch / divisor)


def bpr_loss(
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    batch_size: int,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    BPR pairwise ranking loss with L2 embedding regularisation.

    Loss = −mean(log σ(s_pos − s_neg)) + decay * L2_reg
    """
    pos_scores = (users * pos_items).sum(dim=1)
    neg_scores = (users * neg_items).sum(dim=1)

    reg = (users**2).sum() + (pos_items**2).sum() + (neg_items**2).sum()
    reg = reg / (2.0 * batch_size)

    mf_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    emb_loss = decay * reg
    return mf_loss, emb_loss, 0.0


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse COO tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices: torch.Tensor = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values: torch.Tensor = torch.from_numpy(sparse_mx.data)
    shape: torch.Size = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
