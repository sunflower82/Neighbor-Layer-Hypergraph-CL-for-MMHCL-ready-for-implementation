from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _maybe_torch_compile(fn):
    if os.environ.get("MMHCL_DISABLE_TORCH_COMPILE", "").strip():
        return fn
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return fn
    try:
        return compile_fn(fn, dynamic=True, fullgraph=False)
    except Exception:
        return fn


def _soft_byol_alignment_impl(
    online: torch.Tensor,
    target: torch.Tensor,
    soft_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    online = F.normalize(online, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    sim = (online * target).sum(dim=-1)
    loss = 2.0 - 2.0 * sim
    if soft_weights is not None:
        loss = loss * soft_weights
    return loss.mean()


soft_byol_alignment = _maybe_torch_compile(_soft_byol_alignment_impl)
