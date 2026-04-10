"""
Profiling-guided activation checkpointing (Section 4, Step 2).

Design rationale (from TEX §4.3):
  - 'No Checkpoint'     → OOM on 24 GB GPU for L=3, batch=1024.
  - 'Naive Checkpoint'  → lowest VRAM (~35%) but +45% time due to full recomputation.
  - 'Profiling-Guided'  → medium VRAM (~50%), only +12% time: checkpoint deep layers,
                           cache shallow layers (g ≤ 2) where neighbor-layer pairs are formed.

This module provides a single helper that honours an explicit per-layer allow-list
instead of checkpointing every layer unconditionally.

PyTorch ≥ 2.0 deprecates the old reentrant mode; we always pass use_reentrant=False.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as ckpt


def profiling_guided_checkpoint(
    module: torch.nn.Module,
    x: torch.Tensor,
    layer_id: int,
    checkpoint_layers: list[int],
) -> torch.Tensor:
    """
    Run `module(x)` with or without gradient checkpointing.

    Args:
        module:            The layer to call.
        x:                 Input tensor.
        layer_id:          Zero-based index of this layer within the encoder.
        checkpoint_layers: List of layer indices where checkpointing is applied.
                           Typically the deeper layers (e.g. [1, 2]) while the
                           shallow ones (l < g) are run normally so their outputs
                           can be cached for the neighbor-layer CL pairs.

    Returns:
        Output tensor from `module(x)`.
    """
    if layer_id in checkpoint_layers:
        # use_reentrant=False: non-reentrant implementation (required for PyTorch ≥ 2.0).
        # Passing the module directly avoids the lambda-closure memory leak present
        # in the original scaffold.
        return ckpt.checkpoint(module, x, use_reentrant=False)
    return module(x)
