"""
Layer-wise encoder with profiling-guided checkpointing — TEX Section 4, Step 2.

Design rationale (from TEX §4.3, Table 2 and pseudocode):

  The two-stage CL architecture requires:
    (a) Shallow layers (l < g, where g ≤ 2) MUST be cached for neighbor-layer
        pair construction.  Checkpointing these layers would discard their
        intermediate activations and prevent forward-pass caching.
    (b) Deep layers (l ≥ g) MAY be checkpointed to bound peak VRAM.

  From the profiling table (TEX §4.3):
    - No checkpoint   → OOM on 24 GB for L=3, B=1024.
    - Full checkpoint → lowest VRAM but +45 % epoch time (excessive recompute).
    - Guided approach → medium VRAM (~50 %), only +12 % time.

  The `checkpoint_fn` callable is injected by the caller (TwoStageTrainer),
  which decides which layer IDs to checkpoint based on the config.  This keeps
  the encoder itself policy-agnostic.

Usage example (inside TwoStageTrainer):
    def _ckpt(module, x, lid):
        return profiling_guided_checkpoint(module, x, lid, cfg.system.checkpoint_layers)
    final, all_layers = encoder(h0, checkpoint_fn=_ckpt)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerwiseEncoder(nn.Module):
    """
    Runs a stack of graph-convolution layers and collects per-layer outputs.

    Args:
        layers:          ModuleList of graph-conv / hypergraph-conv layers.
                         Each layer must accept a single tensor and return a tensor.
        max_g_layers:    Maximum number of *shallow* layers (0-indexed) whose
                         outputs are cached for neighbor-layer CL pairs.
                         Outputs for layers with index in [0, max_g_layers] are
                         always stored (including the input embedding at index 0).
        use_checkpoint:  Global flag.  If False, all layers run without
                         checkpointing regardless of the checkpoint_fn argument.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        max_g_layers: int = 2,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.max_g_layers = max_g_layers
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        h0: torch.Tensor,
        checkpoint_fn=None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            h0:             Initial embedding [N, d].
            checkpoint_fn:  Optional callable(module, x, layer_id) → tensor.
                            If None or use_checkpoint=False, all layers run
                            normally (no recomputation).

        Returns:
            (final_h, cached_layers) where:
              - final_h:       Last layer output [N, d].
              - cached_layers: List of tensors for layers 0 .. max_g_layers
                               (inclusive).  Layer 0 is h0 itself.
                               Used to build neighbor-layer CL pairs.
        """
        h = h0
        # Always cache h0 — forms the anchor for the first CL pair (l=0 → l=1)
        cached: list[torch.Tensor] = [h0]

        for layer_id, layer in enumerate(self.layers):
            # Shallow layers (l < max_g_layers) must NOT be checkpointed so that
            # PyTorch retains their activations in the computation graph for the
            # neighbor-layer CL loss backward pass.
            use_ckpt = (
                self.use_checkpoint
                and checkpoint_fn is not None
                and layer_id >= self.max_g_layers  # only checkpoint deep layers
            )

            if use_ckpt:
                h = checkpoint_fn(layer, h, layer_id)
            else:
                h = layer(h)

            # Cache outputs for layers 0 .. max_g_layers (so pairs go up to g)
            if layer_id < self.max_g_layers:
                cached.append(h)

        return h, cached
