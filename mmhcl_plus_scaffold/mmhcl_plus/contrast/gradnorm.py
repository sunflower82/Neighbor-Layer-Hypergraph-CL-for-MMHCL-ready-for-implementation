"""
GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., ICML 2018).

Reference: https://arxiv.org/abs/1711.02257

Design (TEX §4.4):
  Given n_tasks losses {L_i} and shared backbone parameters W, GradNorm:
    1.  Maintains learnable task weights {w_i} (clamped positive).
    2.  Each forward pass computes per-task gradient norms:
            G̃_i  = ‖ ∇_W L_i ‖₂          (gradient norm of *unweighted* loss)
            G_i   = w_i · G̃_i              (gradient norm of *weighted* loss)
    3.  Computes a target gradient norm for task i:
            Ḡ    = mean(G_i)  (detached)
            r_i   = L_i(t) / L_i(0)        (relative training rate)
            r̃_i  = r_i / mean(r_i)         (normalized rate)
            T_i   = Ḡ · r̃_i^α             (target gradient norm)
    4.  GradNorm loss: L_GN = Σ_i |G_i − T_i|₁
        Since G̃_i is detached, ∂L_GN/∂w_i = G̃_i · sign(G_i − T_i).
        This means L_GN only updates task weights, not backbone parameters.
    5.  Main loss: L_total = Σ_i w_i · L_i
        Combined (total + L_GN).backward() updates:
          • backbone parameters θ  via L_total  (grad from w_i·L_i w.r.t. θ)
          • task weights w_i        via both L_total (grad = L_i)
                                       and L_GN    (grad = G̃_i·sign(…))

Backward compatibility:
  The public API is unchanged: combine(losses, shared_params=None).
  Passing shared_params=None falls back to the previous softmax-weighted combine,
  making this a safe drop-in replacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradNormLossBalancer(nn.Module):
    """
    Real GradNorm dynamic loss balancer.

    Parameters
    ----------
    n_tasks : int
        Number of task losses to balance.
    alpha : float
        GradNorm restoring force exponent. Higher α enforces stricter
        gradient norm equalization across tasks.
    init_weights : list[float] | None
        Initial task weight values.  Defaults to all-ones.

    Usage
    -----
    >>> balancer = GradNormLossBalancer(n_tasks=5, alpha=1.5).cuda()
    >>> # shared_params = list(model.embedding_tables())
    >>> total, w = balancer.combine(
    ...     [loss_bpr, loss_u2u, loss_i2i, loss_align, loss_dir],
    ...     shared_params=shared_params,
    ... )
    >>> total.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        n_tasks: int = 5,
        alpha: float = 1.5,
        init_weights: list[float] | None = None,
    ) -> None:
        super().__init__()

        if init_weights is not None:
            init = torch.tensor(init_weights, dtype=torch.float32)
        else:
            init = torch.ones(n_tasks, dtype=torch.float32)

        # Raw (unconstrained) task weights; we clamp to positive in forward
        self.weights = nn.Parameter(init)
        self.alpha = alpha
        self.n_tasks = n_tasks

        # Initial per-task loss values (set on first call; used for relative rate r_i)
        self.register_buffer("l0", torch.zeros(n_tasks, dtype=torch.float32))
        self.register_buffer("l0_initialized", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def normalized_weights(self) -> torch.Tensor:
        """
        Clamp raw weights to (0, ∞) and renormalize to sum = n_tasks.
        This keeps weights interpretable as multipliers with mean = 1.
        """
        w = self.weights.clamp(min=1e-3)
        return w * self.n_tasks / w.sum()

    def _init_l0(self, loss_vec: torch.Tensor) -> None:
        """Record initial losses on the very first call."""
        if not self.l0_initialized.item():
            self.l0 = loss_vec.detach().clone()
            self.l0_initialized.fill_(True)

    # ------------------------------------------------------------------
    # Core: combine
    # ------------------------------------------------------------------

    def combine(
        self,
        losses: list[torch.Tensor],
        shared_params: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the combined weighted loss (optionally with GradNorm correction).

        Parameters
        ----------
        losses : list of n_tasks scalar Tensors
            Per-task losses (must all be on the same device).
        shared_params : list of Tensor | None
            Backbone parameters used to compute per-task gradient norms.
            If None, falls back to a simple learned-weight combination
            (backward-compatible behaviour).

        Returns
        -------
        total : scalar Tensor
            The combined loss ready for .backward().  When shared_params is
            provided, this already includes the GradNorm correction term so
            that a single .backward() call updates both θ and w_i correctly.
        weights : Tensor [n_tasks]
            Detached normalized task weights (for logging only).
        """
        device = losses[0].device
        task_losses = torch.stack(losses)  # [n_tasks]

        self._init_l0(task_losses)

        w = self.normalized_weights()  # [n_tasks], kept in graph

        # Weighted sum — backprop reaches both θ (through task_losses) and w
        total = (w * task_losses).sum()

        if shared_params is None or not self.training:
            return total, w.detach()

        # ── Real GradNorm ──────────────────────────────────────────────────

        # Step 1: compute per-task gradient norms of *unweighted* losses
        #         w.r.t. shared backbone parameters.
        g_tilde: list[torch.Tensor] = []
        for loss_i in losses:
            try:
                grads = torch.autograd.grad(
                    loss_i,
                    shared_params,
                    retain_graph=True,
                    create_graph=False,   # G̃_i detached from w → only w gets grad
                    allow_unused=True,
                )
                gn2 = sum(
                    (g.detach() ** 2).sum()
                    for g in grads
                    if g is not None
                )
                g_tilde.append(gn2.sqrt().to(device))
            except Exception:
                # Safety fallback (e.g. unused task or graph already freed)
                g_tilde.append(task_losses.new_zeros(()))

        G_tilde = torch.stack(g_tilde).detach()   # [n_tasks], no grad to model

        # Step 2: weighted gradient norms — G_i = w_i · G̃_i (w_i in graph)
        G = w * G_tilde   # [n_tasks]

        # Step 3: target gradient norms
        G_bar = G.detach().mean()                          # scalar, detached
        r = task_losses.detach() / (self.l0.to(device) + 1e-8)
        r_tilde = r / r.mean().clamp_min(1e-8)            # [n_tasks], normalized rates
        target = (G_bar * r_tilde.pow(self.alpha)).detach()  # [n_tasks]

        # Step 4: GradNorm loss — L1 between weighted grad norms and targets.
        #         Gradient w.r.t. w_i: G̃_i · sign(G_i − target_i)
        #         Gradient w.r.t. θ: 0 (because G̃_i is detached)
        L_gn = torch.abs(G - target).sum()

        # Combined: total + L_GN
        #   θ-gradient   comes only from total   (through w_i · L_i → θ)
        #   w-gradient   comes from total (L_i)  AND L_GN (G̃_i · sign(…))
        return total + L_gn, w.detach()
