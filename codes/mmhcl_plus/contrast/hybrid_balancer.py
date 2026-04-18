"""
Hybrid Loss Balancer --- Revision 5.2 (ablation-aware).

Combines Uncertainty Weighting (Kendall et al., CVPR 2018) for robust
early-stage initialization with GradNorm (Chen et al., ICML 2018) for
precise late-stage gradient equalization across the 5 core MMHCL+
objectives.

Design (Rev5.2 §3.5):
  Phase 1 (epoch < transition_epoch)           : Uncertainty Weighting
  Phase 2 (epoch >= transition_epoch)          : GradNorm takes over
  Smooth linear blend over ``blend_epochs`` to avoid discontinuity.

In Rev5.2 the balancer is additionally required to support the
C-series ablations from ``mmhcl_plus_ablation_guide_full_translation``:

  * ``mode="hybrid"``      : default, Uncertainty -> GradNorm transition.
  * ``mode="uncertainty"`` : Uncertainty Weighting only (C1_uncertainty).
  * ``mode="gradnorm"``    : GradNorm only (C2_gradnorm).
  * ``mode="fixed"``       : w_k = 1.0 for every task (C3_fixed).

The 5 default tasks are BPR, u2u (VICReg), i2i (InfoNCE), align
(Soft BYOL), and Dirichlet.  Ego-Final is re-introduced as a 6th task
only for the A7_ego_final ablation variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_VALID_MODES: tuple[str, ...] = ("hybrid", "uncertainty", "gradnorm", "fixed")


class HybridLossBalancer(nn.Module):
    """Ablation-aware loss balancer for MMHCL+.

    Parameters
    ----------
    num_tasks : int
        Number of loss components.  Default 5 for Rev5.2 (``BPR, u2u,
        i2i, align, dirichlet``); 6 when the A7 ego-final anchor is
        re-enabled; smaller values for ablations that remove a branch.
    alpha : float
        GradNorm restoring-force exponent (default 1.5).
    transition_epoch : int
        Epoch at which GradNorm starts to take over (hybrid mode only).
    blend_epochs : int
        Number of epochs over which the uncertainty -> gradnorm blend
        is interpolated linearly.
    mode : str
        One of ``{"hybrid", "uncertainty", "gradnorm", "fixed"}``.
    """

    def __init__(
        self,
        num_tasks: int = 5,
        alpha: float = 1.5,
        transition_epoch: int = 40,
        blend_epochs: int = 20,
        mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid balancer mode '{mode}'. Expected one of {_VALID_MODES}."
            )
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.transition_epoch = transition_epoch
        self.blend_epochs = blend_epochs
        self.mode = mode

        # Uncertainty Weighting parameters (Kendall et al., 2018)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # GradNorm parameters (Chen et al., 2018)
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        self.register_buffer("l0", torch.zeros(num_tasks))
        self.register_buffer("l0_initialized", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    #  Sub-objectives
    # ------------------------------------------------------------------
    def _uncertainty_forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros(1, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total.squeeze()

    def _fixed_forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(losses).sum()

    def _gradnorm_forward(
        self,
        losses: list[torch.Tensor],
        shared_params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        device = losses[0].device
        task_losses = torch.stack(losses)

        if not self.l0_initialized.item():
            self.l0 = task_losses.detach().clone()
            self.l0_initialized.fill_(True)

        w = self.task_weights.clamp(min=1e-3)
        w = w * self.num_tasks / w.sum()
        total = (w * task_losses).sum()

        if shared_params is None or not self.training:
            return total

        g_tilde: list[torch.Tensor] = []
        for loss_i in losses:
            try:
                grads = torch.autograd.grad(
                    loss_i, shared_params,
                    retain_graph=True, create_graph=False, allow_unused=True,
                )
                gn2 = sum((g.detach() ** 2).sum() for g in grads if g is not None)
                g_tilde.append(gn2.sqrt().to(device))
            except Exception:
                g_tilde.append(task_losses.new_zeros(()))

        G_tilde = torch.stack(g_tilde).detach()
        G = w * G_tilde
        G_bar = G.detach().mean()
        r = task_losses.detach() / (self.l0.to(device) + 1e-8)
        r_tilde = r / r.mean().clamp_min(1e-8)
        target = (G_bar * r_tilde.pow(self.alpha)).detach()
        L_gn = torch.abs(G - target).sum()

        return total + L_gn

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        losses: list[torch.Tensor],
        epoch: int = 0,
        shared_params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Aggregate ``losses`` according to the configured ``mode``."""
        if self.mode == "fixed":
            return self._fixed_forward(losses)
        if self.mode == "uncertainty":
            return self._uncertainty_forward(losses)
        if self.mode == "gradnorm":
            return self._gradnorm_forward(losses, shared_params)

        # hybrid: uncertainty -> gradnorm with a linear blend
        if epoch < self.transition_epoch:
            return self._uncertainty_forward(losses)
        if epoch < self.transition_epoch + self.blend_epochs:
            blend_ratio = (epoch - self.transition_epoch) / float(self.blend_epochs)
            L_unc = self._uncertainty_forward(losses)
            L_gn = self._gradnorm_forward(losses, shared_params)
            return (1.0 - blend_ratio) * L_unc + blend_ratio * L_gn
        return self._gradnorm_forward(losses, shared_params)

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return current effective task weights (for W&B logging)."""
        unc_w = torch.exp(-self.log_vars).detach()
        gn_w = self.task_weights.clamp(min=1e-3).detach()
        gn_w = gn_w * self.num_tasks / gn_w.sum()
        return {"uncertainty": unc_w, "gradnorm": gn_w}

    def get_phase(self, epoch: int) -> str:
        """Return a short human-readable label for the current phase."""
        if self.mode != "hybrid":
            return self.mode
        if epoch < self.transition_epoch:
            return "uncertainty"
        if epoch < self.transition_epoch + self.blend_epochs:
            ratio = (epoch - self.transition_epoch) / float(self.blend_epochs)
            return f"blend({ratio:.2f})"
        return "gradnorm"
