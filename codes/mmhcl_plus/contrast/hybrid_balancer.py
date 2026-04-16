"""
Hybrid Loss Balancer — Revision 5.2

Combines Uncertainty Weighting (Kendall et al., CVPR 2018) for robust early-stage
initialization with GradNorm (Chen et al., ICML 2018) for precise late-stage
gradient equalization across 5 core objectives.

Design (Rev5.2 §3.5):
  Phase 1 (epoch < transition_epoch): Uncertainty Weighting only
  Phase 2 (epoch >= transition_epoch): GradNorm takes over
  Smooth linear blend over blend_epochs to avoid discontinuity

5 tasks: BPR, u2u (VICReg), i2i (InfoNCE), align (Soft BYOL), dirichlet.
Ego-Final Anchor is eliminated in Rev5.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridLossBalancer(nn.Module):
    """
    Hybrid Loss Balancer: Uncertainty Weighting → GradNorm transition.

    Parameters
    ----------
    num_tasks : int
        Number of loss components (default: 5 for Rev5.2).
    alpha : float
        GradNorm restoring force exponent (default: 1.5).
    transition_epoch : int
        Epoch at which GradNorm begins to activate (default: 40).
    blend_epochs : int
        Number of epochs to linearly blend from Uncertainty to GradNorm (default: 20).
    """

    def __init__(
        self,
        num_tasks: int = 5,
        alpha: float = 1.5,
        transition_epoch: int = 40,
        blend_epochs: int = 20,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.transition_epoch = transition_epoch
        self.blend_epochs = blend_epochs

        # Uncertainty Weighting parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # GradNorm parameters
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        self.register_buffer("l0", torch.zeros(num_tasks))
        self.register_buffer("l0_initialized", torch.zeros(1, dtype=torch.bool))

    def _uncertainty_forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros(1, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total.squeeze()

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

    def forward(
        self,
        losses: list[torch.Tensor],
        epoch: int = 0,
        shared_params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if epoch < self.transition_epoch:
            return self._uncertainty_forward(losses)
        elif epoch < self.transition_epoch + self.blend_epochs:
            blend_ratio = (epoch - self.transition_epoch) / float(self.blend_epochs)
            L_unc = self._uncertainty_forward(losses)
            L_gn = self._gradnorm_forward(losses, shared_params)
            return (1.0 - blend_ratio) * L_unc + blend_ratio * L_gn
        else:
            return self._gradnorm_forward(losses, shared_params)

    def get_weights(self) -> dict[str, torch.Tensor]:
        unc_w = torch.exp(-self.log_vars).detach()
        gn_w = self.task_weights.clamp(min=1e-3).detach()
        gn_w = gn_w * self.num_tasks / gn_w.sum()
        return {"uncertainty": unc_w, "gradnorm": gn_w}

    def get_phase(self, epoch: int) -> str:
        if epoch < self.transition_epoch:
            return "uncertainty"
        elif epoch < self.transition_epoch + self.blend_epochs:
            ratio = (epoch - self.transition_epoch) / float(self.blend_epochs)
            return f"blend({ratio:.2f})"
        else:
            return "gradnorm"
