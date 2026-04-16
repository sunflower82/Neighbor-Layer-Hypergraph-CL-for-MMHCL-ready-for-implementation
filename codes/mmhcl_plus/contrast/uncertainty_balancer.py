"""
Homoscedastic Task Uncertainty Weighting (Kendall et al., CVPR 2018).

Replaces GradNorm from Rev44: no second-order gradients required.
Balances 6 loss objectives via learnable log-variance scalars.

The total loss is:
    L_total = sum_i ( 1/(2*sigma_i^2) * L_i + log(1 + sigma_i) )

This formulation automatically down-weights noisy tasks, converges
faster than GradNorm, and executes in a single backward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyLossBalancer(nn.Module):
    """
    Homoscedastic Task Uncertainty Weighting.

    Parameters
    ----------
    num_tasks : int
        Number of loss components (default: 5 for Rev5.2).
    """

    def __init__(self, num_tasks: int = 5) -> None:
        super().__init__()
        # log(sigma^2) parameterization for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: List of scalar loss tensors, length == num_tasks.
        Returns:
            Weighted total loss scalar.
        """
        total_loss: torch.Tensor = torch.zeros(1, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total_loss.squeeze()

    def get_weights(self) -> torch.Tensor:
        """Return current precision weights (1/sigma^2) for logging."""
        return torch.exp(-self.log_vars).detach()
