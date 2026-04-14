import torch.nn as nn

from mmhcl_plus.config import BARLOW_PROJ_DIM


class ExpandedProjector(nn.Module):
    def __init__(
        self, in_dim: int = 64, hidden_dim: int = 2048, out_dim: int = BARLOW_PROJ_DIM
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
