import torch.nn as nn

# TEX §4.4: expanded projector output dimension.
# Must match BARLOW_PROJ_DIM in mmhcl_plus/contrast/losses.py.
# Declared here as a module-level constant so that callers (parser.py,
# config.py, train_mmhcl_plus.py) can reference it without hard-coding 8192.
BARLOW_PROJ_DIM: int = 8192


class ExpandedProjector(nn.Module):
    def __init__(self, in_dim: int = 64, hidden_dim: int = 2048, out_dim: int = BARLOW_PROJ_DIM):
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
