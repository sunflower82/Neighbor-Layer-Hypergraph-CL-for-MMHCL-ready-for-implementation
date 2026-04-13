import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, hyper_x, bip_x):
        cat = torch.cat([hyper_x, bip_x], dim=-1)
        gate = self.gate(cat)
        fused = gate * hyper_x + (1.0 - gate) * bip_x
        return self.out(torch.cat([fused, cat[..., :hyper_x.size(-1)]], dim=-1))
