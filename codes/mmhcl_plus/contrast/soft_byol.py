import torch
import torch.nn.functional as F

def soft_byol_alignment(online, target, soft_weights=None):
    online = F.normalize(online, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    sim = (online * target).sum(dim=-1)
    loss = 2.0 - 2.0 * sim
    if soft_weights is not None:
        loss = loss * soft_weights
    return loss.mean()
