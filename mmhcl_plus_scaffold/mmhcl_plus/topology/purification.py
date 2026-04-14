import torch


def sampled_jaccard(a_idx, b_idx):
    a, b = set(a_idx.tolist()), set(b_idx.tolist())
    inter = len(a & b)
    union = max(len(a | b), 1)
    return inter / union


def percentile_soft_weight(scores: torch.Tensor, q: float = 0.8):
    threshold = torch.quantile(scores, q)
    scale = torch.clamp(scores - threshold, min=0.0)
    denom = scale.max().clamp_min(1e-8)
    return scale / denom


def popularity_penalty(bridge_score, eps=1e-8):
    return 1.0 / (1.0 + bridge_score + eps)


def soft_topology_weight(sim, percentile_score, jaccard_score, bridge_score, tau=0.2):
    raw = percentile_score * sim * jaccard_score * popularity_penalty(bridge_score)
    if raw.dim() == 1:
        return torch.softmax(raw / tau, dim=0)
    return torch.softmax(raw / tau, dim=-1)
