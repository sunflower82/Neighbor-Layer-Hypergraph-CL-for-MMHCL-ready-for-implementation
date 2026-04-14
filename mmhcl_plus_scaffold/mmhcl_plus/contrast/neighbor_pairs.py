def build_neighbor_layer_pairs(layer_embs, max_hops=None):
    pairs = []
    upper = (
        len(layer_embs) - 1 if max_hops is None else min(len(layer_embs) - 1, max_hops)
    )
    for layer_idx in range(upper):
        online = layer_embs[layer_idx]
        target = layer_embs[layer_idx + 1].detach()
        pairs.append((online, target))
    return pairs
