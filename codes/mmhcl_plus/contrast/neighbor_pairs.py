def build_neighbor_layer_pairs(layer_embs, max_hops=None):
    pairs = []
    upper = len(layer_embs) - 1 if max_hops is None else min(len(layer_embs) - 1, max_hops)
    for l in range(upper):
        online = layer_embs[l]
        target = layer_embs[l + 1].detach()
        pairs.append((online, target))
    return pairs
