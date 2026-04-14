import torch


def dense_to_sparse_coo(x: torch.Tensor):
    idx = x.nonzero(as_tuple=False).T
    val = x[idx[0], idx[1]]
    return torch.sparse_coo_tensor(idx, val, x.shape, device=x.device).coalesce()


def build_identity_minus_theta_block(theta: torch.Tensor, batch_idx: torch.Tensor):
    theta_block = theta.index_select(0, batch_idx).index_select(1, batch_idx)
    eye = torch.eye(theta_block.size(0), device=theta.device, dtype=theta.dtype)
    block = eye - theta_block
    return dense_to_sparse_coo(block)
