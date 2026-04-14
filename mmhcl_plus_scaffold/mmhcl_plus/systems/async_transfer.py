def to_device_async(batch_cpu, device):
    """
    Move a batch dict to ``device`` with non-blocking copies.

    Callers that build batches via ``DataLoader(..., pin_memory=True)`` should
    not call ``pin_memory()`` again here (MMHCL+ Optimization Report — Opt. 4/5).
    """
    batch_gpu = {}
    for k, v in batch_cpu.items():
        if hasattr(v, "to"):
            batch_gpu[k] = v.to(device, non_blocking=True)
        else:
            batch_gpu[k] = v
    return batch_gpu
