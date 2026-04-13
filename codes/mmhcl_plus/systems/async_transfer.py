def to_device_async(batch_cpu, device):
    batch_gpu = {}
    for k, v in batch_cpu.items():
        if hasattr(v, 'pin_memory'):
            v = v.pin_memory()
        batch_gpu[k] = v.to(device, non_blocking=True) if hasattr(v, 'to') else v
    return batch_gpu
