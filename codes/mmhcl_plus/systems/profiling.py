import time
import torch

def profile_step(fn, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_mb = 0.0
    elapsed = time.time() - start
    return out, {'peak_vram_mb': peak_mb, 'elapsed_sec': elapsed}
